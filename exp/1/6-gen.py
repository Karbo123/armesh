"""generate meshes"""

import json
import uuid
import hydra
import os, sys
import os.path as osp
from retrying import retry
from tqdm import tqdm, trange
from functools import partial
from omegaconf import OmegaConf
from contextlib import contextmanager
from libpsc import PSC, export_visual_mesh_from_psc, write_psc

import torch
import torch.nn as nn
import pytorch_lightning as pl

# these warnings are annoying
import warnings
from numba.core.errors import NumbaTypeSafetyWarning

# so disable them
warnings.filterwarnings("ignore", category=FutureWarning, module="flash_attn")
warnings.filterwarnings("ignore", category=NumbaTypeSafetyWarning)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if True:
    FOLDER_PY = osp.abspath(
        osp.dirname(__file__)
    )  # the folder of experimental python files
    ROOT = osp.abspath(osp.join(FOLDER_PY, "../.."))  # the very root of this project

    # import everything
    sys.path.append(FOLDER_PY)
    sys.path.append(ROOT)


from exp_utils import (
    constrained_decoding,
    is_appending_valid,
    Status,
    CausalTransformerDecoder,
)
from codebase.bpe import BPE
from codebase.tokenize import decode_op2


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@contextmanager
def cuda_timer():
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    yield
    end.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start.elapsed_time(end)
    print(f"[Timer] {elapsed_time_ms:.3f} ms")


@retry(
    stop_max_attempt_number=None,
    wait_fixed=100,
    retry_on_exception=lambda x: isinstance(x, TimeoutError),
)
@torch.inference_mode()
def decode_loop(
    gen_fn,  # the network function: seq (list) --> next token logit (list)
    bpe,
    #
    # diversity
    k_p_t=(512, 0.95, 1),
    top_ratio_candidate=0.01,
    top_ratio_termination=0.001,
    #
    # termination
    max_splits=100,  # max number of vertex splitting operations
    max_length=100,  # max number of tokens per operation
    timeout=30,  # the timeout (sec) for one operation decoding
):

    psc = PSC([0, 0, 0])
    op_lst = []

    seq_prev = [bpe.vocab_size]  # SOS
    for _ in trange(max_splits):

        def _gen_fn(seq):
            return gen_fn(seq_prev + seq)

        seq, status = constrained_decoding(
            _gen_fn,
            partial(is_appending_valid, bpe=bpe, psc=psc),
            max_len=max_length,
            k_p_t=k_p_t,
            top_ratio_candidate=top_ratio_candidate,
            top_ratio_termination=top_ratio_termination,
            timeout=timeout,
        )

        if status == Status.FOUND:
            op = decode_op2(bpe.decode(seq))
            op["vtid"] = psc.num(0)  # the number of verts
            op_lst.append(op)

            # apply split
            psc.gvspl(
                op["vsid"], op["vtid"], op["code"], op["position_bit"], op["delta_p"]
            )
            seq_prev += seq

        else:
            assert status == Status.TERMINATE
            break

    return psc, op_lst


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def read_manyline_jsonl(file_path, category=None):
    """read the dataset (jsonl with many lines)"""
    merged = {}
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    if isinstance(category, str) and (category not in data["name"]):
                        continue
                    if callable(category) and (not category(data["name"])):
                        continue
                    # because we only need the keys
                    merged[data["name"]] = None
                except json.JSONDecodeError:
                    continue
    return merged


def to_category_id(category_english_name):
    """convert shapenet english name to id string
    https://gist.github.com/tejaskhot/15ae62827d6e43b91a4b0c5c850c168e
    """
    mapping = {
        "bench": "02828884",
        "lamp": "03636649",
        "chair": "03001627",
        "table": "04379243",
    }
    if (category_english_name is None) or category_english_name == "all":  # use all
        return None
    if category_english_name.isdigit():  # already a category id
        assert category_english_name in mapping.values()
        return category_english_name
    if "lambda" in category_english_name:
        return eval(category_english_name)
    return mapping[category_english_name]  # convert to a category id


def remove_longest_repeated_substring(s: str, min_length: int):
    def build_suffix_array(s):
        return sorted(range(len(s)), key=lambda i: s[i:])

    def build_lcp_array(s, sa):
        n = len(s)
        rank = [0] * n
        for i, idx in enumerate(sa):
            rank[idx] = i

        h = 0
        lcp = [0] * (n - 1)
        for i in range(n):
            if rank[i] == 0:
                continue
            j = sa[rank[i] - 1]
            while i + h < n and j + h < n and s[i + h] == s[j + h]:
                h += 1
            lcp[rank[i] - 1] = h
            if h > 0:
                h -= 1
        return lcp

    n = len(s)
    if n < min_length:
        return "", s

    sa = build_suffix_array(s)
    lcp = build_lcp_array(s, sa)

    max_len = 0
    substr_start = -1
    for i, l in enumerate(lcp):
        if l >= min_length and l > max_len:
            max_len = l
            substr_start = sa[i]

    if max_len == 0:
        return "", s

    a = s[substr_start : substr_start + max_len]
    if a.startswith("~~"):
        a = a[2:]
    if a.endswith("~~"):
        a = a[:-2]
    max_len = len(a)

    result = []
    i = 0
    while i < len(s):
        if s[i : i + max_len] == a:
            i += max_len
        else:
            result.append(s[i])
            i += 1

    return a, "".join(result)


@hydra.main(version_base=None, config_path=FOLDER_PY, config_name="cfg")
def main_fn(cfg):
    assert (
        cfg.get("idx_obj_lst") is not None
    ), "you must specify the object index for generation"

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    bpe_file = osp.join(ROOT, cfg.folder_dataset, "op_tok_bpe.jsonl")
    bpe_tokens = read_manyline_jsonl(bpe_file, category=to_category_id(cfg.category))
    bpe_tokens_keys = list(bpe_tokens.keys())
    name_lst = bpe_tokens_keys[:: cfg.num_variants_per_shape]
    num_obj = len(name_lst)
    print("dataset has been loaded!")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # the BPE
    bpe = BPE.load(osp.join(ROOT, cfg.folder_dataset, "vocab.json"), with_dec=True)
    assert bpe.vocab_size == cfg.vocab_total - 2
    vocab_total = cfg.vocab_total  # with: SOS, EOS
    print("bpe vocab has been loaded!")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # the minimal model
    class Model(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = CausalTransformerDecoder(
                d_model=cfg.d_model, n_layers=cfg.n_layers, num_classes=vocab_total
            )
            self.embedding_self = nn.Embedding(vocab_total, embedding_dim=cfg.d_model)
            self.embedding_cross = nn.Embedding(
                num_obj, embedding_dim=cfg.num_cross_tokens * cfg.d_model
            )
            self.empty_token = nn.Parameter(
                torch.randn(cfg.num_cross_tokens * cfg.d_model)
            )
            self.norm_cond = nn.RMSNorm(cfg.d_model)

    # load the model
    model = Model().cuda()
    checkpoint_folder = cfg.get("checkpoint_folder", "checkpoint")
    model_checkpoint_path = osp.join(
        cfg.resume, f"{checkpoint_folder}/mp_rank_00_model_states.pt"
    )
    print(f"load model from {model_checkpoint_path}")
    checkpoint = torch.load(model_checkpoint_path, map_location="cuda")
    model.load_state_dict(checkpoint["module"])
    model.eval()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    out_folder = cfg.get("out_folder", "gen")
    out_folder = osp.join(
        ROOT, "out", osp.normpath(cfg.resume).split(osp.sep)[1], out_folder
    )
    os.makedirs(out_folder, exist_ok=True)

    def interpolation(feats_1, feats_2, w_1=1.0):
        assert 0 <= w_1 <= 1
        return feats_1 * w_1 + feats_2 * (1 - w_1)

    # try to generate one object
    def process_one_obj(idx_obj):

        # the conditioning tokens for this object
        def _get_feat(item):
            if item is None:  # uncond
                tokens_cond = model.empty_token
            elif isinstance(item, int):  # cond
                tokens_cond = model.embedding_cross(torch.tensor(item, device="cuda"))
            return tokens_cond

        if not isinstance(idx_obj, (list, tuple)):
            tokens_cond = _get_feat(idx_obj)
        else:  # interpolate beween two shapes
            tokens_1 = _get_feat(idx_obj[0])
            tokens_2 = _get_feat(idx_obj[1])
            w_1 = idx_obj[2]
            tokens_cond = interpolation(tokens_1, tokens_2, w_1=w_1)
        tokens_cond = model.norm_cond(tokens_cond.reshape(-1, cfg.d_model))

        @torch.inference_mode()
        def model_fn(
            input_ids,
        ):
            # TODO
            #   we may use prefixed kvcache to speed up
            #   but a little bit complicated

            # do inference
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                x = model.model(
                    x=model.embedding_self(
                        torch.tensor(input_ids, device="cuda").long()
                    ),
                    y=tokens_cond,
                    nums_x=torch.tensor([len(input_ids)], device="cuda"),
                    nums_y=torch.tensor([cfg.num_cross_tokens], device="cuda"),
                )

            # only return the last token
            x = x[-1].float()
            x[-2] = -1e6  # no SOS
            # x[-1] = -1e6  # no EOS
            return x.tolist()  # (vocab_total, )

        # perform constrained decoding
        psc, op_lst = decode_loop(
            model_fn,
            bpe,
            k_p_t=cfg.k_p_t,
            top_ratio_candidate=cfg.top_ratio_candidate,
            top_ratio_termination=cfg.top_ratio_termination,
            max_splits=cfg.max_splits,
            max_length=cfg.max_length,
            timeout=cfg.timeout,
        )

        # export a mesh
        if isinstance(idx_obj, int):
            name = f"idx={idx_obj}~~" + name_lst[idx_obj]
        elif idx_obj is None:
            name = "unconditional--" + uuid.uuid4().hex
        elif isinstance(idx_obj, (list, tuple)):
            names = [None, None]
            for i in range(2):
                idx_obj_i = idx_obj[i]
                if isinstance(idx_obj_i, int):
                    n = f"idx={idx_obj_i}~~" + name_lst[idx_obj_i]
                elif idx_obj_i is None:
                    n = "unconditional"
                names[i] = n
            name = f"s1={names[0]}~~s2={names[1]}~~w1={idx_obj[2]}"
            # too avoid too long file name; otherwise we cannot open file
            longest, name = remove_longest_repeated_substring(name, min_length=10)
            name = f"{longest}~~{name}"

        path_ply = osp.join(out_folder, f"{name}.ply")
        export_visual_mesh_from_psc(path_ply, psc, radius=0.01)
        print(f"write ply mesh to: {path_ply}")

        path_psc = osp.join(out_folder, f"{name}.psc")
        write_psc(path_psc, op_lst, [0, 0, 0])
        print(f"write psc to: {path_psc}")

    # # # # # # # # # # # # # # # # # # # # #
    # only take a subset
    category_sft_fn = to_category_id(cfg.category_sft)
    idx_sub_mapping = dict()
    i = 0
    for j, n in enumerate(name_lst):
        if category_sft_fn(n):
            idx_sub_mapping[i] = j
            i += 1

    # the objects needed to be generated
    idx_obj_lst = cfg.idx_obj_lst
    if isinstance(idx_obj_lst, str):
        idx_obj_lst = eval(idx_obj_lst)

    # for each object
    for idx_obj in tqdm(idx_obj_lst, desc="processing"):
        is_basic_type = isinstance(idx_obj, int) or (idx_obj is None)
        idx_obj = idx_sub_mapping.get(idx_obj, idx_obj)
        if isinstance(idx_obj, int):
            print(f"dealing with {name_lst[idx_obj]}")
        process_one_obj(
            OmegaConf.to_container(idx_obj, resolve=True)
            if not is_basic_type
            else idx_obj
        )


if __name__ == "__main__":
    main_fn()
