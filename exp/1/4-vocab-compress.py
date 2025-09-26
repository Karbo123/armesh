"""merge three BPEs, and apply BPE compression"""

import os
import sys
import json
import struct
import hashlib
import numpy as np
import os.path as osp
from tqdm import tqdm
from glob import glob
from fire import Fire
from copy import deepcopy
from shutil import rmtree
from libpsc import load_psc
from joblib import Parallel, delayed


ROOT = osp.abspath(osp.join(osp.dirname(__file__), "../.."))
sys.path.append(ROOT)
from codebase.tokenize import encode_op2
from codebase.bpe import BPE


def clamp(x, min_val=-2, max_val=2):
    """floating point number of an offset vector cannot be too large"""
    return float(max(min_val, min(x, max_val)))


def random_perturb_op(op, idx, minmax=[0.9, 1.1]):
    """randomly scale the offset vector, deterministically based on "op" and "idx" """
    op_new = deepcopy(op)
    hash_digest = hashlib.md5(f"{idx}{op_new}{idx**2}".encode("utf-8")).digest()
    seed_raw = struct.unpack("Q", hash_digest[:8])[0]
    seed = (seed_raw + idx) % (2**32)
    random_number = np.random.RandomState(seed).rand()
    scale = random_number * (minmax[1] - minmax[0]) + minmax[0]
    op_new["delta_p"] = [clamp(x * scale) for x in op_new["delta_p"]]
    return op_new


def worker(folder, folder_jsonl, name, bpe, num_aug=0):
    """a single worker to process one object only"""
    path_jsonl = osp.join(folder_jsonl, f"{name}.jsonl")
    if osp.exists(path_jsonl) and osp.getsize(path_jsonl) >= 1024:  # at least: 1KB
        print(f"exist, skip: {path_jsonl}")
        return

    # NOTE : we directly load from psc file because it can be done in parallel
    op_lst = load_psc(osp.join(folder, f"{name}.psc"))["vsplits"]
    if num_aug >= 1:
        op_lst_original = deepcopy(op_lst)
    op_lst = [encode_op2(op) for op in op_lst]
    op_lst = bpe.encode(op_lst)
    with open(path_jsonl, "w") as f:
        json.dump({"name": name, "op_lst": op_lst}, f)

    # we may randomly and slightly scale the offset vector
    if num_aug >= 1:
        with open(path_jsonl, "a") as f:
            for idx_aug in range(num_aug):
                op_lst = [
                    encode_op2(random_perturb_op(op, idx=idx_aug))
                    for op in op_lst_original
                ]
                op_lst = bpe.encode(op_lst)
                f.write("\n")
                json.dump({"name": f"{name}--aug{idx_aug}", "op_lst": op_lst}, f)


def main_fn(
    folder="assets/obj100_out", num_workers=8, num_aug=3, expected_vocab_size=16382
):
    folder = osp.join(ROOT, folder)

    # load all info
    with open(osp.join(folder, "object_info.json"), "r") as f:
        object_info = json.load(f)

    print(f"need to process {len(object_info)} objects")

    # load sub-BPEs
    def load_sub_vocab(name):
        path_vocab_sub = osp.join(folder, f"vocab-{name}.json")
        with open(path_vocab_sub, "r") as f:
            return json.load(f)

    # merge sub-BPEs
    d_vocab_0 = load_sub_vocab("bit-high")
    d_vocab_1 = load_sub_vocab("code-vfe")
    d_vocab_01 = dict(
        vocab_size=d_vocab_1["merges"][-1][2] + 1,
        merges=d_vocab_0["merges"] + d_vocab_1["merges"],
        id2token={**d_vocab_0["id2token"], **d_vocab_1["id2token"]},
    )
    if expected_vocab_size is not None:
        assert d_vocab_01["vocab_size"] == expected_vocab_size
        assert len(d_vocab_01["merges"]) == expected_vocab_size - 523
    path_vocab = osp.join(folder, "vocab.json")
    with open(path_vocab, "w") as f:
        json.dump(d_vocab_01, f)

    # load bpe
    assert osp.exists(path_vocab)
    bpe = BPE.load(path_vocab)

    # tmp folder
    folder_jsonl = osp.join(folder, "tmp-jsonl-files")
    os.makedirs(folder_jsonl, exist_ok=True)

    # for each object
    if num_workers > 1:
        Parallel(n_jobs=num_workers, backend="loky")(
            delayed(worker)(folder, folder_jsonl, name, bpe, num_aug)
            for name in tqdm(object_info.keys(), desc="tokenizing")
        )
    else:
        for name in tqdm(object_info.keys(), desc="tokenizing"):
            worker(folder, folder_jsonl, name, bpe, num_aug)

    # merge jsonl files
    path_jsonl = osp.join(folder, "op_tok_bpe.jsonl")
    with open(path_jsonl, "w") as outfile:
        for file in tqdm(glob(osp.join(folder_jsonl, "*.jsonl")), desc="merging jsonl"):
            with open(file, "r") as infile:
                outfile.write(infile.read() + "\n")
    print(f"write BPE-tokenized jsonl file to: {path_jsonl}")

    # remove tmp folder
    rmtree(folder_jsonl)


if __name__ == "__main__":
    Fire(main_fn)
