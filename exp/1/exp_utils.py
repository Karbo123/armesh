import datrie
import numpy as np
from math import ceil
from enum import Enum
from time import time
from copy import deepcopy
from collections import deque
from typing import Union, List
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn.modules.mha import MHA

# https://github.com/Dao-AILab/flash-attention/issues/1227#issuecomment-2467160236
from flash_attn.ops.fused_dense import FusedMLP

from codebase.bpe import BPE
from codebase.utils import batched_range, start_end, isin_with_index
from codebase.tokenize import decode_int


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class TransformerDecoderLayer(nn.Module):
    """Inspired by: https://github.dev/meta-llama/llama/blob/main/llama/model.py"""

    def __init__(self, d_model=256, layer_idx=None):
        super().__init__()
        self.d_model = d_model
        assert d_model % 64 == 0

        # attention
        self.self_attn = MHA(
            embed_dim=d_model,
            num_heads=d_model // 64,
            qkv_proj_bias=False,
            out_proj_bias=False,
            dropout=0.0, # default is 0.1, but set to 0.0 for chair and table only
            causal=True,
            layer_idx=layer_idx,
            use_alibi=True,  # use relative positional encoding
            fused_bias_fc=True,
            use_flash_attn=True,
        )

        # ffn
        self.ffn = FusedMLP(
            in_features=d_model,
            bias1=False,
            bias2=False,
        )

        # normalization
        self.norm_attn = nn.RMSNorm(d_model)
        self.norm_ffn = nn.RMSNorm(d_model)

    @torch.inference_mode()
    def inference_with_kvcache(self, x, inference_params):
        """inference
        Params:
            "x" can be prefixed with "y"
            x.shape == (bs, seqlen, d_model)
        """
        x = self(x, cu_seqlens=None, max_seqlen=None, inference_params=inference_params)
        return x

    def forward(self, x, cu_seqlens, max_seqlen, inference_params=None):
        """
        Params:
            x: (total, d_model)
            cu_seqlens: (batch_size + 1,)
            max_seqlen: int
        """

        # attention
        x = x + self.self_attn(
            self.norm_attn(x), cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, inference_params=inference_params
        )

        # ffn
        x = x + self.ffn(self.norm_ffn(x))

        return x


class CausalTransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model=512,
        n_layers=6,
        num_classes=8192,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model=d_model, layer_idx=idx) for idx in range(n_layers)]
        )
        self.mapping = nn.Linear(d_model, num_classes)

    @torch.inference_mode()
    def inference_with_kvcache(self, x, inference_params):
        """inference
        Params:
            "x" can be prefixed with "y"
            x.shape == (bs, seqlen, d_model)
        """
        for layer in self.layers:
            x = layer.inference_with_kvcache(x, inference_params)
        x = self.mapping(x)
        return x

    def _forward_transformer(self, x, cu_seqlens, max_seqlen):
        for layer in self.layers:
            x = layer(x, cu_seqlens, max_seqlen)
        return x

    def forward(self, x, y, nums_x, nums_y):
        """
        Params:
            x      (Tensor): (num_total_x, d_model)     stacked input features for auto-regressive generation
            y      (Tensor): (num_total_y, d_model)     stacked input features for conditioning
            nums_x (Tensor): (batch_size, )             sequence length of each batch for generation
            nums_y (Tensor): (batch_size, )             sequence length of each batch for conditioning
        """
        nums_yx = torch.stack([nums_y, nums_x], dim=1)  # (batch_size, 2)
        start_yx, end_yx = start_end(nums_yx.reshape(-1))  # (batch_size * 2, )
        start_y, start_x = start_yx.reshape(-1, 2).unbind(dim=1)  # (batch_size, )   each
        end_y, end_x = end_yx.reshape(-1, 2).unbind(dim=1)  # (batch_size, )   each

        range_y = batched_range(start_y, end_y)
        range_x = batched_range(start_x, end_x)

        num_total_yx = len(y) + len(x)
        inputs = x.new_empty([num_total_yx, x.shape[1]])
        inputs[range_y] = y
        inputs[range_x] = x

        # forward
        lengths = nums_yx.sum(dim=1)  # length of each batch
        outputs = self._forward_transformer(
            inputs,
            cu_seqlens=F.pad(lengths.cumsum(dim=0), (1, 0)).to(torch.int32),
            max_seqlen=lengths.max().item(),
        )

        # only care about the auto-regressive sequence
        x = outputs[range_x]  # (num_total_x, d_model)
        x = self.mapping(x)
        return x


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def top_k_top_p_sampling(
    logits: torch.Tensor,
    top_k: int = 512,
    top_p: float = 0.95,
    temperature: float = 1.0,
) -> torch.Tensor:
    if temperature != 1.0:
        logits = logits / temperature

    probs = torch.softmax(logits, dim=-1)
    vocab_size = logits.size(0)

    # --- Top-k filter ---
    if top_k > 0:
        top_k = min(top_k, vocab_size)
        _, top_k_indices = torch.topk(probs, k=top_k)
        mask_k = torch.zeros_like(probs, dtype=torch.bool)
        mask_k[top_k_indices] = True
    else:
        mask_k = torch.ones_like(probs, dtype=torch.bool)

    # --- Top-p filter ---
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=0)
        cutoff = (cum_probs > top_p).nonzero(as_tuple=True)[0]
        if cutoff.numel() > 0:
            cutoff_idx = cutoff[0].item()
            keep_indices = sorted_indices[: cutoff_idx + 1]
        else:
            keep_indices = sorted_indices
        mask_p = torch.zeros_like(probs, dtype=torch.bool)
        mask_p[keep_indices] = True
    else:
        mask_p = torch.ones_like(probs, dtype=torch.bool)

    # --- Combined top-k & top-p ---
    mask = mask_k & mask_p
    candidate_indices = mask.nonzero(as_tuple=True)[0]
    filtered_probs = probs[candidate_indices]
    filtered_probs = filtered_probs / filtered_probs.sum()

    max_samples = candidate_indices.numel()
    if max_samples > 0:
        sampled = torch.multinomial(filtered_probs, num_samples=max_samples, replacement=False)
        sampled_indices = candidate_indices[sampled]
    else:
        sampled_indices = torch.empty(0, dtype=torch.long)

    # --- Get remaining indices sorted by logits ---
    all_indices = torch.arange(vocab_size)
    mask_all = torch.ones(vocab_size, dtype=torch.bool)
    mask_all[sampled_indices] = False
    remaining_indices = all_indices[mask_all]
    remaining_indices = remaining_indices[logits[remaining_indices].argsort(descending=True)]

    # --- Final permutation ---
    final_indices = torch.cat([sampled_indices, remaining_indices], dim=0)

    # --- Sanity check ---
    assert final_indices.numel() == vocab_size
    assert final_indices.unique().numel() == vocab_size
    assert (final_indices.unique() == torch.arange(len(logits))).all()
    return final_indices


def constrained_decoding(
    gen_fn,
    is_valid,
    max_len=100,
    k_p_t=(512, 0.95, 1.0),
    top_ratio_candidate=0.01,
    top_ratio_termination=0.001,
    timeout=30,
):
    """constrained decoding by DFS tree traversal"""
    t0 = time()

    def seq2key(seq):
        return "|".join(map(str, seq)) + "|"

    # first try with some top-k candidates
    # if still cannot find, try all candidates
    for ratio in (top_ratio_candidate, 1.0):
        stack = deque([([], InfoSeq())])
        has_tried = datrie.BaseTrie(alphabet="0123456789|")

        while stack:
            time_used = time() - t0
            if time_used > timeout:
                print(f"timeout: took {time_used:.2f} (>{timeout:.2f}) seconds, retrying")
                raise TimeoutError

            seq, cache = stack.pop()

            logits = gen_fn(seq)
            topk_indices = top_k_top_p_sampling(
                torch.tensor(logits),
                top_k=k_p_t[0],
                top_p=k_p_t[1],
                temperature=k_p_t[2],
            ).tolist()
            topk_indices = topk_indices[: ceil(ratio * len(logits))]

            # a single thread
            lst_new_seq = []
            for i, tok in enumerate(topk_indices):
                new_seq = seq + [tok]
                key = seq2key(new_seq)

                # 不能以 不可行的 为前缀
                if len(has_tried.prefixes(key)) > 0:
                    continue

                # 只需检查 new token
                result, cache_new = is_valid(seq, cache, tok)

                if result == Status.FOUND:
                    # 最终 找到了 一个完整的操作序列
                    return new_seq, result

                elif result == Status.TERMINATE:
                    if i <= int(top_ratio_termination * len(topk_indices)) and len(lst_new_seq) == 0:
                        # 位于前几个 并且 尚未存在其他候选  那么就停止生成
                        return new_seq, result

                elif result == Status.NOT_VALID:
                    # 以它为前缀不可行
                    has_tried[key] = 1
                    continue

                # 合法路径，继续扩展
                elif result == Status.VALID and len(new_seq) <= max_len:
                    lst_new_seq.append((new_seq, cache_new))

            # 拓展不了 被迫停止
            if len(lst_new_seq) == 0 and ratio == 1.0:
                return new_seq, Status.TERMINATE

            # 追加到 栈
            stack.extend(reversed(lst_new_seq))


class Status(Enum):
    NOT_VALID = 0  # the appending is NOT consistent
    VALID = 1  # the appending is consistent
    FOUND = 2  # the appending is the end of the operation
    TERMINATE = 3  # the appending is the end of ALL the operation


class Stage(Enum):
    VSID = 0  # vsid
    OFFSET = 1  # offset
    CODE_V = 2  # code - v
    CODE_F = 3  # code - f
    CODE_E = 4  # code - e
    BIT = 5  # bit


@dataclass
class InfoSeq:  # dynamic cache for tree search
    decoded: List[int] = field(default_factory=list)
    stage: Stage = Stage.VSID  # the stage of the new element
    idx_v: Union[int, None] = None
    incid_ef: Union[np.ndarray, None] = None


def is_appending_valid(seq: List[int], info_seq: InfoSeq, new_elem: int, bpe: BPE, psc):
    """to see if appending the new element to the sequence is valid"""
    id_SOS = bpe.vocab_size
    id_EOS = bpe.vocab_size + 1

    # reach special tokens
    if new_elem in (id_SOS, id_EOS):
        if len(seq) == 0 and new_elem == id_EOS:  # EOS
            return Status.TERMINATE, info_seq
        return Status.NOT_VALID, info_seq

    # see if one decoded token is valid
    def is_valid_one(seq_prev, info_seq, idx, id_tok):
        """see if the decoded new token is valid"""

        def add_new(info_seq):
            info_seq.decoded += [id_tok]
            return info_seq

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # check VSID
        if info_seq.stage == Stage.VSID:
            assert idx <= 1

            if not (0 <= id_tok <= 255):
                return Status.NOT_VALID, info_seq

            if idx == 0:
                if decode_int([id_tok, 0]) >= psc.num(0):  # high-bit is too large
                    return Status.NOT_VALID, info_seq

            else:
                # see if the vertex idx falls in range
                idx_v = decode_int(seq_prev + [id_tok])
                if idx_v >= psc.num(0):
                    return Status.NOT_VALID, info_seq
                info_seq.idx_v = idx_v
                info_seq.stage = Stage.OFFSET

            return Status.VALID, add_new(info_seq)

        assert idx >= 2

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        if info_seq.stage == Stage.OFFSET:

            if not (256 <= id_tok <= 511):
                return Status.NOT_VALID, info_seq

            # move to next stage
            if idx == 7:
                info_seq.stage = Stage.CODE_V

            return Status.VALID, add_new(info_seq)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        if info_seq.stage == Stage.CODE_V:
            assert idx >= 8

            if not (512 <= id_tok <= 513):
                return Status.NOT_VALID, info_seq

            # retrieve the neighborhood
            # this is not fast, so we postpone it to here
            edges, faces = map(torch.tensor, psc.write()[1:])
            edges_local, faces_local = psc.get_star(info_seq.idx_v)
            info_seq.incid_ef = get_incid_ef(edges, faces, edges_local, faces_local)

            # move to next stage
            info_seq.stage = Stage.CODE_F

            return Status.VALID, add_new(info_seq)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        if info_seq.stage == Stage.CODE_F:
            assert idx >= 9

            # move to next stage
            if info_seq.incid_ef.shape[1] == 0:
                info_seq.stage = Stage.CODE_E

            else:
                if not (514 <= id_tok <= 516):
                    return Status.NOT_VALID, info_seq

                # move to next stage
                num_f = info_seq.incid_ef.shape[1]
                if idx == 8 + num_f:
                    info_seq.stage = Stage.CODE_E

                return Status.VALID, add_new(info_seq)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        if info_seq.stage == Stage.CODE_E:

            # move to next stage
            if info_seq.incid_ef.shape[0] == 0:
                info_seq.stage = Stage.BIT

            else:
                if not (517 <= id_tok <= 520):
                    return Status.NOT_VALID, info_seq

                # constraint:   v=0  ==>  e!=3
                if info_seq.decoded[8] == 512 and id_tok == 520:
                    return Status.NOT_VALID, info_seq

                # constraints from faces
                num_f = info_seq.incid_ef.shape[1]
                for idx_f in range(9, 9 + num_f):
                    e_local_lst = info_seq.incid_ef[:, idx_f - 9]
                    assert sum(e_local_lst) == 2
                    e1e2 = e_local_lst.nonzero().squeeze().tolist()
                    assert len(e1e2) == 2
                    idx_e_lst = [(9 + num_f + e) for e in e1e2]
                    idx_e_lst = [e for e in idx_e_lst if e == idx]
                    if len(idx_e_lst) == 0:
                        continue

                    # constraints
                    if info_seq.decoded[idx_f] == 514 and id_tok == 518:
                        return Status.NOT_VALID, info_seq
                    elif info_seq.decoded[idx_f] == 515 and id_tok == 517:
                        return Status.NOT_VALID, info_seq
                    elif info_seq.decoded[idx_f] == 516 and 517 <= id_tok <= 518:
                        return Status.NOT_VALID, info_seq

                # move to next stage
                num_e = info_seq.incid_ef.shape[0]
                if idx == 8 + num_f + num_e:
                    info_seq.stage = Stage.BIT

                return Status.VALID, add_new(info_seq)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        if info_seq.stage == Stage.BIT:
            if not (521 <= id_tok <= 522):
                return Status.NOT_VALID, info_seq

            return Status.FOUND, add_new(info_seq)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # decode the token
    #   may contain many decoded tokens
    decoded_new_lst = bpe.decode_one(new_elem)

    # for each decoded token, see if it is valid
    info_seq_new = deepcopy(info_seq)
    for idx, decoded_new in enumerate(decoded_new_lst):
        result, info_seq_new = is_valid_one(
            info_seq.decoded + decoded_new_lst[:idx], info_seq_new, len(info_seq.decoded) + idx, decoded_new
        )
        # early stop
        if result != Status.VALID:
            return result, info_seq_new  # NOT_VALID / FOUND / TERMINATE

    return Status.VALID, info_seq_new


def get_incid_ef(edges, faces, edges_local, faces_local):
    def compute_incid_ef(edges, faces, index_only=False):
        """compute "incid_ef" -- incidence matrix (edge to face) of the splitting points' neighborhood
            this is very useful in imposing topology classification rules
        Params:
            edges (Tensor) : (num_local_e, 2)  integer   local edges from all batches
            faces (Tensor) : (num_local_f, 3)  integer   local faces from all batches
        """
        assert edges.dtype == torch.long and faces.dtype == torch.long
        if len(edges) == 0 or len(faces) == 0:
            if index_only:
                return torch.tensor([]).long(), torch.tensor([]).long()
            return torch.zeros([len(edges), len(faces)], dtype=torch.long)

        # sort
        e = edges.sort(dim=1).values
        f = faces.sort(dim=1).values
        e3 = torch.cat(
            [f[:, [0, 1]], f[:, [1, 2]], f[:, [0, 2]]],
            dim=0,
        )

        # flatten to 1d
        val_max = max(edges.max(), faces.max()) + 1
        to_1d = lambda x: x[:, 0] + val_max * x[:, 1]
        e_1d, e3_1d = to_1d(e), to_1d(e3)  # (num_local_e, )    (3 * num_local_f, )

        # a face has at most three border edges
        idx_e = isin_with_index(e3_1d, e_1d).reshape(3, -1)  # (3, num_local_f)
        i, i_f = torch.where(idx_e != torch.iinfo(torch.long).max)
        i_e = idx_e[i, i_f]

        # incidence matrix
        if index_only:
            return i_e, i_f
        incid_ef = torch.zeros([len(edges), len(faces)], dtype=torch.long)
        incid_ef[i_e, i_f] = 1
        return incid_ef

    def get_index(simplex_local, simplex_all):
        """get simplex index"""
        if len(simplex_local) == 0:
            return np.asarray([])
        idx = (
            (np.asarray(simplex_local)[:, None, :] == np.sort(simplex_all, axis=1)[None, :, :]).all(axis=2).nonzero()
        )  # (num_local, num_all)
        assert (idx[0] == np.arange(len(idx[0]))).all()  # assume "simplex_all" is unique, contains only one
        return idx[1]

    idx_local_e = get_index(edges_local, edges.cpu().numpy())
    idx_local_f = get_index(faces_local, faces.cpu().numpy())

    incid_ef = compute_incid_ef(
        edges[idx_local_e].long(),
        faces[idx_local_f].long(),
    )

    return incid_ef
