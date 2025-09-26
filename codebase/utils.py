"""some very useful utilities"""

import re
import math
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Sampler

from pytorch_lightning.utilities import rank_zero_only


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def extract_fn(query, pattern):
    return re.search(pattern, query).groups()


def to_list_of_type_fn(t, lst):
    return list(map(t, lst))


SPACE = r"\s+"
PART = rf"{SPACE}(\S+)"

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def load_psc(file_path: str):
    """load psc file"""
    out = dict(
        center=[0, 0, 0],  # the starting point location
        vsplits=[],  # the operation sequence to split vertex
    )

    with open(file_path, "r") as f:
        lines_all = f.readlines()

    idx_line, simplex_loaded = 0, False
    while idx_line < len(lines_all):
        line = lines_all[idx_line].strip()

        if any(line.startswith(c) for c in ("#", "[", "m")):  # not important line
            idx_line += 1

        elif line.startswith("Simplex"):  # starting point location
            out["center"] = to_list_of_type_fn(
                float,
                extract_fn(line, rf"Simplex{SPACE}\S+{SPACE}\S+{PART}{PART}{PART}"),
            )
            simplex_loaded = True
            idx_line += 1

        elif simplex_loaded:  # load one vsplit info
            # vertex id: source, target
            vsid, vtid = to_list_of_type_fn(
                int, extract_fn(lines_all[idx_line + 0].strip(), rf"(\S+){PART}")
            )

            # a string code
            code = lines_all[idx_line + 1].strip()

            # position bit: (see "SplitRecord::applySplit")
            #   always set "vt" as "vs_p + delta_p"
            #   0 ==> midpoint ==> also set "vs_p" as "vs_p - delta_p"
            #   1 ==>          ==> keep     "vs_p" unchanged
            # position offset
            position_bit, *delta_p = to_list_of_type_fn(
                float,
                extract_fn(
                    lines_all[idx_line + 2].strip(), rf"(\S+){PART}{PART}{PART}"
                ),
            )
            position_bit = int(position_bit)

            # wait until meeting two "-1"
            offset, cnt_neg1 = 0, 0
            while cnt_neg1 < 2:
                line = lines_all[idx_line + 3 + offset].strip()
                if line == "-1":
                    cnt_neg1 += 1
                offset += 1

            # save results
            out["vsplits"].append(
                dict(
                    vsid=vsid - 1,  # >= 0
                    vtid=vtid - 1,  # >= 0
                    code=code,
                    position_bit=position_bit,
                    delta_p=delta_p,
                )
            )

            idx_line += 3 + offset

    return out


def load_sc(file_path: str):
    """load sc file
    Returns:
        pack (list of ndarray):
            dim ==> ndarray
             0  ==> vertex coordinates
             1  ==> edges: relative to points
             2  ==> faces: relative to points
    """
    pack = [[], [], []]

    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("Simplex 0"):
                pack[0].append(
                    to_list_of_type_fn(
                        float,
                        extract_fn(
                            line, rf"Simplex{SPACE}0{SPACE}\S+{PART}{PART}{PART}"
                        ),
                    )
                )
            elif line.startswith("Simplex 1"):
                pack[1].append(
                    to_list_of_type_fn(
                        int, extract_fn(line, rf"Simplex{SPACE}1{SPACE}\S+{PART}{PART}")
                    )
                )
            elif line.startswith("Simplex 2"):
                pack[2].append(
                    to_list_of_type_fn(
                        int,
                        extract_fn(
                            line, rf"Simplex{SPACE}2{SPACE}\S+{PART}{PART}{PART}"
                        ),
                    )
                )

    # NOTE : indices are sorted along the row
    pack[0] = np.asarray(pack[0])
    pack[1] = np.asarray(pack[1]) - 1  # >=0
    pack[2] = np.asarray(pack[2]) - 1  # >=0
    pack[2] = pack_to_faces(pack)  # relative to verts

    return pack


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def pack_to_faces(pack):
    """convert the "line index" to "vert index"
    i.e., triangle mesh's "faces" variable
    """
    if len(pack[2]) == 0:  # no faces
        return np.zeros([0, 3]).astype(np.int64)
    faces = np.asarray(
        [np.unique(f) for f in pack[1][pack[2]].reshape(-1, 6)]
    )  # (num_faces, 3)
    return faces


def compute_taken_mask(pack):
    """compute mask of used vertices/edges"""
    # see if verts have been used by edges/faces
    v_taken = np.unique(
        np.concatenate([pack[1].reshape(-1), pack[2].reshape(-1)], axis=0)
    )
    msk_v = np.isin(np.arange(len(pack[0])), v_taken)

    # see if edges have been used by faces
    msk_e = np.zeros([len(pack[1])], dtype=np.bool)
    if len(pack[2]) >= 1:
        e_taken = np.unique(
            np.concatenate(
                [pack[2][:, [0, 1]], pack[2][:, [1, 2]], pack[2][:, [0, 2]]],
                axis=0,
            ),
            axis=0,
        )
        dtype = [("col1", np.int64), ("col2", np.int64)]
        msk_e = np.isin(
            pack[1].astype(np.int64).view(dtype), e_taken.astype(np.int64).view(dtype)
        ).reshape(-1)

    return msk_v, msk_e


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@torch.no_grad()
def quantize(x, minmax=(0.0, 4.0), bits=10):
    """Quantize the input array/tensor into categorical labels with 2^bits levels.

    Params:
        x (ndarray or Tensor): the input data to be quantized, shape == (...)
        minmax (tuple): the range of values in x (min, max)
        bits (int): number of bits for quantization

    Returns:
        y (ndarray or Tensor): the quantized labels, shape == (...,)
    """
    is_numpy = isinstance(x, np.ndarray)

    # Convert to tensor for unified processing
    if is_numpy:
        x_tensor = torch.from_numpy(x)
    else:
        x_tensor = x

    levels = 2**bits
    min_val, max_val = minmax

    # Clamp to minmax range
    x_clamped = torch.clamp(x_tensor, min_val, max_val)

    # Normalize to [0, 1]
    x_norm = (x_clamped - min_val) / (max_val - min_val)

    # Quantize to discrete levels
    y = (x_norm * (levels - 1)).round().long()

    if is_numpy:
        return y.numpy()
    return y


@torch.no_grad()
def dequantize(y, minmax=(0.0, 4.0), bits=10):
    """Dequantize the categorical labels back to the original continuous range.

    Params:
        y (ndarray or Tensor): the quantized labels, shape == (...)
        minmax (tuple): the range of values to restore
        bits (int): number of bits used in quantization

    Returns:
        x (ndarray or Tensor): the dequantized values, shape == (...)
    """
    is_numpy = isinstance(y, np.ndarray)

    if is_numpy:
        y_tensor = torch.from_numpy(y)
    else:
        y_tensor = y

    levels = 2**bits
    min_val, max_val = minmax

    # Map back to normalized range
    x_norm = y_tensor.float() / (levels - 1)

    # Restore to original range
    x = x_norm * (max_val - min_val) + min_val

    if is_numpy:
        return x.numpy()
    return x


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def batched_range(start, end):
    """batched range
    Params:
        start (Tensor): (num_batch, )   the starting index
        end   (Tensor): (num_batch, )   the ending index
    equivalent to:
        "torch.cat([torch.arange(s, e) for s, e in zip(start, end)])"
    """
    num = end - start
    return torch.arange(num.sum(), device=start.device) + (
        start - F.pad(num.cumsum(dim=0), (1, 0))[:-1]
    ).repeat_interleave(num)


def pad_inputs_and_create_mask(inputs, lengths):
    """pad the input and create a src_key_padding_mask

    Params:
        inputs  (Tensor): (num_total, dim)
        lengths (Tensor): (batch_size, )
    """
    # this is much faster than "pad_sequence"
    inputs = torch.nested.nested_tensor_from_jagged(
        values=inputs,
        lengths=lengths,  # (num_batch, )
    ).to_padded_tensor(
        0.0
    )  # (num_batch, max_len, dim)

    max_len = inputs.shape[1]
    src_key_padding_mask = (
        torch.arange(max_len, device=lengths.device)[None, :] >= lengths[:, None]
    )

    return inputs, src_key_padding_mask


def start_end(num):
    """compute the accumulative start/end indices"""
    cumsum = num.cumsum(dim=0)
    return F.pad(cumsum[:-1], (1, 0)), cumsum


def isin_with_index(elements: torch.Tensor, test_elements: torch.Tensor):
    """
    返回 elements 在 test_elements 中的第一个出现位置, 找不到则返回 "torch.iinfo(torch.long).max"
    """
    assert elements.ndim == 1 and test_elements.ndim == 1

    max_long = torch.iinfo(torch.long).max

    # 找 unique 值
    unique_vals, inverse = torch.unique(test_elements, return_inverse=True)

    # 找每个 unique 值首次出现的位置
    first_occurrence = torch.full(
        (unique_vals.size(0),),
        test_elements.size(0),
        device=elements.device,
        dtype=torch.long,
    )
    idxs = torch.arange(test_elements.size(0), device=elements.device)
    first_occurrence = first_occurrence.scatter_reduce(
        0, inverse, idxs, reduce="amin", include_self=True
    )

    # 建立映射表
    mapping_size = max(elements.max(), unique_vals.max()).item() + 1
    mapping = torch.full((mapping_size,), -1, device=elements.device, dtype=torch.long)
    mapping[unique_vals] = first_occurrence

    # 查找
    in_range = (elements >= 0) & (elements < mapping_size) & (mapping[elements] != -1)
    result = torch.full(
        (elements.size(0),), max_long, device=elements.device, dtype=torch.long
    )
    result[in_range] = mapping[elements[in_range]]

    return result


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class LengthAwareBatchSampler(Sampler):
    def __init__(self, lengths, num_total_max, shuffle=True, drop_last=False):
        self.lengths = lengths
        self.num_total_max = num_total_max
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            random.Random(self.epoch).shuffle(indices)

        batch = []
        batch_total = 0
        for idx in indices:
            l = self.lengths[idx]
            if batch_total + l > self.num_total_max:
                if batch:
                    yield batch
                batch = [idx]
                batch_total = l
            else:
                batch.append(idx)
                batch_total += l

        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        return (sum(self.lengths) + self.num_total_max - 1) // self.num_total_max


class LengthAwareDistributedBatchSampler(Sampler):
    def __init__(
        self, lengths, num_total_max, num_replicas, rank, shuffle=True, drop_last=False
    ):
        self.lengths = lengths
        self.num_total_max = num_total_max
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0

        self.num_samples = int(math.ceil(len(self.lengths) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            random.Random(self.epoch).shuffle(indices)

        # pad to make evenly divisible
        if len(indices) < self.total_size:
            padding_size = self.total_size - len(indices)
            indices += indices[:padding_size]

        indices = indices[self.rank : self.total_size : self.num_replicas]

        batch = []
        batch_total = 0
        for idx in indices:
            l = self.lengths[idx]
            if batch_total + l > self.num_total_max:
                if batch:
                    yield batch
                batch = [idx]
                batch_total = l
            else:
                batch.append(idx)
                batch_total += l

        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        return (
            (sum(self.lengths) + self.num_total_max - 1)
            // self.num_total_max
            // self.num_replicas
        )


def get_length_aware_sampler(
    lengths, num_total_max, num_rank=1, local_rank=0, shuffle=True, drop_last=False
):
    """to achieve workload balance during sampling"""
    if num_rank > 1:
        sampler = LengthAwareDistributedBatchSampler(
            lengths,
            num_total_max=num_total_max,
            num_replicas=num_rank,
            rank=local_rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
    else:
        sampler = LengthAwareBatchSampler(
            lengths, num_total_max=num_total_max, shuffle=shuffle, drop_last=drop_last
        )
    return sampler


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@rank_zero_only
def safe_shutdown():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
