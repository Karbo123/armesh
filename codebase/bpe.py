import re
import json
import os.path as osp
from tqdm import tqdm
from numba import njit, prange, types
from numba.typed import List, Dict
from functools import lru_cache

PairInt = types.UniTuple(types.int32, 2)


# Collect bigram statistics in parallel
@njit(parallel=True)
def get_stats_parallel(sequences):
    num_threads = len(sequences)
    stats_list = [Dict.empty(key_type=PairInt, value_type=types.int32) for _ in range(num_threads)]

    for i in prange(num_threads):
        seq = sequences[i]
        stats = stats_list[i]
        for j in range(len(seq) - 1):
            pair = (seq[j], seq[j + 1])
            if pair in stats:
                stats[pair] += 1
            else:
                stats[pair] = 1

    # Merge per-thread stats into a single dict
    merged_stats = Dict.empty(key_type=PairInt, value_type=types.int32)
    for stats in stats_list:
        for pair, count in stats.items():
            if pair in merged_stats:
                merged_stats[pair] += count
            else:
                merged_stats[pair] = count

    return merged_stats


@njit
def merge_sequence(seq, pair, new_token):
    # Ensure output list has int32 elements
    output = List.empty_list(types.int32)
    i = 0
    while i < len(seq):
        if i < len(seq) - 1 and (seq[i], seq[i + 1]) == pair:
            output.append(new_token)
            i += 2
        else:
            output.append(seq[i])
            i += 1
    return output


@njit(parallel=True)
def apply_merge_to_all_parallel(sequences, pair, new_token):
    out = List()
    for _ in range(len(sequences)):
        out.append(List.empty_list(types.int32))

    for i in prange(len(sequences)):
        merged = merge_sequence(sequences[i], pair, new_token)
        out[i] = merged  # Now both sides are List[int32]

    return out


# Find the most frequent pair in the stats
@njit
def find_best_pair(stats):
    best_pair = (-1, -1)
    max_count = -1
    for pair, count in stats.items():
        if count > max_count:
            best_pair = pair
            max_count = count
    return best_pair


# Public interface: takes and returns native Python types
def bpe_train(sequences_py, vocab_size):
    # Convert input to Numba typed list
    sequences = List()
    for seq in sequences_py:
        sequences.append(List(seq))

    merges = List()  # list of (a, b, new_token)
    id2token = Dict.empty(key_type=types.int32, value_type=PairInt)

    current_token = max([max(seq) for seq in sequences_py]) + 1

    # Estimate number of steps for progress bar
    max_steps = vocab_size - current_token
    pbar = tqdm(total=max_steps, desc="Training BPE", unit="merge")

    while current_token < vocab_size:
        stats = get_stats_parallel(sequences)
        if len(stats) == 0:
            break
        best_pair = find_best_pair(stats)
        merges.append((best_pair[0], best_pair[1], current_token))
        id2token[current_token] = best_pair
        sequences = apply_merge_to_all_parallel(sequences, best_pair, current_token)
        current_token += 1
        pbar.update(1)

    pbar.close()

    # Convert back to native Python lists
    final_sequences = [list(seq) for seq in sequences]
    merges_out = [tuple(m) for m in merges]
    id2token_out = {k: (v[0], v[1]) for k, v in id2token.items()}

    return merges_out, id2token_out, final_sequences


# Public BPE encode function
def bpe_encode(seqs, merges):
    return _bpe_encode(seqs, merges)


# Public BPE decode function
def bpe_decode(seq, id2token_dict):
    id2token = Dict.empty(key_type=types.int32, value_type=PairInt)
    for k, v in id2token_dict.items():
        id2token[k] = (v[0], v[1])
    return list(_bpe_decode(List(seq), id2token))


@lru_cache(maxsize=None)
def compile_pattern(a, b):
    pattern = rf"(?<![0-9]){a}, {b}(?![0-9])"
    return re.compile(pattern)


def _bpe_encode(seqs, merges):
    """this is faster then the previous"""
    s = str(seqs)

    for a, b, new_token in merges:
        compiled_pattern = compile_pattern(a, b)
        s = compiled_pattern.sub(str(new_token), s)

    result = eval(s)
    return result


# Internal BPE decode (Numba-optimized)
@njit
def _bpe_decode(seq, id2token):
    result = List()
    for tok in seq:
        stack = List([tok])
        while len(stack) > 0:
            t = stack.pop()
            if t in id2token:
                a, b = id2token[t]
                stack.append(b)
                stack.append(a)
            else:
                result.append(t)
    return result


class BPE:
    """
    Byte Pair Encoding (BPE) class for training and applying token merges
    """

    def __init__(self, vocab_size):
        """
        Initialize the BPE instance with a target vocabulary size
        """
        self.vocab_size = vocab_size
        self.merges = []
        self.id2token = {}

    def train(self, sequences):
        """
        Train BPE merges on a list of sequences.
        Each sequence is a list of integer tokens.
        """
        self.merges, self.id2token, encoded_sequences = bpe_train(sequences, self.vocab_size)
        return encoded_sequences

    def encode(self, sequences):
        """
        Encode a sequence using the learned merges.
        Input: list of integers
        Output: list of integers
        """
        return bpe_encode(sequences, self.merges)

    def decode(self, sequence):
        """
        Decode a sequence using the learned merges.
        Input: list of integers
        Output: list of integers (original form)
        """
        return bpe_decode(sequence, self.id2token)

    def decode_one(self, id_token):
        """
        Decode one token
        Input: an integer
        Output: list of integers
        """
        if not hasattr(self, "decode_mapping"):
            self.try_make_decode_mapping()
        return self.decode_mapping[id_token]

    def save(self, filepath):
        """
        Save the BPE model to a JSON file.
        Includes vocab_size, merges, and id2token.
        """
        data = {
            "vocab_size": self.vocab_size,
            "merges": [list(m) for m in self.merges],  # Convert tuple to list
            "id2token": {str(k): list(v) for k, v in self.id2token.items()},  # Convert keys to str, values to list
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filepath, with_dec=False):
        """
        Load a BPE model from a JSON file and return a BPE instance.
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        bpe = cls(data["vocab_size"])
        bpe.merges = [tuple(m) for m in data["merges"]]
        bpe.id2token = {int(k): tuple(v) for k, v in data["id2token"].items()}
        bpe.filepath = filepath
        if with_dec:
            bpe.try_make_decode_mapping()
        return bpe

    def try_make_decode_mapping(self):
        """used for fast decoding, by flattening the decoding tree"""
        assert hasattr(self, "filepath")
        filepath_dec = osp.splitext(self.filepath)[0] + "_dec.json"
        if osp.exists(filepath_dec):
            with open(filepath_dec, "r") as f:
                self.decode_mapping = json.load(f)
            self.decode_mapping = {int(k): v for k, v in self.decode_mapping.items()}
        else:
            self.decode_mapping = {k: self.decode([k]) for k in self.id2token.keys()}
            self.decode_mapping = {**{k: [k] for k in range(523)}, **self.decode_mapping}
            with open(filepath_dec, "w") as f:
                json.dump(self.decode_mapping, f)

    def use_subset(self, vocab_size):
        """not use all vocab"""
        assert self.vocab_size >= vocab_size
        idx = [x[2] for x in self.merges].index(vocab_size)
        self.merges = self.merges[:idx]
        self.id2token = {k: v for k, v in self.id2token.items() if k < vocab_size}
        self.vocab_size = vocab_size
