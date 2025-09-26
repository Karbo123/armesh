"""train a BPE for sub-parts of the sequences"""

import sys
import random
import os.path as osp
from fire import Fire
import json

# root of this repo
ROOT = osp.abspath(osp.join(osp.dirname(__file__), "../.."))
sys.path.append(ROOT)
from codebase.bpe import BPE


def main_fn(
    folder="assets/obj100_out",
    name="bit-high",
    num_new_vocab=4096,
    id_new_start=523,
    range_start=2,
    range_end=5,
    sample=30000,  # use larger number if possible
):
    folder = osp.join(ROOT, folder)

    # the sequences before BPE
    data_all = []
    path_op_tok = osp.join(folder, "op_tok.json")
    print(f"loading from: {path_op_tok}")
    with open(path_op_tok, "r") as f:
        for _line in f:
            line = _line.strip()
            if line:
                seq_lst = next(iter(json.loads(line).values()))
                for seq in seq_lst:
                    data_all.append(seq[range_start:range_end])
    print("done")

    print(f"collected {len(data_all)} data")
    if sample is not None:
        random.seed(42**range_start + range_end)
        data_all = random.sample(data_all, sample)
        print(f"but downsampled to {len(data_all)} data")

    # train a BPE
    num_existing = max([max(seq) for seq in data_all]) + 1
    bpe = BPE(num_existing + num_new_vocab)
    bpe.train(data_all)
    assert len(bpe.merges) == num_new_vocab

    # apply token offset
    id_first_new_vocab = bpe.merges[0][2]
    assert id_first_new_vocab == num_existing
    replace_fn = lambda x: (
        x - id_first_new_vocab + id_new_start if x >= id_first_new_vocab else x
    )
    for i in range(len(bpe.merges)):
        bpe.merges[i] = tuple(replace_fn(x) for x in bpe.merges[i])
    bpe.id2token = {
        replace_fn(k): tuple(replace_fn(x) for x in v) for k, v in bpe.id2token.items()
    }
    bpe.vocab_size = None

    # the BPE vocab
    path_vocab = osp.join(folder, f"vocab-{name}.json")
    bpe.save(path_vocab)
    print(f"write vocab to: {path_vocab}")


if __name__ == "__main__":
    Fire(main_fn)
