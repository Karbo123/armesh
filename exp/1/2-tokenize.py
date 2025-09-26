"""tokenize every shape, and write "op_tok.json" """

import sys
import json
import os.path as osp
from tqdm import tqdm
from fire import Fire
from libpsc import load_psc

# root of this repo
ROOT = osp.abspath(osp.join(osp.dirname(__file__), "../.."))
sys.path.append(ROOT)
from codebase.tokenize import encode_op2


def main_fn(folder="assets/obj100_out"):
    folder = osp.join(ROOT, folder)

    with open(osp.join(folder, "object_info.json"), "r") as f:
        object_info = json.load(f)

    # the sequences before BPE
    path_op_tok = osp.join(folder, "op_tok.json")
    print(f"going to write raw tokenized operation sequences to: {path_op_tok}")
    with open(path_op_tok, "w") as f:
        for name in tqdm(object_info.keys(), desc="processing"):
            op_lst = load_psc(osp.join(folder, f"{name}.psc"))["vsplits"]
            json.dump({name: [encode_op2(op) for op in op_lst]}, f)
            f.write("\n")
    print("done!")


if __name__ == "__main__":
    Fire(main_fn)
