"""transform logits to make them satisfy constraints"""

import torch


def hardmax(x, dim):
    """hardmax"""
    hard = torch.zeros_like(x)
    idx = x.argmax(dim=dim, keepdim=True)
    hard.scatter_(dim, idx, 1.0)
    return hard


def constrained_decoding(v, e, f, incid_ef, e_batch):
    """used to transform the logits in order to satisfy the rules
    Params:
        v         (Tensor):   (num_v, 2)       logits    "num_v" is also the batch size
        e         (Tensor):   (num_e, 4)       logits    all "e" must be connected to "v"
        f         (Tensor):   (num_f, 3)       logits
        incid_ef  (Tensor):   (num_e, num_f)   incidence matrix, with 1 being connected
        e_batch   (Tensor):   (num_e,)         the value indicating which batch the edge belongs to
    """
    assert v.ndim == 2 and v.shape[1] == 2
    assert e.ndim == 2 and e.shape[1] == 4
    assert f.ndim == 2 and f.shape[1] == 3
    assert incid_ef.shape[0] == len(e) and incid_ef.shape[1] == len(f)
    assert e_batch.ndim == 1 and len(e_batch) == len(e)

    # setting this to a positive value will later set the corresponding edge value to a large negative value
    e_rm = torch.zeros_like(e)

    #  v=0   ==>   e!=3
    prob_v_is_0 = hardmax(v, dim=1)[:, 0]  # (num_v,)
    e_rm[:, 3] = e_rm[:, 3] + prob_v_is_0[e_batch]

    # specific face constraints not detailed in their paper
    if incid_ef.numel() > 0:
        prob_f = hardmax(f, dim=1)  # (num_f, 3)
        prob_e = incid_ef.float() @ prob_f  # (num_e, 3)

        # if  (e1, e2 in f=0)  then  e1,e2 != 1
        prob_e_not_1 = prob_e[:, 0]  # (num_e,)
        e_rm[:, 1] = e_rm[:, 1] + prob_e_not_1

        # if  (e1, e2 in f=1)  then  e1,e2 != 0
        prob_e_not_0 = prob_e[:, 1]  # (num_e,)
        e_rm[:, 0] = e_rm[:, 0] + prob_e_not_0

        # if  (e1, e2 in f=2)  then  e1,e2 != 0/1
        prob_e_not_01 = prob_e[:, 2]  # (num_e,)
        e_rm[:, :2] = e_rm[:, :2] + prob_e_not_01[:, None]

    # only modify "e"
    e_mag = e.max(dim=1).values - e.min(dim=1).values  # (num_e,)
    # multiplied by 10 to ensure it is sufficiently large
    e_mag = e_mag.mul(10)[:, None]  # (num_e, 1)
    e = e - e_mag * (e_rm > 0)

    return e


if __name__ == "__main__":

    import torch

    v = torch.tensor([5, 1]).float()
    e = torch.tensor(
        [
            [9, 5, 5, 10],
            [1, 0, 7, 4],
            [6, 7, 5, 2],
        ]
    ).float()
    f = torch.tensor(
        [
            [4, 3, 5],
            [1, 2, 7],
            [4, 8, 4],
            [5, 9, 0],
        ]
    ).float()
    incid_ef = torch.tensor(
        [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    print("v =", v.argmax(dim=0))  # v = tensor(0)
    print("e =", e.argmax(dim=1))  # e = tensor([3, 2, 1])
    print("f =", f.argmax(dim=1))  # f = tensor([2, 2, 1, 1])
    print("= = = = = = = ")

    e = constrained_decoding(v, e, f, incid_ef)
    print("v =", v.argmax(dim=0))  # v = tensor(0)
    print("e =", e.argmax(dim=1))  # e = tensor([2, 2, 1])
    print("f =", f.argmax(dim=1))  # f = tensor([2, 2, 1, 1])
