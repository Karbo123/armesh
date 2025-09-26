"""tokenize the vsplit operations (without BPE applied)"""

import trimesh
from libpsc import obtain_psc_sequence_from_mesh


def process(path):
    mesh = trimesh.load(path, process=False)

    # normalized into a unit cube: [0, 1]^3
    verts_min = mesh.vertices.min(axis=0, keepdims=True)
    verts_max = mesh.vertices.max(axis=0, keepdims=True)
    mesh.vertices = (mesh.vertices - verts_min) / verts_max

    op_lst, pt = obtain_psc_sequence_from_mesh(
        mesh,
    )

    return op_lst, pt


def encode_int(x: int) -> list:
    assert 0 <= x <= 65535, "x must be in [0, 65535]"
    return [x >> 8, x & 0xFF]


def decode_int(l: list) -> int:
    assert len(l) == 2, "must have only 2 elements"
    return l[0] << 8 | l[1]


def encode_fp(x: float) -> list:
    assert -2.0 <= x <= 2.0
    x = (x + 2) / 4  # 0~1
    return encode_int(round(x * 65535))


def decode_fp(l: list) -> float:
    assert len(l) == 2, "must have only 2 elements"
    x = decode_int(l)
    x /= 65535
    x = x * 4 - 2
    return x


def encode_topo_lst(topo_lst):
    assert len(topo_lst) % 2 == 0, "must have even number of elements"
    mapping = {
        "90": 0,
        "91": 1,
        "80": 2,
        "81": 3,
        "82": 4,
        "83": 5,
        "70": 6,
        "71": 7,
        "72": 8,
    }
    cate_lst = [mapping[topo_lst[i : i + 2]] for i in range(0, len(topo_lst), 2)]
    return cate_lst


def decode_topo_lst(cate_lst):
    assert all(0 <= x and x <= 9 for x in cate_lst)  # 0~8
    mapping = ["90", "91", "80", "81", "82", "83", "70", "71", "72", ""]
    topo_lst = "".join([mapping[x] for x in cate_lst])
    return topo_lst


def encode_topo_lst2(topo_lst):
    assert len(topo_lst) % 2 == 0, "must have even number of elements"
    mapping = {
        "90": 0,
        "91": 1,
        "70": 2,
        "71": 3,
        "72": 4,
        "80": 5,
        "81": 6,
        "82": 7,
        "83": 8,
    }
    cate_lst = [mapping[topo_lst[i : i + 2]] for i in range(0, len(topo_lst), 2)]
    return cate_lst


def decode_topo_lst2(cate_lst):
    assert all(0 <= x and x <= 9 for x in cate_lst)  # 0~8
    mapping = ["90", "91", "70", "71", "72", "80", "81", "82", "83", ""]
    topo_lst = "".join([mapping[x] for x in cate_lst])
    return topo_lst


def offset_lst(lst, offset=0):
    return [x + offset for x in lst]


def encode_op(op):
    s = encode_int(op["vsid"])  # 256
    s += offset_lst(encode_topo_lst(op["code"]), offset=256)  # 9
    s += offset_lst(encode_fp(op["delta_p"][0]), offset=265)  # 256
    s += offset_lst(encode_fp(op["delta_p"][1]), offset=265)  # 256
    s += offset_lst(encode_fp(op["delta_p"][2]), offset=265)  # 256
    s += [521] if op["position_bit"] == 0 else [522]  # 2
    return s


def decode_op(s):
    op = dict()
    op["vsid"] = decode_int(s[0:2])
    op["code"] = decode_topo_lst([x - 256 for x in s[2:-7]])
    op["delta_p"] = [
        decode_fp([x - 265 for x in s[-7:-5]]),
        decode_fp([x - 265 for x in s[-5:-3]]),
        decode_fp([x - 265 for x in s[-3:-1]]),
    ]
    assert s[-1] == 521 or s[-1] == 522
    op["position_bit"] = 0 if s[-1] == 521 else 1
    return op


def split_list(lst, a_minmax=(0, 1), b_minmax=(2, 5), c_minmax=(6, 8)):
    lst_a, lst_b, lst_c = [], [], []

    for num in lst:
        if a_minmax[0] <= num <= a_minmax[1]:
            lst_a.append(num)
        elif b_minmax[0] <= num <= b_minmax[1]:
            lst_b.append(num)
        elif c_minmax[0] <= num <= c_minmax[1]:
            lst_c.append(num)

    assert lst == lst_a + lst_b + lst_c
    return lst_a, lst_b, lst_c


def encode_op2(op):
    s = encode_int(op["vsid"])
    #
    two_bytes_x = encode_fp(op["delta_p"][0])
    two_bytes_y = encode_fp(op["delta_p"][1])
    two_bytes_z = encode_fp(op["delta_p"][2])
    s += offset_lst(
        [
            two_bytes_x[0],
            two_bytes_y[0],
            two_bytes_z[0],
            two_bytes_x[1],
            two_bytes_y[1],
            two_bytes_z[1],
        ],
        offset=256,
    )
    #
    topo_lst = encode_topo_lst2(op["code"])
    topo_lst_v, topo_lst_e, topo_lst_f = split_list(topo_lst, (0, 1), (5, 8), (2, 4))
    topo_lst = topo_lst_v + topo_lst_f + topo_lst_e
    s += offset_lst(topo_lst, offset=512)
    #
    s += [521] if op["position_bit"] == 0 else [522]
    return s


def decode_op2(s):
    op = dict()
    op["vsid"] = decode_int(s[0:2])
    op["delta_p"] = [
        decode_fp(offset_lst([s[2], s[5]], offset=-256)),
        decode_fp(offset_lst([s[3], s[6]], offset=-256)),
        decode_fp(offset_lst([s[4], s[7]], offset=-256)),
    ]
    #
    topo_lst = offset_lst(s[8:-1], offset=-512)
    topo_lst_v, topo_lst_f, topo_lst_e = split_list(topo_lst, (0, 1), (2, 4), (5, 8))
    topo_lst = topo_lst_v + topo_lst_e + topo_lst_f
    op["code"] = decode_topo_lst2(topo_lst)
    #
    assert s[-1] == 521 or s[-1] == 522
    op["position_bit"] = 0 if s[-1] == 521 else 1
    return op
