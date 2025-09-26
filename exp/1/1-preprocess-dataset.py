"""build psc files from meshes, and write "object_info.json" """

import os
import json
import signal
import trimesh
import platform
import numpy as np
import os.path as osp
from tqdm import tqdm
from fire import Fire
from joblib import Parallel, delayed
from libpsc import obtain_psc_sequence_from_mesh, reconstruct, write_psc


# root of this repo
ROOT = osp.abspath(osp.join(osp.dirname(__file__), "../.."))


def find_files_by_extensions(folder, extensions):
    """
    recursively find all files with given extensions in the specified folder and its subfolders.

    parameters:
        folder (str): the root folder path to search.
        extensions (list): a list of file extensions to match, e.g., ['.ply', '.obj'].

    returns:
        list: a list of relative paths of all matched files.
    """
    # initialize an empty list to store results
    result = []

    # traverse the folder and its subfolders using os.walk
    for root, dirs, files in os.walk(folder):
        for file in files:
            # check if the file ends with any of the given extensions
            if any(file.endswith(ext) for ext in extensions):
                # compute the relative path of the file with respect to the root folder
                relative_path = os.path.relpath(os.path.join(root, file), folder)
                # add the relative path to the result list
                result.append(relative_path)

    # return the list of matched file paths
    return result


def use_suffix(path, suffix=".psc"):
    a, _ = osp.splitext(path)
    return a + suffix


def as_mesh(scene_or_mesh):
    """convert a scene to a mesh"""
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = trimesh.Trimesh()  # empty scene, this may cause error
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(
                    trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()
                )
            )
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh


def add_noise(mesh, eps=1e-4):
    pt_min, pt_max = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    size_min = (pt_max - pt_min).min()
    mesh.vertices += np.random.RandomState(42).randn(*mesh.vertices.shape) * max(
        size_min * eps, 1e-4
    )
    return mesh


def worker(folder, folder_out, mesh_path, num_vert_max, eps):

    mesh = trimesh.load(osp.join(folder, mesh_path))
    mesh = as_mesh(mesh)

    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()

    if len(mesh.vertices) > num_vert_max * 5:  # at most simplified to 20%
        # NOTE :
        #   when the input mesh is too large, doing mesh decimation is likely to get the process killed
        #   which is problematic with multi-threading
        #   so we limit the mesh complexity
        print(
            f"the raw mesh is too large (#V={len(mesh.vertices)}, #F={len(mesh.faces)}), skipped: {mesh_path}"
        )
        return [None, None]

    # if too many vertices, apply our simplification algorithm to reduce complexity
    if len(mesh.vertices) > num_vert_max:

        mesh = add_noise(mesh, eps=eps)
        op_lst, pt = obtain_psc_sequence_from_mesh(
            mesh,
            std_v=0,
        )

        psc = reconstruct(op_lst[: num_vert_max - 1], pt)
        verts, _, faces = map(np.asarray, psc.write())
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
        assert len(mesh.vertices) <= num_vert_max

    # if too few vertices
    if len(mesh.vertices) < 3 or len(mesh.faces) < 1:
        print(
            f"too few vertices (#V={len(mesh.vertices)}, #F={len(mesh.faces)}), skipped: {mesh_path}"
        )
        return [None, None]

    print(f"handling with: {mesh_path}")

    # compute PSC sequence
    mesh = add_noise(mesh, eps=eps)
    op_lst, pt = obtain_psc_sequence_from_mesh(
        mesh,
        std_v=0,
    )

    # compute normalization factor
    scale = float(abs(mesh.vertices - np.asarray(pt)[None]).max())
    for i in range(len(op_lst)):
        for j in range(3):
            op_lst[i]["delta_p"][j] /= scale

    # write the psc file
    mesh_path_flatten = f"{osp.basename(folder)}--" + mesh_path.replace(
        "/", "--"
    ).replace("\\", "--")
    write_psc(
        use_suffix(osp.join(folder_out, mesh_path_flatten), suffix=".psc"),
        op_lst,
        [0, 0, 0],
    )

    # save info
    key = use_suffix(mesh_path_flatten, suffix="")
    val = dict(
        # reconstruction from [0,0,0] results in an object within [-1,1]^3
        # applying the following further will convert back to the original scale:
        #   vertices * scale + center
        center=pt,  # the starting point
        scale=scale,  # the scale
    )
    return [key, val]


class TimeoutException(Exception):
    pass


def handler(signum, frame):
    raise TimeoutException()


def worker_with_timeout(*args, timeout=300):  # timeout in seconds
    if platform.system() == "Windows":
        try:
            result = worker(*args)
            return result
        except Exception as e:
            print(f"error: {e}, skipped: {args}")
            return [None, None]

    else:
        # NOTE : only support UNIX/Linux/macOS
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout)

        try:
            result = worker(*args)
            signal.alarm(0)
            return result
        except TimeoutException:
            print(f"timeout in worker with args {args}")
            return [None, None]
        except Exception as e:
            print(f"error: {e}, skipped: {args}")
            return [None, None]


def main_fn(
    folder="assets/obj100",
    num_vert_max=2048,
    num_workers=8,
    eps=1e-4,
    add_category=None,
):
    """main function
    Params:
        add_category (str): add a category
            considered new category if "add_category in path"
            this will not affect existing files, just add more data
    """
    folder = osp.join(ROOT, folder)
    folder_out = osp.join(ROOT, f"{folder}_out")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # create output folder
    if osp.exists(folder_out) and add_category is None:
        print(f"output folder already exists: {folder_out}")
        print("please delete it before proceeding")
        exit(1)

    os.makedirs(folder_out, exist_ok=True)
    print(f"processed files will be written to: {folder_out}")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # read all mesh file paths

    all_mesh_files = find_files_by_extensions(folder, [".ply", ".obj", ".stl", ".off"])
    if add_category:
        all_mesh_files = [path for path in all_mesh_files if (add_category in path)]
    print(f"found {len(all_mesh_files)} meshes")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # process each file

    # for each mesh, parallel
    if num_workers > 1:
        outs = Parallel(n_jobs=num_workers, backend="multiprocessing")(
            delayed(worker_with_timeout)(
                folder, folder_out, mesh_path, num_vert_max, eps
            )
            for mesh_path in tqdm(all_mesh_files, desc="processing")
        )
    else:
        outs = [
            worker_with_timeout(folder, folder_out, mesh_path, num_vert_max, eps)
            for mesh_path in tqdm(all_mesh_files, desc="processing")
        ]

    # gather into a dict
    print("preparing object info...")
    object_info = {k: v for k, v in outs if k is not None}

    # write object info json file
    print("writing object info into a file...")
    path_object_info = osp.join(folder_out, "object_info.json")
    if add_category is None:
        with open(path_object_info, "w") as f:
            json.dump(object_info, f)
    else:
        with open(path_object_info, "r") as f:
            object_info_old = json.load(f)
        with open(path_object_info, "w") as f:
            json.dump({**object_info_old, **object_info}, f)
    print(f"write object info to: {path_object_info}")


if __name__ == "__main__":
    Fire(main_fn)
