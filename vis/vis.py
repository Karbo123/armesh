"""visualize the psc file by opening up an interactive windows
    TODO : we may use it to record video (https://github.com/hhoppe/Mesh-processing-library?tab=readme-ov-file#geometry-viewer)
example:
    python vis.py xxx.psc --v=0.01 --e=0.003                         # interative viewer
    python vis.py xxx.psc --v=0.01 --e=0.003 --video_frames=360      # video recording
    python vis.py xxx.psc --vis_radius=0.05 --ratio=1.0              # save as file
"""

import os
import platform
import numpy as np
import os.path as osp
from fire import Fire
from easygui import filesavebox
from tempfile import TemporaryDirectory
from libpsc import load_psc, reconstruct, export_visual_mesh_from_psc


def ask_for_saving_path_gui(default_file_name):
    # default download path
    default_download_path = osp.expanduser("~/Downloads")

    # pop out a window
    file_path = filesavebox(
        title="Save File",
        default=osp.join(default_download_path, default_file_name),
        filetypes=[f"*{osp.splitext(default_file_name)[1]}"],
    )

    # if cancel, returning None
    return file_path


def write_a3d(range_min, range_max):
    """write a 3d bounding box"""
    vertices = np.asarray(
        [
            [range_min[0], range_min[1], range_min[2]],
            [range_max[0], range_min[1], range_min[2]],
            [range_max[0], range_max[1], range_min[2]],
            [range_min[0], range_max[1], range_min[2]],
            [range_min[0], range_min[1], range_max[2]],
            [range_max[0], range_min[1], range_max[2]],
            [range_max[0], range_max[1], range_max[2]],
            [range_min[0], range_max[1], range_max[2]],
        ]
    )
    faces = np.asarray(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [1, 2, 6],
            [1, 6, 5],
            [0, 3, 7],
            [0, 7, 4],
        ]
    )
    # convert into a string
    s = ""
    for i, v in enumerate(vertices):
        s += f"Vertex {i + 1} {float(v[0])} {float(v[1])} {float(v[2])}\n"
    for i, f in enumerate(faces):
        s += f"Face {i + 1} {int(f[0]) + 1} {int(f[1]) + 1} {int(f[2]) + 1}\n"
    return s


def update_camera_pose(up_init, eye_init, center, angle_height, angle_rotate):
    """
    Update camera pose by rotating around a fixed target point (center).
    First rotate horizontally around a fixed global up axis,
    then tilt vertically along the local right axis on the rotated path.

    Args:
        up_init (ndarray): initial up vector (should remain unchanged as global up)
        front_init (ndarray): initial front vector (not directly used)
        right_init (ndarray): initial right vector (not directly used)
        eye_init (ndarray): initial camera position
        center (ndarray): target point the camera always looks at
        angle_height (float): vertical tilt in degrees (positive = look downward)
        angle_rotate (float): horizontal orbit in degrees

    Returns:
        up_new, front_new, right_new, eye_new: updated orthonormal basis and camera position
    """

    def rotate_around_axis(v, axis, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        axis = axis / np.linalg.norm(axis)
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)
        cross = np.cross(axis, v)
        dot = np.dot(axis, v)
        return v * cos_theta + cross * sin_theta + axis * dot * (1 - cos_theta)

    # Compute vector from center to camera
    cam_vec = eye_init - center
    dist = np.linalg.norm(cam_vec)
    cam_vec /= dist

    # Step 1: horizontal orbit around fixed up vector
    cam_vec = rotate_around_axis(cam_vec, up_init, -angle_rotate)

    # Step 2: vertical tilt around local right axis
    # Note: positive angle_height => look down => move camera downward => rotate cam_vec upward
    right_axis = np.cross(up_init, cam_vec)
    right_axis /= np.linalg.norm(right_axis)
    cam_vec = rotate_around_axis(
        cam_vec, right_axis, -angle_height
    )  # negative to match "look down" logic

    # New eye position
    eye_new = center + cam_vec * dist

    # Recompute orthonormal basis
    front_new = center - eye_new
    front_new /= np.linalg.norm(front_new)

    right_new = np.cross(up_init, front_new)
    right_new /= np.linalg.norm(right_new)

    up_new = np.cross(front_new, right_new)
    up_new /= np.linalg.norm(up_new)

    return up_new, front_new, right_new, eye_new


def write_s3d(
    range_min, range_max, view_up="Y", fov=60, angle_height=0, angle_rotate=0
):
    # to numpy
    min_pt = np.array(range_min, dtype=np.float32)
    max_pt = np.array(range_max, dtype=np.float32)
    center = (min_pt + max_pt) / 2
    size = max_pt - min_pt

    # the up direction
    if view_up.upper() == "X":
        up = np.asarray([1, 0, 0]).astype(np.float32)
        front = np.asarray([0, -1, 0]).astype(np.float32)
        right = np.asarray([0, 0, 1]).astype(np.float32)
        length = max(size[0], size[2])
        nearest = center - front * size[1] / 2
    elif view_up.upper() == "Y":
        up = np.asarray([0, 1, 0]).astype(np.float32)
        front = np.asarray([0, 0, -1]).astype(np.float32)
        right = np.asarray([1, 0, 0]).astype(np.float32)
        length = max(size[0], size[1])
        nearest = center - front * size[2] / 2
    elif view_up.upper() == "Z":
        up = np.asarray([0, 0, 1]).astype(np.float32)
        front = np.asarray([-1, 0, 0]).astype(np.float32)
        right = np.asarray([0, 1, 0]).astype(np.float32)
        length = max(size[1], size[2])
        nearest = center - front * size[0] / 2
    else:
        raise ValueError("view_up must be 'X', 'Y', or 'Z'")

    # zoom factor : tan(fov/2)
    zoom = np.tan(np.deg2rad(fov) / 2)

    # set the eye position
    dist = length / (2 * zoom)
    eye = nearest - front * dist

    # rotate by angles
    up, front, right, eye = update_camera_pose(
        up, eye, center, angle_height, angle_rotate
    )

    # make output
    values = np.concatenate([front, -right, up, eye, [zoom]])
    values_str = " ".join(f"{v:.6f}" for v in values)
    return f"F 0 {values_str}\n"


def main_fn(path, v=0.01, e=0.003, ratio=0.0, vis_radius=0.05, video_frames=0, exe=None):
    d = load_psc(path)
    if ratio > 0:
        psc = reconstruct(d["vsplits"], d["center"], ratio=ratio)
        save_path = ask_for_saving_path_gui(f"mesh-ratio={ratio:.5f}.ply")
        if save_path is not None:
            export_visual_mesh_from_psc(save_path, psc, radius=vis_radius)
            print(f"saved mesh to: {save_path}")
        return
    psc = reconstruct(d["vsplits"], d["center"])
    verts = np.asarray(psc.write()[0])
    range_min = np.min(verts, axis=0)
    range_max = np.max(verts, axis=0)
    v_size = float(np.linalg.norm(range_max - range_min) * v)
    e_size = float(np.linalg.norm(range_max - range_min) * e)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    with TemporaryDirectory() as folder:
        path_a3d = osp.join(folder, "temp.a3d")
        with open(path_a3d, "w", newline="") as f_a3d:
            f_a3d.write(write_a3d(range_min, range_max))

        # path of the executable
        path_exe = exe or osp.abspath(
            osp.join(
                osp.dirname(__file__),
                "..",
                "assets",
                "G3dOGL",
                "G3dOGL.exe" if platform.system() == "Windows" else "G3dOGL",
            )
        )
        assert osp.exists(path_exe), f"cannot find the executable: {path_exe}"

        # explanation for key:
        #   l   ==>   toggle auto_level
        #   j   ==>   find good viewing location
        #   J   ==>   auto rotate
        cmd = (
            f"{path_exe} -psc_mode {path} {path_a3d} "
            + f"-key ljJ -lightambient .6 -geom 1100x850+100+50 -bigfont "
            + f"-minpointradius {v_size} -minedgeradius {e_size} -usedefaultcolor 1 "
        )

        # record a video
        if video_frames > 0:
            save_video_path = ask_for_saving_path_gui(f"video-{video_frames}.mp4")
            if save_video_path is not None:
                cmd += f"-video {video_frames} {save_video_path} "

        print(f"running command: \n{cmd}\n\n")
        os.system(cmd)


if __name__ == "__main__":
    Fire(main_fn)
