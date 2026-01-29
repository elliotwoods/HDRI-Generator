from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


def compute_camera_dirs(H: int, W: int, fov_h_deg: float, fov_v_deg: float) -> np.ndarray:
    fov_h = np.deg2rad(fov_h_deg)
    fov_v = np.deg2rad(fov_v_deg)
    xs = (np.arange(W) + 0.5) / W * 2.0 - 1.0
    ys = 1.0 - (np.arange(H) + 0.5) / H * 2.0
    x = np.tan(fov_h / 2.0) * xs[None, :]
    y = np.tan(fov_v / 2.0) * ys[:, None]
    x = np.broadcast_to(x, (H, W))
    y = np.broadcast_to(y, (H, W))
    z = np.ones((H, W), dtype=np.float32)
    dirs = np.stack([x, y, z], axis=-1).astype(np.float32)
    norm = np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-6
    return dirs / norm


def _dirs_to_cubemap_uv(dirs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    abs_dirs = np.abs(dirs)
    max_axis = np.argmax(abs_dirs, axis=-1)
    u = np.zeros(dirs.shape[:2], dtype=np.float32)
    v = np.zeros(dirs.shape[:2], dtype=np.float32)
    face = np.empty(dirs.shape[:2], dtype="<U2")

    x = dirs[..., 0]
    y = dirs[..., 1]
    z = dirs[..., 2]

    pos_x = (max_axis == 0) & (x > 0)
    neg_x = (max_axis == 0) & (x <= 0)
    pos_y = (max_axis == 1) & (y > 0)
    neg_y = (max_axis == 1) & (y <= 0)
    pos_z = (max_axis == 2) & (z > 0)
    neg_z = (max_axis == 2) & (z <= 0)

    face[pos_x] = "px"
    u[pos_x] = -z[pos_x] / abs_dirs[..., 0][pos_x]
    v[pos_x] = -y[pos_x] / abs_dirs[..., 0][pos_x]

    face[neg_x] = "nx"
    u[neg_x] = z[neg_x] / abs_dirs[..., 0][neg_x]
    v[neg_x] = -y[neg_x] / abs_dirs[..., 0][neg_x]

    face[pos_y] = "py"
    u[pos_y] = x[pos_y] / abs_dirs[..., 1][pos_y]
    v[pos_y] = z[pos_y] / abs_dirs[..., 1][pos_y]

    face[neg_y] = "ny"
    u[neg_y] = x[neg_y] / abs_dirs[..., 1][neg_y]
    v[neg_y] = -z[neg_y] / abs_dirs[..., 1][neg_y]

    face[pos_z] = "pz"
    u[pos_z] = x[pos_z] / abs_dirs[..., 2][pos_z]
    v[pos_z] = -y[pos_z] / abs_dirs[..., 2][pos_z]

    face[neg_z] = "nz"
    u[neg_z] = -x[neg_z] / abs_dirs[..., 2][neg_z]
    v[neg_z] = -y[neg_z] / abs_dirs[..., 2][neg_z]

    return face, u, v


def _face_dirs(face: str, cube_res: int) -> np.ndarray:
    u = (np.arange(cube_res) + 0.5) / cube_res * 2.0 - 1.0
    v = 1.0 - (np.arange(cube_res) + 0.5) / cube_res * 2.0
    uu, vv = np.meshgrid(u, v)
    if face == "px":
        dirs = np.stack([np.ones_like(uu), vv, -uu], axis=-1)
    elif face == "nx":
        dirs = np.stack([-np.ones_like(uu), vv, uu], axis=-1)
    elif face == "py":
        dirs = np.stack([uu, np.ones_like(uu), vv], axis=-1)
    elif face == "ny":
        dirs = np.stack([uu, -np.ones_like(uu), -vv], axis=-1)
    elif face == "pz":
        dirs = np.stack([uu, vv, np.ones_like(uu)], axis=-1)
    else:  # nz
        dirs = np.stack([-uu, vv, -np.ones_like(uu)], axis=-1)
    norm = np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-6
    return (dirs / norm).astype(np.float32)


def _project_face_remap(
    img_lin: np.ndarray, face: str, cube_res: int, fov_h_deg: float, fov_v_deg: float
) -> Tuple[np.ndarray, np.ndarray]:
    H, W = img_lin.shape[:2]
    dirs = _face_dirs(face, cube_res)
    x = dirs[..., 0]
    y = dirs[..., 1]
    z = dirs[..., 2]
    valid = z > 1e-6
    x_ndc = (x / z) / np.tan(np.deg2rad(fov_h_deg) / 2.0)
    y_ndc = (y / z) / np.tan(np.deg2rad(fov_v_deg) / 2.0)
    valid &= (np.abs(x_ndc) <= 1.0) & (np.abs(y_ndc) <= 1.0)
    map_x = (x_ndc + 1.0) * 0.5 * (W - 1)
    map_y = (1.0 - y_ndc) * 0.5 * (H - 1)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    face_img = cv2.remap(img_lin, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return face_img.astype(np.float32), valid


def project_to_cubemap(
    img_lin: np.ndarray,
    fov_h_deg: float,
    fov_v_deg: float,
    cube_res: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    faces = {name: np.zeros((cube_res, cube_res, 3), dtype=np.float32) for name in ["px", "nx", "py", "ny", "pz", "nz"]}
    masks = {name: np.zeros((cube_res, cube_res), dtype=bool) for name in faces}
    if cv2 is None:
        H, W = img_lin.shape[:2]
        dirs_cam = compute_camera_dirs(H, W, fov_h_deg, fov_v_deg)
        face, u, v = _dirs_to_cubemap_uv(dirs_cam)
        u_px = ((u + 1.0) * 0.5 * (cube_res - 1)).astype(np.int32)
        v_px = ((v + 1.0) * 0.5 * (cube_res - 1)).astype(np.int32)
        for j in range(H):
            for i in range(W):
                f = face[j, i]
                x = u_px[j, i]
                y = v_px[j, i]
                if f in faces:
                    faces[f][y, x] = img_lin[j, i]
                    masks[f][y, x] = True
        return faces, masks

    for face in faces.keys():
        face_img, valid = _project_face_remap(img_lin, face, cube_res, fov_h_deg, fov_v_deg)
        faces[face] = face_img
        masks[face] = valid
    return faces, masks


def cubemap_to_equirectangular(faces: Dict[str, np.ndarray], width: int, height: int) -> np.ndarray:
    lon = (np.arange(width) + 0.5) / width * 2.0 * np.pi - np.pi
    lat = (np.arange(height) + 0.5) / height * np.pi - np.pi / 2.0
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    x = np.cos(lat_grid) * np.sin(lon_grid)
    y = np.sin(lat_grid)
    z = np.cos(lat_grid) * np.cos(lon_grid)

    dirs = np.stack([x, y, z], axis=-1).astype(np.float32)
    face, u, v = _dirs_to_cubemap_uv(dirs)
    cube_res = next(iter(faces.values())).shape[0]
    u_px = ((u + 1.0) * 0.5 * (cube_res - 1)).astype(np.int32)
    v_px = ((v + 1.0) * 0.5 * (cube_res - 1)).astype(np.int32)

    pano = np.zeros((height, width, 3), dtype=np.float32)
    for f in ["px", "nx", "py", "ny", "pz", "nz"]:
        mask = face == f
        if not np.any(mask):
            continue
        pano[mask] = faces[f][v_px[mask], u_px[mask]]
    return pano

