from typing import Tuple
import numpy as np
import cv2
from .structs import AlignedBox2f, CameraModel, PinholePlaneCameraModel
from .geometry import *

def construct_crop_camera(
    box: AlignedBox2f,
    camera_model_c2w: CameraModel,
    viewport_size: Tuple[int, int],
    viewport_rel_pad: float = 0.0,
) -> CameraModel:
    """Constructs a virtual pinhole camera from the specified 2D bounding box.

    Args:
        camera_model_c2w: Original camera model with extrinsics set to the
            camera->world transformation.

        viewport_crop_size: Viewport size of the new camera.
        viewport_scaling_factor: Requested scaling of the viewport.
    Returns:
        A virtual pinhole camera whose optical axis passes through the center
        of the specified 2D bounding box and whose focal length is set such as
        the sphere representing the bounding box (+ requested padding) is visible
        in the camera viewport.
    """

    # Get centroid and radius of the reference sphere (the virtual camera will
    # be constructed such as the projection of the sphere fits the viewport.
    f = 0.5 * (camera_model_c2w.f[0] + camera_model_c2w.f[1])
    cx, cy = camera_model_c2w.c
    box_corners_in_c = np.array(
        [
            [box.left - cx, box.top - cy, f],
            [box.right - cx, box.top - cy, f],
            [box.left - cx, box.bottom - cy, f],
            [box.right - cx, box.bottom - cy, f],
        ]
    )
    box_corners_in_c /= np.linalg.norm(box_corners_in_c, axis=1, keepdims=True)
    centroid_in_c = np.mean(box_corners_in_c, axis=0)
    centroid_in_c_h = np.hstack([centroid_in_c, 1]).reshape((4, 1))
    centroid_in_w = camera_model_c2w.T_world_from_eye.dot(centroid_in_c_h)[:3, 0]

    radius = np.linalg.norm(box_corners_in_c - centroid_in_c, axis=1).max()

    # Transformations from world to the original and virtual cameras.
    trans_w2c = np.linalg.inv(camera_model_c2w.T_world_from_eye)
    trans_w2vc = gen_look_at_matrix(trans_w2c, centroid_in_w)

    # Transform the centroid from world to the virtual camera.
    centroid_in_vc = transform_3d_points_numpy(
        trans_w2vc, np.expand_dims(centroid_in_w, axis=0)
    ).squeeze()

    # Project the sphere radius to the image plane of the virtual camera and
    # enlarge it by the specified padding. This defines the 2D extent that
    # should be visible in the virtual camera.
    fx_fy_orig = np.array(camera_model_c2w.f, dtype=np.float32)
    radius_2d = fx_fy_orig * radius / centroid_in_vc[2]
    extent_2d = (1.0 + viewport_rel_pad) * radius_2d

    cx_cy = np.array(viewport_size, dtype=np.float32) / 2.0 - 0.5

    # Set the focal length such as all projected points fit the viewport of the
    # virtual camera.
    fx_fy = fx_fy_orig * cx_cy / extent_2d

    # Parameters of the virtual camera.
    return PinholePlaneCameraModel(
        width=viewport_size[0],
        height=viewport_size[1],
        f=tuple(fx_fy),
        c=tuple(cx_cy),
        T_world_from_eye=np.linalg.inv(trans_w2vc),
    )

def warp_image(
    src_camera: CameraModel,
    dst_camera: CameraModel,
    src_image: np.ndarray,
    interpolation: int = cv2.INTER_LINEAR,
    depth_check: bool = True,
    factor_to_downsample: int = 1,
) -> np.ndarray:
    """
    Warp an image from the source camera to the destination camera.

    Parameters
    ----------
    src_camera :
        Source camera model
    dst_camera :
        Destination camera model
    src_image :
        Source image
    interpolation :
        Interpolation method
    depth_check :
        If True, mask out points with negative z coordinates
    factor_to_downsample :
        If this value is greater than 1, it will downsample the input image prior to warping.
        This improves downsampling performance, in an attempt to replicate
        area interpolation for crop+undistortion warps.
    """

    W, H = dst_camera.width, dst_camera.height
    px, py = np.meshgrid(np.arange(W), np.arange(H))
    dst_win_pts = np.column_stack((px.flatten(), py.flatten()))

    dst_eye_pts = dst_camera.window_to_eye(dst_win_pts)
    world_pts = dst_camera.eye_to_world(dst_eye_pts)
    src_eye_pts = src_camera.world_to_eye(world_pts)
    src_win_pts = src_camera.eye_to_window(src_eye_pts)

    # Mask out points with negative z coordinates
    if depth_check:
        mask = src_eye_pts[:, 2] < 0
        src_win_pts[mask] = -1

    src_win_pts = src_win_pts.astype(np.float32)

    map_x = src_win_pts[:, 0].reshape((H, W))
    map_y = src_win_pts[:, 1].reshape((H, W))
    
    # handle the mask image
    if src_image.dtype == bool:
        src_image = src_image.astype(np.uint8) * 255

    return cv2.remap(src_image, map_x, map_y, interpolation)

def warp_depth_image(
    src_camera: CameraModel,
    dst_camera: CameraModel,
    src_depth_image: np.ndarray,
    depth_check: bool = True,
) -> np.ndarray:

    # Copy the source depth image.
    depth_image = np.array(src_depth_image)

    # If the camera extrinsics changed, update the depth values.
    if not np.allclose(src_camera.T_world_from_eye, dst_camera.T_world_from_eye):

        # Image coordinates with valid depth values.
        valid_mask = depth_image > 0
        ys, xs = np.nonzero(valid_mask)

        # Transform the source depth image to a point cloud.
        pts_in_src = src_camera.window_to_eye(np.vstack([xs, ys]).T)
        pts_in_src *= np.expand_dims(depth_image[valid_mask] / pts_in_src[:, 2], axis=1)

        # Transform the point cloud from the source to the target camera.
        pts_in_w = src_camera.eye_to_world(pts_in_src)
        pts_in_trg = dst_camera.world_to_eye(pts_in_w)

        depth_image[valid_mask] = pts_in_trg[:, 2]

    # Warp the depth image to the target camera.
    return warp_image(
        src_camera=src_camera,
        dst_camera=dst_camera,
        src_image=depth_image,
        interpolation=cv2.INTER_NEAREST,
        depth_check=depth_check,
    )