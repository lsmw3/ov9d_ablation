import numpy as np
import torch
import cv2


def get_2d_coord_np(width, height, low=0, high=1, fmt="CHW"):
    """
    Args:
        width:
        height:
    Returns:
        xy: (2, height, width)
    """
    # coords values are in [low, high]  [0,1] or [-1,1]
    x = np.linspace(low, high, width, dtype=np.float32)
    y = np.linspace(low, high, height, dtype=np.float32)
    xy = np.asarray(np.meshgrid(x, y))
    if fmt == "HWC":
        xy = xy.transpose(1, 2, 0)
    elif fmt == "CHW":
        pass
    else:
        raise ValueError(f"Unknown format: {fmt}")
    return xy


def crop_resize_by_warp_affine(img, center, scale, output_size, rot=0, interpolation=cv2.INTER_LINEAR):
    """
    output_size: int or (w, h)
    NOTE: if img is (h,w,1), the output will be (h,w)
    """
    if isinstance(scale, (int, float)):
        scale = (scale, scale)
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img, trans, (int(output_size[0]), int(output_size[1])), flags=interpolation)

    return dst_img


def get_affine_transform(center, scale, rot, output_size, inv=False):
    """
    adapted from CenterNet: https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
    center: ndarray: (cx, cy), bbox center
    scale: (w, h), bbox size
    rot: angle in deg
    output_size: int or (w, h)
    """
    if isinstance(center, (tuple, list)):
        center = np.array(center, dtype=np.float32)

    if isinstance(scale, (int, float)):
        scale = np.array([scale, scale], dtype=np.float32)

    if isinstance(output_size, (int, float)):
        output_size = (output_size, output_size)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def draw_3d_bbox_with_coordinate_frame(image, keypoints, m2c_R, m2c_t, camera_matrix, dist_coeffs=None):
    """
    Draw 3D bounding box and coordinate frame on a 2D image.
    
    Args:
        image: The input 2D image
        keypoints: Array of shape (9, 3) where the first point is the center and the rest are corners
        m2c_R: 3x3 rotation matrix from model to camera frame
        m2c_t: 3x1 translation vector from model to camera frame
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients (optional)
    
    Returns:
        image_with_bbox: The image with drawn 3D bbox and coordinate frame
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)
    
    # Create a copy of the input image to draw on
    image_with_bbox = image.copy()
    
    # Extract the center point and corner points
    center_3d = keypoints[0]
    corners_3d = keypoints[1:9]
    
    # Define the bounding box edges (pairs of corner indices)
    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),  # bottom face
        (4, 5), (5, 7), (7, 6), (6, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # connecting edges
    ]
    
    # Transform keypoints from object frame to camera frame
    corners_cam = []
    for corner in corners_3d:
        corner_cam = m2c_R @ corner + m2c_t
        corners_cam.append(corner_cam)
    corners_cam = np.array(corners_cam)
    
    # Transform center from object frame to camera frame
    center_cam = m2c_R @ center_3d + m2c_t
    
    # Project points to image plane
    corners_2d, _ = cv2.projectPoints(corners_cam, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
    corners_2d = corners_2d.reshape(-1, 2)
    
    center_2d, _ = cv2.projectPoints(center_cam.reshape(1, 3), np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
    center_2d = center_2d.reshape(-1, 2)[0]
    
    # Define coordinate axes in object frame (X, Y, Z with length of 0.1m or adjust based on bbox size)
    axis_length = np.max(np.linalg.norm(corners_3d - center_3d, axis=1)) * 0.7
    axes_3d = np.array([
        [axis_length, 0, 0],  # X-axis (red)
        [0, axis_length, 0],  # Y-axis (green)
        [0, 0, axis_length]   # Z-axis (blue)
    ])
    
    # Transform and project coordinate axes
    axes_cam = []
    for axis in axes_3d:
        axis_cam = m2c_R @ axis + m2c_t
        axes_cam.append(axis_cam)
    axes_cam = np.array(axes_cam)
    
    # Add center point to get end points of axes
    axes_end_cam = center_cam + axes_cam
    axes_end_points = []
    for axis_end in axes_end_cam:
        axis_2d, _ = cv2.projectPoints(axis_end.reshape(1, 3), np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
        axes_end_points.append(axis_2d.reshape(-1, 2)[0])
    axes_end_points = np.array(axes_end_points)
    
    # Draw the bounding box edges
    for i, j in edges:
        pt1 = tuple(map(int, corners_2d[i]))
        pt2 = tuple(map(int, corners_2d[j]))
        cv2.line(image_with_bbox, pt1, pt2, (255, 0, 255), 2)  # magenta for bbox
    
    # Draw coordinate axes
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # RGB colors for X (red), Y (green), Z (blue)
    
    for i, (end_point, color) in enumerate(zip(axes_end_points, colors)):
        start_point = tuple(map(int, center_2d))
        end_point = tuple(map(int, end_point))
        cv2.line(image_with_bbox, start_point, end_point, color, 3)
        
        # # Label the axes
        # text_pos = (int((start_point[0] + end_point[0]) * 0.6), 
        #            int((start_point[1] + end_point[1]) * 0.6))
        # axis_label = ['X', 'Y', 'Z'][i]
        # cv2.putText(image_with_bbox, axis_label, text_pos, 
        #            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Draw and label the center point
    cv2.circle(image_with_bbox, tuple(map(int, center_2d)), 5, (0, 165, 255), -1)  # orange center
    
    return image_with_bbox
