import numpy as np
import torch
import cv2
import torch.nn.functional as F
from PIL import Image, ImageDraw
from pytorch3d import _C


def get_coarse_mask(mask, scale_factor=1/7):

        if len(mask.shape) == 2:
            mask = mask[None][None]
        elif len(mask.shape) == 3:
            mask = mask[None]
        else:
            raise ValueError("mask should be 2D or 3D arrays")
        
        if isinstance(mask, np.ndarray):
            mask = torch.tensor(mask)
        
        mask_c = F.interpolate(mask.float(),
                            scale_factor=scale_factor,
                            mode='nearest',
                            recompute_scale_factor=False)[0].squeeze(0)

        mask_c = mask_c.bool().numpy()

        return mask_c


def filter_small_edge_obj(bbox, img_w, img_h, vis_threshold=0.3):
    x, y, w, h = bbox

    x_min = x - (w / 2)
    y_min = y - (h / 2)
    x_max = x + (w / 2)
    y_max = y + (h / 2)

    full_area = w * h
    if full_area <= 0:
        return True
    
    visible_x_min = max(0, x_min)
    visible_y_min = max(0, y_min)
    visible_x_max = min(img_w, x_max)
    visible_y_max = min(img_h, y_max)
    
    visible_width = max(0, visible_x_max - visible_x_min)
    visible_height = max(0, visible_y_max - visible_y_min)
    visible_area = visible_width * visible_height
    
    visibility_ratio = visible_area / full_area
    if visibility_ratio <= vis_threshold:
        return True
    else:
        return False


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


def center_to_pixel_bbox(center, size):
    """
    Convert bounding box centers to 2D pixel bounding boxes.
    
    Args:
        center: Tensor or array of shape (2,) representing the center coordinates (x, y)
        size: Integer or tuple representing the width and height of the bounding box
    
    Returns:
        Tensor or array of shape (2, size, size) representing 2D pixel bounding boxes
    """
    
    # Handle both int size and (width, height) tuple
    if isinstance(size, int):
        width, height = size, size
    else:
        width, height = size
    
    # Create empty bounding boxes
    if isinstance(center, torch.Tensor):
        bbox = torch.zeros((2, height, width)).to(center)
    else:
        bbox = np.zeros((2, height, width))
    
    # For each box in the batch
    # Get center coordinates
    cx, cy = center
    
    # Calculate top-left corner
    x0 = cx - width // 2
    y0 = cy - height // 2
    
    # Generate coordinate grids
    for y in range(height):
        for x in range(width):
            # Store absolute pixel coordinates
            bbox[0, y, x] = x0 + x  # x-coordinate
            bbox[1, y, x] = y0 + y  # y-coordinate
    
    return bbox


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


def draw_3d_bbox_on_image_array(image_array, points, K=None):
    """
    Draw a 3D bounding box with center point and coordinate axes on an image.
    
    Args:
        image_array: The input image as a numpy array
        points: Array of shape (9, 3) where the first point is the center and the rest are corners
                OR already projected points with shape (9, 2)
        K: Camera intrinsic matrix (3x3). If None, points are assumed to be already projected
    
    Returns:
        Image with the 3D bounding box, center point, and coordinate axes drawn
    """
    image = image_array.copy()
    
    # Check if we need to project the points
    if K is not None:
        pts_cam = points.T  # shape (3, N)
        pts_img_homog = K @ pts_cam  # shape (3, N)
        pts_2d = (pts_img_homog[:2, :] / pts_img_homog[2:3, :]).T  # shape (N, 2)
    else:
        pts_2d = points
    
    # Extract center point and corners
    center_2d = pts_2d[0]
    corners_2d = pts_2d[1:9]
    
    # Define the bounding box edges (pairs of corner indices)
    # Adjust indices for the original function (0-indexed) to match our data (1-indexed due to center)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # connecting edges
    ]
    
    # Draw the bounding box edges
    for start, end in edges:
        # Add 1 to indices because corners start at index 1 in pts_2d
        pt1 = tuple(corners_2d[start].astype(int))
        pt2 = tuple(corners_2d[end].astype(int))
        cv2.line(image, pt1, pt2, color=(255, 0, 255), thickness=2)  # Magenta for bbox
    
    # Draw corner points
    for point in corners_2d:
        cv2.circle(image, tuple(point.astype(int)), radius=3, color=(0, 255, 255), thickness=-1)  # Cyan corners
    
    # If we have the 3D camera-frame coordinates, compute and draw coordinate axes
    if K is not None:
        # Extract the center and first three corners in 3D camera coordinates
        center_3d = points[0]
        corners_3d = points[1:9]
        
        # Calculate the size of the bounding box for scaling the axes
        bbox_size = np.max(np.linalg.norm(corners_3d - center_3d, axis=1))
        axis_length = bbox_size * 0.7  # Scale axis length to 70% of the maximum distance
        
        # Define coordinate axes in camera frame (relative to center)
        axes_3d = np.array([
            [axis_length, 0, 0],  # X-axis (red)
            [0, axis_length, 0],  # Y-axis (green)
            [0, 0, axis_length]   # Z-axis (blue)
        ])
        
        # Add center to get the end points of axes in camera frame
        axes_end_3d = center_3d + axes_3d
        
        # Project axes end points to image
        axes_end_homog = K @ axes_end_3d.T
        axes_end_2d = (axes_end_homog[:2, :] / axes_end_homog[2:3, :]).T
        
        # Draw coordinate axes
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # RGB colors for X (red), Y (green), Z (blue)
        
        for i, (end_point, color) in enumerate(zip(axes_end_2d, colors)):
            start_point = tuple(center_2d.astype(int))
            end_point = tuple(end_point.astype(int))
            cv2.line(image, start_point, end_point, color, 3)
            
            # Add text labels for axes
            text_pos = (int((start_point[0] + end_point[0]) * 0.6), 
                       int((start_point[1] + end_point[1]) * 0.6))
            axis_label = ['X', 'Y', 'Z'][i]
            cv2.putText(image, axis_label, text_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Draw center point last so it's on top
    cv2.circle(image, tuple(center_2d.astype(int)), radius=5, color=(0, 165, 255), thickness=-1)  # Orange center
    
    return image


def draw_bbox_on_image_array(image_array, bbox, output_path=None):
    """
    Draw a bounding box on an image array and save it as a PNG file.
    
    Args:
        image_array (np.ndarray): NumPy array representing the image
        bbox (np.ndarray or list): Bounding box coordinates in format [x, y, w, h]
                     where (x, y) is the top-left corner and (w, h) is the width and height
        output_path (str, optional): Path where the output image will be saved.
                                    If not provided, the function will return the modified array
    
    Returns:
        np.ndarray: The modified image array with the bounding box drawn on it
                   (if output_path is None), otherwise None
    """
    try:
        # Convert numpy array to PIL Image
        # Ensure the array is in the correct format (uint8)
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
            
        # Handle grayscale images
        if len(image_array.shape) == 2:
            pil_image = Image.fromarray(image_array, mode='L')
            # Convert to RGB for drawing colored bbox
            pil_image = pil_image.convert('RGB')
        elif len(image_array.shape) == 3 and image_array.shape[2] == 1:
            pil_image = Image.fromarray(image_array.squeeze(), mode='L')
            pil_image = pil_image.convert('RGB')
        elif len(image_array.shape) == 3 and image_array.shape[2] == 3:
            pil_image = Image.fromarray(image_array, mode='RGB')
        elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
            pil_image = Image.fromarray(image_array, mode='RGBA')
        else:
            raise ValueError(f"Unsupported image array shape: {image_array.shape}")
        
        # Create a drawing context
        draw = ImageDraw.Draw(pil_image)
        
        # Extract bbox coordinates
        x, y, w, h = bbox
        
        # Calculate bottom right corner
        x2, y2 = x + w, y + h
        
        # Draw the rectangle (bounding box)
        # Using a red color with width=2
        draw.rectangle([x, y, x2, y2], outline="red", width=2)
        
        # If output path is provided, save the image
        if output_path:
            # Ensure output has .png extension
            if not output_path.lower().endswith('.png'):
                output_path += '.png'
                
            pil_image.save(output_path)
            print(f"Image with bounding box saved to {output_path}")
            return None
        else:
            # Convert back to numpy array
            result_array = np.array(pil_image)
            return result_array
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def _check_coplanar(boxes: torch.Tensor, eps: float = 1e-4) -> torch.BoolTensor:
    """
    Checks that plane vertices are coplanar.
    Returns a bool tensor of size B, where True indicates a box is coplanar.
    """
    faces = torch.tensor(_box_planes, dtype=torch.int64, device=boxes.device)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    P, V = faces.shape
    # (B, P, 4, 3) -> (B, P, 3)
    v0, v1, v2, v3 = verts.reshape(B, P, V, 3).unbind(2)

    # Compute the normal
    e0 = F.normalize(v1 - v0, dim=-1)
    e1 = F.normalize(v2 - v0, dim=-1)
    normal = F.normalize(torch.cross(e0, e1, dim=-1), dim=-1)

    # Check the fourth vertex is also on the same plane
    mat1 = (v3 - v0).view(B, 1, -1)  # (B, 1, P*3)
    mat2 = normal.view(B, -1, 1)  # (B, P*3, 1)
    
    return (mat1.bmm(mat2).abs() < eps).view(B)


def _check_nonzero(boxes: torch.Tensor, eps: float = 1e-8) -> torch.BoolTensor:
    """
    Checks that the sides of the box have a non zero area.
    Returns a bool tensor of size B, where True indicates a box is nonzero.
    """
    faces = torch.tensor(_box_triangles, dtype=torch.int64, device=boxes.device)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    T, V = faces.shape
    # (B, T, 3, 3) -> (B, T, 3)
    v0, v1, v2 = verts.reshape(B, T, V, 3).unbind(2)

    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (B, T, 3)
    face_areas = normals.norm(dim=-1) / 2

    return (face_areas > eps).all(1).view(B)


def box3d_overlap(
    boxes_dt: torch.Tensor, boxes_gt: torch.Tensor, 
    eps_coplanar: float = 1e-4, eps_nonzero: float = 1e-8
) -> torch.Tensor:
    """
    Computes the intersection of 3D boxes_dt and boxes_gt.

    Inputs boxes_dt, boxes_gt are tensors of shape (B, 8, 3)
    (where B doesn't have to be the same for boxes_dt and boxes_gt),
    containing the 8 corners of the boxes, as follows:

        (4) +---------+. (5)
            | ` .     |  ` .
            | (0) +---+-----+ (1)
            |     |   |     |
        (7) +-----+---+. (6)|
            ` .   |     ` . |
            (3) ` +---------+ (2)


    NOTE: Throughout this implementation, we assume that boxes
    are defined by their 8 corners exactly in the order specified in the
    diagram above for the function to give correct results. In addition
    the vertices on each plane must be coplanar.
    As an alternative to the diagram, this is a unit bounding
    box which has the correct vertex ordering:

    box_corner_vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]

    Args:
        boxes_dt: tensor of shape (N, 8, 3) of the coordinates of the 1st boxes
        boxes_gt: tensor of shape (M, 8, 3) of the coordinates of the 2nd boxes
    Returns:
        iou: (N, M) tensor of the intersection over union which is
            defined as: `iou = vol / (vol1 + vol2 - vol)`
    """
    # Make sure predictions are coplanar and nonzero 
    invalid_coplanar = ~_check_coplanar(boxes_dt, eps=eps_coplanar)
    invalid_nonzero  = ~_check_nonzero(boxes_dt, eps=eps_nonzero)

    ious = _C.iou_box3d(boxes_dt, boxes_gt)[1]

    # Offending boxes are set to zero IoU
    if invalid_coplanar.any():
        ious[invalid_coplanar] = 0
        print('Warning: skipping {:d} non-coplanar boxes at eval.'.format(int(invalid_coplanar.float().sum())))
    
    if invalid_nonzero.any():
        ious[invalid_nonzero] = 0
        print('Warning: skipping {:d} zero volume boxes at eval.'.format(int(invalid_nonzero.float().sum())))

    return ious