import os
import cv2
import json
import numpy as np
from PIL import Image
from scipy.stats import truncnorm
from scipy.spatial.transform import Rotation as R
from dataset.base_dataset import BaseDataset
from utils.utils import get_2d_coord_np, crop_resize_by_warp_affine, get_coarse_mask
import zipfile
import torch


class objectron_2(BaseDataset):
    def __init__(self, data_path, data_name, data_type, feat_3d_path, virtual_focal,
                 is_train=True, scale_size=420):
        super().__init__()

        self.scale_size = scale_size
        self.virtual_focal = virtual_focal

        self.is_train = is_train
        self.objectron_path = os.path.join(data_path, "objectron_origin")
        self.omninocs_objectron_path = os.path.join(data_path, "omninocs_release_objectron")
        self.omninocs_annotation_path = os.path.join(self.omninocs_objectron_path, "seperate_annotation")
        with open(os.path.join(self.omninocs_objectron_path, "frame_mask_obj_list.json"), 'r') as f:
            self.omninocs_frame_mask_obj_list = json.load(f)

        # # Load zip file
        # self.rgb_data_zip_list = {}
        # for file in os.listdir(self.objectron_path):
        #     if file.endswith('.zip'):
        #         category_name = file.split('.')[0]
        #         self.rgb_data_zip_list[category_name] = zipfile.ZipFile(os.path.join(self.objectron_path, file), 'r')
        # self.omninocs_zip_data = zipfile.ZipFile(os.path.join(self.omninocs_objectron_path, 'objectron.zip'), 'r')
        
        self.rgb_data_zip_paths = {
            file.split('.')[0]: os.path.join(self.objectron_path, file)
            for file in os.listdir(self.objectron_path) if file.endswith('.zip')
        }
        self.omninocs_zip_path = os.path.join(self.omninocs_objectron_path, 'objectron.zip')
        # Cache for storing ZIP files per worker
        self._worker_zip_cache = {}

        self.data_list = []

        with open(f"dataset_collect/objectron_valid_instances_{data_type}.json", "r") as f:
            valid_instances = json.load(f)

        for valid_instance in valid_instances:
            scene_id = valid_instance["scene_id"]
            scene_path = os.path.join(self.omninocs_annotation_path, scene_id)

            object_id = valid_instance["object_id"]
            object_json = f"{object_id}.json"
            with open(os.path.join(scene_path, object_json), 'r') as f:
                omninocs_annotation_list_of_object = json.load(f)

            frame = omninocs_annotation_list_of_object[valid_instance["frame_idx"]]
            info_dict = {
                'scene': scene_id,
                'annotation': frame,
                'object_id': object_id
            }

            self.data_list.append(info_dict)
        
        phase = 'train' if is_train else 'test'
        print("Dataset: OmniNOCS Objectron")
        print("# of %s images: %d" % (phase, len(self.data_list)))

    def __len__(self):
        return len(self.data_list)

    def _get_worker_zip(self):
        """Ensure each worker opens its own ZIP file once and keeps it open."""
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0  # Assign ID 0 for single-threaded mode

        # Check if the worker already has ZIP files open
        if worker_id not in self._worker_zip_cache:
            self._worker_zip_cache[worker_id] = {}

            # Open RGB ZIPs per worker
            for category, zip_path in self.rgb_data_zip_paths.items():
                self._worker_zip_cache[worker_id][category] = zipfile.ZipFile(zip_path, 'r')

            # Open Omninocs ZIP per worker
            self._worker_zip_cache[worker_id]["omninocs"] = zipfile.ZipFile(self.omninocs_zip_path, 'r')

        return self._worker_zip_cache[worker_id]

    def __getitem__(self, idx):
        frame_info = self.data_list[idx]
        # scene, view, cam, gt, gt_info, meta, feat_3d = info['scene'], info['view'], info['cam'], info['gt'], info['gt_info'], info['meta'], info['3d_feat']
        frame_annotation = frame_info['annotation']
        category_name = frame_annotation['image_name'].split('/')[0]

        cam_K = [
            frame_annotation["intrinsics"]["fx"], 0.0, frame_annotation["intrinsics"]["cx"],
            0.0, frame_annotation["intrinsics"]["fy"], frame_annotation["intrinsics"]["cy"],
            0.0, 0.0, 1.0
        ]
        cam_K = np.asarray(cam_K).reshape(3, 3)

        object_annotation = {}
        for obj in frame_annotation['objects']:
            if obj["object_id"] == frame_info["object_id"]:
                object_annotation = obj
                break

        cam_R_m2c, cam_t_m2c, obj_id = object_annotation['rotation'], object_annotation['translation'], object_annotation['object_id']
        cam_R_m2c = np.asarray(cam_R_m2c).reshape(3, 3)
        cam_t_m2c = np.asarray(cam_t_m2c).reshape(1, 3)
        keypoints_3d = self.get_kpt_3d(object_annotation) # key points in 3d model frame
        diag = np.linalg.norm(keypoints_3d[1] - keypoints_3d[8])

        keypoints_2d = self.get_kpt_2d(keypoints_3d, cam_R_m2c, cam_t_m2c, cam_K) # (9, 3), key points on 2d pixel plane
        # obj_z_gt = keypoints_2d[0, 2] # depth of model centroid in camera frame
        keypoints_2d = keypoints_2d[..., 0:2] / keypoints_2d[..., 2:] # (9, 2), homogeneous 2d key points coordinates
        # obj_centroid_2d = keypoints_2d[0, :2]

        bbox = [np.min(keypoints_2d[:, 0]), np.min(keypoints_2d[:, 1]), 
                np.max(keypoints_2d[:, 0])-np.min(keypoints_2d[:, 0]),
                np.max(keypoints_2d[:, 1])-np.min(keypoints_2d[:, 1])]

        rgb_path = frame_annotation["image_name"] + '.png'
        mask_path = frame_annotation["omninocs_name"] + '_instances.png'
        nocs_path = frame_annotation["omninocs_name"] + '_nocs.png'
        
        # Get worker-specific ZIP files
        zip_files = self._get_worker_zip()

        # with self.rgb_data_zip_list[category_name].open(rgb_path) as rgb_file:
        with zip_files[category_name].open(rgb_path) as rgb_file:
            image = np.frombuffer(rgb_file.read(), np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W = image.shape[:2]
        
        # with self.omninocs_zip_data.open(mask_path) as mask_file:
        with zip_files["omninocs"].open(mask_path) as mask_file:
            mask = np.frombuffer(mask_file.read(), np.uint8)
            mask = cv2.imdecode(mask, cv2.IMREAD_UNCHANGED)

        # with self.omninocs_zip_data.open(nocs_path) as nocs_file:
        with zip_files["omninocs"].open(nocs_path) as nocs_file:
            nocs_image = np.frombuffer(nocs_file.read(), np.uint8)
            nocs_image = cv2.imdecode(nocs_image, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
            
        if W != 360:
            pad_width = 360 - W  # Amount of padding needed on the right
            image = np.pad(image, ((0, 0), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
            mask = np.pad(mask, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
            nocs_image = np.pad(nocs_image, ((0, 0), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
            W = 360

        raw_scene = image.copy()
        scene_padded_resized, bbox_resized, cam_K_resized, mask_resized, nocs_resized, scale_ratio = self.preprocess(raw_scene, bbox, cam_K, mask, nocs_image)

        keypoints_2d_resized = self.get_kpt_2d(keypoints_3d, cam_R_m2c, cam_t_m2c, cam_K_resized) # (9, 3), key points on 2d pixel plane
        obj_z_gt = keypoints_2d_resized[0, 2] # depth of model centroid in camera frame
        keypoints_2d_resized = keypoints_2d_resized[..., 0:2] / keypoints_2d_resized[..., 2:] # (9, 2), homogeneous 2d key points coordinates
        obj_centroid_2d_resized = keypoints_2d_resized[0, :2]

        # c, s = self.xywh2cs(bbox)
        c, s = self.xywh2cs(bbox_resized)

        initial_focal = (cam_K[0, 0] + cam_K[1, 1]) / 2
        focal_ratio = initial_focal / self.virtual_focal

        ratio_scalar = scale_ratio * focal_ratio

        # relative translation offset from the bbox centeroid to object centroid on 2d pixel plane
        delta_c = obj_centroid_2d_resized - c
        trans_ratio = np.asarray([delta_c[0] / s, delta_c[1] / s, obj_z_gt / ratio_scalar]).astype(np.float32)
        
        rgb, c_h_, c_w_, s_, roi_coord_2d = self.zoom_in_v2(scene_padded_resized, c, s, res=self.scale_size, return_roi_coord=True) # center-cropped rgb
        
        mask, *_ = self.zoom_in_v2(mask_resized, c, s, res=self.scale_size, interpolate=cv2.INTER_NEAREST) # center-cropped mask
        mask = (mask == frame_info["object_id"])
        rgb[mask[:, :, None] != [True, True, True]] = 0 # further align the background, especially in case the cropped rgb is outside the boundary of raw image
    
        nocs, *_ = self.zoom_in_v2(nocs_resized, c, s, res=self.scale_size)

        nocs[np.logical_not(mask)] = 0
        mask[np.sum(np.logical_or(nocs > 1, nocs < 0), axis=-1) != 0] = False
    
        mask_vis = get_coarse_mask(mask, scale_factor=8)
        nocs_vis = np.stack([cv2.resize(nocs[..., i], (256, 256), interpolation=cv2.INTER_CUBIC) for i in range(3)], axis=-1)
        nocs_vis[np.logical_not(mask_vis)] = 0

        nocs_mask = np.ones_like(nocs)
        bg_pixels = np.all(nocs == [0, 0, 0], axis=-1)
        nocs_mask[bg_pixels] = [0, 0, 0]
        
        c = np.array([c_w_, c_h_])
        s = s_

        out_dict = {
            'raw_scene': raw_scene.transpose((2, 0, 1)), # (3, H, W), dtype = np.uint8
            'input_scene': (scene_padded_resized.transpose((2, 0, 1)) / 255).astype(np.float32), # (3, 490, 490)
            'bbox_size': s,
            'bbox_center': c,
            'gt_bbox_2d': np.array(bbox_resized).astype(np.float32), # (4,)
            'roi_coord_2d': roi_coord_2d.astype(np.float32), # (2, 32, 32)
            'gt_r': cam_R_m2c.astype(np.float32),
            'gt_t': cam_t_m2c.reshape(-1).astype(np.float32),
            'cam_K': cam_K.astype(np.float32),
            'cam_K_resized': cam_K_resized.astype(np.float32),
            'resize_ratio': np.array([ratio_scalar], dtype=np.float32),
            'gt_trans_ratio': trans_ratio.reshape(-1).astype(np.float32),
            'mask': mask, # (32, 32)
            'nocs': nocs.transpose((2, 0, 1)).astype(np.float32), # (3, 32, 32)
            'nocs_mask': nocs_mask[:, :, 0], # (32, 32)
            'nocs_vis': nocs_vis.transpose((2, 0, 1)).astype(np.float32), # (3, 256, 256)
            'mask_vis': mask_vis, # (256, 256)
            'kps_3d_m': keypoints_3d.astype(np.float32), # (9, 3)
            'class_name': category_name,
            # '3d_feat': feat_3d.astype(np.float32) # (1024, 387)
            'img_name': frame_annotation["image_name"]
        }

        return out_dict

    def get_kpt_2d(self, kpt_3d, R_m2c, t_m2c, K):
        kpt_2d = (kpt_3d @ R_m2c.T + t_m2c) @ K.T
        return kpt_2d

    @staticmethod
    def get_kpt_3d(object_annotation):
        size = object_annotation["size"]

        # Define 3D bounding box in object space
        l, w, h = size[0] / 2, size[1] / 2, size[2] / 2
        keypoints = np.array([
            [0.0, 0.0, 0.0], [-l, -w, -h], [-l, -w, h], [-l, w, -h], [-l, w, h],  
            [l, -w, -h], [l, -w, h], [l, w, -h], [l, w, h]      
        ])
        return keypoints

    @staticmethod
    def get_intr(h, w):
        fx = fy = 1422.222
        res_raw = 1024
        f_x = f_y = fx * h / res_raw
        K = np.array([f_x, 0, w / 2, 0, f_y, h / 2, 0, 0, 1]).reshape(3, 3)
        return K
    
    @staticmethod
    def read_camera_matrix_single(json_file):
        with open(json_file, 'r', encoding='utf8') as reader:
            json_content = json.load(reader)
        camera_matrix = np.eye(4)
        camera_matrix[:3, 0] = np.array(json_content['x'])
        camera_matrix[:3, 1] = -np.array(json_content['y'])
        camera_matrix[:3, 2] = -np.array(json_content['z'])
        camera_matrix[:3, 3] = np.array(json_content['origin'])

        c2w = camera_matrix
        flip_yz = np.eye(4)
        flip_yz[1, 1] = -1
        flip_yz[2, 2] = -1
        c2w = np.matmul(c2w, flip_yz)
        
        T_ = np.eye(4)
        T_[:3, :3] = R.from_euler('x', -90, degrees=True).as_matrix()
        c2w = np.matmul(T_, c2w)

        w2c = np.linalg.inv(c2w)

        return w2c[0:3, 0:3], w2c[0:3, 3].reshape(1, 1, 3) * 1000
    
    @staticmethod
    def K_dpt2cld(dpt, cam_scale, K):
        dpt = dpt.astype(np.float32)
        dpt /= cam_scale

        Kinv = np.linalg.inv(K)

        h, w = dpt.shape[0], dpt.shape[1]

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        ones = np.ones((h, w), dtype=np.float32)
        x2d = np.stack((x, y, ones), axis=2).reshape(w*h, 3)

        # backproj
        R = np.dot(Kinv, x2d.transpose())

        # compute 3D points
        X = R * np.tile(dpt.reshape(1, w*h), (3, 1))
        X = np.array(X).transpose()

        X = X.reshape(h, w, 3)
        return X
    
    @staticmethod
    def xywh2cs_dzi(xywh, base_ratio=1.5, sigma=1, shift_ratio=0.25, box_ratio=0.25, wh_max=480):
        # copy from
        # https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi/blob/master/lib/utils/img.py
        x, y, w, h = xywh
        shift = truncnorm.rvs(-shift_ratio / sigma, shift_ratio / sigma, scale=sigma, size=2)
        scale = 1+truncnorm.rvs(-box_ratio / sigma, box_ratio / sigma, scale=sigma, size=1)
        assert scale > 0
        center = np.array([x+w*(0.5+shift[1]), y+h*(0.5+shift[0])])
        wh = max(w, h) * base_ratio * scale
        if wh_max != None:
            wh = min(wh, wh_max)
        return center, wh

    @staticmethod
    def xywh2cs(xywh, base_ratio=1.0):
        x, y, w, h = xywh
        center = np.array((x+0.5*w, y+0.5*h)) # [c_w, c_h]
        wh = max(w, h) * base_ratio
        return center, wh
    
    @staticmethod
    def zoom_in_v2(im, c, s, res=480, interpolate=cv2.INTER_LINEAR, return_roi_coord=False):
        """
        copy from
        https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi/blob/master/lib/utils/img.py
        zoom in on the object with center c and size s, and resize to resolution res.
        :param im: nd.array, single-channel or 3-channel image
        :param c: (w, h), object center
        :param s: scalar, object size
        :param res: target resolution
        :param channel:
        :param interpolate:
        :return: zoomed object patch
        """
        c_w, c_h = c
        c_w, c_h, s, res = int(c_w), int(c_h), int(s), int(res)
        ndim = im.ndim
        if ndim == 2:
            im = im[..., np.newaxis]

        max_h, max_w = im.shape[0:2]
        s = s

        im_crop = crop_resize_by_warp_affine(im, np.array([c_w, c_h]), s, res).astype(np.float32)

        if return_roi_coord:
            coord_2d = get_2d_coord_np(max_w, max_h, fmt="HWC")
            roi_coord_2d = crop_resize_by_warp_affine(coord_2d, np.array([c_w, c_h]), s, res, interpolation=interpolate).transpose(2, 0, 1).astype(np.float32) # HWC -> CHW
            return im_crop, c_h, c_w, s, roi_coord_2d

        return im_crop, c_h, c_w, s
    
    def preprocess(self, image, bbox, K, mask, nocs, target_size=(490, 490)):
        """
        Pad image to square and resize while adjusting bounding boxes
        
        Args:
            image: numpy array of shape (H, W, 3)
            bboxes: list of [x, y, w, h] coordinates in original image
            K: initial camera instrinsic
            mask: numpy array of shape (H, W)
            nocs: numpy array of shape (H, W, 3)
            target_size: desired output size (height, width)
        
        Returns:
            padded_resized_img: numpy array of padded and resized image
            adjusted_bboxes: list of adjusted bounding boxes
        """
        H, W, _ = image.shape
        max_dim = max(H, W)
        
        # Create a square black canvas
        padded_img = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        padded_mask = np.zeros((max_dim, max_dim), dtype=mask.dtype)
        padded_nocs = np.zeros((max_dim, max_dim, 3), dtype=nocs.dtype)
        
        # Paste original image onto canvas (centered)
        pad_top = (max_dim - H) // 2
        pad_left = (max_dim - W) // 2

        padded_img[pad_top:pad_top+H, pad_left:pad_left+W, :] = image
        padded_mask[pad_top:pad_top+H, pad_left:pad_left+W] = mask
        padded_nocs[pad_top:pad_top+H, pad_left:pad_left+W, :] = nocs

        K_pad = K.copy()
        K_pad[0,2] += pad_left
        K_pad[1,2] += pad_top
        
        # Calculate scale factor for resizing
        scale_factor = target_size[0] / max_dim
        
        # Adjust bboxes for padding and resizing
        x, y, w, h = bbox
        
        # Add padding offsets to top-left coordinates
        x_padded = x + pad_left
        y_padded = y + pad_top
        
        # Width and height remain the same after padding
        w_padded = w
        h_padded = h
        
        # Apply scaling for resize
        x_adj = x_padded * scale_factor
        y_adj = y_padded * scale_factor
        w_adj = w_padded * scale_factor
        h_adj = h_padded * scale_factor
            
        adjusted_bboxes = np.array([x_adj, y_adj, w_adj, h_adj])

        K_resized = K_pad.copy()
        K_resized[0,:] *= scale_factor
        K_resized[1,:] *= scale_factor
        
        # Resize images
        S_h, S_w = target_size
        padded_resized_img = cv2.resize(padded_img, (S_w, S_h), interpolation=cv2.INTER_LINEAR)
        padded_resized_mask = cv2.resize(padded_mask, (S_w, S_h), interpolation=cv2.INTER_NEAREST)
        padded_resized_nocs = cv2.resize(padded_nocs, (S_w, S_h), interpolation=cv2.INTER_LINEAR)

        
        return padded_resized_img, adjusted_bboxes, K_resized, padded_resized_mask, padded_resized_nocs, scale_factor
