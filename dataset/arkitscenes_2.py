import os
import cv2
import json
import numpy as np
from PIL import Image
from scipy.stats import truncnorm
from scipy.spatial.transform import Rotation as R
from dataset.base_dataset import BaseDataset
from utils.utils import get_2d_coord_np, crop_resize_by_warp_affine, get_coarse_mask, draw_bbox_on_image_array
import torch


class arkitscenes_2(BaseDataset):
    def __init__(self, data_path, data_name, data_type, feat_3d_path,
                 is_train=True, scale_size=420):
        super().__init__()

        self.scale_size = scale_size

        self.is_train = is_train
        self.arkitscenes_path = os.path.join(data_path, "ARKitScenes")
        self.omninocs_arkitscenes_path = os.path.join(data_path, "omninocs_release_ARKitScenes")
        self.omninocs_annotation_path = os.path.join(self.omninocs_arkitscenes_path, "seperate_annotation")
        with open(os.path.join(self.omninocs_arkitscenes_path, "frame_mask_obj_list.json"), 'r') as f:
            self.omninocs_frame_mask_obj_list = json.load(f)

        self.data_list = []

        with open(f"dataset_collect/arkitscenes_valid_instances_{data_type}.json", "r") as f:
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
        print("Dataset: OmniNOCS ARKitScenes")
        print("# of %s images: %d" % (phase, len(self.data_list)))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        frame_info = self.data_list[idx]
        # scene, view, cam, gt, gt_info, meta, feat_3d = info['scene'], info['view'], info['cam'], info['gt'], info['gt_info'], info['meta'], info['3d_feat']
        frame_annotation = frame_info['annotation']
        assert len(frame_annotation["objects"]) == 1

        category_name = frame_annotation["objects"][0]["category"]

        cam_K = [
            frame_annotation["intrinsics"]["fx"], 0.0, frame_annotation["intrinsics"]["cx"],
            0.0, frame_annotation["intrinsics"]["fy"], frame_annotation["intrinsics"]["cy"],
            0.0, 0.0, 1.0
        ]
        cam_K = np.asarray(cam_K).reshape(3, 3)

        # object_annotation = {}
        # for obj in frame_annotation['objects']:
        #     if obj["object_id"] == frame_info["object_id"]:
        #         object_annotation = obj
        #         break

        assert frame_annotation['objects'][0]["object_id"] == frame_info["object_id"]
        object_annotation = frame_annotation['objects'][0]

        cam_R_m2c, cam_t_m2c, obj_id = object_annotation['rotation'], object_annotation['translation'], object_annotation['object_id']
        cam_R_m2c = np.asarray(cam_R_m2c).reshape(3, 3)
        cam_t_m2c = np.asarray(cam_t_m2c).reshape(1, 1, 3)
        keypoints3d = self.get_keypoints(object_annotation) # key points in 3d model frame

        diag = np.linalg.norm(keypoints3d[0, 1] - keypoints3d[0, 8])
        keypoints_2d = (keypoints3d @ cam_R_m2c.T + cam_t_m2c) @ cam_K.T # n * 9 * 3, key points on 2d pixel plane
        obj_z_gt = keypoints_2d[0, 0, 2] # depth of model centroid in camera frame
        keypoints_2d = keypoints_2d[..., 0:2] / keypoints_2d[..., 2:] # n * 9 * 2, homogeneous 2d key points coordinates
        obj_centroid_2d = keypoints_2d[0, 0, :2]

        bbox = [np.min(keypoints_2d[0, :, 0]), np.min(keypoints_2d[0, :, 1]), 
                np.max(keypoints_2d[0, :, 0])-np.min(keypoints_2d[0, :, 0]),
                np.max(keypoints_2d[0, :, 1])-np.min(keypoints_2d[0, :, 1])]

        rgb_path = frame_annotation["image_name"]
        mask_path = frame_annotation["omninocs_name"] + '_instances.png'
        nocs_path = frame_annotation["omninocs_name"] + '_nocs.png'

        image = cv2.imread(os.path.join(self.arkitscenes_path, rgb_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W = image.shape[:2]

        mask_raw = cv2.imread(os.path.join(self.omninocs_arkitscenes_path, mask_path), cv2.IMREAD_UNCHANGED)

        nocs_image_raw = cv2.imread(os.path.join(self.omninocs_arkitscenes_path, nocs_path), cv2.IMREAD_COLOR)
        nocs_image_raw = cv2.cvtColor(nocs_image_raw, cv2.COLOR_BGR2RGB)
        nocs_image_raw = nocs_image_raw.astype(np.float32) / 255.0
        
        mask = cv2.resize(mask_raw, (W, H), cv2.INTER_NEAREST)
        nocs_image = np.stack([cv2.resize(nocs_image_raw[..., i], (W, H), interpolation=cv2.INTER_CUBIC) for i in range(3)], axis=-1)
    
        raw_scene = image.copy()
        scene_padded_resized, bbox_resized = self.pad_and_resize_with_bbox_adjustment(raw_scene, bbox)

        # #################
        # draw_bbox_on_image_array(image, bbox, f"test_pose/z_img_{idx}.png")
        # draw_bbox_on_image_array(scene_padded_resized, bbox_resized, f"test_pose/z_img_resized_{idx}.png")
        # #################

        image[mask != frame_info["object_id"]] = 0 # remove background

        c, s = self.xywh2cs(bbox, wh_max=self.scale_size)
        c_resized, s_resized = self.xywh2cs(bbox_resized, wh_max=self.scale_size)

        # relative translation offset from the bbox centeroid to object centroid on 2d pixel plane
        delta_c = obj_centroid_2d - c
        trans_ratio = np.asarray([delta_c[0] / s, delta_c[1] / s, obj_z_gt / (self.scale_size / s)]).astype(np.float32)

        interpolate = cv2.INTER_NEAREST
        # interpolate = cv2.INTER_LINEAR
        rgb, c_h_, c_w_, s_, roi_coord_2d = self.zoom_in_v2(image, c, s, res=self.scale_size, return_roi_coord=True) # center-cropped rgb

        mask, *_ = self.zoom_in_v2(mask, c, s, res=self.scale_size, interpolate=interpolate) # center-cropped mask
        mask = (mask == frame_info["object_id"])
        rgb[mask[:, :, None] != [True, True, True]] = 0 # further align the background, especially in case the cropped rgb is outside the boundary of raw image

        nocs, *_ = self.zoom_in_v2(nocs_image, c, s, res=self.scale_size, interpolate=interpolate) 
        nocs[np.logical_not(mask)] = 0
        nocs_resized = np.stack([cv2.resize(nocs[..., i], (self.scale_size//14, self.scale_size//14), interpolation=cv2.INTER_CUBIC) for i in range(3)], axis=-1)

        mask[np.sum(np.logical_or(nocs > 1, nocs < 0), axis=-1) != 0] = False
        mask_resized = get_coarse_mask(mask, scale_factor=(1 / 14))
        nocs_resized[np.logical_not(mask_resized)] = 0

        nocs_mask = np.ones_like(nocs_resized)
        bg_pixels = np.all(nocs_resized == [0, 0, 0], axis=-1)
        nocs_mask[bg_pixels] = [0, 0, 0]

        c = np.array([c_w_, c_h_])
        s = s_
        keypoints_2d = (keypoints_2d - c.reshape(1, 1, 2)) / s  # * self.scale_size

        if self.is_train:
            rgb = self.augment_training_data(rgb.astype(np.uint8))

        # if mask_resized.sum() < 32 or nocs_mask[:, :, 0].sum() < 32:
        #     idx += 1
        #     if idx >= len(self.data_list):
        #         idx = 0
        # else:
        out_dict = {
            'raw_scene': raw_scene.transpose((2, 0, 1)), # (3, H, W), dtype = np.uint8
            'input_scene': (scene_padded_resized.transpose((2, 0, 1)) / 255).astype(np.float32), # (3, 490, 490)
            'image': (rgb.transpose((2, 0, 1)) / 255).astype(np.float32), # (3, 490, 490)
            'bbox_size': s,
            'bbox_center': c,
            'bbox_size_resized': s_resized,
            'bbox_center_resized': c_resized,
            'gt_bbox_2d': np.array(bbox).astype(np.float32), # (4,)
            'roi_coord_2d': roi_coord_2d.astype(np.float32), # (2, 490, 490)
            'gt_r': cam_R_m2c.astype(np.float32),
            'gt_t': cam_t_m2c.reshape(-1).astype(np.float32),
            'cam': cam_K.astype(np.float32),
            'resize_ratio': np.array([self.scale_size / s], dtype=np.float32),
            'gt_trans_ratio': trans_ratio.reshape(-1).astype(np.float32),
            'mask': mask, # (490, 490)
            'mask_resized': mask_resized, # (32, 32)
            'nocs': nocs.transpose((2, 0, 1)).astype(np.float32), # (3, 490, 490)
            'nocs_resized': nocs_resized.transpose((2, 0, 1)).astype(np.float32), # (3, 35, 35)
            'nocs_mask': nocs_mask[:, :, 0],
            # 'kps': kp_i.astype(np.float32),
            'kps_3d_m': keypoints3d[0].astype(np.float32), # (9, 3)
            'kps_3d_center': keypoints3d[0, 0].astype(np.float32), # (3)
            'kps_3d_dig': np.array([diag], dtype=np.float32),
            'class_name': category_name,
            # '3d_feat': feat_3d.astype(np.float32) # (1024, 387)
            'img_name': frame_annotation["image_name"]
        }

        return out_dict

    @staticmethod
    def get_keypoints(object_annotation, dt=5):
        size = object_annotation["size"]

        # Define 3D bounding box in object space
        l, w, h = size[0] / 2, size[1] / 2, size[2] / 2
        keypoints = np.array([
            [0.0, 0.0, 0.0], [-l, -w, -h], [-l, -w, h], [-l, w, -h], [-l, w, h],  
            [l, -w, -h], [l, -w, h], [l, w, -h], [l, w, h]      
        ])
        keypoints = [keypoints]
        if 'symmetries_discrete' in object_annotation:
            mats = [np.asarray(mat_list).reshape(4, 4) for mat_list in object_annotation['symmetries_discrete']]
            for mat in mats:
                curr = keypoints[0] @ mat[0:3, 0:3].T + mat[0:3, 3:].T
                keypoints.append(curr)
        elif 'symmetries_continuous' in object_annotation:
            # todo: consider multiple symmetries
            ao = object_annotation['symmetries_continuous'][0]
            axis = np.asarray(ao['axis'])
            offset = np.asarray(ao['offset'])
            angles = np.deg2rad(np.arange(dt, 180, dt))
            rotvecs = axis.reshape(1, 3) * angles.reshape(-1, 1)
            # https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Rotation_vector
            rots = R.from_rotvec(rotvecs).as_matrix()
            for rot in rots:
                curr = keypoints[0] @ rot.T + offset.reshape(1, 3)
                keypoints.append(curr)
        keypoints = np.stack(keypoints, axis=0)
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
    def xywh2cs(xywh, base_ratio=1.0, wh_max=480):
        x, y, w, h = xywh
        center = np.array((x+0.5*w, y+0.5*h)) # [c_w, c_h]
        wh = max(w, h) * base_ratio
        # if wh_max != None:
        #     wh = min(wh, wh_max)
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

    def pad_and_resize_with_bbox_adjustment(self, image, bbox, target_size=(490, 490)):
        """
        Pad image to square and resize while adjusting bounding boxes
        
        Args:
            image: numpy array of shape (H, W, 3)
            bboxes: list of [x, y, w, h] coordinates in original image
            target_size: desired output size (height, width)
        
        Returns:
            padded_resized_img: numpy array of padded and resized image
            adjusted_bboxes: list of adjusted bounding boxes
        """
        H, W, _ = image.shape
        max_dim = max(H, W)
        
        # Create a square black canvas
        padded_img = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        
        # Paste original image onto canvas (centered)
        pad_top = (max_dim - H) // 2
        pad_left = (max_dim - W) // 2
        padded_img[pad_top:pad_top+H, pad_left:pad_left+W, :] = image
        
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
        
        # Resize image
        padded_resized_img = Image.fromarray(padded_img).resize(target_size, Image.BILINEAR)
        padded_resized_img = np.array(padded_resized_img)
        
        return padded_resized_img, adjusted_bboxes
