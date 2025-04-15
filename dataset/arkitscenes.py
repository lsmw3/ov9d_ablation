import os
import cv2
import json
import numpy as np
from scipy.stats import truncnorm
from scipy.spatial.transform import Rotation as R
from dataset.base_dataset import BaseDataset
from utils.utils import get_2d_coord_np, crop_resize_by_warp_affine, get_coarse_mask
import torch


class arkitscenes(BaseDataset):
    def __init__(self, data_path, data_name, data_type, feat_3d_path, xyz_bin: int=64,
                 is_train=True, scale_size=420, num_view=50):
        super().__init__()

        self.scale_size = scale_size
        self.xyz_bin = xyz_bin

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

        # valid_instances = []

        # self.data_path = os.path.join(self.omninocs_arkitscenes_path, f"arkitscenes_{data_type}.json")
        # with open(self.data_path, "r") as f:
        #     self.scene_list = json.load(f)
        # for scene_id in self.scene_list:
        #     scene_path = os.path.join(self.omninocs_annotation_path, scene_id)
        #     if not os.path.isdir(scene_path):
        #         continue

        #     for object_json in os.listdir(scene_path):
        #         object_id = int(object_json.split('.')[0])
        #         with open(os.path.join(scene_path, object_json), 'r') as f:
        #             omninocs_annotation_list_of_object = json.load(f)
        #         for frame in omninocs_annotation_list_of_object:
        #             if object_id not in self.omninocs_frame_mask_obj_list.get(frame["image_name"], []):
        #                 continue
        #             # feat_3d_points = np.load(os.path.join(feat_3d_path, scene_id, "3d_feat.npy"))
        #             info_dict = {
        #                 'scene': scene_id,
        #                 'annotation': frame,
        #                 'object_id': object_id
        #                 # '3d_feat': feat_3d_points # (1024, 387)
        #             }
        #             if not self.check_valid_instance(info_dict):
        #                 continue
        #             else:
        #                 valid_instance = {
        #                     'scene_id': scene_id,
        #                     'object_id': object_id,
        #                     'frame_idx': omninocs_annotation_list_of_object.index(frame)
        #                 }
                    
        #             valid_instances.append(valid_instance)

        #             self.data_list.append(info_dict)

        # with open(f"dataset_collect/arkitscenes_valid_instances_{data_type}.json", "w") as f:
        #     json.dump(valid_instances, f, indent=2)
        
        phase = 'train' if is_train else 'test'
        print("Dataset: OmniNOCS ARKitScenes")
        print("# of %s images: %d" % (phase, len(self.data_list)))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        while True:
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
                
            # if W != 360:
            #     pad_width = 360 - W  # Amount of padding needed on the right
            #     image = np.pad(image, ((0, 0), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
            #     mask = np.pad(mask, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
            #     nocs_image = np.pad(nocs_image, ((0, 0), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
            #     W = 360
            raw_image = image.copy()

            mask = cv2.resize(mask_raw, (W, H), cv2.INTER_NEAREST)
            nocs_image = np.stack([cv2.resize(nocs_image_raw[..., i], (W, H), interpolation=cv2.INTER_CUBIC) for i in range(3)], axis=-1)

            image[mask != frame_info["object_id"]] = 0 # remove background

            c, s = self.xywh2cs(bbox, wh_max=self.scale_size)

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

            # label each x, y, z channel into intergers between [0, bin-1], and the bg is bin
            nocs_x, nocs_y, nocs_z = nocs[:, :, 0], nocs[:, :, 1], nocs[:, :, 2]
            for nocs_plane in (nocs_x, nocs_y, nocs_z):
                nocs_plane[nocs_plane < 0] = 0
                nocs_plane[nocs_plane > 0.999999] = 0.999999
            gt_x_bin = np.asarray(nocs_x * self.xyz_bin, dtype=np.uint8) # intergers interval [0, bin-1], bin is the background
            gt_y_bin = np.asarray(nocs_y * self.xyz_bin, dtype=np.uint8)
            gt_z_bin = np.asarray(nocs_z * self.xyz_bin, dtype=np.uint8)
            gt_x_bin[np.logical_not(mask)] = self.xyz_bin
            gt_y_bin[np.logical_not(mask)] = self.xyz_bin
            gt_z_bin[np.logical_not(mask)] = self.xyz_bin
            gt_xyz_bin = np.stack([gt_x_bin, gt_y_bin, gt_z_bin], axis=0) # (3, 480, 480)

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

            gt_coord_x = np.arange(W)
            gt_coord_y = np.arange(H)
            gt_coord_xy = np.asarray(np.meshgrid(gt_coord_x, gt_coord_y)).transpose(1, 2 ,0)
            gt_coord_2d = crop_resize_by_warp_affine(gt_coord_xy, c, s, self.scale_size, interpolation=cv2.INTER_NEAREST).astype(np.float32) # HWC
            
            dis_sym = np.zeros((3, 4, 4))
            if 'symmetries_discrete' in object_annotation:
                mats = np.asarray([np.asarray(mat_list).reshape(4, 4) for mat_list in object_annotation['symmetries_discrete']])
                dis_sym[:mats.shape[0]] = mats
            con_sym = np.zeros((3, 6))
            if 'symmetries_continuous' in object_annotation:
                for i, ao in enumerate(object_annotation['symmetries_continuous']):
                    axis = np.asarray(ao['axis'])
                    offset = np.asarray(ao['offset'])
                    con_sym[i] = np.concatenate([axis, offset])

            if self.is_train:
                rgb = self.augment_training_data(rgb.astype(np.uint8))
            
            if mask_resized.sum() < 32 or nocs_mask[:, :, 0].sum() < 32:
                idx += 1
                if idx >= len(self.data_list):
                    idx = 0
            else:
                out_dict = {
                    'raw_scene': raw_image.transpose((2, 0, 1)), # (3, H, W), dtype = np.uint8
                    'image': (rgb.transpose((2, 0, 1)) / 255).astype(np.float32), # (3, 490, 490)
                    'gt_xyz_bin': gt_xyz_bin, # (3, H, W), dtype = np.uint8
                    'gt_coord_2d': gt_coord_2d.astype(np.float32), # (490, 490, 2)
                    'bbox_size': s,
                    'bbox_center': c,
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
                    'dis_sym': dis_sym.astype(np.float32),
                    'con_sym': con_sym.astype(np.float32),
                    'class_name': category_name,
                    # '3d_feat': feat_3d.astype(np.float32) # (1024, 387)
                    'img_name': frame_annotation["image_name"]
                }
                break

        return out_dict

    def check_valid_instance(self, frame_info):
        frame_annotation = frame_info['annotation']

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

        c, s = self.xywh2cs(bbox, wh_max=self.scale_size)

        mask, *_ = self.zoom_in_v2(mask, c, s, res=self.scale_size, interpolate=cv2.INTER_NEAREST)
        mask = (mask == frame_info["object_id"])

        nocs, *_ = self.zoom_in_v2(nocs_image, c, s, res=self.scale_size, interpolate=cv2.INTER_NEAREST)

        nocs[np.logical_not(mask)] = 0
        nocs_resized = np.stack([cv2.resize(nocs[..., i], (self.scale_size//14, self.scale_size//14), interpolation=cv2.INTER_CUBIC) for i in range(3)], axis=-1)
        
        mask[np.sum(np.logical_or(nocs > 1, nocs < 0), axis=-1) != 0] = False
        mask_resized = get_coarse_mask(mask, scale_factor=(1 / 14))
        nocs_resized[np.logical_not(mask_resized)] = 0

        nocs_mask = np.ones_like(nocs_resized)
        bg_pixels = np.all(nocs_resized == [0, 0, 0], axis=-1)
        nocs_mask[bg_pixels] = [0, 0, 0]

        if mask_resized.sum() < 15 or nocs_mask[:, :, 0].sum() < 15:
            return False
        else:
            return True

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
    def xywh2cs(xywh, base_ratio=1.1, wh_max=480):
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
