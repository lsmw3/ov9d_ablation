import os
import cv2
import json
import numpy as np
from scipy.stats import truncnorm
from scipy.spatial.transform import Rotation as R
from dataset.base_dataset import BaseDataset
from utils.utils import get_2d_coord_np, crop_resize_by_warp_affine, filter_small_edge_obj, get_coarse_mask


class oo3d9dmulti(BaseDataset):
    def __init__(self, data_path, data_name, data_type, feat_3d_path, xyz_bin: int=64, raw_h: int=480, raw_w: int=640,
                 is_train=True, scale_size=490, num_view=50):
        super().__init__()

        self.scale_size = scale_size
        self.resize = 224
        self.xyz_bin = xyz_bin

        self.is_train = is_train
        self.data_path = os.path.join(data_path, data_type)
        self.data_list = []
        with open(os.path.join(data_path, 'models_info_with_symmetry.json'), 'r') as f:
            self.models_info = json.load(f)
        with open(os.path.join(data_path, 'class_list.json'), 'r') as f:
            class_list = json.load(f)
            self.class_dict = {k+1: v for k, v in zip(range(len(class_list)), class_list)}
        with open(os.path.join(data_path, 'oid2cid.json'), 'r') as f:
            oid_2_cid = json.load(f)
        for scene_id in os.listdir(self.data_path):
            with open(os.path.join(self.data_path, scene_id, 'scene_camera.json'), 'r') as f:
                scene_camera = json.load(f)
            with open(os.path.join(self.data_path, scene_id, 'scene_gt.json'), 'r') as f:
                scene_gt = json.load(f)
            with open(os.path.join(self.data_path, scene_id, 'scene_gt_info.json'), 'r') as f:
                scene_gt_info = json.load(f)

            # feat_3d_points = np.load(os.path.join(feat_3d_path, scene_id, "3d_feat.npy"))

            curr_num_view = min(num_view, len(scene_camera.keys()))
            view_ids = np.array(list(scene_camera.keys()))
            # np.random.shuffle(view_ids)
            i = 0
            for view_id in view_ids:
                if i >= curr_num_view:
                    break
                objects_ids_in_scene = np.arange(len(scene_gt_info[view_id]))
                for obj_id in objects_ids_in_scene:
                    if scene_gt_info[view_id][obj_id]['bbox_visib'][2] < 50 or scene_gt_info[view_id][0]['bbox_visib'][3] < 50:
                        continue
                    # if filter_small_edge_obj(scene_gt_info[view_id][obj_id]['bbox_visib'], raw_w, raw_h):
                    #     continue
                    self.data_list.append(
                        {
                            'scene': scene_id,
                            'view': f'{int(view_id):{0}{6}}',
                            'object_in_scene': f'{int(obj_id):{0}{6}}',
                            'cam': scene_camera[view_id],
                            'gt': scene_gt[view_id][obj_id],
                            'gt_info': scene_gt_info[view_id][obj_id],
                            'meta': self.models_info[str(scene_gt[view_id][obj_id]["obj_id"])],
                            'class_id': oid_2_cid[str(scene_gt[view_id][obj_id]["obj_id"])]
                            # '3d_feat': feat_3d_points # (1024, 387)
                        }
                    )
                i += 1
        
        phase = 'train' if is_train else 'test'
        print("Dataset: OmniObject3D Render")
        print("# of %s images: %d" % (phase, len(self.data_list)))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        info = self.data_list[idx]
        # scene, view, cam, gt, gt_info, meta, feat_3d = info['scene'], info['view'], info['cam'], info['gt'], info['gt_info'], info['meta'], info['3d_feat']
        scene, view, obj_in_scene, cam, gt, gt_info, meta, class_id = info['scene'], info['view'], info['object_in_scene'], info['cam'], info['gt'], info['gt_info'], info['meta'], info['class_id']

        cam_K = np.asarray(cam['cam_K']).reshape(3, 3)
        cam_R_m2c, cam_t_m2c, obj_id = gt['cam_R_m2c'], gt['cam_t_m2c'], gt['obj_id']
        cam_R_m2c = np.asarray(cam_R_m2c).reshape(3, 3)
        cam_t_m2c = np.asarray(cam_t_m2c).reshape(1, 1, 3)
        kps3d = self.get_keypoints(meta) # key points in 3d model frame
        diag = np.linalg.norm(kps3d[0, 1] - kps3d[0, 8])
        kp_i = (kps3d @ cam_R_m2c.T + cam_t_m2c) @ cam_K.T # n * 9 * 3, key points on 2d pixel plane
        obj_z_gt = kp_i[0, 0, 2] # depth of model centroid in camera frame
        kp_i = kp_i[..., 0:2] / kp_i[..., 2:] # n * 9 * 2, homogeneous 2d key points coordinates
        obj_centroid_2d = kp_i[0, 0, :2]

        bbox = gt_info['bbox_visib']

        rgb_path = os.path.join(self.data_path, scene, 'rgb', view+'.png')
        if not os.path.exists(rgb_path):
            rgb_path = os.path.join(self.data_path, scene, 'rgb', view+'.jpg')
        depth_path = os.path.join(self.data_path, scene, 'depth', view+'.png')
        mask_path = os.path.join(self.data_path, scene, 'mask_visib', '_'.join([view, obj_in_scene])+'.png')

        image = cv2.imread(rgb_path)
        H, W = image.shape[:2] # 480, 640
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        raw_image = image.copy()
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        pcl_c = self.K_dpt2cld(depth, 1/cam['depth_scale'], cam_K) # points in camera frame
        pcl_m = (pcl_c - cam_t_m2c).dot(cam_R_m2c) # points in model frame
        mask = cv2.imread(mask_path)

        # vis_img = draw_3d_bbox_with_coordinate_frame(image, kps3d[0], cam_R_m2c, cam_t_m2c.reshape(-1), cam_K)

        image[mask != 255] = 0 # remove background

        if self.is_train:
            c, s = self.xywh2cs_dzi(bbox, wh_max=self.scale_size)
        else:
            c, s = self.xywh2cs(bbox, wh_max=self.scale_size)

        # relative translation offset from the bbox centeroid to object centroid on 2d pixel plane
        delta_c = obj_centroid_2d - c
        trans_ratio = np.asarray([delta_c[0] / s, delta_c[1] / s, obj_z_gt / (self.scale_size / s)]).astype(np.float32)
        
        interpolate = cv2.INTER_NEAREST
        # interpolate = cv2.INTER_LINEAR
        rgb, c_h_, c_w_, s_, roi_coord_2d = self.zoom_in_v2(image, c, s, res=self.scale_size, return_roi_coord=True) # center-cropped rgb
        pcl_m, *_ = self.zoom_in_v2(pcl_m.astype(np.float32), c, s, res=self.scale_size, interpolate=interpolate) # center-cropped point cloud in model frame, (480, 480, 3)
        mask, *_ = self.zoom_in_v2(mask, c, s, res=self.scale_size, interpolate=interpolate) # center-cropped mask
        mask = mask[..., 0] >= 250
        rgb[mask[:, :, None] != [True, True, True]] = 0 # further align the background, especially in case the cropped rgb is outside the boundary of raw image
        center = (kps3d[0, 1] + kps3d[0, 8]) / 2 # seems this is the true center of the objects in 3d model frame (actually same as kps3d[0, 0])
        nocs = (pcl_m - center.reshape(1, 1, 3)) / diag + 0.5

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
        # rgb_resized = np.stack([cv2.resize(rgb[..., i], (self.resize, self.resize), interpolation=cv2.INTER_CUBIC) for i in range(3)], axis=-1)
        # mask_rgb_resized = get_coarse_mask(mask, scale_factor=(self.resize / self.scale_size))
        # rgb_resized[mask_rgb_resized[:, :, None] != [True, True, True]] = 0
        nocs_resized[np.logical_not(mask_resized)] = 0

        c = np.array([c_w_, c_h_])
        s = s_
        kp_i = (kp_i - c.reshape(1, 1, 2)) / s  # * self.scale_size

        gt_coord_x = np.arange(W)
        gt_coord_y = np.arange(H)
        gt_coord_xy = np.asarray(np.meshgrid(gt_coord_x, gt_coord_y)).transpose(1, 2 ,0)
        gt_coord_2d = crop_resize_by_warp_affine(gt_coord_xy, c, s, self.scale_size, interpolation=cv2.INTER_NEAREST).astype(np.float32) # HWC
        
        dis_sym = np.zeros((3, 4, 4))
        if 'symmetries_discrete' in self.models_info[f'{obj_id}']:
            mats = np.asarray([np.asarray(mat_list).reshape(4, 4) for mat_list in self.models_info[f'{obj_id}']['symmetries_discrete']])
            dis_sym[:mats.shape[0]] = mats
        con_sym = np.zeros((3, 6))
        if 'symmetries_continuous' in self.models_info[f'{obj_id}']:
            for i, ao in enumerate(self.models_info[f'{obj_id}']['symmetries_continuous']):
                axis = np.asarray(ao['axis'])
                offset = np.asarray(ao['offset'])
                con_sym[i] = np.concatenate([axis, offset])

        if self.is_train:
            rgb = self.augment_training_data(rgb.astype(np.uint8))

        out_dict = {
            'raw_scene': raw_image.transpose((2, 0, 1)), # (3, H, W), dtype = np.uint8
            'image': (rgb.transpose((2, 0, 1)) / 255).astype(np.float32), # (3, 490, 490)
            # 'input_image': (rgb_resized.transpose((2, 0, 1)) / 255).astype(np.float32), # (3, 224, 224)
            'gt_xyz_bin': gt_xyz_bin, # (3, H, W), dtype = np.uint8
            'gt_coord_2d': gt_coord_2d.astype(np.float32), # (490, 490, 2)
            'bbox_size': s,
            'bbox_center': c,
            'roi_coord_2d': roi_coord_2d.astype(np.float32), # (2, 490, 490)
            'gt_r': cam_R_m2c.astype(np.float32),
            'gt_t': cam_t_m2c.reshape(-1).astype(np.float32),
            'cam': cam_K.astype(np.float32),
            'resize_ratio': np.array([self.scale_size / s], dtype=np.float32),
            'gt_trans_ratio': trans_ratio.reshape(-1).astype(np.float32),
            'mask': mask, # (490, 490)
            'mask_resized': mask_resized, # (32, 32)
            'nocs': nocs.transpose((2, 0, 1)).astype(np.float32), # (3, 490, 490)
            'nocs_resized': nocs_resized.transpose((2, 0, 1)).astype(np.float32), # (3, 32, 32)
            # 'kps': kp_i.astype(np.float32),
            'kps_3d_m': kps3d[0].astype(np.float32), # (9, 3)
            'kps_3d_center': center.astype(np.float32), # (3)
            'kps_3d_dig': np.array([diag], dtype=np.float32),
            'dis_sym': dis_sym.astype(np.float32),
            'con_sym': con_sym.astype(np.float32),
            'filename': '-'.join([scene, view]),
            'class_id': class_id,
            'obj_id': obj_id,
            'pcl_model': pcl_m.astype(np.float32), # (490, 490, 3)
            # '3d_feat': feat_3d.astype(np.float32) # (1024, 387)
        }

        # if not self.is_train:
        #     kps3d = kps3d - center.reshape(1, 1, 3)
        #     pcl_c, *_ = self.zoom_in_v2(pcl_c, c, s, res=self.scale_size, interpolate=interpolate)
        #     extra_gt = {
        #         'dis_image': (rgb.transpose((2, 0, 1)) / 255).astype(np.float32),
        #         "kps3d": kps3d,
        #         "cam_K": cam_K,
        #         "cam_R_m2c": cam_R_m2c,
        #         "cam_t_m2c": cam_t_m2c,
        #         "c": c,
        #         "s": s,
        #         "diag": diag, 
        #         "pcl_c": pcl_c.astype(np.float32), 
        #     }
        #     out_dict.update(extra_gt)

        return out_dict

    @staticmethod
    def get_keypoints(model_info, dt=5):
        mins = [model_info['min_x'], model_info['min_y'], model_info['min_z']]
        sizes = [model_info['size_x'], model_info['size_y'], model_info['size_z']]
        maxs = [mins[i]+sizes[i] for i in range(len(mins))]
        base = [c.reshape(-1) for c in np.meshgrid(*zip(mins, maxs), indexing='ij')]
        base = np.stack(base, axis=-1) # 8 corners of the 3d bounding box of the object
        centroid = np.mean(base, axis=0, keepdims=True) # center of the 3d bounding box
        base = np.concatenate([centroid, base], axis=0)
        keypoints = [base]
        if 'symmetries_discrete' in model_info:
            mats = [np.asarray(mat_list).reshape(4, 4) for mat_list in model_info['symmetries_discrete']]
            for mat in mats:
                curr = keypoints[0] @ mat[0:3, 0:3].T + mat[0:3, 3:].T
                keypoints.append(curr)
        elif 'symmetries_continuous' in model_info:
            # todo: consider multiple symmetries
            ao = model_info['symmetries_continuous'][0]
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
    def xywh2cs(xywh, base_ratio=1.5, wh_max=480):
        x, y, w, h = xywh
        center = np.array((x+0.5*w, y+0.5*h)) # [c_w, c_h]
        wh = max(w, h) * base_ratio
        if wh_max != None:
            wh = min(wh, wh_max)
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
