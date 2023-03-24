import os
import io
import json
import copy
import torch
from math import pi
import numpy as np
from scipy.interpolate import interp1d
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils import warp, generate_random_params_for_warp
from view_transform import calibration

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from data import PlanningDataset


class PoseidonSequenceDataset(PlanningDataset):
    def __init__(self, split_txt_path, prefix, mode, use_memcache=True, return_origin=False):
        self.split_txt_path = split_txt_path
        self.prefix = prefix

        self.samples = open(split_txt_path).readlines()
        self.samples = [i.strip() for i in self.samples]

        assert mode in ('train', 'val', 'demo')
        self.mode = mode
        if self.mode == 'demo':
            print('PoseidonSequenceDataset: DEMO mode is on.')

        self.fix_seq_length = 100 if mode == 'train' else 100

        self.transforms = transforms.Compose(
            [
                # transforms.Resize((900 // 2, 1600 // 2)),
                # transforms.Resize((9 * 32, 16 * 32)),
                transforms.Resize((128, 256)),
                #transforms.Resize((256, 512)),
                transforms.ToTensor(),
                #transforms.Normalize([0.3890, 0.3937, 0.3851],
                #                     [0.2172, 0.2141, 0.2209]),
            ]
        )

        self.warp_matrix = calibration(extrinsic_matrix=np.array([[ 0, -1,  0,    0],
                                                                  [ 0,  0, -1, 1.546],
                                                                  [ 1,  0,  0,    0],
                                                                  [ 0,  0,  0,    1]]),
                                       cam_intrinsics=np.array([[1204, 0, 960],
                                                                [0, 1204, 540],
                                                                [0,   0,   1]]),
                                       device_frame_from_road_frame=np.hstack((np.diag([1, -1, -1]), [[0], [0], [1.546]])))

        self.use_memcache = use_memcache
        if self.use_memcache:
            self._init_mc_()

        self.return_origin = return_origin

        self.num_pts = 10 * 30  # 10 s * 30 Hz = 300 frames
        self.t_anchors = np.array(
            (0.        ,  0.00976562,  0.0390625 ,  0.08789062,  0.15625   ,
             0.24414062,  0.3515625 ,  0.47851562,  0.625     ,  0.79101562,
             0.9765625 ,  1.18164062,  1.40625   ,  1.65039062,  1.9140625 ,
             2.19726562,  2.5       ,  2.82226562,  3.1640625 ,  3.52539062,
             3.90625   ,  4.30664062,  4.7265625 ,  5.16601562,  5.625     ,
             6.10351562,  6.6015625 ,  7.11914062,  7.65625   ,  8.21289062,
             8.7890625 ,  9.38476562, 10.)
        )
        self.t_idx = np.linspace(0, 10, num=self.num_pts)


    def _get_cv2_vid(self, path):
        if self.use_memcache:
            path = self.client.generate_presigned_url(str(path), client_method='get_object', expires_in=3600)
        return cv2.VideoCapture(path)

    def _get_numpy(self, path):
        if self.use_memcache:
            bytes = io.BytesIO(memoryview(self.client.get(str(path))))
            return np.lib.format.read_array(bytes)
        else:
            return np.load(path)

    def _get_images(self, seq_sample_path):
        imgdir = os.path.join(seq_sample_path, 'images')
        return os.listdir(imgdir)
    
    def _get_future_poses(self, seq_sample_path):
        uid = seq_sample_path.split('/')[-1]
        traj = np.load(os.path.join(seq_sample_path, 'perception', f'{uid}_trajectory.npy'))
        return traj

    def _distort(self,cam_info,in_point):
        fx = cam_info['focal_u']
        fy = cam_info['focal_v']
        cx = cam_info['center_u']
        cy = cam_info['center_v']
        k = cam_info['distort']['param']
        x_wor = (in_point[0]-cx)/fx
        y_wor = (in_point[1]-cy)/fy
        x=np.array([x_wor,y_wor])
        r2 = x.dot(x)
        r=sqrt(r2)
        theta = np.atan(r)
        theta2 = theta * theta 
        theta3 = theta2 * theta 
        theta4 = theta2 * theta2
        theta5 = theta4 * theta
        theta6 = theta3 * theta3 
        theta7 = theta6 * theta
        theta8 = theta4 * theta4 
        theta9 = theta8 * theta  
        theta_d = theta + k[0] * theta3 + k[1] * theta5 + k[2] * theta7 + k[3] * theta9 
        inv_r =1.0/r if r > 10^-8 else 1.0 
        cdist = theta_d * inv_r if r > 10^-8 else 1.0 

        xd1 = x*cdist
        return xd1[0]*fx+cx,xd1[1]*fy+cy

    def _compute_self_vcsgnd2img(self,cam_info):
        self.Rotation = R.from_euler('zyx', (cam_info['roll'],cam_info['pitch'],cam_info['yaw']),degrees=False).as_matrix()
        self.Translation = np.asarray([cam_info['camera_x'],cam_info['camera_y'],cam_info['camera_z']]).reshape(3,1)

        axis_camhr_to_camstd = np.asarray([0, -1, 0, 0, 0, -1, 1, 0, 0]).reshape(3,3)
        Kstd = np.asarray([cam_info['focal_u'],0,cam_info['center_u'],
                            0,cam_info['focal_v'],cam_info['center_v'],
                            0,0,1]).reshape(3,3)
        self.camhr2img = Kstd.dot(axis_camhr_to_camstd)
        self.distort_param = cam_info['distort']['param']

    def _get_mat_vcsgnd2img(self,cam_info):
        yaw = cam_info['yaw']
        pitch = cam_info['pitch']
        roll = cam_info['roll']
        camera_x = cam_info['camera_x']
        camera_y = cam_info['camera_y']
        camera_z = cam_info['camera_z']
        r = R.from_euler('zyx', (yaw,pitch,roll),degrees=False)
        self.mat_vcsgnd2camgnd = r.as_matrix()   
        self.camera_x = cam_info['camera_x']        
        self.mat_vcsgnd2img = np.asarray(cam_info["mat"]["mat_vcsgnd2img"]).reshape(3,3)  
        self.mat_vcsgnd2local = r.as_matrix()
        self.mat_local2img = np.asarray(cam_info["mat"]["mat_vcsgnd2img"]).reshape(3,3)
        self.mat_img2local = np.asarray(cam_info["mat"]["mat_img2local"]).reshape(3,3)
        self.mat_gnd2img = np.asarray(cam_info["mat"]["mat_gnd2img"]).reshape(3,3)
        self.mat_img2gnd = np.asarray(cam_info["mat"]["mat_img2gnd"]).reshape(3,3)        
        # self.create_mat_vcsgnd2img = self.local2img.dot(self.mat_vcsgnd2local)
        # print(self.mat_vcsgnd2img)
        # print(self.create_mat_vcsgnd2img)

    def _campt2img(self,cam_info,camx,camy):
        img_x = camy*cam_info["focal_u"]/camx + cam_info["center_u"]
        img_y = cam_info['camera_z']*cam_info["focal_v"]/camx + cam_info["center_v"]
        return int(img_x),int(img_y)          

    def _split_yuv(self, frame):
        H = (frame.shape[0]*2)//3
        W = frame.shape[1]
        parsed = np.zeros((H//2, W//2, 6), dtype=np.uint8)

        parsed[..., 0] = frame[0:H:2, 0::2]
        parsed[..., 1] = frame[1:H:2, 0::2]
        parsed[..., 2] = frame[0:H:2, 1::2]
        parsed[..., 3] = frame[1:H:2, 1::2]
        parsed[..., 4] = frame[H:H+H//4].reshape((-1, H//2,W//2))
        parsed[..., 5] = frame[H+H//4:H+H//2].reshape((-1, H//2,W//2))
        #cv2.imwrite('y1.jpg', parsed[0])
        #cv2.imwrite('y2.jpg', parsed[1])
        #cv2.imwrite('y3.jpg', parsed[2])
        #cv2.imwrite('y4.jpg', parsed[3])
        #cv2.imwrite('u.jpg', parsed[4])
        #cv2.imwrite('v.jpg', parsed[5])

        return parsed

    def _transform_yuv(self, yuv):
        i1 = self.transforms(Image.fromarray(yuv[..., 0:3]))[None]
        i2 = self.transforms(Image.fromarray(yuv[..., 3:]))[None]
        return torch.cat([i1, i2], dim=1)

    def __getitem__(self, idx):
        seq_sample_path = self.prefix + self.samples[idx]
        imnames = self._get_images(seq_sample_path)

        seq_length = len(imnames)
        seq_length_delta = np.random.randint(1, seq_length - self.fix_seq_length)

        seq_start_idx = seq_length_delta
        seq_end_idx = seq_length_delta + self.fix_seq_length

        imgs = []  # <--- select frames here
        origin_imgs = []
        for imname in imnames[seq_start_idx-1: seq_end_idx]: # contains one more img
            impath = os.path.join(seq_sample_path, 'images', imname)
            frame = cv2.imread(impath)
            if frame is None:
                print(f'can not open image file {imname}')
                frame = np.zeros((512, 256, 3), dtype='uint8')

            imgs.append(frame)
            #cv2.imshow('frame', frame)
            #cv2.waitKey(0)
            if self.return_origin:
                origin_imgs.append(frame)

        # seq_input_img
        imgs = [cv2.warpPerspective(src=img, M=self.warp_matrix, dsize=(512,256), flags=cv2.WARP_INVERSE_MAP) for img in imgs]
        img_yuvs = [cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420) for img in imgs]
        img_yuvs = [self._split_yuv(yuv) for yuv in img_yuvs]
        img_yuvs = [self._transform_yuv(yuv) for yuv in img_yuvs]

        # poses
        trajs = self._get_future_poses(seq_sample_path)
        fposes = trajs[seq_start_idx: seq_end_idx]

        future_poses = []
        for i in range(self.fix_seq_length):
            interp_positions = fposes[i]
            future_poses.append(interp_positions)

            ## debug
            ##if i == self.fix_seq_length - 1:
            #c_str = '{"focal_u": 2353.0527, "focal_v": 2353.0527, "center_u": 1917.296, "center_v": 1087.0253, "camera_x": -1.892, "camera_y": 0.0, "camera_z": 1.546, "pitch": 0.011249786, "yaw": 0.00076591276, "roll": 0.004828244, "type": 0, "fov": 120.0, "version": 513, "valid_height": {"left_y": 1785, "right_y": 1785}, "distort": {"param": [-0.13210864, 0.8884662, 4.866906e-05, 6.136231e-05, 0.12150907, 0.4133815, 0.76438034, 0.60310525]}, "vcs": {"rotation": [0.0, 0.0, 0.0], "translation": [3.875, 0.0, 0.0]}, "mat": {"mat_gnd2img": [2351.452, 1918.9762, 3681.6326, -12.251949, 1060.4854, 5662.884, -0.00082021905, 0.9999364, 1.9092655], "mat_img2gnd": [0.00042497474, -2.0517969e-06, -0.8133923, -2.1901242e-06, -0.0005248337, 1.5608804, 1.3295988e-06, 0.00027486935, -0.29406512], "mat_vcsgnd2img": [1918.9762, -2351.452, -3754.4001, 1060.4854, 12.251949, 1553.5029, 0.9999364, 0.00082021905, -1.965488], "mat_img2vcsgnd": [2.9620714e-06, 0.00054028514, 0.4213782, -0.00042497474, 2.0517841e-06, 0.8133923, 1.329599e-06, 0.00027486938, -0.29406512], "mat_local2img": [1918.9762, -2351.452, 3681.6326, 1060.4854, 12.251949, 5662.884, 0.9999364, 0.00082021905, 1.9092655], "mat_img2local": [-2.1901242e-06, -0.0005248337, 1.5608804, -0.00042497474, 2.0517969e-06, 0.8133923, 1.3295988e-06, 0.00027486935, -0.29406512]}, "vendor": "ar0820"}'
            #camera = json.loads(c_str)
            #self._get_mat_vcsgnd2img(camera)
            #self._compute_self_vcsgnd2img(camera)
            #for world_pt in interp_positions:# traj:
            #    pt = copy.deepcopy(world_pt)
            #    pt[2] = 1
            #    img_pt = self.mat_vcsgnd2img.dot(pt)
            #    img_x = int(img_pt[0]/img_pt[2]/2)
            #    img_y = int(img_pt[1]/img_pt[2]/2)   
            #    cv2.circle(origin_imgs[i], (img_x, img_y), 5, (0,0,255), 5)
            #resized_img = cv2.resize(origin_imgs[i], (int(origin_imgs[i].shape[1]/4), int(origin_imgs[i].shape[0]/4)))
            #cv2.imwrite(f'{idx}_{i}_traj.jpg', resized_img)

            #    #print(seq_sample_path, idx)
            #    #plt.scatter(interp_positions[:,0], interp_positions[:,1], s=1)
            #    #plt.scatter(interp_positions[:,0], interp_positions[:,1], s=5)
            #    #plt.savefig(f'{idx}_{i}_time_traj.jpg')
            #    #plt.clf()
            
        input_img = torch.cat(img_yuvs, dim=0)  # [N+1, 3, H, W]
        input_img = torch.cat((input_img[:-1, ...], input_img[1:, ...]), dim=1)
        future_poses = torch.tensor(np.array(future_poses), dtype=torch.float32)

        prompt_traj = copy.deepcopy(future_poses)
        random_scale = np.random.rand(prompt_traj.shape[0],3) * 2
        random_scale = np.expand_dims(random_scale, axis=1).repeat(prompt_traj.shape[1], axis=1)
        prompt_traj = prompt_traj * random_scale
        #if np.random.rand() > 0.5:
        #    prompt_traj = prompt_traj * 1e-6
        prompt_traj = prompt_traj.type_as(future_poses)

        rtn_dict = dict(
            seq_input_img=input_img,  # torch.Size([N, 12, 128, 256])
            seq_future_poses=future_poses,  # torch.Size([N, num_pts, 3])
            seq_prompt_traj=prompt_traj, # torch.Size([N, num_pts, 3])
            # camera_intrinsic=torch.tensor(seq_samples[0]['camera_intrinsic']),
            # camera_extrinsic=torch.tensor(seq_samples[0]['camera_extrinsic']),
            # camera_translation_inv=torch.tensor(seq_samples[0]['camera_translation_inv']),
            # camera_rotation_matrix_inv=torch.tensor(seq_samples[0]['camera_rotation_matrix_inv']),
        )

        # For DEMO
        if self.return_origin:
            #origin_imgs = origin_imgs[seq_start_idx: seq_end_idx]
            origin_imgs = [torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[None] for img in origin_imgs]
            origin_imgs = torch.cat(origin_imgs, dim=0)  # N, H_ori, W_ori, 3
            rtn_dict['origin_imgs'] = origin_imgs

        return rtn_dict
