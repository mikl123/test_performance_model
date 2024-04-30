import logging
import os
import cv2
import torch
from copy import deepcopy
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import math
import numpy as np

from alnet import ALNet
from soft_detect import DKD
import time

configs = {
    'alike-t': {'c1': 8, 'c2': 16, 'c3': 32, 'c4': 64, 'dim': 64, 'single_head': True, 'radius': 2,
                'model_path': os.path.join(os.path.split(__file__)[0], 'models', 'alike-t.pth')},
    'alike-s': {'c1': 8, 'c2': 16, 'c3': 48, 'c4': 96, 'dim': 96, 'single_head': True, 'radius': 2,
                'model_path': os.path.join(os.path.split(__file__)[0], 'models', 'alike-s.pth')},
    'alike-n': {'c1': 16, 'c2': 32, 'c3': 64, 'c4': 128, 'dim': 128, 'single_head': True, 'radius': 2,
                'model_path': os.path.join(os.path.split(__file__)[0], 'models', 'alike-n.pth')},
    'alike-l': {'c1': 32, 'c2': 64, 'c3': 128, 'c4': 128, 'dim': 128, 'single_head': False, 'radius': 2,
                'model_path': os.path.join(os.path.split(__file__)[0], 'models', 'alike-l.pth')},
}


class ALike(ALNet):
    def __init__(self,
                 # ================================== feature encoder
                 c1: int = 32, c2: int = 64, c3: int = 128, c4: int = 128, dim: int = 128,
                 single_head: bool = False,
                 # ================================== detect parameters
                 radius: int = 2,
                 top_k: int = 500, scores_th: float = 0.5,
                 n_limit: int = 5000,
                 device: str = 'cpu',
                 model_path: str = ''
                 ):
        super().__init__(c1, c2, c3, c4, dim, single_head)
        self.radius = radius
        self.top_k = top_k
        self.n_limit = n_limit
        self.scores_th = scores_th
       
        self.device = device

        if model_path != '':
            state_dict = torch.load(model_path, self.device)
            self.load_state_dict(state_dict)
            self.to(self.device)
            self.eval()
            logging.info(f'Loaded model parameters from {model_path}')
            logging.info(
                f"Number of model parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e3}KB")

    def extract_dense_map(self, image,h,w, ret_dict=False):
        # ====================================================
        # check image size, should be integer multiples of 2^5
        # if it is not a integer multiples of 2^5, padding zeros
        device = image.device
        x = super().forward(image)
        return x

    def forward(self, img, image_size_max=99999, sort=False, sub_pixel=False):
        """
        :param img: np.array HxWx3, RGB
        :param image_size_max: maximum image size, otherwise, the image will be resized
        :param sort: sort keypoints by scores
        :param sub_pixel: whether to use sub-pixel accuracy
        :return: a dictionary with 'keypoints', 'descriptors', 'scores', and 'time'
        """
        H = torch.tensor(480)
        W = torch.tensor(640)
        three = torch.tensor(3)
        # ==================== image size constraint
        image = img.clone()
        # ==================== convert image to tensor
        # image =image.to(self.device).to(torch.float32).permute(2, 0, 1)[None] / 255.0
        # ==================== extract keypoints
        with torch.no_grad():
            x = self.extract_dense_map(image,H,W)
        
        x = torch.tensor(x)
        start = time.time()
        descriptor_map, scores_map = torch.split(x, [x.shape[1] - 1, 1], dim = 1)
        scores_map = torch.sigmoid(scores_map)
        print(time.time() - start)
        print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        descriptor_map = torch.nn.functional.normalize(descriptor_map, p=2, dim=1)
        print("score")
        print(scores_map.shape)
        print("descriptor")
        print(descriptor_map.shape)
        return scores_map, descriptor_map
