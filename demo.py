import copy
import os
import cv2
import glob
import logging
import argparse
import numpy as np
from tqdm import tqdm
from alike_step2 import ALike, configs
import onnxruntime
import torch
import copy
import json
import sys
import time
class ImageLoader(object):
    def __init__(self, filepath: str):
        self.N = 3000
        if filepath.startswith('camera'):
            camera = int(filepath[6:])
            self.cap = cv2.VideoCapture(camera)
            if not self.cap.isOpened():
                raise IOError(f"Can't open camera {camera}!")
            logging.info(f'Opened camera {camera}')
            self.mode = 'camera'
        elif os.path.exists(filepath):
            if os.path.isfile(filepath):
                self.cap = cv2.VideoCapture(filepath)
                if not self.cap.isOpened():
                    raise IOError(f"Can't open video {filepath}!")
                rate = self.cap.get(cv2.CAP_PROP_FPS)
                self.N = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
                duration = self.N / rate
                logging.info(f'Opened video {filepath}')
                logging.info(f'Frames: {self.N}, FPS: {rate}, Duration: {duration}s')
                self.mode = 'video'
            else:
                self.images = glob.glob(os.path.join(filepath, '*.png')) + \
                              glob.glob(os.path.join(filepath, '*.jpg')) + \
                              glob.glob(os.path.join(filepath, '*.ppm'))
                self.images.sort()
                self.N = len(self.images)
                logging.info(f'Loading {self.N} images')
                self.mode = 'images'
        else:
            raise IOError('Error filepath (camerax/path of images/path of videos): ', filepath)

    def __getitem__(self, item):
        if self.mode == 'camera' or self.mode == 'video':
            if item > self.N:
                return None
            ret, img = self.cap.read()
            if not ret:
                raise "Can't read image from camera"
            if self.mode == 'video':
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, item)
        elif self.mode == 'images':
            filename = self.images[item]
            img = cv2.imread(filename)
            if img is None:
                raise Exception('Error reading image %s' % filename)        
        return img

    def __len__(self):
        return self.N

time_step1 = 0
time_step2=  0
class SimpleTracker(object):
    def __init__(self):
        self.pts_prev = None
        self.desc_prev = None

    def update(self, img, pts, desc):
        N_matches = 0
        if self.pts_prev is None:
            self.pts_prev = pts
            self.desc_prev = desc

            out = copy.deepcopy(img)
            for pt1 in pts:
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                cv2.circle(out, p1, 1, (0, 0, 255), -1, lineType=16)
        else:
            matches = self.mnn_mather(self.desc_prev, desc)
            mpts1, mpts2 = self.pts_prev[matches[:, 0]], pts[matches[:, 1]]
            N_matches = len(matches)

            out = copy.deepcopy(img)
            for pt1, pt2 in zip(mpts1, mpts2):
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])))
                cv2.line(out, p1, p2, (0, 255, 0), lineType=16)
                cv2.circle(out, p2, 1, (0, 0, 255), -1, lineType=16)

            self.pts_prev = pts
            self.desc_prev = desc

        return out, N_matches

    def mnn_mather(self, desc1, desc2):
        sim = desc1 @ desc2.transpose()
        sim[sim < 0.9] = 0
        nn12 = np.argmax(sim, axis=1)
        nn21 = np.argmax(sim, axis=0)
        ids1 = np.arange(0, sim.shape[0])
        mask = (ids1 == nn21[nn12])
        matches = np.stack([ids1[mask], nn12[mask]])
        return matches.transpose()

def predict(step_1, step_2, data):
    start1 = time.time()
    data = np.expand_dims(data, axis=0)
    data = np.transpose(data, (0, 3, 1, 2))/ 255.0
    output = step_1.run(None, {"img": data.astype(np.float32)})
    print(str(time.time() - start1) + "step1")
    start1 = time.time()
    res = step_2(output[0],output[1])
    print(str(time.time() - start1) + "step2")
    return res
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = {"model":"alike-t",
        "input":r".\assets\tum",
        "device":"cpu",
        "top_k":-1,
        "scores_th":0.2,
        "n_limit":5000,
        "no_display":False
        }
    image_loader = ImageLoader(args["input"])
    model = ALike(**configs[args["model"]],
                  device=args["device"],
                  top_k=args["top_k"],
                  scores_th=args["scores_th"],
                  n_limit=args["n_limit"])
    onnx_model_path = "step1.onnx"
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    tracker = SimpleTracker()
    if not args["no_display"]:
        logging.info("Press 'q' to stop!")
        cv2.namedWindow(args["model"])
    runtime = []
    progress_bar = tqdm(image_loader)
    for img in progress_bar:
        if img is None:
            break
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = copy.deepcopy(img_rgb)
        pred = predict(ort_session, model, img_rgb)
        kpts = pred["keypoints"]
        desc = pred["descriptors"]
        out, N_matches = tracker.update(img, kpts, desc)
        if not args["no_display"]:
            cv2.imshow(args["model"], out)
            if cv2.waitKey(1) == ord('q'):
                break
    logging.info('Finished!')
    if not args["no_display"]:
        logging.info('Press any key to exit!')
        cv2.waitKey()

