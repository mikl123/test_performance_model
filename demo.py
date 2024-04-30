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
    global time_step1, time_step2
    data = np.expand_dims(data, axis=0)
    data = np.transpose(data, (0, 3, 1, 2))/ 255.0
    

    start_time = time.time()
    output = step_1.run(None, {"img": data.astype(np.float32)})
    print(type(output))
    # converted_list = [arr.tolist() for arr in output]

    # with open('calibration/calibration.json', 'w') as f:
    #         json.dump(converted_list, f)
    # sys.exit(0)
    execution_time = time.time() - start_time
    print("Step 1 Execution time:", execution_time, "seconds")
    time_step1 += execution_time


    start_time = time.time()
    # print(output[1])
    res = step_2.run(None, {"score_map": output[0], "descriptor_map": output[1]})
    # print(res.shape)
    # res = step_2.forward(output[0], output[1])
    print("======================")
    print(res[0].shape)
    print(res[1].shape)
    print("======================")
    execution_time = time.time() - start_time
    print("Step 2 execution time:", execution_time, "seconds")
    time_step2 += execution_time

    return res
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ALike Demo.')
    parser.add_argument('input', type=str, default='',
                        help='Image directory or movie file or "camera0" (for webcam0).')
    parser.add_argument('--model', choices=['alike-t', 'alike-s', 'alike-n', 'alike-l'], default="alike-t",
                        help="The model configuration")
    parser.add_argument('--device', type=str, default='cpu', help="Running device (default: cuda).")
    parser.add_argument('--top_k', type=int, default=-1,
                        help='Detect top K keypoints. -1 for threshold based mode, >0 for top K mode. (default: -1)')
    parser.add_argument('--scores_th', type=float, default=0.2,
                        help='Detector score threshold (default: 0.2).')
    parser.add_argument('--n_limit', type=int, default=5000,
                        help='Maximum number of keypoints to be detected (default: 5000).')
    parser.add_argument('--no_display', action='store_true',
                        help='Do not display images to screen. Useful if running remotely (default: False).')
    parser.add_argument('--no_sub_pixel', action='store_true',
                        help='Do not detect sub-pixel keypoints (default: False).')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    image_loader = ImageLoader(args.input)
    model = ALike(**configs[args.model],
                  device=args.device,
                  top_k=args.top_k,
                  scores_th=args.scores_th,
                  n_limit=args.n_limit)
    
    onnx_model_path = "step1.onnx"
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    
    onnx_model_path1 = "step2.onnx"
    ort_session1 = onnxruntime.InferenceSession(onnx_model_path1)
    
    tracker = SimpleTracker()

    if not args.no_display:
        logging.info("Press 'q' to stop!")
        cv2.namedWindow(args.model)

    runtime = []
    iterations = 5
    progress_bar = tqdm(image_loader)
    for img in progress_bar:
        if img is None:
            break
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # pred = model(img_rgb, sub_pixel=not args.no_sub_pixel)
        img_rgb = copy.deepcopy(img_rgb)
        
        print(img_rgb.shape)
        for i in range(iterations):
            pred = predict(ort_session, ort_session1, img_rgb)
        kpts = pred[0]
        desc = pred[1]
        
        out, N_matches = tracker.update(img, kpts, desc)

        if not args.no_display:
            cv2.imshow(args.model, out)
            if cv2.waitKey(1) == ord('q'):
                break
    
    average_all = model.access_dkd().average_all
    average_detect_keypoints =model.access_dkd().average_detect_keypoints
    average_sample_descriptors =model.access_dkd().average_sample_descriptors
    average_nms = model.access_dkd().average_nms
    average_keypoint_other = model.access_dkd().average_keypoint_other

    print("--------------Results-------------")
    print("Step1:")
    print(np.sum(time_step1) / iterations)
    print("Step2:")
    print(np.sum(time_step2) / iterations)
    # print("DKD:")
    # print(np.sum(average_all) / iterations)

    # print("Detect Keypoints:")
    # print(np.sum(average_detect_keypoints) / iterations)

    # print("Sample Descriptors:")
    # print(np.sum(average_sample_descriptors) / iterations)

    # print("NMS_default:")
    # print(np.sum(average_nms) / iterations)

    # print("Average keypoints other:")
    # print(np.sum(average_keypoint_other) / iterations)

    logging.info('Finished!')
    if not args.no_display:
        logging.info('Press any key to exit!')
        cv2.waitKey()

