import copy
import os
import cv2
import glob
import logging
import argparse
import numpy as np
from tqdm import tqdm
from alike1 import ALike, configs
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


if __name__ == '__main__':
    iterations = 5
    logging.basicConfig(level=logging.INFO)
    no_display = False
    model_name = "alike-t"
    image_loader = ImageLoader("./assets/tum")
    model = ALike(**configs["alike-t"],
                  device="cpu",
                  top_k=-1,
                  scores_th=0.2,
                  n_limit=5000)
    tracker = SimpleTracker()
    res = []
    if not no_display:
        logging.info("Press 'q' to stop!")
        cv2.namedWindow(model_name)

    runtime = []
    progress_bar = tqdm(image_loader)
    overall_time = []
    res = []
    for img in progress_bar:
        if img is None:
            break
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        start_time = time.time()
        for i in range(iterations):
            pred = model(img_rgb, sub_pixel=not True)
        end_time = time.time()
        execution_time = end_time - start_time
        res.append(execution_time)
        print("Execution time(all flow):", execution_time, "seconds")
        kpts = pred['keypoints']
        desc = pred['descriptors']
        runtime.append(pred['time'])

        out, N_matches = tracker.update(img, kpts, desc)

        ave_fps = (1. / np.stack(runtime)).mean()
        status = f"Fps:{ave_fps:.1f}, Keypoints/Matches: {len(kpts)}/{N_matches}"
        progress_bar.set_description(status)

        if not no_display:
            cv2.setWindowTitle(model_name, model_name + ': ' + status)
            cv2.imshow(model_name, out)
            if cv2.waitKey(1) == ord('q'):
                break
        overall_time.append(sum(res))

    average_all = model.access_dkd().average_all
    average_detect_keypoints =model.access_dkd().average_detect_keypoints
    average_sample_descriptors =model.access_dkd().average_sample_descriptors
    average_nms = model.access_dkd().average_nms
    average_keypoint_other = model.access_dkd().average_keypoint_other
    nms = model.access_dkd().nms
    print("===============================================================")
    print(nms)
    print("===============================================================")

    print("Feature encoding")
    print(np.sum(model.time_encoder) / iterations)
    print("Feature aggregation")
    print(np.sum(model.feature_aggregation) / iterations)
    print("Feature extraction")
    print(np.sum(model.feature_extraction) / iterations)


    print("---------------")
    print("Step 1")
    print(np.sum(model.step1) / iterations)
    print("Step 2")
    print(np.sum(model.step2) / iterations)


    print("--------------Results-------------")
    print("DKD:")
    print(np.sum(average_all) / iterations)

    print("Detect Keypoints:")
    print(np.sum(average_detect_keypoints) / iterations)

    print("Sample Descriptors:")
    print(np.sum(average_sample_descriptors) / iterations)

    print("NMS_default:")
    print(np.sum(average_nms) / iterations)

    print("Average keypoints other:")
    print(np.sum(average_keypoint_other) / iterations)


    with open("result1.txt", "w",encoding="utf-8") as file:
        file.write(str(sum(overall_time)))
    logging.info('Finished!')
    if not no_display:
        logging.info('Press any key to exit!')
        cv2.waitKey()


   
