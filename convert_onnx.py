
import logging
import argparse
import numpy as np
from tqdm import tqdm
from alike_step1 import ALike, configs
import torch
import torchvision.models as models
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

    model = ALike(**configs[args.model],
                  device=args.device,
                  top_k=args.top_k,
                  scores_th=args.scores_th,
                  n_limit=args.n_limit)
    img = torch.randn(480, 640, 3)

    torch.onnx.export(model,               # model being run
                   img,                         # model input (or a tuple for multiple inputs)
                  "step1.onnx",   # where to save the model (can be a file or file-like object)
                  opset_version=11,   # the ONNX version to export the model to
                  input_names = ['img'],   # the model's input names
                  output_names = ['output']
                  )