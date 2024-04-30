
import logging
import argparse
import numpy as np
from tqdm import tqdm
from alike_step2_onnx import ALike, configs
import torch
import torchvision.models as models
import json

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
    score_map = torch.randn(1, 1, 480, 640)
    descriptor_map = torch.randn(1, 64, 480, 640)
    # with open('calibration/calibration.json', 'r') as f:
    #     loaded_list = json.load(f)
    # # loaded_list = np.array(loaded_list)
    # score_map = torch.tensor(loaded_list[0])
    # descriptor_map = torch.tensor(loaded_list[1])

    torch.onnx.export(model,  
                                   # model being run
                   (score_map, descriptor_map),    # model input (or a tuple for multiple inputs)
                  "step2.onnx",   # where to save the model (can be a file or file-like object)
                  opset_version=17,   # the ONNX version to export the model to
                  dynamic_axes={
                        'output1': {1: 'output1_variable_dim_0'},
                        'output2': {1: 'output2_variable_dim_0'},
                    },
                  input_names = ['score_map', 'descriptor_map'],   # the model's input names
                  output_names = ['output1', "output2"]
                  
                  )
    print("infreing ")
    # onnx.shape_inference.infer_shapes_path("step1.onnx", "step1.onnx")