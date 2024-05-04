import onnx
from onnx.tools import update_model_dims

onnx.shape_inference.infer_shapes_path("test/step1.onnx", "step1.onnx")