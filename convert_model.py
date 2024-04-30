import onnx
from onnx.tools import update_model_dims

onnx.shape_inference.infer_shapes_path("test/step2.onnx", "step2.onnx")