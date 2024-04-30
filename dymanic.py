import onnx

ONNX_PATH = 'test/step2.onnx'
model = onnx.load(ONNX_PATH)

print(model.graph.input[0].type.tensor_type.shape)
# model.graph.input[0].type.tensor_type.shape.dim[2].ClearField('dim_value')
# model.graph.input[0].type.tensor_type.shape.dim[3].ClearField('dim_value')

ONNX_PATH = 'dynamic_model_step2.onnx'
onnx.save(model, ONNX_PATH)
