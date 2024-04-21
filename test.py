import onnxruntime
import numpy as np

# Load the ONNX model
ort_session = onnxruntime.InferenceSession("step1.onnx")

# Generate different input data
input_data_1 = np.random.randn(224, 224, 3).astype(np.float32)
input_data_2 = np.random.randn(224, 224, 3).astype(np.float32)

# Run inference with the first input data
outputs_1 = ort_session.run(None, {'img': input_data_1})

# Run inference with the second input data
outputs_2 = ort_session.run(None, {'img': input_data_2})

# Print the outputs to check if they are different
print("Output 1:", outputs_1[0])
print("Sadsadsasdasdassad")
print("Output 2:", outputs_2[0])