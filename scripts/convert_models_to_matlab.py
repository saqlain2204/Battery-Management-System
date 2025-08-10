import joblib
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Load the .joblib model
model = joblib.load("../models/SOC_LinearRegression_v1.0_test_result_trial_end_v1.0_20250804_004225.joblib")

# Define the input type (adjust shape based on your model's input)
# shape=(None, n_features) where n_features is your feature count
n_features = 4  # change this to match your model
initial_type = [("input", FloatTensorType([None, n_features]))]

# Convert to ONNX
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Save the ONNX model
with open("../models/matlab_compatible/SOC_LinearRegression_v1.0_test_result_trial_end_v1.0_20250804_004225.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Conversion completed! Saved as ../models/matlab_compatible/SOC_LinearRegression_v1.0_test_result_trial_end_v1.0_20250804_004225.onnx")
