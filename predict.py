import joblib
import pandas as pd

# Load the saved best model
model = joblib.load("best_model.joblib")

# New flower measurements
new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=[
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]
)

# Make prediction
prediction = model.predict(new_flower)

print("Predicted Iris Species:", prediction[0])