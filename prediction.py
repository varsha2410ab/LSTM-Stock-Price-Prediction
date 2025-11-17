import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from preprocessing import preprocess_data

# Load the trained model
model = load_model("lstm_model.h5")

# Preprocess the data again
x, y, scaler = preprocess_data("data.csv", sequence_length=60)

# Predict on the data
predicted = model.predict(x)

# Inverse transform to get original price scale
predicted_prices = scaler.inverse_transform(predicted.reshape(-1, 1))
actual_prices = scaler.inverse_transform(y.reshape(-1, 1))

# Print sample prediction
print("First predicted price:", predicted_prices[0][0])
print("Actual price:", actual_prices[0][0])

# === STEP 9A: Visualization ===
plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label="Actual Price", color='blue')
plt.plot(predicted_prices, label="Predicted Price", color='red')
plt.title("Stock Price Prediction: Actual vs Predicted")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === STEP 9B: Save to CSV ===
results_df = pd.DataFrame({
    "Actual Price": actual_prices.flatten(),
    "Predicted Price": predicted_prices.flatten()
})
results_df.to_csv("prediction_results.csv", index=False)
print("Results saved to prediction_results.csv")
