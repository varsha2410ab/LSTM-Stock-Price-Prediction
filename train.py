from preprocessing import preprocess_data
from model import create_model

x, y, scaler = preprocess_data("data.csv")

x = x.reshape((x.shape[0], x.shape[1], 1))  # reshape for LSTM input

model = create_model((x.shape[1], 1))
model.fit(x, y, epochs=10, batch_size=32)

model.save("lstm_model.h5")
print("Model training complete and saved as lstm_model.h5")
