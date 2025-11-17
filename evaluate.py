# Evaluation Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_model(model, x, y, scaler):
    predictions = model.predict(x.reshape((x.shape[0], x.shape[1], 1)))
    predictions = scaler.inverse_transform(predictions)
    y_true = scaler.inverse_transform(y.reshape(-1, 1))

    rmse = mean_squared_error(y_true, predictions, squared=False)
    mae = mean_absolute_error(y_true, predictions)
    mape = np.mean(np.abs((y_true - predictions) / y_true)) * 100
    direction_accuracy = np.mean((np.sign(predictions[1:] - predictions[:-1]) == np.sign(y_true[1:] - y_true[:-1])).astype(int))

    return rmse, mae, mape, direction_accuracy
