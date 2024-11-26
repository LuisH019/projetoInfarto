import joblib
import numpy as np

def predict(input_data):
    model_lgbm = joblib.load('model_lgbm.pkl')
    scaler = joblib.load('scaler.pkl')

    input_data_scaled = scaler.transform(input_data)

    prediction = model_lgbm.predict(input_data_scaled)
    if prediction == 1:
        pred_answer = "Doença cardíaca"
    else:
        pred_answer = "Sem doença cardíaca"

    probabilities = model_lgbm.predict_proba(input_data_scaled)
    if prediction == 1:
        probability = np.max(probabilities)
    else:
        probability = np.min(probabilities)

    return pred_answer, round(probability * 100, 2)

