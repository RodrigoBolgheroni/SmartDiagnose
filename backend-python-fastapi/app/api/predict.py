from fastapi import APIRouter, HTTPException
from app.schemas.predict_schema import PredictRequest
import numpy as np
import joblib
import os
from keras.models import load_model

router = APIRouter()

# Caminhos
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.keras')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')

# Carregamento
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

@router.post("/predict")
def predict(req: PredictRequest):
    sintomas_dict = req.sintomas

    try:
        # Ordenar e montar input
        features = list(sintomas_dict.values())
        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)

        pred_prob = model.predict(X_scaled)[0]
        pred_class = np.argmax(pred_prob)
        doenca = label_encoder.inverse_transform([pred_class])[0]

        return {
            "doenca_predita": doenca,
            "confianca": float(np.max(pred_prob))
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
