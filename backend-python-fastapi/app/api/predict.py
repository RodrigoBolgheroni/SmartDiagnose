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
FEATURE_PATH = os.path.join(MODEL_DIR, 'features.pkl')

# Carregamento
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)
feature_names = joblib.load(FEATURE_PATH) 


@router.post("/predict")
def predict(req: PredictRequest):
    sintomas_dict = req.sintomas

    try:
        input_data = [sintomas_dict.get(feature, 0) for feature in feature_names]

        X = np.array(input_data).reshape(1, -1)
        X_scaled = scaler.transform(X)

        pred_prob = model.predict(X_scaled)[0]

        top_indices = pred_prob.argsort()[-5:][::-1]  
        top_doencas = label_encoder.inverse_transform(top_indices)
        top_probs = pred_prob[top_indices]

        top_5_result = {doenca: float(prob) for doenca, prob in zip(top_doencas, top_probs)}

        pred_class = np.argmax(pred_prob)
        doenca_predita = label_encoder.inverse_transform([pred_class])[0]
        confianca = float(np.max(pred_prob))

        return {
            "doenca_predita": doenca_predita,
            "confianca": confianca,
            "top_5_doencas": top_5_result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

