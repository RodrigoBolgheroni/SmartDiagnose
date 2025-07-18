from pydantic import BaseModel
from typing import Dict

class PredictRequest(BaseModel):
    sintomas: Dict[str, float]
