from pydantic import BaseModel, Field
from typing import List


class NLPRequest(BaseModel):
    text: str = Field(..., min_length=3,
                      example="My delivery was 45 minutes late due to traffic.")


class CategoryScore(BaseModel):
    category: str
    probability: float


class NLPResponse(BaseModel):
    predicted_category: str
    predicted_label:    int
    confidence:         float
    all_scores:         List[CategoryScore]
    model_version:      str = "1.0.0"
