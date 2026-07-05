from pydantic import BaseModel, Field
from typing import Optional


class QuoteRequest(BaseModel):
    pickup_zone:     str   = Field(..., example="East Village")
    dropoff_zone:    str   = Field(..., example="Clinton East")
    pickup_borough:  str   = Field(..., example="Manhattan")
    dropoff_borough: str   = Field(..., example="Manhattan")
    trip_distance:   float = Field(..., gt=0, example=2.7)
    order_hour:      int   = Field(..., ge=0, le=23, example=9)
    day_of_week:     int   = Field(..., ge=0, le=6, example=2)
    month:           int   = Field(..., ge=1, le=12, example=3)
    temp:            float = Field(default=10.0, example=5.0)
    prcp:            float = Field(default=0.0,  ge=0.0, example=0.0)
    wspd:            float = Field(default=10.0, ge=0.0, example=10.0)
    snow:            float = Field(default=0.0,  ge=0.0, example=0.0)


class QuoteResponse(BaseModel):
    eta_p50_min:      float
    eta_p10_min:      float
    eta_p90_min:      float
    base_price:       float
    surge_multiplier: float
    final_price:      float
    model_version:    str = "1.0.0"
