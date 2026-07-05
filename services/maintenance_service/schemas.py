from pydantic import BaseModel, Field


class MaintenanceRequest(BaseModel):
    vehicle_id:          str   = Field(..., example="V0042")
    odometer_km:         float = Field(..., ge=0,   example=125000)
    avg_speed_kmh:       float = Field(..., ge=0,   example=52.0)
    max_speed_kmh:       float = Field(..., ge=0,   example=112.0)
    engine_temp_mean_c:  float = Field(...,          example=94.0)
    engine_temp_max_c:   float = Field(...,          example=108.0)
    battery_voltage_mean:float = Field(...,          example=13.1)
    battery_voltage_min: float = Field(...,          example=11.8)
    vibration_mean_ms2:  float = Field(..., ge=0,   example=0.9)
    vibration_max_ms2:   float = Field(..., ge=0,   example=2.1)
    brake_pressure_mean: float = Field(..., ge=0,   example=7.8)
    hard_brakes_count:   int   = Field(..., ge=0,   example=8)
    idle_hours:          float = Field(..., ge=0,   example=120.0)
    trips_count:         int   = Field(..., ge=0,   example=310)


class MaintenanceResponse(BaseModel):
    vehicle_id:         str
    risk_score:         float  # 0–100
    risk_level:         str    # LOW / MEDIUM / HIGH / CRITICAL
    failure_probability:float  # 0–1 from RF classifier
    recommendation:     str
    model_version:      str = "1.0.0"
