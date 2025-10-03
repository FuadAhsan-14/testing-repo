from pydantic import BaseModel, Field

class AnomalySchema(BaseModel):
    is_anomaly: bool = Field(..., description= "True or False")
    # anomaly_reason: str = Field(..., description= "The progress thinking and the reason of anomaly")
    anomaly_reason: str = Field(..., description= "A detailed step-by-step explanation of how the anomaly conclusion is reached.")
