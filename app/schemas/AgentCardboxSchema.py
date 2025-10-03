from pydantic import BaseModel, Field

class CardboxSchema(BaseModel):
    total_item: int = Field(description="The total number of items counted directly from the image.")
    total_item_confidence: int = Field(description="The confidence score (0-100) for the total_item count.")
    reasoning: str = Field(description="A detailed step-by-step reasoning for how the count was determined.")
