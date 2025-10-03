from pydantic import BaseModel, Field

class BatchSchema(BaseModel):
    item_name: str = Field(..., description = "This is the name of the item")
    batch_number: str = Field(..., description="The extracted and mapped batch number. Fill it with None value if the field is not present in the image.")
    batch_number_confidence: str = Field(..., description="Your confidence value of extracting the batch number of the items")
    expired_date: str = Field(..., description="The extracted expiration date of the items in the image. Fill it with None value if the field is not present in the image.")
    expired_date_confidence: str = Field(..., description="Your confidence value of extracting the expired date of the items.")
    reason: str = Field(..., description="The explanation detailing information findins and choices.") 