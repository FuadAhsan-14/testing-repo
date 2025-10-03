from pydantic import BaseModel, Field
from typing import List

class PrimarySchema(BaseModel):
    item_name: str = Field(..., description= "product name and composition")
    is_big_brown_cardboard_box: bool = Field(..., description= "True or False")
    quantity_image : List[int] = Field(..., description= "image numbers")
    batch_number_expired_date_image : List[int] = Field(..., description= "image numbers")
    quantity_details: str = Field(..., description= "packaging details")
    first_nested_unit: int = Field(..., description= "first nested unit value of quantity details")
    second_nested_unit: int = Field(..., description= "second nested unit value of quantity details")
    third_nested_unit: int = Field(..., description= "third nested unit value of quantity details")
