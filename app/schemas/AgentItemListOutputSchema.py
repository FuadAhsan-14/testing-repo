from typing import List, Optional
from pydantic import BaseModel, Field

class ItemListSchema(BaseModel):
    item_name: str = Field(
        ...,
        description="This field is the item name of each unit in the invoice"
    )
    quantity: Optional[int] = Field(
        None,
        description="The quantity of the item in the invoice."
    )
    uom: Optional[str] = Field(
        None,
        description="This field is the uom of the unit in the invoice"
    )
    batch_number: Optional[str] = Field(
        None,
        description="This Field is the batch number of the invoice in the image"
    )
    expired_date: Optional[str] = Field(
        None,
        description="This field is the expired date of the invoice in the image"
    )

class AgentItemListOutput(BaseModel):
    """
    Schema for the Agent output.
    """

    item_list: List[ItemListSchema] = Field(..., description="A list of item details.")