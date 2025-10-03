from pydantic import BaseModel, Field
from typing import List

class MatchingSchema(BaseModel):
    po_item_list: List[str] = Field(..., description="The source items that were part of the mapping.")
    faktur_item_list: List[str] = Field(..., description="The target items, reordered to match their source counterparts.")