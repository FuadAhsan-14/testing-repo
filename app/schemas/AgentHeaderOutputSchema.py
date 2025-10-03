from pydantic import BaseModel, Field

class AgentHeaderOutput(BaseModel):
    """
    Schema for the Agent output.
    """

    invoice_date: str = Field(
        ...,
        description="The date of the invoice."
    )
    
    invoice_number: str = Field(
        ...,
        description="The invoice number."
    )
    
    po_number: str = Field(
        ...,
        description="The purchase order number."
    )
