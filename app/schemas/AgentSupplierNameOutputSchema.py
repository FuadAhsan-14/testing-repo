from pydantic import BaseModel, Field

class AgentSupplierNameOutput(BaseModel):
    """
    Schema for the Supplier Name Agent output.
    """

    supplier_name: str = Field(
        ...,
        description="Extracted supplier name from the invoice."
    )
