from pydantic import BaseModel, Field

class AgentPrimaryOutput(BaseModel):
    """
    Schema for the Agent output.
    """

    is_cardboard: bool = Field(
        ...,
        description="Indicates whether the output is for a cardboard box."
    )

    quantity_images: int = Field(
        ...,
        description="The number of images in the output."
    )

    bn_ed_images: list = Field(
        ...,
        description="A list of images containing batch number and expiration date information."
    )
