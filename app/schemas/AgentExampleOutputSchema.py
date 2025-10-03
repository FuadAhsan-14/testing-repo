from pydantic import BaseModel, Field

class AgentExampleOutput(BaseModel):
    """
    Schema for the Agent output.
    """

    response_message: str = Field(
        ...,
        description="Create the response message for the user."
    )
