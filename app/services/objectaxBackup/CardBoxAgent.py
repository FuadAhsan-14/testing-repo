import asyncio
from datetime import datetime
import json
import time
import uuid
from google.cloud import storage
from langchain_core.messages import HumanMessage
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.output_parsers import JsonOutputParser
from core.BaseAgent import BaseAgent
from config.setting import env
from config.credentials import google_credential
from app.schemas.AgentCardboxSchema import CardboxSchema
from app.models.ObjectState import State_ax
from app.utils.Http.HttpResponseUtils import response_success, response_error 
from app.utils.Helper import prepare_images_for_llm, _postprocess_response


PROMPT_BOX_CLAUDE= """You are an expert AI inventory analyst specializing in pharmaceutical product identification and quantity estimation. Your primary goal is to provide accurate and reliable information for big cardboard boxes quantities.
To count the quantities, follow these steps:
1. How many cardboard boxes are in the image? fill the answer value in the field 'total_item'.
2. Give your confidence value for your counting result.
3. The output should be formatted as a JSON instance that conforms to the JSON schema below.
4. Don't Include Reasoning In Output: Output should be in JSON format.
"""


class CardboxAgentService(BaseAgent):
    """An agent responsible for getting Cardboard Box Informations"""
    def __init__(self, llm, **kwargs):
        super().__init__(
            llm=llm,
            prompt_template=PROMPT_BOX_CLAUDE,
            output_model=CardboxSchema,
            use_structured_output=True,
            # enable_caching=True,
            # # enable_timing=True,
            # gcs_client=gcs_client,
            # gcs_bucket=gcs_bucket,
            **kwargs
        )

        # Add caching for models
        self.llm_cache = {}
        self.llmc_cache = {}

    async def __call__(self, state: State_ax):
        try:

            quantity_image = state["primary"]["quantity_image"]
            # print(quantity_image)
            # if isinstance(quantity_image, int):
                # quantity_image = [quantity_image]

            images_link = state["url"]
            image_quantity_n = [images_link[i-1] for i in quantity_image]
            # print(image_quantity_n)
            # Use BaseAgent helper
            # content_parts, url_list = await self.prepare_images_for_llm(
            #     image_quantity_n,
            #     text_content=str(state["primary"]),
            #     use_compression=True
            # )
            
            content_parts, _ = await prepare_images_for_llm(
                image_quantity_n,
                use_compression=True
            )
            
            # Run chain
            raw, parsed = await self.arun_chain(content_parts)
            
            # ai_msg = await self.run_llm_with_images(
            #     content_parts,
            #     extra_inputs={"json": state["primary"]}
            # )

            ai_msg = parsed.__dict__

            processed_msg = _postprocess_response(ai_msg, state)
            output = {"quantity": processed_msg}

            print("cardbox_output: ", output)  
            return output
        
        except Exception as e:
            print("Error terjadi pada Cardbox Agent")
            # if state.get("url"):
            #     self.cleanup_gcs_images(state["url"])
            raise response_error(str(e))