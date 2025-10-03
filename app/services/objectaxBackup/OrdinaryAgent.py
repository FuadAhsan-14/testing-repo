import asyncio
from datetime import datetime
import json
import time
import uuid
from google.cloud import storage
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.output_parsers import JsonOutputParser
from core.BaseAgent import BaseAgent
from config.setting import env
from config.credentials import google_credential
from app.schemas.AgentOrdinarySchema import OrdinarySchema
from app.models.ObjectState import State_ax
from app.utils.Http.HttpResponseUtils import response_success, response_error 
from app.utils.Helper import prepare_images_for_llm, _postprocess_response, _compress_image_async

PROMPT_QUANTITY_CLAUDE= """Core Objective
You are a highly specialized pharmaceutical item counter with exceptional visual analysis capabilities. Your primary task is to deliver an extremely accurate count of pharmaceutical items in any given image, even under challenging visual conditions.

Detailed Counting Instructions
1. Perform an exhaustive visual scan of the entire image.
2. Count items with absolute precision, considering:
   - Partially obscured items
   - Stacked or overlapping boxes
   - Irregular lighting conditions
   - Varied box orientations
   - Different packaging styles
   - Shadows or visual obstructions
3. Count the item per stack first then add it up into total_item

Counting Methodology
- Systematically analyze visual cues.
- Count each unique item, even if partially visible.
- Verify counts using multiple visual confirmation techniques.
- Explain the precise rationale behind the final count.

Specific Pharmaceutical Counting Strategies
- Identify unique product identifiers.
- Recognize different box orientations (top, side, front).
- Account for stacking and partial visibility.
- Detect subtle visual distinctions between individual items.

Critical Guidelines
- Prioritize accuracy over speed.
- Be transparent about the counting methodology.
- Handle ambiguous scenarios with methodical visual analysis.
- Never guess or estimate without clear visual evidence.

Special Considerations for Challenging Images
- If image quality is poor, explain the specific challenges.
- Describe any assumptions made during counting.
- Provide a confidence level according to the challenge on your counting process.
- Lower your confidence level when the image is challenging for you to count accurately such as when there are obsctucted item.

Prohibited Actions
- Do NOT inflate or deflate item counts.
- Do NOT make assumptions without visual evidence.
- Do NOT assume the number of items is the same between one stack and another.
- Always base counts on clear, confirmable visual data.

Treat each image as a critical pharmaceutical inventory verification task — precision is paramount."""

class OrdinaryBoxAgentService(BaseAgent):
    """An agent responsible for ordinary box quantity extraction from pharmaceutical images."""
    
    def __init__(self, llm, **kwargs):
        super().__init__(
            llm=llm,
            prompt_template=PROMPT_QUANTITY_CLAUDE,
            output_model=OrdinarySchema,
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
            image_quantity_n = [images_link[i] for i in quantity_image]
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

            print("ordinary_output: ", output)  
            return output
        
        except Exception as e:
            print("Error terjadi pada Ordinary Agent")
            # if state.get("url"):
            #     self.cleanup_gcs_images(state["url"])
            raise response_error(str(e))
    