import asyncio
import time
from urllib.parse import urlparse
import uuid
from google.cloud import storage
from langchain_core.messages import HumanMessage
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.output_parsers import JsonOutputParser
from config.setting import env
from config.credentials import google_credential
from core.BaseAgent import BaseAgent
from app.schemas.AgentPrimarySchema import PrimarySchema
from app.models.ObjectState import State_ax
from app.utils.Http.HttpResponseUtils import response_success, response_error 
from app.utils.Helper import _compress_image_async, create_image_content_from_urls, upload_images_to_gcs, prepare_images_for_llm

primary_prompt="""You are an expert AI inventory analyst specializing in pharmaceutical product. Your primary goal is to analyze image of pharmaceutical products and extract the corresponding information. You will receive 1 to 3 images of the SAME PRODUCT from different viewpoints.
CRITICAL: Adopt a "Systematic Verification" framework. This means you will approach each task with a structured, step-by-step methodology, constantly verifying your assumptions and calculations.

To get accurate information of the items, follow these steps
1. Analyze the images entirely.
2. Count the quantity of the items available in each images
3. Item Name (Precise and Complete): Correctly identify the name of each unique item based on the text present on the packaging. 
4. Big Brown Cardboard Box: Determine if the item is packaged in INDUSTRIAL-STYLE BROWN CARDBOARD BOXES, used as the primary packaging for a pharmaceutical product and labeled with key details like batch number and expiration date. If a brown box is present in the background or surrounding the item, but the item itself is not IN the brown box, set "is_big_brown_cardboard_box" to False. Only set it to True if the item is CONTAINED WITHIN a large brown cardboard box.
5. Quantity Image:
    - Carefully count each item visible in each image. Ensure you are not double-counting or missing any items. If an item is partially obscured, but you can reasonably infer its presence, count it.
    - Compare the number of items in each images
    - Select the image where the number of items is highest. Your primary goal is to identify the image with the HIGHEST possible number of items. Even if one image has only one more item than another, you must select the image with the higher count.
    - It doesn't matter if you choose the image that are difficult to accurately count the total number of items.
    - STRICTLY Choose ONLY 1 IMAGE, If there are 2 or more images that have the same number of items, choose only 1 of the image.
    - This field cannot be empty!
6. Batch/Expiry Image: Select the image(s) that show the batch number and expiry date most clearly. This field cannot be empty! there must be an image that is showing the batch number and expired date.
7. Quantity Details: Examine the packaging to determine how the product is packaged within the box (e.g., blisters, bottles, vials). Provide a string description of the internal packaging, describe the packaging based on any quantity information available on the external labels. This could include the number of items per box, or groupings like 'dozens' or 'packs'. If no packaging information is discernible from either internal views or external labels, then and only then should you return an empty string "".
8. Nested Unit value:
    - From quantity details, broke it down into three nested unit values: first_nested_unit, second_nested_unit, and third_nested_unit.
    - A 'nested unit' refers to any grouping of items within the outermost box excluding volume and weight. If the box contains individual items directly (e.g., bottles, vials, tablets), those individual items are considered the 'first nested unit'. For example, "24 boxes @ 12 strips @ 10 tablets" indicates three levels of packaging. "ISI: 12 BOTOL" indicates ONE level of packaging, where the bottle is the individual item. Even if there's only one type of item inside the box, it still represents a nested unit.
    - IMPORTANT: STRICTLY DO NOT include any values that represent VOLUME (e.g., mL, L, cc) or WEIGHT (e.g., mg, g, kg) on any nested unit.
    - If there are grouping units like "dozen" (dz, Dz, doz), "gross", "pair", or other grouping units, convert it into integer and PUT it in ONE NESTED UNIT (e.g. "10 dz" the nested unit are "first_nested_unit": 10, "second_nested_unit": 12.).
9. Output should be only containing JSON, do not put any reasoning on the output.

Final Verification:
[ ] 'image_quantity' filled with image that contain the highest number of items
[ ] Do not include value that represent volume or weight in any of nested unit
"""

class PrimaryAgentService(BaseAgent):
    """An agent responsible to extract primary information from images using existing URLs."""
    def __init__(self, llm, **kwargs):
        super().__init__(
            llm=llm,
            prompt_template=primary_prompt,
            output_model=PrimarySchema,
            use_structured_output=True,
            # enable_caching=True,
            # enable_timing=True,
            # gcs_client=gcs_client,
            # gcs_bucket=gcs_bucket,
            **kwargs
        )
        # # Add caching for models
        # self.llm_cache = {}
        # self.llmc_cache = {}

    async def __call__(self, state: State_ax):
        try:
            fewshot_examples = """
            <EXAMPLE>
            INPUT 1 (Imagine 3 images of a box of Amoxicillin):
                *  Image 1: Front view of the box, showing the name "Amoxicillin 500mg Capsules" and the quantity "100 capsules". Shows 2 visible boxes.
                *  Image 2: Side view of the box, showing the batch number and expiry date. Shows 1 visible box.
                *  Image 3: Another side view of the box. Shows 5 visible boxes.
            OUTPUT 1:
                ```json
                {{"item_name" : "Amoxicillin 500mg Capsules [Brand Name]", "is_big_brown_cardboard_box" : false, "quantity_image" : [3], "batch_number_expired_date_image" : [2], "quantity_details" : "1 bottle x 100 capsules", "first_nested_unit": 1, "second_nested_unit": 100, "third_nested_unit": 0}}
                ```
            INPUT 2 (Imagine 3 images of a single bottle inside a cardboard box):
                *  Image 1: Front view of the cardboard box, showing the name.
                *  Image 2: Open cardboard box, showing the bottle inside.
                *  Image 3: Close-up of the bottle, showing batch number and expiry.
            OUTPUT 2:
                ```json
                {{"item_name" : "Amoxicillin 500mg Capsules [Brand Name]", "is_big_brown_cardboard_box" : true, "quantity_image" : [1], "batch_number_expired_date_image" : [3], "quantity_details" : "1 bottle x 500 ml", "first_nested_unit": 1, "second_nested_unit": 0, "third_nested_unit": 0}}
                ```
            INPUT 3 (Imagine 3 images of a single bottle with a brown box in the background):**
                *  Image 1: Front view of the bottle, with a brown box partially visible in the background.
                *  Image 2: Side view of the bottle, with the brown box still in the background.
                *  Image 3: Close-up of the bottle, showing batch number and expiry.
            OUTPUT 3:
                ```json
                {{"item_name" : "Amoxicillin 500mg Capsules [Brand Name]", "is_big_brown_cardboard_box" : false, "quantity_image" : [1], "batch_number_expired_date_image" : [3], "quantity_details" : "1 bottle", "first_nested_unit": 1, "second_nested_unit": 0, "third_nested_unit": 0}}
                ```
            INPUT 4 (Imagine 3 images of two brown cardboard box):**
                *  Image 1: Front view of the box, showing the name "Paracetamol 100mg Tablet" and the quantity "20 dz @ 10 Tablet". Shows 2 visible boxes.
                *  Image 2: Side view of the box, showing the clearer label of batch number and expiry date.
                *  Image 3: Another side view of the box.
            OUTPUT 4:
                ```json
                {{"item_name" : "Paracetamol 100mg Tablet", "is_big_brown_cardboard_box" : true, "quantity_image" : [1], "batch_number_expired_date_image" : [2], "quantity_details" : "20 Dz @ 10 Tablet", "first_nested_unit": 20, "second_nested_unit": 12, "third_nested_unit": 10}}
                ```
            </EXAMPLE>
            """

            self.rebind_prompt_variable(
                fewshot_examples = fewshot_examples
            )
            
            url_list = state["url"]
            
            if not url_list:
                raise ValueError("No URLs provided in state")
            # print(url_list)

            # Create image content using BaseAgent method
            content_parts, _ = await prepare_images_for_llm(
                url_list,
                use_compression=True
            )
            
            # Use BaseAgent's caching and chain execution
            raw, parsed = await self.arun_chain(content_parts)
            
            ai_msg = parsed.__dict__
            output = {
                "primary": ai_msg,
            }
            
            print("primary_output:", output)
            return output
            
        except Exception as e:
            # Clean up on error if needed
            print("Error terjadi pada Primary Agent")
        
            raise response_error(str(e))