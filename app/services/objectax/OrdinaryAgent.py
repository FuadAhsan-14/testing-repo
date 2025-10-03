import asyncio
from datetime import datetime
import json
import time
from urllib.parse import urlparse
import uuid
from google.cloud import storage
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser, JsonOutputToolsParser
from langchain.output_parsers import OutputFixingParser
from core.BaseAgent import BaseAgent
from config.setting import env
from config.credentials import google_credential
from app.schemas.AgentOrdinarySchema import OrdinarySchema
from app.models.ObjectState import State_ax
from app.utils.Http.HttpResponseUtils import response_success, response_error 
from app.utils.Helper import prepare_images_for_llm, _postprocess_response, _compress_image_async, compute_usage

prompt_ordinary_claude = """Core Objective
You are a highly specialized pharmaceutical item counter with exceptional visual analysis skills. Your task is to return an exact count of the smallest saleable pharmaceutical items visible in the image, even under challenging conditions.

Detailed Instructions

1. Essential Counting Principles: 
   - Avoid analyzing the image based on the front row or the back row.
   - If both product information (alphanumeric characters, expiration date) and the product name appear in the image, treat them as separate objects when counting.
   - Each object may has the different structure, angle, orientation, or position.
   - Object may undergo rotation by a certain degree.
   - If a section exhibits a different pattern, treat that section as a distinct object.
   - Inspect edges and corners for easy-to-miss or slightly cropped items (still count them).
   - Ignore shelves, trays, straps/rubber bands, and shipping cartons.

   Specific Visual Cues to Watch For:
   - Partially obscured items, overlapping items, and items with different orientations.
   - Irregular lighting, shadows, glare, or visual obstructions.
   - Different packaging styles of the same product.
   - Shadow discrimination: Shadows cast by items are NOT separate items - look for consistent product dimensions and labeling

2. Common Error Traps (explicitly check)
   - Missing/partial edge rows or columns.
   - Ignore shadows, reflections, and glare highlights as objects; count only discrete tops with clear edges
   - Perspective causing unequal apparent spacing—still count discrete tops.
   - Check for height/width differences between stacks.
   - Look for a single "topper" item that makes one stack taller.

3. Dot-Based Thinking Method
   Perform a chain-of-thought analysis, noting the arrangement of items and identify each distinct product unit, especially in challenging situations like overlaps, partial visibility, or varied packaging. Mention any ambiguities.
   
   Key Definitions
   - Pharmaceutical Product Unit: A single, distinct item of pharmaceutical product. This includes, but is not limited to, individual boxes, vials, or bottles. 
   - Dot Thinking: Use dots (•) internally during your reasoning process to mentally mark each identified pharmaceutical product unit.
   
   Based on your analysis:
   - Exhaustive Detection: Scan the entire image systematically to locate all units. This includes units that are partially obscured, stacked, or in poor lighting.
   - Internal Dot Assignment: During your reasoning, mentally assign one dot (•) for every identified pharmaceutical product unit.
   - One Dot per Object: Ensure each dot represents exactly one product unit in your thinking process.
   - Visual Mapping: Describe the location/arrangement of items as you mentally place dots to ensure comprehensive coverage.
   - Final Count: Count all the dots you've mentally placed to get the total.

Output Requirements
Provide a JSON-formatted response with two key elements:
{{{{
    "total_item": <exact number by counting the dots>,
    "reasoning": "<detailed explanation of counting process, including systematic scan pattern, how dots were mentally assigned, and the dot-based thinking process>"
}}}}

4. DO NOT PUT ANY REASONING outside of "reasoning" field.

Special Considerations
- If image quality is poor or items are obstructed, specify the exact challenge and how it affected counting and confidence.
- When placing dots, be especially careful with overlapping or partially visible items.

Example Dot Thinking Process:
- Scan row 1: Found 3 items → mentally assign • • •
- Scan row 2: Found 2 items → mentally assign • •  
- Internal dot count: 5 dots total
- total_item: 5

Prohibited Actions
- Do not inflate/deflate counts without rationale.
- Do not make assumptions unsupported by the image.
- Do not output anything outside the required JSON.
- Do not place mental dots for shadows, reflections, or non-pharmaceutical items.

Treat every image like a critical pharmaceutical inventory verification—precision is paramount.
"""

class OrdinaryBoxAgentService(BaseAgent):
    """An agent responsible for ordinary box quantity extraction from pharmaceutical images."""
    
    def __init__(self, llm, **kwargs):
        super().__init__(
            llm=llm,
            prompt_template=prompt_ordinary_claude,
            output_model=OrdinarySchema, 
            use_structured_output=False,
            **kwargs
        )
        self._cached_examples = None

        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=self.output_model) if self.output_model else StrOutputParser()
        self.model_name = getattr(llm, "model_id", "unknown")
        self.examples = []
        asyncio.create_task(self.setup_examples())
    
    async def setup_examples(self):
        """Load few-shot examples for better LLM performance."""
        if self._cached_examples is not None:
            self.examples = self._cached_examples
            return
        
        example_configs = [
            {
                "url": "https://raw.githubusercontent.com/FuadAhsan-14/testing-repo/16ac3b7d0e81177364c1daf848002bb1ac1dfee8/36c.jpg",
                "output": {
                    "total_item": 10,
                    "reasoning": "I performed a systematic scan of the pharmaceutical boxes using dot-based thinking. The picture was taken from a slightly rightward position, causing the objects to appear slanted and to have unequal heights. The right stack is positioned slightly more forward on the shelf than the left stack, so its front edges protrude a bit further. Starting from the top-left and moving systematically: Top row - I identified 2 boxes of Jardiance Duo stacked horizontally. Second row - Found 2 more boxes in similar arrangement. Third row - Located 2 additional boxes. Fourth row - Identified 2 boxes. Fifth row - Found 2 boxes. All boxes show consistent Jardiance Duo Empagliflozin/Metformin labeling with 12.5 mg/1000 mg dosage and registration number DKL2356100217D1. Each box represents a distinct pharmaceutical product unit with clear edges and separate packaging. Total internal dot count: 10 dots, representing 10 individual pharmaceutical product boxes."
                }
            },
            {
                "url": "https://raw.githubusercontent.com/FuadAhsan-14/testing-repo/16ac3b7d0e81177364c1daf848002bb1ac1dfee8/81a.jpg", 
                "output": {
                    "total_item": 25,
                    "reasoning": "Using systematic dot-based thinking, I scanned the image from left to right, top to bottom to identify all pharmaceutical product units arranged in a grid pattern. I found 25 distinct pharmaceutical boxes with batch numbers and pricing information. Row 1 (top): 1 box - assigned dot •. Row 2: 2 boxes - assigned dots • •. Row 3: 2 boxes - assigned dots • •. Row 4: 4 boxes - assigned dots • • • •. Row 5: 4 boxes - assigned dots • • • •. Row 6: 4 boxes - assigned dots • • • •. Row 7: 4 boxes - assigned dots • • • •. Row 8 (bottom): 4 boxes - assigned dots • • • •. Critical considerations that prevented counting errors: 1) Did not assume perfect rectangular grid - carefully counted each row independently despite irregular arrangement. 2) Avoided visual processing shortcuts - although all boxes appear to be the same product with similar text layouts, each represents a distinct physical pharmaceutical unit. 3) Overcame text orientation confusion - some boxes display upside-down text, but this does not affect the physical unit count. 4) Used systematic row-by-row verification - double-checked each row count to ensure no boxes were missed or double-counted. 5) Focused on box boundaries rather than text content - distinguished individual units by their physical edges, not by text readability. Total internal dot count: 25 dots, representing 25 individual pharmaceutical product boxes in an irregular grid arrangement (1+2+2+4+4+4+4+4=25)."
                }
            },
            {
                "url": "https://raw.githubusercontent.com/FuadAhsan-14/testing-repo/16ac3b7d0e81177364c1daf848002bb1ac1dfee8/87a.jpg",
                "text": "[{'item_name': 'Festaric Febuxostat 40 Film coated tablet 40 mg', 'is_big_brown_cardboard_box': False, 'quantity_image': [1], 'batch_number_expired_date_image': [2], 'quantity_details': '3 strips x 10 film coated tablets', 'first_nested_unit': 3, 'second_nested_unit': 10, 'third_nested_unit': 0}]",
                "output": {
                    "total_item": 17,
                    "reasoning": "Using systematic dot-based thinking, I scanned the image from left to right, top to bottom to identify all pharmaceutical product units. I found 17 distinct pharmaceutical boxes arranged in two uneven vertical stacks of Festaric 40 (Febuxostat) products. Left stack (bottom to top): 9 boxes - each distinct box edge assigned dots • • • • • • • • •. Right stack (bottom to top): 8 boxes - each distinct box edge assigned dots • • • • • • • •. Critical considerations that ensured accurate counting: 1) Did not assume symmetrical stacking - carefully counted each stack independently by identifying horizontal dividing lines between boxes. 2) Avoided double-counting due to camera angle - while multiple faces of boxes are visible, each physical package was counted only once. 3) Distinguished between stacked boxes versus single boxes with multiple visible faces - each horizontal separation line indicates a separate physical pharmaceutical unit. 4) Ignored the surface/background and focused only on the pharmaceutical product packages. 5) Verified count by re-scanning each stack separately to confirm the uneven arrangement (9 vs 8 boxes). Total internal dot count: 17 dots, representing 17 individual pharmaceutical product boxes in an asymmetrical 9+8 configuration."
                }
            },
            {
                "url": "https://raw.githubusercontent.com/FuadAhsan-14/testing-repo/16ac3b7d0e81177364c1daf848002bb1ac1dfee8/88a.jpg",
                "output": {
                    "total_item": 17,
                    "reasoning": "I performed a systematic scan of the pharmaceutical boxes using dot-based thinking. I found two stacks with different heights. After further observation, the difference in height between the stacks was due to one additional object being placed on the left stack. The image shows white pharmaceutical boxes with 'Festaric Febuxostat 40' labeling arranged in two main stacks. Left stack analysis: Starting from top - I can see 9 distinct boxes stacked vertically, each with clear edges and separate packaging. Right stack analysis: I can identify 8 distinct boxes stacked vertically with two different patterns, 5 objects below with the label 'Festaric Febuxostat 40' and 3 white box objects above, each with clearly defined boundaries. No shadows, reflections, or non-pharmaceutical items were counted. Total internal dot count: 9 + 8 = 17 dots, representing 17 individual pharmaceutical product units."
                }
            }
        ]
        
        # Load base64 examples
        for config in example_configs:
            try:
                base64_content = await _compress_image_async(config["url"])
                if base64_content:
                    self.examples.append(
                        (
                            HumanMessage(content=[
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_content}"}},
                                # {"type": "text", "text": config["text"]}
                            ]),
                            AIMessage(content=json.dumps(config["output"], ensure_ascii=False))
                        )
                    )
            except Exception as e:
                print(f"Warning: {e}")
        
        self._cached_examples = self.examples
    
    async def __call__(self, state: State_ax):
        # """An agent responsible for getting Ordinary Box Informations."""
        try:
            run_start = time.time()
            # Extract quantity images
            quantity_image = state["primary"]["quantity_image"]
            if isinstance(quantity_image, int):
                quantity_image = [quantity_image]
            
            images_link = state["url"]
            image_quantity_n = [
                images_link[i - 1] for i in quantity_image 
                if 1 <= i <= len(images_link)
            ]
            
            # Prepare images for LLM with compression
            content_parts, _ = await prepare_images_for_llm(
                image_quantity_n,
                use_compression=True
            )

            # few-shot as messages
            messages = [("system", prompt_ordinary_claude)]
            for human, ai in self.examples:
                messages.append(human)
                messages.append(ai)

            # add the real query
            messages.append(HumanMessage(content=content_parts))

            final_prompt = ChatPromptTemplate.from_messages(messages)
            chain = final_prompt | self.llm | RunnableParallel(
                raw=RunnablePassthrough(),
                parsed=OutputFixingParser.from_llm(self.llm, self.parser)
            )

            # With few-shot examples
            response = await chain.ainvoke({})
            raw = response["raw"]
            parsed = response["parsed"]
            
            # # Without few-shot examples
            # raw, parsed = await self.arun_chain(content_parts)
            
            print("Raw ordinary box response:", raw)
            print("Parsed ordinary box response:", parsed)

            get_metadata = True
            if get_metadata:
                # print("Raw response:", raw)
                # Convert AIMessage to dict before computing usage
                raw_dict = {
                    "response_metadata": raw.response_metadata,
                    "usage_metadata": getattr(raw, 'usage_metadata', {})
                }
                
                usage = compute_usage(raw_dict, self.model_name)

                ai_msg = parsed
                processed_msg = _postprocess_response(ai_msg, state)

                runtime = time.time() - run_start
                output = {
                    "quantity": processed_msg,
                    "quantity_usage": usage,
                    "quantity_runtime": runtime,
                }
                print("quantity_output: ", output)

            else: 
            
                ai_msg = json.loads(parsed.content)

                processed_msg = _postprocess_response(ai_msg, state)

                output = {"quantity": processed_msg}

                print("ordinary_box_output: ", output)
            
            return output
            
        except Exception as e:
            # Cleanup on error
            print("Error terjadi pada Ordinary Box Agent")
            raise response_error(str(e))
    