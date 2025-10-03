import asyncio
import time
from typing import Any, Dict, List, Optional
import uuid
from google.cloud import storage
from langchain_core.messages import HumanMessage
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.output_parsers import JsonOutputParser
from config.setting import env
from core.BaseAgent import BaseAgent
from config.credentials import google_credential
from app.schemas.AgentAnomalySchema import AnomalySchema
from app.models.ObjectState import State_ax
from app.utils.Http.HttpResponseUtils import response_success, response_error 
from app.utils.Helper import prepare_images_for_llm, compute_usage

test_anomaly_prompt_v4 = """
# Role & Goal
You are a specialized AI analyst for pharmaceutical inventory management. Your task is to analyze 1-4 input images and detect specific anomalies using OCR and visual analysis.

# Rules & Constraints
1. IGNORE any information that considered as a background, such as loose paper, leaflet, invoice, poster, manual, or printed sheet visible in an image.
2. Account for various text formats and pharmaceutical packaging styles
3. Consider partial dates and abbreviated text
4. Distinguish between same product (multiple units) vs different products
5. Prioritize accuracy over speed in anomaly detection
6. Maintain systematic verification throughout analysis

# Process / Steps
Step 1: Image Content Verification
    For each image, verify it shows:
    - Pharmaceutical product packaging
    - The pharmaceutical product itself
    - Boxes containing pharmaceutical products

Step 2: Batch Number Detection
    Search for batch identifiers in each image:
    - Orientation: Horizontal or vertical text
    - Common labels: "Batch Number", "Batch No.", "Lot No.", "BN", "BN:", "LOT:", or just alphanumeric code without label
    - Format: Alphanumeric or numeric codes (e.g., ICFZE123, 223244, LOT456)
    - Location: Typically near dates (expiry, manufacturing)
    - Note: When the batch number cannot be read clearly (e.g., due to poor lighting, partially obscured, or other challenges), assume it is the same as the others.

Step 3: Product Identification Analysis
    Extract key identifiers from each image:
    - Product name
    - Batch number 
    - Active ingredient
    - Dosage/strength
    - Exclude any other identifiers like SN, S/N, Reg. No. 

Step 4: Anomaly Detection Rules
    Batch Number Anomaly
    - Trigger: NO batch number visible in ANY image of the set
    - Pass: At least one image contains a visible batch number

    Non-Pharmaceutical Anomaly
    - Trigger: ANY image shows only documents or non-pharmaceutical items
    - Pass: All images show pharmaceutical products, packaging, or product boxes

    Multiple Products (Single Image) Anomaly
    - Trigger: ANY single image contains 2 or more distinct pharmaceutical products
    - Example: Multiple Products Anomaly: One image showing two batch number "Batch number: ABC123" AND "Batch number: XYZ789"
    - Not an anomaly: Multiple units of the same product

    Different Products (Image Set) Anomaly
    - Assumption: If product details are missing in one image but present in others, assume they match.  
    - Trigger: Images show different values from the same identifiers (batch number vs batch number, NOT batch number vs product name. GTIN vs GTIN, NOT GTIN vs Product Name. Product Name vs Product Name, NOT Product Name vs GTIN)
    - Example: Not an anomaly: Image 1 shows "GTIN: 07613510", "Batch Number: 2234012" and "Expiry Date: 12 2025". Image 2 shows boxes of Lioresal Baclofen".

Step 5: Analysis Process
1. Sequential Image Review: Examine each image systematically
2. Text Region Focus: Prioritize areas with visible text and labels
3. Visual Cue Recognition: Look for pharmaceutical indicators (medical terminology)
4. Cross-Reference Validation: Compare product details across all images
5. Anomaly Classification: Apply detection rules to classify findings
    """

class AnomalyAgentService(BaseAgent):
    """Simplified Anomaly Detection Agent."""
    
    def __init__(self, llm, **kwargs):
        # Setup GCS client and bucket
        super().__init__(
            llm=llm,
            prompt_template=test_anomaly_prompt_v4,  
            output_model=AnomalySchema,              
            use_structured_output=False,
            **kwargs
        )
        self.model_name = getattr(llm, "model_name", "unknown")

    async def __call__(self, state) -> Dict:
        """Process anomaly detection."""
        try:
            run_start = time.time()
            
            # files = state["images"]
            # po_number = state.get("po_number")

            url_list = state["url"]
            
            content_parts, _ = await prepare_images_for_llm(url_list)

            # result = await self.run_llm_with_images(content_parts)
            
            raw, parsed = await self.arun_chain(content_parts)
            
            get_metadata = True
            if get_metadata:
                if parsed['is_anomaly']:
                    print("⚠️ Anomaly detected!")
                # Convert AIMessage to dict before computing usage
                raw_dict = {
                    "response_metadata": raw.response_metadata,
                    "usage_metadata": getattr(raw, 'usage_metadata', {})
                }
                
                usage = compute_usage(raw_dict, self.model_name)
                runtime = time.time() - run_start
                output = {
                    "anomaly": parsed,
                    "url": url_list,
                    "anomaly_usage": usage,
                    "anomaly_runtime": runtime,
                }
                print("anomaly_output: ", output)

            else: 
                if parsed.is_anomaly:
                    print("⚠️ Anomaly detected!")

                ai_msg = parsed.__dict__
                output = {
                    "anomaly": ai_msg,
                    "url": url_list,
                }
                
                print("anomaly_output: ", output)
                # print(f"anomaly_usage: \n {usage}")    
            return output
            
        except Exception as e:
            print("Error terjadi pada Anomaly Agent")
            raise e
        