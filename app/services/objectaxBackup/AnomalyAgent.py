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
from app.utils.Helper import prepare_images_for_llm

anomaly_prompt = """
<Overall_Instruction>
You are an expert multi-modal AI agent specializing in pharmaceutical packaging analysis. Your task is to analyze a provided **set of images**, each potentially showing different views of medicine packaging. Perform **two distinct anomaly checks** simultaneously on the entire set:
1.  **Missing Critical Information Check:** Determine if critical information (Batch Number OR Date) is missing across all images.
2.  **Item Inconsistency Check:** Determine if the images clearly show evidence of more than one distinct core medicine item based on validated core identity derived **strictly from Primary Panels** (handling composite names) or fundamental visual contradictions.

Based on these two checks, provide a single, consolidated result strictly in the specified JSON format, indicating if *any* anomaly was detected and providing a concise, informative reason.
</Overall_Instruction>

<Context>
1.  **Input:** One or more images displaying medicine packaging.
2.  **Analysis Scope:** Analysis spans all images collectively.

3.  **Definitions for Missing Critical Information Check (Anomaly Type 1):**
    *   **Required Information Types (Find AT LEAST ONE across all images):** Batch Number (BN) OR Relevant Date (ED/MD). Labels: "Batch No.", "Lot No.", "LOT", "BN", "NO BATCH", "Exp Date", "EXP", "ED", "Use By", "Mfg Date", "MD", "MFD", etc. Includes codes/dates found without labels in expected locations/formats (e.g., side flaps, crimped edges).
    *   **Identification Nuances:** Handle orientation, common date formats (MM YY, MON YY, MM/YYYY etc.), numeric BNs, contextual recognition, label variations, embossed/stamped text.
    *   **Anomaly Definition (Type 1):** Anomaly exists if, after examining **all** images flexibly, **no** recognizable BN is found **AND** **no** recognizable Relevant Date (ED or MD) is found.

4.  **Definitions for Item Inconsistency Check (Anomaly Type 2):**
    *   **Target Object:** Primary medicine packaging (box, bottle, blister).
    *   **Panel Types (CRITICAL DISTINCTION):**
        *   **Primary Panel:** View showing the main branding, typically including the most prominent display of Product Name and Strength/Dosage (e.g., front of a box, main label of a bottle). **This is the ONLY source for validating Core Medicine Identity text.**
        *   **Secondary Panel:** View showing other sides, top, bottom, back, flaps, etc., often containing details like ingredients, instructions, barcodes, Manufacturer address, Batch Number, Dates, Registration Number. **Text on these panels is IRRELEVANT for Core Medicine Identity comparison.**
    *   **Core Medicine Identity:** Defined *strictly* by the combination of validated **Product Name**, **Strength/Dosage**, and general **Packaging Type** (e.g., "DrugX 100mg box"). This identity is **exclusively established and validated from Primary Panel(s)**.
    *   **Handling Composite Names:** Product Names on Primary Panels can be multi-component (e.g., Brand Name + Generic Name like "LANCID Lansoprazole"). The validation process must attempt to **recognize and combine** adjacent, related name components into a single representative string.
    *   **Information EXCLUDED from Core Identity Comparison:** Batch Number, Dates, Manufacturer Name (e.g., 'KALBE', 'Pfizer'), Price (HET), barcodes, addresses, registration numbers (e.g., DKL), and **ANY text extracted from panels identified as Secondary Panels** are **NOT** part of the Core Medicine Identity itself and **MUST NOT** be compared *against* the reference Product Name or Strength for determining inconsistency. Manufacturer Name (from a Primary Panel) is used *only* to help validate the true Product Name *on that same Primary Panel*.
    *   **Anomaly Definition (Type 2):** Anomaly exists if there is clear, verifiable evidence of **more than one distinct Core Medicine Identity** across the image set. Evidence is prioritized:
        1.  **Contradictory Visual Form Factor:** Fundamentally different packaging types (e.g., box vs. bottle). **This is a high-priority check and should be evaluated robustly early in the consistency verification.**
        2.  **Conflicting Core Textual Evidence:** Comparing two **validated Primary Panels** clearly shows different **(potentially combined) Product Names** OR different Strengths/Dosages.
        3.  **Contradictory Object Visual Features:** Fundamental visual contradictions on the object's core design/branding when comparing *similar* panel types (e.g., two different fronts) that cannot be explained by normal variations.
        4.  **Significant Surface Difference (Lowest Priority):** Only considered if no object/text contradiction exists; suggests items might be different if surfaces drastically differ.
</Context>

<Anomaly_Types_To_Detect>
1.  **Anomaly Type 1: Missing Critical Information:** (BN or Date missing across all images).
2.  **Anomaly Type 2: Item Inconsistency:** (Evidence of >1 distinct core item based on Primary Panel text or visual contradiction).
</Anomaly_Types_To_Detect>

<Steps>
1.  **Initial Image Analysis & Comprehensive OCR:**
    a.  Examine *each* image. Identify primary object(s) and immediate surface.
    b.  **CRITICAL:** For each image, classify the visible panel type(s) as reliably as possible: **Primary** or **Secondary**. This classification dictates how the image's text will be used in Step 3.
    c.  Perform comprehensive OCR on *all* visible text in *each* image, capturing text strings and their approximate locations/bounding boxes. Extract potential candidates for: BN, Dates, Product Names, Strengths/Dosages, Manufacturer Names, Other secondary text.
    d.  Store extracted text (with location info) associated with its image and its **classified panel type (Primary/Secondary)**.

2.  **Analyze for Anomaly Type 1 (Missing Critical Information):**
    a.  Gather *all* potential BN candidates extracted from *all* images (Step 1c).
    b.  Gather *all* potential Date (ED/MD) candidates extracted from *all* images (Step 1c).
    c.  **Determine Global Found Status:**
        i.  Apply flexible identification rules (Context 3) to the gathered candidates.
        ii. Set `found_bn = true` if *at least one* valid BN was identified across all images, otherwise `found_bn = false`.
        iii. Set `found_date = true` if *at least one* valid Relevant Date (ED or MD) was identified across all images, otherwise `found_date = false`.
        iv. **Crucial Check:** Ensure `found_bn` and `found_date` accurately reflect whether *any* valid information of that type was found *anywhere* in the image set.
    d.  **Determine Anomaly Status:** Evaluate the condition `internal_anomaly_type1 = !(found_bn || found_date)`.
    e.  If `internal_anomaly_type1` is true, note reason: `internal_reason_type1 = "Neither Batch Number nor Expiration/Manufacturing Date could be identified in any of the images."`.

3.  **Analyze for Anomaly Type 2 (Item Inconsistency - Strict Panel-Based Logic with Semantic Name Check):**
    a.  **Isolate and Validate Primary Panel Information:**
        i.  Create a list `validated_primary_panels` containing information ONLY from images classified as **Primary** in Step 1b. Each entry should ideally contain the validated name, strength, and source image identifier.
        ii. If `validated_primary_panels` is empty (no primary panels identified), set `reference_established = false`. Proceed directly to Step 3.c focusing ONLY on visual checks.
        iii. For each potential primary panel view identified in Step 1b:
            - Extract its potential Product Name candidate strings (with locations), Strength/Dosage(s), and Manufacturer Name from Step 1d based on the OCR results for this specific primary panel view.
            - **Validate & Combine/Select Product Name:**
                1.  **Filter potential Product Name candidates:**
                    - **MANDATORY MANUFACTURER EXCLUSION:** Identify the Manufacturer Name (e.g., 'KALBE', 'OGBdexa logo text', 'Phizer', 'OTTO') on this Primary Panel. This Manufacturer Name **MUST BE COMPLETELY EXCLUDED** from `valid_name_candidates`. It **CANNOT** be considered part of the product name.
                    - **MANDATORY CODE/DATE/BN/PRICE EXCLUSION:** Additionally, strings extracted from this Primary Panel that *strongly match* common formats for Batch Numbers (e.g., alphanumeric codes often preceded by "Batch No.", "Lot No.", "LOT", "BN", "NO BATCH"), Dates (e.g., 'EXP', 'MFD', MM YY, MON YYYY), Registration Numbers (e.g., 'ABC'+numbers), or Price Tags (e.g., 'HET', 'Rp.') **MUST BE COMPLETELY EXCLUDED** from `valid_name_candidates`. They **CANNOT** be considered product names.
                    - **Derive Final Candidates:** Let the result be the `valid_name_candidates` list after applying *both* mandatory exclusions above.
                2.  **Check for Composite Name:** `IF count(valid_name_candidates) > 1:`
                    - Analyze the spatial relationship (location/proximity) of these `valid_name_candidates` on the original image.
                    - `IF` candidates are visually adjacent and appear to form a single composite name (e.g., Brand above/beside Generic like "LANCID" and "Lansoprazole"):
                        - Combine them into a single `composite_name` string (e.g., "LANCID Lansoprazole").
                        - Set `panel_info.validated_name = composite_name`.
                    - `ELSE` (valid candidates exist but are not adjacent/related): Select the most prominent or contextually primary candidate *that represents the medicine itself* as `panel_info.validated_name`. (Handle potential ambiguity if necessary).
                3.  `ELSE IF count(valid_name_candidates) == 1:`
                    - Set `panel_info.validated_name = the single valid candidate`.
                4.  `ELSE:`
                    - Set `panel_info.validated_name = null`.
            - **Validate Strength/Dosage:** Check if plausible based on context and common formats. Store as `panel_info.validated_strength` (or null).
            - If a valid name or strength was found, add this `panel_info` (containing validated name, strength, and source identifier) to the `validated_primary_panels` list.

    b.  **Establish Reference Core Identity (STRICTLY from Primary Panel Data ONLY):**
        **CRITICAL CONTEXT:** For this entire step (Step 3b), you **MUST COMPLETELY IGNORE** any information previously identified or extracted from images classified as **Secondary Panels** (e.g., Batch Numbers, Dates, HET prices, detailed addresses found on sides/top/bottom/back). The *sole purpose* here is to define the reference based *exclusively* on the best representation found within the `validated_primary_panels` list. Do not let secondary information influence your choice or understanding of the reference identity.
        i.  **Select Reference Candidate STRICTLY from Primary Panels:** If `validated_primary_panels` (derived *only* from images classified as Primary) is not empty, select the `panel_info` from this list *only*. Base the selection on the following **prioritized criteria** applied *only* to the primary panel candidates:
            *   **Mandatory Exclusion of Manufacturer & Codes (Re-check):** First, re-confirm that the `validated_name` in any considered `panel_info` is *not* the Manufacturer Name and does *not* resemble a BN/Date/Code/Price format (as per Step 3a filtering). Any such candidates are ineligible.
            *   **Priority 1 (Semantic Drug Name):** Among *eligible* candidates, choose the `panel_info` whose `validated_name` most clearly and completely represents a **drug name** (Brand, Generic, or composite like 'VANCOMYCIN HCI'). This takes precedence over visual prominence alone.
            *   **Priority 2: Association with Strength:** Prefer eligible drug name candidates logically associated with a `validated_strength`.
            *   **Priority 3: Typical Placement:** Consider eligible drug names in typical branding locations.
        ii. **Attempt to Define Reference:**
            *   Let `selected_panel_info` be the candidate chosen based on the criteria above (if any).
            *   `IF` a suitable `selected_panel_info` representing a drug name was found:
                - Tentatively define `potential_Reference_Product_Name` using its `validated_name` (apply composite rule).
                - Tentatively define `potential_Reference_Strength` from `validated_strength`.
                - Tentatively define `potential_Reference_Packaging_Type`.
                - Mark `potential_reference_found = true`.
            *   `ELSE` (No suitable primary panel info found):
                - Mark `potential_reference_found = false`.
        iii. **Verify Reference Origin and Validity:**
            *   `IF potential_reference_found is true:`
                - **Check 1 (Origin):** Was the `potential_Reference_Product_Name` derived *exclusively* from information validated from a **Primary Panel** in Step 3a? (This should always be true if Step 3b.i was followed correctly, but verify).
                - **Check 2 (Semantic Validity):** Does the `potential_Reference_Product_Name` clearly represent a **drug name** and *not* a manufacturer, BN, date, code, price, or other non-drug identifier?
                - **Decision:**
                    - `IF` Check 1 is YES and Check 2 is YES:
                        - Finalize the reference:
                            - `Reference_Product_Name = potential_Reference_Product_Name`
                            - `Reference_Strength = potential_Reference_Strength`
                            - `Reference_Packaging_Type = potential_Reference_Packaging_Type`
                            - `reference_established = true`
                    - `ELSE` (Either check failed - Origin wrong or Semantically invalid):
                        - **Abort Reference:** Set `reference_established = false`. Log internal reason for failure (e.g., "Reference candidate failed semantic check" or "Reference candidate origin suspect").
            *   `ELSE` (`potential_reference_found` was already false):
                - Set `reference_established = false`.

        iv. If `validated_primary_panels` was initially empty, ensure `reference_established = false`.

    c.  **Verify Consistency Across All Images (Strict Panel Logic with Explicit Brand Tolerance):**
        i.  Initialize `anomaly_found_type2 = false`, `conflict_reason = ""`.

        ii. **Global Packaging Type Consistency Check (High Priority):**
            - Examine the visually identifiable packaging types across *all* images (e.g., box, bottle/vial, blister). Attempt to classify the fundamental form factor shown in each image.
            - Determine the set of distinct fundamental packaging types identified across the image set.
            - **IF** more than one distinct fundamental packaging type (e.g., both 'box' AND 'bottle/vial') is clearly identified:
                - Set `anomaly_found_type2 = true`.
                - Set `conflict_reason = "visual contradiction: Multiple distinct packaging types found across images (e.g., boxes and bottles/vials)."`.
                - **Immediately proceed to Step 4.** (Anomaly Type 2 confirmed, no further inconsistency checks needed).
            - **ELSE IF** only one consistent fundamental packaging type is identified across all images where type is clear:
                - Store this as `consistent_packaging_type` for potential later reference.
                - Proceed to the next step (`iii`).
            - **ELSE** (Type unclear in some/all images, or only one image provided):
                - Proceed to the next step (`iii`). The per-image checks might still catch inconsistencies if a reference is established.

        iii. **Iterative Per-Image Consistency Verification (If no global type conflict found):**
            - `IF anomaly_found_type2 is true: Proceed directly to Step 4.` // Double check in case logic flow needs it
            - Iterate through *each image* in the input set:
                - Let `current_image_panel_type` be the classification from Step 1b (Primary or Secondary).
                - Let `current_image_visual_type` be the visually identified packaging type for this image.

                - **Packaging Type Check (Per-Image vs Reference/Consistency - Lower priority than global check):**
                    - `IF reference_established is true AND Reference_Packaging_Type is defined:`
                        - Compare `current_image_visual_type` with `Reference_Packaging_Type`.
                        - `IF` they fundamentally contradict (and this wasn't caught by the global check ii):
                            - Set `anomaly_found_type2 = true`, `conflict_reason = "visual contradiction: Packaging type mismatch ('[current_image_visual_type]') vs reference ('[Reference_Packaging_Type]')."`, break. // Exit loop
                    - `ELSE IF consistent_packaging_type was defined in step ii:`
                        - Compare `current_image_visual_type` with `consistent_packaging_type`.
                        - `IF` they fundamentally contradict:
                            - Set `anomaly_found_type2 = true`, `conflict_reason = "visual contradiction: Packaging type mismatch ('[current_image_visual_type]') vs consistent type found ('[consistent_packaging_type]')."`, break. // Exit loop
                - **Textual Conflict Check (ONLY if Current Image is PRIMARY - Applying Semantic Name Comparison with Explicit Brand Tolerance):**
                    - `// CRITICAL: This check ONLY runs if the current image view is classified as a PRIMARY panel.`
                    - `IF reference_established is true AND current_image_panel_type is Primary:`
                        - Find the corresponding `validated_panel_info` for the current image within the `validated_primary_panels` list (this ensures we use data validated in Step 3a).
                        - `IF validated_panel_info exists:` // Ensure we found the validated info for this primary panel
                            - `// Perform SEMANTIC COMPARISON for Product Name (Primary vs Reference Primary)`
                            - `IF validated_panel_info.validated_name is not null AND Reference_Product_Name is not null:`
                                # ... (Keep the detailed semantic comparison logic as is) ...
                                - `ELSE` (Core substances are fundamentally different...):
                                    - **Flag Anomaly:** Set `anomaly_found_type2 = true`, `conflict_reason = "conflicting textual evidence (Semantic): Found Primary Panel Product Name '[validated_panel_info.validated_name]' which refers to a different drug substance than the reference '[Reference_Product_Name]'."`, break. // Exit loop

                            - `// STRENGTH COMPARISON (Primary vs Reference Primary - check only if names are semantically identical OR different core substances anomaly wasn't flagged)`
                            - `IF !anomaly_found_type2 AND validated_panel_info.validated_strength is not null AND Reference_Strength is not null:`
                                # ... (Keep strength comparison logic as is) ...
                                - `IF` they represent **different dosage values**:
                                    Set anomaly: `anomaly_found_type2 = true`, `conflict_reason = "conflicting textual evidence: Found Primary Panel Strength '[validated_panel_info.validated_strength]' which differs from the reference strength '[Reference_Strength]'."`, break. // Exit loop

                - `// If anomaly already found by type or text, skip lower priority checks for this image`
                - `IF anomaly_found_type2: continue; // Go to next image`

            - `// If anomaly already found by type or text, skip lower priority checks for this image`
            - `IF anomaly_found_type2: continue; // Go to next image`

            - **Visual Object Feature Check (All Images):**
                - `IF !anomaly_found_type2:` // Only run if no higher priority anomaly (like packaging type mismatch) found yet
                    - **Compare the overall visual characteristics** of the packaging object in `current_image` against the reference object(s), **paying close attention to core branding elements and overall design, regardless of whether the current view is Primary or Secondary.**
                    - Look specifically for **strong indicators of a different product**, such as:
                        - **Clearly Different Primary Brand Name/Logo:** Is a prominent brand name or logo visible on the `current_image` (even if on a secondary panel) that **fundamentally contradicts** the `Reference_Product_Name` or associated branding (e.g., detecting "DrugX Name" logo/text when the reference is "DrugY Name")?
                        - **Markedly Different Overall Design:** Does the color scheme, graphic layout, and typography style visible on `current_image` present a **stark contrast** to the reference object's design, suggesting a completely different product line?
                        - **Contradictory Manufacturer (if visible and reference manufacturer known):** Does a clearly visible manufacturer on `current_image` differ from the manufacturer associated with the reference primary panel (if identifiable)? (Lower priority indicator).
                    - **Decision:** `IF` one or more of these strong visual contradictions are present, indicating with high confidence that `current_image` shows a **fundamentally different core product** than the reference:
                        - Set `anomaly_found_type2 = true`.
                        - Set `conflict_reason = "visual contradiction: Core branding/design features (e.g., different brand name/logo like '[Detected Feature e.g., DrugX Name]', colors, layout) strongly suggest a different item compared to the reference '[Reference_Product_Name]'."`.
                        - `// Consider adding 'break' here if this visual confirmation is deemed high-confidence enough to stop iteration.`
            - **Surface Check (All Images, Lowest Priority):** If `anomaly_found_type2` is still false, does the surface texture, significant damage (beyond normal wear), or other major surface characteristics visible on `current_image` strongly suggest it's a different physical item compared to the reference images? If yes, set potential anomaly: `anomaly_found_type2 = true`, `conflict_reason = "surface difference: Item surfaces suggest potentially different items."`

    d.  **Determine Status & Reason for Anomaly Type 2:**
        - `internal_anomaly_type2 = anomaly_found_type2`.
        - If `internal_anomaly_type2` is true, `internal_reason_type2 = "Item inconsistency detected due to " + conflict_reason`. Use the reason from the highest priority conflict found during the iteration (Semantic Name Difference [Different Substance] > Strength Difference > Packaging Type > Visual Feature > Surface). Note that a difference only in brand name should NOT result in `internal_anomaly_type2 = true`.
        
4.  **Synthesize Final Output:**
    a.  `final_is_anomaly = internal_anomaly_type1 || internal_anomaly_type2`.
    b.  Construct `final_anomaly_reason`:
        *   If `false`: "No anomalies detected."
        *   If `true`:
            *   Only Type 1: `internal_reason_type1`.
            *   Only Type 2: `internal_reason_type2`.
            *   Both Type 1 and Type 2: "Multiple anomalies detected: 1. [internal_reason_type1]. 2. [internal_reason_type2]."

5.  **Format Output:** Strict JSON as specified.
</Steps>

<FewShotExamples>
{fewshot_examples}
</FewShotExamples>
"""

class AnomalyAgentService(BaseAgent):
    """Simplified Anomaly Detection Agent."""
    
    def __init__(self, llm, **kwargs):
        
        super().__init__(
            llm=llm,
            prompt_template=anomaly_prompt,  
            output_model=AnomalySchema,              
            use_structured_output=True,
            # agent_name="AnomalyAgentService",
            # gcs_client=gcs_client,
            # gcs_bucket=gcs_bucket,
            **kwargs
        )

    async def __call__(self, state) -> Dict:
        """Process anomaly detection."""
        try:
            fewshot_examples = """
            <Example 1>
            <Input_Description 1>
            *   **Image 1:** Menampilkan panel depan (Primary Panel) sebuah kotak obat. Terdapat logo produsen "MfgLogo", teks besar "BrandX DrugName", dan di bawahnya "Tablet 50 mg". Ada juga detail visual lainnya seperti warna desain spesifik.
            *   **Image 2:** Menampilkan panel samping (Secondary Panel) dari beberapa kotak obat yang identik. Setiap panel menunjukkan teks "BN LOT12345", "MFD 01 2024", "EXP 12 2026", dan "HET IDR 75.000".
            </Input_Description 1>
            <Output 1>
            ```json
            {{
            "is_anomaly": false,
            "anomaly_reason": "No anomalies detected."
            }}
            </Output 1>
            <Explanation>Analysis: Image 1 is the Primary Panel establishing the core identity as 'BrandX DrugName 50 mg' (after validation). Image 2 is the Secondary Panel providing the BN ('LOT12345') and Dates (MFD/EXP). The BN/Date/HET information from the Secondary Panel is *not* used to establish the core drug reference identity and is *not* compared against 'BrandX DrugName' to check for substance inconsistency. Since BN and Dates were found on the secondary panel, and only one core drug identity is visible from the primary panel, there is no anomaly.</Explanation>
            </Example 1>

            <Example 2>
            <Input_Description 2>
            * **Image 1:** Menampilkan panel depan (Primary Panel) dari beberapa kotak obat identik. Teks utama adalah "GenericDrugZ HCl Powder 250 mg". Di bagian bawah tertera nama produsen "PharmaCorp". Terdapat juga logo generik.
            * **Image 2:** Menampilkan panel atas (Secondary Panel) dari banyak kotak obat yang identik. Setiap panel menunjukkan teks "Batch: A9876B", "Use By: OCT 2025", "Reg No: DKL1234567890A1", dan logo "PharmaCorp".
            </Input_Description 2>
            <Output 2>
            ```json
            {{
            "is_anomaly": false,
            "anomaly_reason": "No anomalies detected."
            }}
            ```
            </Output 2>
            <Explanation>Analysis: Image 1 is the Primary Panel establishing the core identity as 'GenericDrugZ HCl 250 mg'. Image 2 is the Secondary Panel providing the Batch Number ('A9876B') and Date ('OCT 2025'). The Batch/Date/Reg No information from the Secondary Panel is *not* used to establish the core drug reference identity and is *not* compared against 'GenericDrugZ HCl' to check for substance inconsistency. Since Batch Number and Date were found on the secondary panel, and only one core drug identity is visible from the primary panel, there is no anomaly.</Explanation>
            </Example 2>
            """

            self.rebind_prompt_variable(
                fewshot_examples = fewshot_examples
            )

            files = state["images"]
            po_number = state.get("po_number")
            
            content_parts, url_list = await prepare_images_for_llm(
                files,
                use_compression=True
            )

            # result = await self.run_llm_with_images(content_parts)
            
            raw, parsed = await self.arun_chain(content_parts)
            
            if parsed.is_anomaly:
                print("⚠️ Anomaly detected!")

            ai_msg = parsed.__dict__
            output = {
                "anomaly": ai_msg,
                "url": url_list
            }
            
            print("anomaly_output: ", output)    
            return output
            
        except Exception as e:
            print("Error terjadi pada Anomaly Agent")
            raise e
        
# class AnomalyAgentService(BaseAgent):
#     """An agent responsible for Anomaly Detection."""
    
#     def __init__(self, llm, **kwargs):
#         # Setup GCS client and bucket
#         credentials = google_credential()
#         from google.cloud import storage
#         gcs_client = storage.Client(project=env.project_name, credentials=credentials)
#         gcs_bucket = gcs_client.get_bucket(env.bucket_name)
        
#         super().__init__(
#             llm=llm,
#             prompt_template=test_anomaly_prompt_v4,
#             output_model=AnomalySchema,
#             use_structured_output=True,
#             enable_caching=True,
#             enable_timing=True,
#             gcs_client=gcs_client,
#             gcs_bucket=gcs_bucket,
#             **kwargs
#         )

#     async def __call__(self, state: State_ax):
#         try:
#             files = state["images"]
#             po_number = state["po_number"]
            
#             # Upload images using BaseAgent method
#             content_parts, url_list = await self.upload_images(files)
            
#             # Create input for the chain
#             human_message = HumanMessage(content=content_parts)
            
#             parsed = await self.arun_chain(state={"input": [human_message]})

#             output = {
#                 "anomaly": [parsed],
#                 "url": url_list
#             }
            
#             if output.get("is_anomaly", True):
#                 print(f"⚠️ Anomaly detected!")

#             print("State: ")
#             print(output.keys())   
#             return output
            
#         except Exception as e:
#             # Clean up on error using BaseAgent method
#             if 'url_list' in locals():
#                 self.cleanup_images(url_list)
#             raise response_error(str(e))

# class AnomalyAgentService(BaseAgent):
#     """An agent responsible for Anomaly Detection"""
#     def __init__(self, llm, **kwargs):
#       super().__init__(
#         llm=llm,
#         prompt_template=test_anomaly_prompt_v4,
#         output_model=AnomalySchema,
#         **kwargs
#       )
#       self.llm = llm
      
#       credentials = google_credential()
#       self.client = storage.Client(project=env.project_name, credentials=credentials)
#       self.bucket_access = self.client.get_bucket(env.bucket_name)

#       # Add caching for models
#       self.llm_cache = {}
#       self.llmc_cache = {}



#     async def _upload_image(self, blob, file_bytes):
#       try:
#         """Helper method to upload an image asynchronously"""
#         # Make this non-blocking
#         loop = asyncio.get_event_loop()
#         await loop.run_in_executor(
#             None, 
#             lambda: blob.upload_from_string(file_bytes, content_type="image/png")
#         )
#         await loop.run_in_executor(None, blob.make_public)
#       except Exception as e:
#         raise response_error(str(e))
    
#     async def __call__(self, state:State_ax):
#       try:
#         t0 = time.perf_counter()
#         # Process batches of images instead of one by one
#         files = state["images"]
#         po_number = state["po_number"]
#         # print("po_number:",po_number)

#         # Parallelize image uploads
#         content_parts = []
#         url_list = []
#         upload_tasks = []
        
#         # Prepare all upload tasks
#         for file_bytes in files:
#           unique_id = uuid.uuid4().hex
#           file_upload_name = f"image_{unique_id}.png"
#           blob = self.bucket_access.blob(file_upload_name)
#           task = asyncio.create_task(self._upload_image(blob, file_bytes))
#           upload_tasks.append((task, blob, file_upload_name))
        
#         # Wait for all uploads to complete
#         for task, blob, _ in upload_tasks:
#           await task
#           url = blob.public_url
#           content_parts.append({
#               "type": "image_url",
#               "image_url": {"url": url},
#           })
#           url_list.append(url)

#         content_parts.append({
#               "type": "text",
#               "text": " "
#           })

#         # Use a cache key based on the image URLs
#         cache_key = hash(tuple(url_list))
#         if cache_key in self.llm_cache:
#           return self.llm_cache[cache_key]
          
#         human_message = HumanMessage(content=content_parts)
#         final_prompt = ChatPromptTemplate.from_messages(
#             [("system", test_anomaly_prompt_v4), MessagesPlaceholder("input")]
#         )

#         parser_anomaly = JsonOutputParser(pydantic_object=AnomalySchema)
#         chain = final_prompt | self.llm | parser_anomaly
#         # chain = final_prompt | self.llmg | parser_anomaly
#         output = {"anomaly": [await chain.ainvoke({"input": [human_message],
#                                                 "anomaly_format_instructions": parser_anomaly.get_format_instructions()
#                                                 })],
#                   "url": url_list
#                 }
                
#         print("anomaly_output:",output)
#         if output["anomaly"][0]["is_anomaly"] == True:
#           print(f"⚠️ Anomaly detected!")
#         # Cache the result
#         self.llm_cache[cache_key] = output

#         print(f"Anomaly detection took {time.perf_counter() - t0:.2f} seconds \n")
#       #   if True:
#       #      print(f"-----PEMERIKSAAN ANOMALY SELESAI-----")
#       #      return None 
#         return output
      
#       except Exception as e:
#         if state["url"]: # Hanya bersihkan jika url_list tidak kosong
#             #print(f"DEBUG: Cleaning up {len(state["url"])} images due to error.")
#             self._cleanup_images(state["url"])
#         else:
#             print("DEBUG: No images to clean up (url_list was empty).")
#             pass 
#         # Clean up on error
#         #self._cleanup_images(url_list)
#         raise response_error(str(e))