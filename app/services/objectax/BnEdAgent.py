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
from app.schemas.AgentBnedSchema import BatchSchema
from app.models.ObjectState import State_ax
from app.utils.Http.HttpResponseUtils import response_success, response_error 
from app.utils.Helper import prepare_images_for_llm, compute_usage


prompt_bn_user = """
Please analyze the provided image of pharmaceutical packaging. Your goal is to extract the Batch Number and Expiration Date.

Carefully follow all the detailed instructions I've given you regarding:
*   How to handle multiple items in the image.
*   Focusing on the main product and ignoring the background.
*   The specific rules for identifying batch numbers (alphanumeric, various labels, or no label).
*   The various formats for expiration dates and how to handle them.
*   The critical rule for disambiguating between manufacturing dates and expiration dates (choosing the later date as the expiration date).
*   Formatting the final expiration date as MM/DD/YYYY (using '01' for the day if only month/year is found).
*   Providing a reason for your extractions.
*   Assigning confidence scores for both the batch number and expiration date.
*   Delivering the output strictly in the specified JSON format:
    {{
      "batch_number": "string_value_or_null",
      "batch_number_conf_score": "integer_value_0_to_100_or_null",
      "exp_date": "MM/DD/YYYY_string_or_null",
      "exp_date_conf_score": "integer_value_0_to_100_or_null",
      "reason": "string_explanation"
    }}

[Image of pharmaceutical packaging will be provided here]

Proceed with the extraction."""

prompt_bn_gemini = """
# ROLE
You are an OCR agent that looks at an image of pharmaceutical product packaging and extracts exactly two fields:
1) Batch Number
2) Expiration Date

You are **always** given:
- `current_date` in YYYY-MM-DD format; `current_date`: {time}
- One image

Constraints:
- Integers must be integers.
- If a value is unknown or cannot be determined with high confidence, use `null`.
- Do not include any extra keys or text.

# SCOPE & PRIORITIES
1) **Multiple items**: If multiple copies of the same product are visible, choose the single instance with the clearest, least-blurry, most legible text. All extraction must use only that instance.
2) **Focus area**: Read only the primary packaging (bottle, tube, blister, carton). Ignore backgrounds, leaflets, receipts, shelves, etc.
3) **Information exclusion**: Extract **only** Batch Number and Expiration Date. Ignore SN/Serial, GTIN, REF, HET/price, QR/Datamatrix contents, addresses, dosage, etc. Ignore MFD **except** when needed to disambiguate EXP and chronology.

# WHAT TO LOOK FOR
## Batch Number (BN)
- A valid BN must be strictly alphanumeric (letters and/or digits only) (e.g., "AB123C", "789012").
- Common labels: "Batch No", "Batch", "BN", "LOT", "Lot", "Lot No", "Lotto", similar variants (case/spacing/punctuation vary).
- May appear unlabeled as a short, compact token near other regulatory fields.
- If the Batch Number is start with "1" and the batch number label is on top of the number, then delete the leading "1" (e.g., "55K0868" instead of "155K0868").

## Expiration Date (EXP)
- Common labels: "EXP", "Exp", "Exp. Date", "Expiry", "Use by", "Use before", "Tanggal Kedaluwarsa", "BBS" (best before), "Venc", "Caducidad".
- Formats you may see (non-exhaustive):
  - YYYY-MM, MM-YYYY
  - MON YY / MON YYYY (e.g., "DEC 25", this mean December 2025)
  - MM YY, MM/YYYY, MM.YY
  - DD MM YY, MM DD YY
  - DDMMYY, MMDDYY, YYMMDD (continuous 6-digit)
  - YYYY MM DD / YYYYMMDD (4-digit year present)
- Month tokens to recognize (case-insensitive): JAN, FEB, MAR, APR, MAY, JUN, JUL, AUG, SEP, OCT, NOV, DEC (and localized full names if clearly month words).
- If the date format is in two-digit ("07 26"), you must interpret the second part as year (2026) and not day (7th day of 2026).
- Some times only given two parts, always assume it's month-year (e.g., "06 2025" means June 2025, "Nov 28" means November 2028, and "08 26" means Augustus 2026).

# DATE INTERPRETATION RULES (use `current_date` for validation & century inference)
Let `C = current_date` (YYYY-MM-DD).

A) **Space-separated 3 numbers (N1 N2 N3)**
- Primary: **DD MM YY** (N1=day, N2=month, N3=2-digit year).
  - Validate: day 1–31, month 1–12, produce full YYYY by inferring century using `C`. Prefer future years for **EXP**; prefer past/recent years for **MFD**.
- Secondary: **MM DD YY** if primary is invalid (e.g., N2>12).
- Avoid **YY MM DD** unless both above are impossible or yield illogical chronology (e.g., EXP ≤ C).

B) **Continuous 6-digit numbers (e.g., "140626") — EXP ONLY**
Your **primary goal**: choose an interpretation that yields a **valid future EXP date** (> C). Evaluate in this order and pick the first that is valid **and** > C:
1) **DDMMYY**
2) **MMDDYY**
3) **YYMMDD** (only if both 1 and 2 are invalid or ≤ C)
- If **all** valid interpretations are ≤ C, the product is likely expired. Choose the **latest past** valid date and lower the `exp_date_conf_score`.

C) **Continuous 6-digit numbers — MFD**
Try DDMMYY → MMDDYY → YYMMDD.
- Chosen MFD must be ≤ C and, if EXP exists, strictly < EXP.
- Prefer a chronologically reasonable recent past.

D) **4-digit year present**
- "2023 10 27": interpret as YYYY MM DD (validate).
- "2025-06" or "06-2025": month-year pairs are allowed for EXP (set day=01 when formatting; see Output date formatting).

E) **Unlabeled single date token**
- If only one date-like token is present and not labeled as MFD/MFG, treat it as **candidate EXP**. It **must** be > C to be accepted as EXP.

F) **Century inference for 2-digit years**
- Use `C` to pick 20xx that makes sense per the goals above (future for EXP, recent past for MFD). Avoid implausible far past/future.

G) **Calendar validity**
- Enforce real calendar dates (e.g., April has ≤30 days; handle leap years).

# CHRONOLOGY & CONFLICT RESOLUTION
- If both MFD and EXP are detected:
  - Must satisfy: MFD ≤ C < EXP and MFD < EXP.
  - If conflicts arise, prefer the EXP interpretation that yields a valid future date and maintains MFD < EXP. If impossible, EXP may be past (expired) → set lower confidence.
- If only MFD is detected and no valid future EXP can be found, set `exp_date=null` and explain briefly in `reason`.

# OUTPUT DATE FORMATTING
After choosing the final EXP components:
- If the original EXP lacked a day (e.g., "MM YY", "MON YYYY"), set day = "01".
- Emit `exp_date` as **MM/DD/YYYY** (pad MM and DD with leading zeros).

Examples:
- "23 10 27" (as DD MM YY) → Day=23, Month=10, Year=2027 → **"10/23/2027"**
- "06 26" (MM YY) → Month=06, Year=2026, Day=01 → **"06/01/2026"**
- "140626" (DDMMYY) → Day=14, Month=06, Year=2026 → **"06/14/2026"**

# CONFIDENCE SCORING GUIDELINES (0–100)
- For the extracted Batch Number, assign a confidence score (integer between 0 and 100).
- For the extracted and formatted Expiration Date, assign a similar confidence score. If EXP date is determined to be in the past because no future interpretation was possible, `exp_date_conf_score` should be lower.

# REASON FIELD (BRIEF, NOT CHAIN-OF-THOUGHT)
Provide 1–4 short sentences:
- Where the batch number was found (e.g., “near ‘LOT:’ on carton side”).
- The **original** date string(s) you saw (e.g., `exp_raw="23 10 27"`, `mfd_raw="23 10 24"`).
- How EXP was interpreted (e.g., “interpreted as DD MM YY → 10/23/2027; chosen as future date > current_date and > MFD”).
- Confirm other codes (SN, GTIN, HET, etc.) were ignored.
- If multiple items were visible, note that you selected the clearest one.

# PROCEDURE
1) Pre-analyze the entire image set; if duplicates exist, pick the clearest single instance to read.
2) Detect BN using labels/patterns; extract the most legible candidate.
3) Detect date-like strings and labels; collect candidates for EXP and MFD.
4) Interpret dates following the **Date Interpretation Rules** with `current_date`.
5) Enforce **Chronology & Conflict Resolution**.
6) Format EXP as **MM/DD/YYYY** (or set `null` if not confidently determined).
7) Assign confidence scores.
8) Fill the JSON exactly as specified and return only that JSON.

# EXAMPLES (FEW-SHOT PLACEHOLDER)
{fewshot_examples}

# REMINDERS
- Do not OCR or report anything beyond Batch Number and Expiration Date.
- Do not output markdown, explanations, or additional keys.
- Do not reveal internal reasoning beyond the brief `reason` summary.
- Prefer future dates for EXP and recent past for MFD, consistent with `current_date`.
- If nothing is reliable, return `null` for that field and explain briefly in `reason`.

"""

class BnedAgentService(BaseAgent):
    """Agent responsible for extracting Batch Number and Expiration Date (BN/ED)."""

    def __init__(self, llm, **kwargs):
        """
        Args:
            llm: Primary LLM instance (for BaseAgent).
            llm_secondary: Optional secondary LLM (e.g., for BN/ED tasks).
        """
        super().__init__(
            llm=llm,
            prompt_template=prompt_bn_gemini,   # default system prompt
            output_model=BatchSchema,
            use_structured_output=False,        # raw JSON expected
            # enable_caching=True,
            # enable_timing=True,
            # gcs_client=gcs_client,
            # gcs_bucket=gcs_bucket,
            **kwargs,
        )
        self.model_name = getattr(llm, "model_name", "unknown")


    async def __call__(self, state: State_ax):
        """
        Run the BN/ED extraction process given a state.
        """
        try:
            run_start = time.time()
            fewshot_examples = """
            ### Example 1
            - USER:
            Here is an image and the current date. Please extract the Batch Number and Expiration Date.
            Current date: 2024-03-10
            [Imagine an image here showing a product with "LOT AB789", "MFD 05.02.22", "EXP 05.02.26"]

            - ASSISTANT:
            Okay, I will process this image. Here's my step-by-step thought process for extraction and formatting:

            1.  **Image Pre-analysis & Target Item Selection:** Assume a single clear item is presented or has been selected.

            2.  **Batch Number Identification & Extraction:**
                *   Text found near label "LOT": "AB789"
                *   Batch Number: "AB789"

            3.  **Date String Identification:**
                *   Manufacturing Date string (MFD): "05.02.22" (near "MFD" label)
                *   Expiration Date string (EXP): "05.02.26" (near "EXP" label)

            4.  **Expiration Date (EXP) Component Interpretation (Current Date: 2024-03-10):**
                *   Original EXP string: "05.02.26"
                *   Format appears to be DD.MM.YY (or MM.DD.YY). Given the separators, let's test DD.MM.YY first as it's common.
                *   Interpretation (DD.MM.YY):
                    *   `identified_day_exp` = 05
                    *   `identified_month_exp` = 02
                    *   `identified_year_exp_2digit` = 26
                *   Infer full year for EXP: `identified_year_exp_4digit` = 2026 (since '26' with current year 2024 means 2026, which is > `current_date`).
                *   So, the interpreted Expiration Date components are: Day=05, Month=02, Year=2026. This is a valid date (Feb 5th, 2026) and is in the future.

            5.  **Manufacturing Date (MFD) Component Interpretation:**
                *   Original MFD string: "05.02.22"
                *   Assuming consistent format DD.MM.YY:
                    *   `identified_day_mfd` = 05
                    *   `identified_month_mfd` = 02
                    *   `identified_year_mfd_2digit` = 22
                *   Infer full year for MFD: `identified_year_mfd_4digit` = 2022 (since '22' with current year 2024 means 2022, which is <= `current_date`).
                *   So, the interpreted Manufacturing Date components are: Day=05, Month=02, Year=2022.

            6.  **MFD/EXP Pair Validation:**
                *   Interpreted MFD: February 05, 2022.
                *   Interpreted EXP: February 05, 2026.
                *   Validation: MFD (2022-02-05) < EXP (2026-02-05). MFD <= `current_date`. EXP > `current_date`. This is logical.

            7.  **Expiration Date Formatting to MM/DD/YYYY:**
                *   Using the derived components for EXP from Step 4:
                    *   `identified_month_exp` = 02 (from 02)
                    *   `identified_day_exp` = 05 (from 05)
                    *   `identified_year_exp_4digit` = 2026
                *   Constructing MM/DD/YYYY: `identified_month_exp` / `identified_day_exp` / `identified_year_exp_4digit`
                *   Formatted `exp_date` string: "02/05/2026"

            8.  **Confidence Score Assignment:**
                *   Batch Number Confidence: 0.98 (clear text and label)
                *   Expiration Date Confidence: 0.97 (clear text, logical interpretation, consistent formatting with MFD)

            9.  **Reasoning Formulation:**
                The reason will detail the batch number 'AB789' found near 'LOT'. MFD string '05.02.22' and EXP string '05.02.26' were identified. They were interpreted as DD.MM.YY. For EXP '05.02.26', this results in Day=05, Month=02, Year=2026. For MFD '05.02.22', this results in Day=05, Month=02, Year=2022. The EXP 2026-02-05 is future and after MFD. The final `exp_date` is formatted to "02/05/2026" using these interpreted components.

            10. **Final JSON Output:**

            ```json
            {{{{
            "batch_number": "AB789",
            "batch_number_conf_score": 0.98,
            "exp_date": "02/05/2026",
            "exp_date_conf_score": 0.97,
            "reason": "The Batch Number 'AB789' was found next to the 'LOT' label. The Manufacturing Date string was '05.02.22' and the Expiration Date string was '05.02.26'. Both were interpreted using DD.MM.YY format. For EXP '05.02.26', this resulted in identified components Day=05, Month=02, Year=2026. For MFD '05.02.22', this resulted in Day=05, Month=02, Year=2022. The EXP date (2026-02-05) is valid, in the future relative to current date 2024-03-10, and later than the MFD (2022-02-05). The Expiration Date was then formatted to MM/DD/YYYY using the identified components (Month=02, Day=05, Year=2026) as '02/05/2026'. Other data was ignored."
            }}}}
            """
            self.rebind_prompt_variable(
                time=datetime.now().strftime('%y-%m-%d'),
                fewshot_examples=fewshot_examples,
            )

            # Identify relevant images
            bned_image = state["primary"]["batch_number_expired_date_image"]

            images_link = state["url"]
            image_bned_n = [images_link[i - 1] for i in bned_image if 1 <= i <= len(images_link)]

            # Prepare images (compressed, with user prompt text)
            content_parts, _ = await prepare_images_for_llm(
                image_bned_n,
                use_compression=True
            )

            content_parts.append({
              "type": "text",
              "text": prompt_bn_user
            })
            
            raw, parsed = await self.arun_chain(content_parts)
            
            get_metadata = True
            if get_metadata:
                # Convert AIMessage to dict before computing usage
                raw_dict = {
                    "response_metadata": raw.response_metadata,
                    "usage_metadata": getattr(raw, 'usage_metadata', {})
                }
                
                usage = compute_usage(raw_dict, self.model_name)
                runtime = time.time() - run_start
                output = {
                    "bn_ed": parsed,
                    "bned_usage": usage,
                    "bned_runtime": runtime,
                }
                print("bned_output: ", output)

            else: 
                ai_msg = parsed.__dict__

                output = {"bn_ed": ai_msg}
                print("bned_output:", output)
            
            return output

        except Exception as e:
            print("Error terjadi pada BNED Agent")
            raise response_error(str(e))
