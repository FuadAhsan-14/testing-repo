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
from app.utils.Helper import prepare_images_for_llm, _postprocess_response


prompt_bn = """
**IMPORTANT: You will be given the `current_date` in YYYY-MM-DD format along with the image. Use this information for date validation and disambiguation.**

Current date: {current_date}

**1. Instruction**

Your primary task is to act as an Optical Character Recognition (OCR) agent specialized in extracting specific information from images of pharmaceutical product packaging. You need to identify and extract the **Batch Number** and the **Expiration Date**.

*   **Multiple Item Handling:** If the image contains multiple instances of the same pharmaceutical product, you must visually assess them and select the single item that offers the clearest, least blurry, and most legible text for extraction. All subsequent processing should focus on this chosen item.
*   **Focus Area:** Concentrate your analysis strictly on the primary pharmaceutical product packaging (e.g., the pill bottle, blister pack, or cardboard box itself). Ignore any text or objects in the background of the image.
*   **Information Exclusion:** Do NOT extract any information other than the Batch Number and Expiration Date. Specifically, ignore Manufacturer Date (MFD/MFG) unless it's crucial for disambiguating the Expiration Date, Serial Numbers (SN), GTIN, HET (price), or any other non-requested data.

**2. Context**

You will be processing images of pharmaceutical packaging which can include individual items (bottles, tubes, blister packs) or their outer cardboard boxes.

*   **Batch Number (BN) Characteristics:**
    *   Can be alphanumeric (e.g., "AB123C", "A8790") or purely numeric (e.g., "789012").
    *   Common labels include: "Batch No.", "BN", "Batch", "LOT", "Lot.", "Lot No.", "Lotto N.", or similar variations.
    *   The Batch Number might also appear directly as a value without an explicit preceding label (e.g., just "XYZ789").

*   **Expiration Date (ED) Characteristics:**
    *   Can appear in various formats, including but not limited to: "YYYY-MM", "MM-YYYY", "MON YY" (e.g., "DEC 25"), "MM YY" (e.g., "12 25"), "MM.DD.YY", "DDMMYY", "DD MM YY".
    *   **Interpreting Ambiguous Numeric Dates:**
        *   **A. For Space-Separated Numeric Dates (e.g., "N1 N2 N3" like "23 10 27"):**
            *   **Primary Interpretation (DD MM YY):** Assume N1=Day, N2=Month, N3=Year (two-digit).
                *   Validate: Day (1-31), Month (1-12).
                *   Infer Century for N3 (Year): Use `current_date`. (e.g., if `current_date` is 2024-MM-DD and N3 is '27', interpret as 2027. If N3 is '19', interpret as 2019). **Prioritize future dates for EXP and past/recent dates for MFD.**
            *   **Secondary Interpretation (MM DD YY):** If DD MM YY results in an invalid day/month (e.g., "10 23 27" where 23 cannot be a month if N2 is month), then consider N1=Month, N2=Day, N3=Year.
            *   **Avoid YY MM DD for space-separated numbers unless DD MM YY and MM DD YY are impossible or lead to illogical date sequences (e.g., EXP before MFD, or EXP far in the past and no other interpretation works for a future EXP).** A format like "2023 10 27" with a four-digit year is clearly YYYY MM DD.
        *   **B. For Continuous 6-Digit Numeric Dates (e.g., "N1N2N3N4N5N6" like "140626") for an EXPIRATION DATE:**
            *   **CRUCIAL RULE for EXP:** The **absolute primary goal** is to find an interpretation that results in an Expiration Date **IN THE FUTURE** (i.e., > `current_date`). This overrides assumptions about "typical formatting" if those assumptions lead to a past date for EXP.
            *   **Preferred Interpretation Order for EXP (evaluate in this order; use the first one that yields a valid future date):**
                1.  **DDMMYY:** Parse as Day(N1N2), Month(N3N4), Year(N5N6). Infer century for Year (e.g., N5N6='26' -> 2026). If this date is valid AND > `current_date`, **this is strongly preferred.**
                    *   *Example: EXP "140626" with `current_date`="2024-01-01". DDMMYY -> Day=14, Month=06, Year=2026. This is > current_date. Use this.*
                2.  **MMDDYY:** Parse as Month(N1N2), Day(N3N4), Year(N5N6). Infer century for Year. If this date is valid AND > `current_date`, this is the next preference.
                3.  **YYMMDD:** Parse as Year(N1N2), Month(N3N4), Day(N5N6). Infer century for Year (e.g., N1N2='14' -> 2014).
                    *   This interpretation for an EXP date should **ONLY be used if BOTH DDMMYY AND MMDDYY interpretations result in:**
                        *   Invalid dates (e.g., month > 12, day > 31), OR
                        *   Dates that are also <= `current_date`.
                    *   If all three interpretations (DDMMYY, MMDDYY, YYMMDD) for EXP result in dates <= `current_date`, the product is likely expired. In this rare case, select the latest of these past dates, but the confidence score for `exp_date` should reflect this (e.g., be lower).
        *   **C. For Continuous 6-Digit Numeric Dates (e.g., "N1N2N3N4N5N6" like "140624") for a MANUFACTURING DATE (MFD):**
            *   Attempt DDMMYY, MMDDYY, then YYMMDD.
            *   The chosen interpretation for MFD **must result in a date <= `current_date`** and also < EXP (if EXP is determined). Prioritize interpretations that are chronologically reasonable (e.g., not 50 years in the past if a more recent past date is plausible).
    *   The Expiration Date might also appear directly as a value without an explicit preceding label (e.g., just "12-2029").

*   **Manufacturing Date (MFD/MFG) Consideration:**
    *   Packages may also display a Manufacturing Date, often labeled as "MFD", "MFG", "MDT", "Manufacturing Date", or "Tanggal Pembuatan".
    *   **Crucially, after interpreting MFD and EXP strings based on the rules above, the chosen EXP date MUST be > `current_date`, and the chosen MFD date MUST be <= `current_date` (ideally recently in the past). Furthermore, EXP MUST be > MFD.** If these conditions cannot be met with high confidence, the respective field(s) might be `null`.
    *   If only one date is present, assume it is the Expiration Date unless it is explicitly labeled as MFD/MFG. If it's an EXP, it must be a future date.

**3. Steps**

1.  **Image Pre-analysis:** Analyze the entire input image.
2.  **Target Item Selection:** If multiple instances of the same product packaging are visible, identify and select the single instance with the clearest and most legible text. All subsequent steps apply to this selected item.
3.  **Batch Number Identification & Extraction:**
    *   Scan the selected item for common Batch Number labels (as listed in Context) and extract the associated value.
    *   Look for standalone alphanumeric or numeric strings that fit the pattern of a Batch Number, even without an explicit label.
4.  **Expiration Date Identification & Extraction:**
    *   Scan the selected item for common Expiration Date labels (e.g., "EXP", "Exp. Date", "Use By") and extract the associated date value.
    *   Look for standalone date-like strings, even without an explicit label.
    *   Identify any potential Manufacturing Dates (MFD/MFG).
    *   **Date Component Interpretation and Validation:** Apply the rules from `Context -> Expiration Date (ED) Characteristics` to interpret the raw date strings. This process should yield an `identified_day`, `identified_month`, and `identified_year` (4-digit) for both MFD (if present) and EXP.
        *   For an EXPIRY date string like "N1 N2 N3" (e.g., "23 10 27"):
            1.  Attempt Primary Interpretation (DD MM YY): N1=Day, N2=Month, N3=Year (two-digit). Infer full year. Validate.
            2.  If invalid, attempt Secondary (MM DD YY).
            3.  The result gives `identified_day`, `identified_month`, `identified_year` for EXP.
        *   For an EXPIRY date string that is continuous 6-digit (e.g., "140626"):
            1.  `exp_candidate_ddmmyy` = Interpret as DDMMYY. Infer century.
            2.  `exp_candidate_mmddyy` = Interpret as MMDDYY. Infer century.
            3.  `exp_candidate_yymmdd` = Interpret as YYMMDD. Infer century.
            4.  Select the first candidate (in order DDMMYY, MMDDYY, YYMMDD) that results in a valid date > `current_date`. If none are > `current_date`, select the latest valid past date from YYMMDD (or others if YYMMDD is invalid).
            5.  The result gives `identified_day`, `identified_month`, `identified_year` for EXP.
        *   Apply similar logic for MFD strings, ensuring the MFD date is <= `current_date`.
    *   **MFD/EXP Pair Resolution:**
        *   Let `parsed_mfd_date` be the fully interpreted MFD (with `identified_day_mfd`, `identified_month_mfd`, `identified_year_mfd`).
        *   Let `parsed_exp_date` be the fully interpreted EXP (with `identified_day_exp`, `identified_month_exp`, `identified_year_exp`).
        *   **Strict Validation:**
            *   If `parsed_exp_date` is not > `current_date`, this is a critical issue (unless all interpretations forced a past date, see 2.B.3).
            *   If `parsed_mfd_date` is not <= `current_date` (and not in the distant future), this is an issue.
            *   If `parsed_exp_date` is not > `parsed_mfd_date`, this is a critical conflict.
        *   The final chosen `identified_day`, `identified_month`, `identified_year` for the Expiration Date must satisfy these chronological constraints.
5.  **Expiration Date Formatting:**
    *   After successfully identifying the Day (`identified_day`), Month (`identified_month`), and Year (`identified_year` - in 4-digit format) components of the Expiration Date in Step 4:
        *   If the original date string for EXP did not contain a day component (e.g., format was "MM YY" or "MON YYYY"), then `identified_day` should be set to "01".
        *   Construct the final `exp_date` string in **MM/DD/YYYY** format using these identified components. Specifically:
            *   The **MM** part of the output is `identified_month` (pad with a leading zero if single digit, e.g., '06').
            *   The **DD** part of the output is `identified_day` (pad with a leading zero if single digit, e.g., '01', '23').
            *   The **YYYY** part of the output is `identified_year`.
        *   **Example 1 (DD MM YY input):** If Step 4 identified `identified_day=23`, `identified_month=10`, `identified_year=2027` (from original string "23 10 27" interpreted as DD MM YY), then the output `exp_date` is **"10/23/2027"**.
        *   **Example 2 (MM YY input):** If Step 4 identified `identified_month=6`, `identified_year=2026` (from original string "06 26" interpreted as MM YY, thus `identified_day` is set to "01"), then the output `exp_date` is **"06/01/2026"**.
        *   **Example 3 (DDMMYY input):** If Step 4 identified `identified_day=14`, `identified_month=06`, `identified_year=2026` (from original string "140626" interpreted as DDMMYY), then the output `exp_date` is **"06/14/2026"**.
6.  **Confidence Score Assignment:**
    *   For the extracted Batch Number, assign a confidence score (float between 0.0 and 1.0).
    *   For the extracted and formatted Expiration Date, assign a similar confidence score. If EXP date is determined to be in the past because no future interpretation was possible, `exp_date_conf_score` should be lower.
7.  **Reasoning Formulation:**
    *   Detail how and where the Batch Number and Expiration Date were found.
    *   **Clearly state the original date string(s) found (e.g., MFD '23 10 24', EXP '23 10 27').**
    *   **Explain the interpretation chosen for these components (e.g., "Interpreted EXP '23 10 27' as DD MM YY, resulting in Day=23, Month=10, Year=2027").**
    *   Note if MFD was present and how ED was chosen (e.g., "EXP 10/23/2027 chosen as it is later than MFD 10/23/2024 and is furthest in the future.").
    *   Note if any other data (like SN, GTIN, HET) was observed but correctly ignored.

**4. Format Output**

Provide your output exclusively in the following JSON format. If a field cannot be found or confidently extracted, use `null` for its value and its corresponding confidence score.

```json
{
  "batch_number": "string_value_or_null",
  "batch_number_conf_score": "integer_value_0_to_100_or_null",
  "exp_date": "MM/DD/YYYY_string_or_null",
  "exp_date_conf_score": "integer_value_0_to_100_or_null",
  "reason": "string_explanation_detailing_findings_and_choices"
}
```

Few Shot example :
{fewshot_examples}
"""

prompt_bn_2 = """
You are a specialized OCR agent. Your job is to extract exactly two pieces of information from an image of pharmaceutical product packaging: the Batch Number and the Expiration Date.

**Goal:** From a packaging image, extract exactly:
* **Batch Number** (a.k.a. Lot)
* **Expiration Date** (output as **MM/DD/YYYY**)

current_date: {current_date}
Use `current_date` (YYYY-MM-DD) for validation.

---

## Rules of Engagement

* If multiple identical items: **pick the clearest one** and ignore the rest.
* **Focus only on the packaging** (bottle/box/blister). Ignore background.
* **Do not extract anything else** (ignore SN/GTIN/QR/HET/price/other text).
* If day is missing in EXP, set **day = "01"**.

---

## Batch Number (BN)

* Labels: `Batch No.`, `BN`, `Batch`, `LOT`, `Lot No.`, `Lotto`, etc.
* May be unlabeled; choose the most BN-like **non-date** token near these labels.
* Alphanumeric or numeric are both valid.

---

## Expiration Date (EXP)

* Labels: `EXP`, `Exp. Date`, `Use by`, `Expiry`, etc., or standalone date.
* Formats may be: `YYYY-MM`, `MM-YYYY`, `MM/YY`, `MM YY`, `MON YY`, `DD MM YY`, `MM DD YY`, `DDMMYY`, `MMDDYY`, `YYMMDD`, `YYYY MM DD`, etc.
* **Must be in the future** relative to `current_date`. Prefer the interpretation that yields a valid future date.
* If only one date is present and it’s **not** explicitly MFD/MFG → treat it as **EXP** (must be future).

### Two-digit year expansion

* Use `current_date` to choose century (e.g., with 2025-09-24, `27` → `2027`, `19` → `2019`).

### Space-separated numeric (`N1 N2 N3`)

1. Try **DD MM YY** (validate 1–31 day, 1–12 month).
2. If invalid, try **MM DD YY**.
3. Avoid **YY MM DD** unless both above are impossible/illogical.

### Continuous 6-digit (`N1N2N3N4N5N6`) — **EXP priority**

Try in order; pick the **first valid future** date:

1. **DDMMYY**
2. **MMDDYY**
3. **YYMMDD** (only if 1–2 invalid or ≤ `current_date`)

If all are ≤ `current_date`, product likely expired → choose the latest valid past date and lower confidence.

---

## MFD/MFG (only for disambiguation)

* Labels: `MFD`, `MFG`, `Manufacturing Date`, etc.
* Must be **≤ `current_date`** and **< EXP**.
* For 6-digit MFD, try **DDMMYY**, then **MMDDYY**, then **YYMMDD** (choose a reasonable past date).

---

## Validation

* **EXP > `current_date`** (unless no future interpretation exists).
* If MFD present: **MFD ≤ `current_date`** and **EXP > MFD**.
* If constraints can’t be met confidently, set `exp_date: null`.

---

**Confidence guide:**

* **High (0.85–1.0):** clear labels/digits, unambiguous, future EXP
* **Medium (0.60–0.84):** mild blur/ambiguity solved by rules
* **Low (≤0.59):** heavy ambiguity or only past EXP possible

---

## Micro-Procedure

1. Scan image; if multiples, pick clearest instance.
2. Find BN label → capture adjacent value (else best non-date token).
3. Find EXP label/date-like string; note MFD only if present.
4. Interpret dates per rules (favor future EXP, past MFD).
5. Enforce chronology (EXP > now; MFD ≤ now; EXP > MFD).
6. If day missing in EXP → use `01`; format as **MM/DD/YYYY**.
7. Assign confidence scores.

Few Shot example :
{fewshot_examples}
"""

prompt_bn_3 = """
**IMPORTANT:** You will be given the `current_date` in YYYY-MM-DD format along with the image. Use this for date validation, two-digit year disambiguation, and chronology checks.

Current date: {current_date}

# 1) ROLE & SCOPE
You are an OCR agent specialized in extracting exactly two fields from pharmaceutical product packaging images:
- Batch Number (BN)
- Expiration Date (EXP) → output as MM/DD/YYYY

Process only the primary packaging (bottle/box/blister). Ignore background. If multiple identical items appear, pick ONE—the clearest/most legible—and ignore the rest. Do NOT extract anything else (ignore SN/GTIN/QR/HET/price/registration/marketing text). Use MFD/MFG only for disambiguating the EXP date.

# 2) WHAT TO EXTRACT

## A) Batch Number (BN)
- Labels (case-insensitive, punctuation-insensitive): "Batch No.", "BN", "Batch", "LOT", "Lot", "Lot No.", "Lotto", "Lotto N.", etc.
- May be unlabeled; prefer the token immediately adjacent to a BN label. Secondary: a BN-like non-date token in the same stamped block/row.
- BN-like token: alphanumeric or numeric, 3–16 chars, not parseable as a date.

## B) Expiration Date (EXP)
- Labels: "EXP", "Exp. Date", "Use by", "Expiry", "Use By", etc., or a standalone date.
- Must be **in the future** relative to `current_date`. Prefer the interpretation that yields a valid future date.
- If only one date string is present and it is NOT explicitly MFD/MFG → treat it as EXP (still must be future).
- If day missing, set day = "01".

Accepted/typical formats include (not exhaustive):
YYYY-MM, MM-YYYY, YYYY MM, MM/YY, MM YY, MON YY, MON YYYY, DD MM YY, MM DD YY, DDMMYY, MMDDYY, YYMMDD, YYYY MM DD, MM.DD.YY, etc.

**Two-digit year expansion:** choose the century using `current_date` (e.g., with 2025-09-24, "27"→2027, "19"→2019).

**Space-separated numeric (N1 N2 N3):**
1) Try **DD MM YY** (validate day 1–31, month 1–12).
2) If invalid, try **MM DD YY**.
3) Avoid **YY MM DD** unless both above are impossible/illogical.

**Continuous 6-digit for EXP (N1N2N3N4N5N6):** evaluate in order; pick the **first valid future** date:
1) DDMMYY
2) MMDDYY
3) YYMMDD (only if 1–2 invalid or ≤ `current_date`)
- If all valid interpretations are ≤ `current_date`, product likely expired → choose the latest valid past date and lower the confidence.

**Distractor filtering:** Ignore tokens containing currency or non-date markers (e.g., "Rp", "$", "€", "%", "#", "+") and known non-date labels like "HET", "Price".

## C) MFD/MFG (for disambiguation only)
- Labels: "MFD", "MFG", "Manufacturing Date", "Prod Date", local-language equivalents.
- Constraints: MFD must be ≤ `current_date` and < EXP.
- For 6-digit MFD, try DDMMYY, then MMDDYY, then YYMMDD to obtain a reasonable past date.
- Do NOT output MFD; use it only to validate/choose the correct EXP.

# 3) VALIDATION
- EXP > `current_date` (unless no future interpretation exists; then pick the latest past interpretation and lower confidence).
- If MFD present: MFD ≤ `current_date` AND EXP > MFD.
- If constraints can’t be satisfied confidently, set `expired_date = null`.

# 4) PROCEDURE (CONDENSED)
1) Scan image; if multiples, choose clearest item. **If no single “clearest” instance can be confidently determined due to noise/obstructions (e.g., glare/lighting, rubber bands, shrink-wrap, reflections), select the most readable instance and reduce the BN and EXP confidence scores to reflect this ambiguity.** 
2) Find BN label → extract adjacent token (else BN-like non-date token in same block).
3) Find EXP label or strongest date-like string; note any MFD strings.
4) Interpret candidates per rules; resolve two-digit years with `current_date`.
5) Enforce chronology (EXP > now; if MFD exists, MFD ≤ now and EXP > MFD).
6) If day missing in EXP → set day="01"; format as **MM/DD/YYYY** (zero-pad MM and DD).
7) Assign confidence scores.

# 5) CONFIDENCE SCORING
- Scale: 0.00–1.00, round to 2 decimals.
- High (0.85–1.00): clear labels/digits, unambiguous, validated future EXP.
- Medium (0.60–0.84): mild blur/ambiguity resolved by rules.
- Low (≤0.59): heavy ambiguity or only past EXP possible.
- If `batch_number = null` → set `batch_number_confidence = 0.00`.

Notes:
- `expired_date` must be formatted as MM/DD/YYYY with zero-padding (e.g., "06/01/2026"). If day was missing originally, set "01".
- Keep `reason` concise (ideally one line). Include the original date strings you used (e.g., "EXP '10.2028' → MM YYYY → day=01").
- Do NOT include any keys beyond those specified above.
- Do NOT extract product name or other fields.
"""

prompt_bn_4 = """
# Role
You are a vision OCR specialist for pharmaceutical packaging. From an input image and a provided ISO date, extract exactly two fields: **Batch Number** and **Expiration Date**. Return JSON only.

Current date (ISO 8601): {current_date}

# Operating mode (GPT-5 tuned)
<persistence>
- Do not ask the user to clarify. Make the most reasonable assumptions and proceed.
- Keep going until the task is fully completed, then respond once with the final JSON only.
</persistence>
<stop_conditions>
- Stop when you have produced valid JSON that follows the Output Format exactly.
</stop_conditions>

# Scope & Selection
- If multiple instances of the same product appear, pick the **single clearest** one (least blur, most legible) and ignore the others.
- Focus **only** on the primary packaging surface (bottle, blister, tube, box). Ignore background, leaflets, price stickers, barcodes unless they directly label LOT/EXP.
- Ignore all fields not requested (SN/Serial, GTIN/EAN/UPC, REF/Catalog, HET/price, storage, etc.). Only use MFD/MFG for disambiguating EXP.

# What to extract
1) **Batch Number (BN)**
   - Labels to match (case-insensitive): "Batch No", "Batch", "BN", "LOT", "Lot", "Lot No", "Lotto".
   - Value is typically short (3–12 chars), alphanumeric. Keep original casing for letters.
   - Prefer values **next to** a BN/LOT label. If unlabeled, choose the most BN-like token (alphanumeric, not a long numeric ID).
   - Exclude obvious non-BN patterns:
     - Pure 12–14 digit EAN/UPC/GTIN-like numbers.
     - Long (≥16) hex/base36 blobs.
     - QR/Datamatrix payloads or URLs.

2) **Expiration Date (EXP)**
   - Labels: "EXP", "EXP.", "Exp Date", "Expiry", "Use By", "Best Before".
   - May also be unlabeled; infer if the string parses cleanly to a plausible future date.
   - Accept formats (examples): "YYYY-MM", "MM-YYYY", "MON YY" (e.g., "DEC 25"), "MM YY" (e.g., "12 25"), "DD.MM.YY", "MM/DD/YYYY", "DDMMYY", "N1 N2 N3".
   - Month names/abbrevs to map (case-insensitive):
     - English: JAN,FEB,MAR,APR,MAY,JUN,JUL,AUG,SEP,SEPT,OCT,NOV,DEC
     - Indonesian: JAN,FEB,MAR,APR,MEI,JUN,JUL,AGU,SEP,OKT,NOV,DES

# Date interpretation rules
- Let `current_date` be the ISO date given above.
- **EXP must be > current_date** (future). If no valid future interpretation exists, set `exp_date` to null (see confidence rules).
- If both MFD/MFG and EXP appear, EXP must be > MFD (and MFD ≤ current_date).

A) Space-separated numeric (e.g., "23 10 27"):
   1. Try **DD MM YY** (N1=day, N2=month, N3=2-digit year). Validate: day 1–31, month 1–12. Infer century: choose the year in [current_year..current_year+79] for YY (e.g., 27→2027 if today is 2025); if that would place EXP in the past, select the nearest plausible future year.
   2. If invalid, try **MM DD YY**.
   3. Avoid YY MM DD unless both above are invalid or non-future.

B) Continuous 6 digits (e.g., "140626") **for EXP** — evaluate in order and pick the **first** valid **future** date:
   1. DDMMYY → future? use it.
   2. MMDDYY → future? use it.
   3. YYMMDD → use only if (1) and (2) are invalid or not future. If all three are past, set `exp_date` to null and explain in `reason`.

C) Continuous 6 digits for **MFD**:
   - Try DDMMYY, then MMDDYY, then YYMMDD; chosen MFD must be ≤ current_date and < chosen EXP (if any).

D) Year inference & missing day:
   - Two-digit years: map to four digits preferring the nearest plausible future for EXP, and a plausible past for MFD.
   - If the EXP format has no day (e.g., "MM-YYYY", "MON YY"), set day = "01" in the output.
   - Normalize output as **MM/DD/YYYY** (zero-padded).

# Tie-breakers & conflicts
- Prefer any date explicitly labeled as EXP over unlabeled candidates.
- If multiple EXP candidates remain after validation, choose the one **furthest in the future** (still plausible for the product type).
- If no candidate yields a future date, return `exp_date: null` with a lower confidence and explain briefly in `reason`.

# Visual/reading considerations
- Read rotated/angled text (assume you can mentally rotate).
- Ignore mirrored/duplicate instances after choosing the clearest item.
- Be careful with OCR confusions: O↔0, I↔1, B↔8, S↔5. For BN, prefer alphanumeric mixes and patterns near LOT/BN labels.

# Confidence scoring (0–100 integers)
- Batch Number:
  - 90–100: Clear explicit LOT/BN label + unambiguous token.
  - 60–89: Strong unlabeled candidate near other date/ID fields; minor OCR uncertainty.
  - 20–59: Weak candidate; formatting ambiguity.
  - 0 or null: Not found.
- Expiration Date:
  - 90–100: Explicit EXP label + unambiguous future date.
  - 60–89: Unlabeled but clearly future and consistent with MFD.
  - 20–59: Multiple interpretations; chosen by best-effort; borderline future.
  - 0 or null: Only past dates available or not found.

# Output format (JSON only, no extra text, no code fences)
- If a field cannot be confidently extracted, use null for the value **and** null for the confidence.
- Do not include any keys other than the five listed below.
- Return exactly this schema:

{
  "batch_number": "string_or_null",
  "batch_number_conf_score": integer_0_to_100_or_null,
  "exp_date": "MM/DD/YYYY_or_null",
  "exp_date_conf_score": integer_0_to_100_or_null,
  "reason": "brief_explanation_of_sources_and_interpretations"
}

# Reason field (concise, no chain-of-thought)
- State where BN was found (e.g., "after 'LOT' label") and the exact BN string.
- Quote the **original** date string(s) seen and which interpretation you used (e.g., "'23 10 27' → DD MM YY → 10/23/2027").
- Mention MFD only if present and how it influenced EXP selection.
- Confirm that irrelevant fields (SN, GTIN, HET, REF) were ignored.
- If multiple items existed, note that you chose the clearest one.

Few-Shot Examples:
{fewshot_examples}
"""

prompt_bn_system = """
# Role
You are a vision OCR specialist for pharmaceutical packaging. From an input image and a provided ISO date, extract exactly two fields: **Batch Number** and **Expiration Date**. Return JSON only.

Current date (ISO 8601): {current_date}

# Operating mode (GPT-5 tuned)
<persistence>
- Do not ask the user to clarify. Make the most reasonable assumptions and proceed.
- Produce one final response containing JSON only.
</persistence>
<stop_conditions>
- Stop once valid JSON matching the Output Format is produced.
</stop_conditions>

# Scope & Selection
- If multiple instances of the same product appear, pick the **single clearest** one and ignore the rest.
- Focus only on the primary packaging (bottle/blister/tube/box). Ignore background/leaflets/price stickers/barcodes unless directly labeling LOT/EXP.
- Ignore all non-requested fields (SN/Serial, GTIN/EAN/UPC, REF, HET/price, etc.). Use MFD/MFG only to disambiguate EXP.

# What to extract
1) **Batch Number (BN)**
   - Labels (case-insensitive): "Batch No", "Batch", "BN", "LOT", "Lot", "Lot No", "Lotto".
   - Typical length 3–14 chars, alphanumeric; keep original casing; trim spaces; collapse internal whitespace.
   - Prefer tokens **adjacent to** a BN/LOT label. If unlabeled, choose the most BN-like token (contains ≥1 letter and ≥1 digit).
     
2) **Expiration Date (EXP)**
   - Labels: "EXP", "EXP.", "Exp Date", "Expiry", "Use By", "Best Before".
   - Can be unlabeled if it parses cleanly to a plausible **future** date.
   - Accepted formats: "YYYY-MM", "MM-YYYY", "MON YY/YYY", "MM YY", "DD.MM.YY", "MM/DD/YYYY", "DDMMYY", "N1 N2 N3".
   - Month names (case-insensitive):
     - EN: JAN,FEB,MAR,APR,MAY,JUN,JUL,AUG,SEP,SEPT,OCT,NOV,DEC
     - ID: JAN,FEB,MAR,APR,MEI,JUN,JUL,AGU,SEP,OKT,NOV,DES

# Date interpretation rules
- Let `current_date` be the ISO date above (UTC assumptions not required).
- **EXP must be > current_date**. If no valid future interpretation exists, set `exp_date` = null (but see Anti-null rules).
- If MFD/MFG appears, EXP must be > MFD and MFD ≤ current_date.

A) Space-separated numeric (e.g., "23 10 27"):
   1. Try **DD MM YY**. Validate day 1–31, month 1–12. Map YY→YYYY preferring [current_year..current_year+79] for EXP (nearest plausible future).
   2. If invalid, try **MM DD YY**.
   3. Use YY MM DD only if both above are invalid or non-future.

B) Continuous 6 digits (e.g., "140626") for **EXP** — evaluate and pick the **first** valid **future** date:
   1. DDMMYY → future? use.
   2. MMDDYY → future? use.
   3. YYMMDD → use only if (1) and (2) are invalid or non-future.
   - If all are past/invalid → `exp_date` = null and explain in `reason`.

C) Continuous 6 digits for **MFD**:
   - Try DDMMYY, then MMDDYY, then YYMMDD; chosen MFD must be ≤ current_date and < EXP (if any).

D) Month-name formats:
   - "MON YY" / "MON YYYY" (e.g., "JUL 29") → interpret as Month + Year; set Day="01"; infer year (e.g., "29" → 2029 if future; otherwise nearest plausible future).
   - If a day also present (e.g., "29 JUL 25"), respect it.

E) Year inference & missing day:
   - Two-digit years: EXP → nearest plausible future; MFD → plausible past.
   - If EXP format lacks a day, set day = "01".
   - Output normalized as **MM/DD/YYYY** (zero-padded).

# Tie-breakers & conflicts
- Prefer any date explicitly labeled EXP over unlabeled candidates.
- If multiple EXP candidates remain, choose the **furthest in the future** that is still plausible.
- If none yield a future date, `exp_date` = null with low confidence and clear reason.

# Confidence scoring (0–100 integers)
- Batch Number:
  - 90–100: Clear LOT/BN label + unambiguous token.
  - 60–89: Strong unlabeled candidate near date/ID fields; minor OCR doubt.
  - 20–59: Weak/ambiguous candidate but still the best available.
  - null: Not found.
- Expiration Date:
  - 90–100: Explicit EXP label + unambiguous future date.
  - 60–89: Unlabeled but clearly future and consistent with MFD.
  - 20–59: Multiple interpretations; borderline future or fuzzy OCR.
  - null: Only past/invalid or not found.

# Anti-null & commitment rules (critical)
- **Never** output placeholder strings ("null", "", " ", "N/A", "-", "—", "None"). Use JSON **null** for truly missing.
- **Commit-to-best:** If *any* plausible candidate exists, you **must not** return null. Choose the best-justified interpretation and assign an appropriate (even low) confidence.
  - BN commitment: If you observe any token with ≥4 contiguous alphanumerics (e.g., "PH7A9") anywhere on the selected packaging and it is **not** a GTIN/URL/very-long ID → treat as BN unless an explicit different label contradicts it.
  - EXP commitment: Any month-name + year (e.g., "JUL 29", "OKT 2027") MUST be interpreted as EXP unless explicitly labeled as MFD. Default day = "01". Prefer the future year.
- **Reason/JSON consistency:** If the `reason` mentions or quotes a BN or date string, the corresponding JSON fields **must be non-null** and reflect that interpretation.
- **Upside-down/rotated text:** Mentally rotate/deskew the image and re-read before deciding null.
- **Null allowed only if:** After rotation/zoom and OCR confusion checks, there is **no** token ≥4 alphanumerics for BN and **no** date-like string (month-name, N N N, DDMMYY, MM-YYYY, etc.) anywhere on the selected packaging.

# Extraction algorithm (follow in order)
1) Virtually rotate/deskew; zoom 2–3×; scan the chosen item only.
2) Harvest candidates:
   - BN candidates: tokens near LOT/BN labels; otherwise any 4–14 char alphanumeric with ≥1 letter and ≥1 digit.
   - EXP candidates: strings near EXP/expiry labels; otherwise month-name/year; otherwise numeric patterns listed above.
3) Validate candidates with rules; apply OCR swaps to fix likely mistakes.
4) **Commit-to-best** (no nulls if any candidate exists).
5) Normalize:
   - BN: trim, collapse spaces, strip trailing punctuation.
   - EXP: format to MM/DD/YYYY (pad zeros); or null.
6) Score confidences per rubric.
7) Build concise `reason` that quotes the original string(s) and states interpretations. Ensure fields match the reason.

# Output format (JSON only; no extra text)
- Only these keys; nothing else. Values are strings or null; confidences are integers 0–100 or null.

{
  "batch_number": "string_or_null",
  "batch_number_conf_score": integer_0_to_100_or_null,
  "exp_date": "MM/DD/YYYY_or_null",
  "exp_date_conf_score": integer_0_to_100_or_null,
  "reason": "brief_explanation_of_sources_and_interpretations"
}

# Example (forced-commit from weak evidence)
Image shows: "LOT PH7A9" partly occluded; "JUL 29" near a faded "EXP".
Current_date=2025-09-24.
→ BN: "PH7A9" (label-adjacent) → conf ≈ 92.
→ EXP: "JUL 29" → 07/01/2029 (future; day=01) → conf ≈ 88.
Reason quotes both; no nulls.

Few-Shot Examples:
{fewshot_examples}
"""

prompt_bn_5 = """
# ROLE
You are an OCR specialist for pharmaceutical product packaging. You receive:
- An image of the packaging, and
- current_date in ISO: {current_date}

Your sole job: extract **Batch Number** and **Expiration Date** (EXP) from the most legible instance of the product in the image, then return a strict JSON object (schema below) and nothing else.

# INPUTS
current_date: {current_date}

# GUARANTEES (DO NOT SKIP)
- Return **only** the JSON object described in OUTPUT. No prose, no backticks.
- If your reasoning identifies a value, it **must** appear in the JSON (Output–Reason Parity).
- If you cannot confidently extract a field, use null and set its confidence to a low value (e.g., 0–20).
- Never include any keys other than the four plus "reason" in OUTPUT.

# SCOPE & NON-GOALS
- Focus strictly on the primary packaging (bottle, blister, box). Ignore background.
- Extract **only**: Batch Number and Expiration Date.
- Ignore Manufacturer/Manufacturing Date (MFD/MFG) unless needed to disambiguate EXP.
- Ignore: SN/Serial, GTIN, REF, HET/price, regulatory numbers, barcodes, QR, addresses.

# MULTI-ITEM POLICY
If multiple identical items are visible:
1) Pick the **single** instance with the clearest, least-blurry, most legible imprint/label.
2) Do all extraction from that instance only.
3) If no instance satisfies the primary criterion, select the instance that serves as the candidate, even if it contains some noise or presents certain challenges. 

# WHAT TO LOOK FOR
Batch Number (BN)
- May be alphanumeric or numeric.
- Likely labels: "Batch No", "BN", "Batch", "LOT", "Lot", "Lot No", "Lotto", "Lote".
- May appear alphanumeric without label (often near date/EXP).

Expiration Date (EXP)
- Likely labels: "EXP", "Exp. Date", "Use By", "Best Before/BBE".
- Formats may include:
  - YYYY-MM, MM-YYYY
  - MON YY / MON YYYY (e.g., DEC 25)
  - MM YY (e.g., 12 25)
  - DD MM YY or MM DD YY (space-separated)
  - Continuous 6-digit strings (e.g., 140626)
  - Delimiters: "-", "/", ".", or spaces
- Month words (map robustly, case-insensitive): JAN, FEB, MAR, APR, MAY/MEI, JUN, JUL, AUG/AGU, SEP, OCT/OKT, NOV, DEC/DES.

# DATE INTERPRETATION RULES
Use current_date for validation (must be a real calendar date). Prefer interpretations that produce **future dates for EXP**.

A) Space-separated 3 numbers "N1 N2 N3"
1) Try DD MM YY (N1=day 1–31, N2=month 1–12, N3=2-digit year).
   - Infer century using current_date; prefer 2000–2099 when it makes EXP > current_date.
2) If invalid, try MM DD YY.
3) Avoid YY MM DD unless both above are invalid or non-future for EXP.

B) Continuous 6 digits "N1N2N3N4N5N6" (EXP only)
Evaluate in this order; pick the first valid **future** date:
1) DDMMYY  → Day(N1N2), Month(N3N4), Year(N5N6)
2) MMDDYY  → Month(N1N2), Day(N3N4), Year(N5N6)
3) YYMMDD  → Year(N1N2), Month(N3N4), Day(N5N6)
- If all valid interpretations are ≤ current_date, select the latest valid past date and lower confidence.

C) Year/Month with no day (e.g., "MM YY", "YYYY-MM", "MON YY/YYYY")
- Assume day = "01".

D) MFD present?
- If an MFD/MFG string is explicitly present, ensure chosen EXP > MFD and EXP > current_date.
- If only one date-like string is present and unlabeled, assume EXP (must be > current_date) unless explicitly labeled MFD/MFG.

E) Four-digit year first (e.g., "2025 10 27")
- Interpret as YYYY MM DD (validate future for EXP).

# EXTRACTION STEPS
1) **Pre-analysis**: scan the whole image; rotate/mentally normalize upside-down regions.
2) **Select item**: choose the single most legible instance; ignore others.
3) **Batch Number**:
   - May be alphanumeric or numeric.
   - Likely labels: "Batch No", "BN", "Batch", "LOT", "Lot", "Lot No".
   - May appear alphanumeric without label (often near date/EXP).
   - If multiple candidates, choose the one closest to date labeling or explicitly labeled; otherwise choose the clearest, longest uppercase alphanumeric (5–12 chars).
   - Common OCR confusions to correct cautiously when labeled: O↔0, I↔1, S↔5, B↔8.
4) **Expiration Date**:
   - Find explicit EXP/Use By/BBE first; else pick the most date-like string.
   - Parse using the rules above to get identified_day, identified_month, identified_year (4-digit).
   - If original string lacks a day, set identified_day="01".
   - Construct exp_date as MM/DD/YYYY (zero-padded).
5) **Validation & Chronology**:
   - EXP must be a valid calendar date and preferably > current_date.
   - If the date ≤ current_date then it's must be likely an MFD.
   - If an MFD is present, require EXP > MFD and MFD ≤ current_date.
   - DO NOT RETURN MFD as EXP.
   6) **Confidence scoring (0–100)**:
   - Batch: label presence (+20), clarity/contrast (+20), no corrections needed (+20), proximity to EXP (+10), single unambiguous candidate (+30).
   - EXP: explicit label (+25), unambiguous format (+25), future vs. current_date (+25), no disambiguation needed (+25).
   - Adjust downward for ambiguity, corrections, or past EXP.
7) **Reason field** (short, factual):
   - Quote the original strings you saw (e.g., EXP '23 10 27').
   - State how you interpreted them (e.g., DD MM YY → 23/10/2027).
   - Note MFD only if it helped disambiguate.
   - Mention ignored info if present (SN, GTIN, etc.).
   - If multiple items existed, note which one you chose and why.

# FINAL GATE (Prevents “found in reasoning but missing in output”)
Before you respond:
- If your reason mentions a concrete batch or date, ensure those exact normalized values appear in the JSON fields.
- If you are unsure about a value, remove it from the reason OR set the JSON field to null with a low confidence (but do not claim the value in reason).
- Do not output anything except the JSON object.

Few-Shot Examples:
{fewshot_examples}
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
            prompt_template=prompt_bn_5,   # default system prompt
            output_model=BatchSchema,
            use_structured_output=True,        # raw JSON expected
            # enable_caching=True,
            # enable_timing=True,
            # gcs_client=gcs_client,
            # gcs_bucket=gcs_bucket,
            **kwargs,
        )

    async def __call__(self, state: State_ax):
        """
        Run the BN/ED extraction process given a state.
        """
        try:
            fewshot_examples = """
            USER:
            Here is an image and the current date. Please extract the Batch Number and Expiration Date.
            Current date: 2024-03-10
            [Imagine an image here showing a product with "LOT AB789", "MFD 05.02.22", "EXP 05.02.26"]

            ASSISTANT:
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
            {{{{
            "batch_number": "AB789",
            "batch_number_conf_score": 0.98,
            "expired_date": "02/05/2026",
            "expired_date_conf_score": 0.97,
            "reason": "The batch number 'AB789' found near 'LOT'. MFD string '05.02.22' and EXP string '05.02.26' were identified. They were interpreted as DD.MM.YY. For EXP '05.02.26', this results in Day=05, Month=02, Year=2026. For MFD '05.02.22', this results in Day=05, Month=02, Year=2022. The EXP 2026-02-05 is future and after MFD. The final `exp_date` is formatted to "02/05/2026" using these interpreted components."
            }}}}
            """
            self.rebind_prompt_variable(
                current_date=datetime.now().strftime('%y-%m-%d'),
                fewshot_examples=fewshot_examples,
            )

            # Identify relevant images
            bned_image = state["primary"]["batch_number_expired_date_image"]
            # if isinstance(bned_image, int):
            #     bned_image = [bned_image]

            images_link = state["url"]
            image_bned_n = [images_link[i - 1] for i in bned_image if 1 <= i <= len(images_link)]

            # Prepare images (compressed, with user prompt text)
            content_parts, _ = await prepare_images_for_llm(
                image_bned_n,
                use_compression=True
            )

            # Build custom BN/ED chain
            # final_prompt = ChatPromptTemplate.from_messages(
            #     [("system", prompt_bn_system), MessagesPlaceholder("input")]
            # )
            # chain = final_prompt | self.llm_secondary

            # # Run chain
            # human_message = HumanMessage(content=content_parts)
            # output_original = await chain.ainvoke({"input": [human_message]})

            raw, parsed = await self.arun_chain(content_parts)
            # print(output_original)
            # Parse BN/ED JSON
            # clean_content = (
            #     output_original.content
            #     .replace("```json", "")
            #     .replace("```", "")
            # )
            # parsed = json.loads(clean_content)

            ai_msg = parsed.__dict__

            output = {"bn_ed": ai_msg}
            print("bned_output:", output)
            return output

        except Exception as e:
            print("Error terjadi pada BNED Agent")
            # if state.get("url"):
            #     self.cleanup_gcs_images(state["url"])
            raise response_error(str(e))
