import datetime

from core.BaseAgent import BaseAgent
from app.schemas.AgentHeaderOutputSchema import AgentHeaderOutput
from langchain_core.messages import HumanMessage

### Gemini Model
PROMPT_TEMPLATE = """
# Role & Goal
You are an AI assistant tasked with extracting **header-level information** from an image of a delivery order or invoice. Your goal is to accurately identify and retrieve key header data points (`supplier_name`, `invoice_date`, `PO_number`, `invoice_number`), then structure them into a predefined JSON format. The provided image is a scanned document, likely a delivery order (DO) or an invoice.

# Rules & Constraints
1.  **JSON Output Only:** Your output MUST be a single, valid JSON object and nothing else. Do not include any explanatory text outside the JSON object.
2.  **Null Value Rule:** If a data field's value is `null`, its value MUST be `null`.
3.  **Date Format Rule:** The `invoice_date` MUST be reformatted into a **`MM/DD/YY`** string in the final JSON output, regardless of its original format on the document.

# Process / Steps
1.  **Identify General Information**
    For each field, extract the value according to the specific rules. If not found, use `null` for both the data field.

2. **Extract Each Header Field**:    
    -   **`supplier_name`**:
        1.  Locate and extract the full name of the **main** supplying company.
        2.  Prioritize the full legal name (e.g., "PT. NAMA PERUSAHAAN UTAMA") over branch names.

    -   **`invoice_date`**:
        1.  **Condition 1: If `supplier_name` is "PT. MILLENNIUM PHARMACON INTERNATIONAL Tbk."**:
            *   The correct `invoice_date` is the value adjacent to the label **`Tgl. Jatuh Tempo`**.
            *   **DO NOT** use the date from the `Tgl` field or dates from the item table.
        2.  **Condition 2: For all other suppliers**:
            *   **Preference A:** Prioritize the date associated with labels **"Tgl Cetak"** or **"Tanggal Cetak"**.
            *   **Preference B:** Only if A is not found, use the date from a general label like **"Tanggal"** or **"Tgl."**.
        3.  After finding the date, reformat it to **`MM/DD/YY`**.

    -   **`PO_number`**:
        1.  **Condition 1: If `supplier_name` is "PT. MILLENNIUM PHARMACON INTERNATIONAL Tbk."**:
            *   Search the **bottom-left quadrant** of the document for a string in the format `PO-XX-XXXXXXXXX`.
            *   **DO NOT** extract values from `Order No.`, `No. Faktur`, `No. Ref`, or `PEF No:`.
        2.  **Condition 2: If `supplier_name` is "PT Merapi Utama Pharma"**:
            *   Locate the label **`KET:`** in the **bottom-left area** and extract the value to its right.
            *   **DO NOT** extract values from `NOMOR` or `SURAT PESANAN`.
        3.  **Condition 3: If `supplier_name` is "PT. KEBAYORAN PHARMA"**:
            *   Extract the value found under the **`Keterangan`** column header.
            *   **DO NOT** extract the value from the `SP / Order` column.
        4.  **Condition 4: For all other suppliers**:
            *   **A. Define PO Format:** A valid PO number meets at least one of these rules:
                *   Starts with (case-insensitive) `"PO-"`, `"PO."`, `"SPO-"`, or contains `"-05-"` or `"NARKOPO-"`.
                *   Is a 9-digit number where the first two digits match the last two digits of the `invoice_date`'s year (e.g., `25xxxxxxx` for year 2025).
                *   Is a 6-digit number where the first two digits match the last two digits of the `invoice_date`'s year (e.g., `25xxxx`).
                *   Is a numeric-only value with 3 to 8 digits.
                *   Starts with a 3-8 digit number, possibly followed by text (e.g., `1234 CITOO`).
                *   Contains at least one letter and is 3+ characters long (e.g., "FOC Bonus").
            *   **B. Extraction Hierarchy:**
                *   **Preference 1 (High-Confidence):** Find a value passing validation next to labels: `"PO"`, `"No.PO"`, `"PO. No"`, `"NOMOR PO"`, `"No. PO Cust"`, `"Customer PO"`. If found, stop.
                *   **Preference 2 (Medium-Confidence):** If nothing found, find a value passing validation next to labels: `"Ket"`, `"Keterangan"`, `"Note"`, `"NO SP"`, `"SP"`, `"SP / Order"`, `"Nomor Pesanan"`. If found, stop.
                *   **Preference 3 (Format-Based Scan):** As a last resort, scan the entire document for any standalone string that passes validation, excluding already identified numbers.

    -   **`invoice_number`**:
        1.  **Condition 1: If `supplier_name` is "PT ENSEVAL PUTERA MEGATRADING Tbk."**:
            *   The `invoice_number` is the value under the **`NO. DOK`** column header.
            *   **DO NOT** extract the value from the `K. DOK` or `NO. SO` columns.
        2.  **Condition 2: For all other suppliers**:
            *   **Preference A:** Search for values next to labels like "Nomor SO", "No. SO", "Faktur No", "FAKTUR", "No. Invoice", or "Sales Order".
            *   **Preference B:** If A fails, search next to generic labels like "Nomor:", "No", or "NO.".
            *   **Preference C:** If B fails, look for a prominent alphanumeric/numeric value in the top area of the document.
            *   **Crucial Exclusion:** The number printed directly **underneath a barcode** is **NEVER** the `invoice_number`.

3.  **Format the Output**
    Compile all extracted header information into a single JSON object.

# Examples
{fewshot_examples}
"""

class GeneralHeaderAgentService(BaseAgent):
    """An agent responsible for ..."""
    def __init__(self, llm, **kwargs):
        super().__init__(
            llm=llm,
            prompt_template=PROMPT_TEMPLATE,
            output_model=AgentHeaderOutput,
            # agent_name="GeneralHeaderAgentService",
            **kwargs
        )

    async def __call__(self, state):
        
        image_url = state.get("image_url")
        
        if not image_url:
            raise ValueError("image_url not found in state.")
        
        fewshot_examples = """
        ### Example 1: Enseval Invoice Number Extraction
        -   **Input Image Snippet:** Shows columns `K. DOK` with value `325001938` and `NO. DOK` with value `109571997`.
        -   **Reasoning:** The supplier is "PT ENSEVAL PUTERA MEGATRADING Tbk.". The rule states to extract the value from the `NO. DOK` column, not `K. DOK`.
        -   **Final Output (partial):**
            {{
                "invoice_number": "109571997"
            }}

        ### Example 2: Millennium PO Number Extraction
        -   **Input Image Snippet:** Shows a bottom-left area with the text `CCP: ... PO-05-250502237`. The main header has `Order No.: 12345`.
        -   **Reasoning:** The supplier is "PT. MILLENNIUM PHARMACON INTERNATIONAL Tbk.". The rule is to find the `PO-XX-XXXXXXXXX` format in the bottom-left area and ignore header fields like `Order No.`.
        -   **Final Output (partial):**
            {{
                "PO_number": "PO-05-250502237"
            }}

        ### Example 3: Millennium Date Extraction and Formatting
        -   **Input Image Snippet:** Shows `Tgl: 19/07/25` and `Tgl. Jatuh Tempo: 01/06/25`.
        -   **Reasoning:** The supplier is "PT. MILLENNIUM PHARMACON INTERNATIONAL Tbk.". The rule is to use the `Tgl. Jatuh Tempo` value. The original date `01/06/25` must be reformatted from DD/MM/YY to MM/DD/YY.
        -   **Final Output (partial):**
            {{
                "invoice_date": "06/01/25"
            }}

        ### Example 4: Merapi PO Number Extraction
        -   **Input Image Snippet:** Shows a bottom-left area with the label `KET:` and the value `1395 CITOO` next to it.
        -   **Reasoning:** The supplier is "PT Merapi Utama Pharma". The rule is to extract the value next to the `KET:` label in the bottom-left corner.
        -   **Final Output (partial):**
            {{
                "PO_number": "1395 CITOO"
            }}
        """
        
        self.rebind_prompt_variable(
            fewshot_examples=fewshot_examples
        )
        
        message_content = [
            {
                "type": "text",
                "text": "Based on the provided invoice image and the detailed instructions, extract the header information."
            },
            {
                "type": "image_url",
                "image_url": {"url": image_url}
            }
        ]

        raw, parsed = await self.arun_chain(input=message_content)

        return {
            "header_data": parsed
        }
