import datetime

from core.BaseAgent import BaseAgent
from app.schemas.AgentHeaderOutputSchema import AgentHeaderOutput
from langchain_core.messages import HumanMessage

### Gemini Model
PROMPT_TEMPLATE = """
# Role & Goal
You are an expert document analysis agent. Your primary goal is to accurately extract header information from an invoice image based on a specific template for the supplier: **{supplier_name}**.

# Rules & Constraints
1.  You must extract values for the following fields: {fields_to_extract}.
2.  Strictly follow the `Rules` and `Hints` provided for each field to ensure accuracy.
3.  If a value for a field cannot be found in the image, return `null` for that field.
4.  Pay close attention to `location_keywords` to find the correct area in the document.
5.  The output must be a valid JSON object matching the requested schema.

# Steps / Process
1.  **Extract `po_number`**:
    -   **Hint:** {po_number_hint}
    -   **Location Keywords:** Look for text like `{po_number_location_keywords}`.
    -   **Rules to follow:** {po_number_rules}

2.  **Extract `invoice_number`**:
    -   **Hint:** {invoice_number_hint}
    -   **Location Keywords:** Look for text like `{invoice_number_location_keywords}`.
    -   **Rules to follow:** {invoice_number_rules}

3.  **Extract `invoice_date`**:
    -   **Hint:** {invoice_date_hint}
    -   **Location Keywords:** Look for text like `{invoice_date_location_keywords}`.
    -   **Date Format:** {invoice_date_format_hint}
    -   **Rules to follow:** {invoice_date_rules}

"""

class TypesenseHeaderAgentService(BaseAgent):
    """An agent responsible for extracting invoice header data based on a dynamic Typesense template."""
    def __init__(self, llm, **kwargs):
        super().__init__(
            llm=llm,
            prompt_template=PROMPT_TEMPLATE,
            output_model=AgentHeaderOutput,
            # agent_name="TypesenseHeaderAgentService",
            **kwargs
        )

    async def __call__(self, state):
        
        image_url = state.get("image_url")
        invoice_data = state.get("invoice_data_from_typesense")

        if not image_url:
            raise ValueError("image_url not found in state.")
        if not invoice_data:
            raise ValueError("invoice_data_from_typesense not found in state. Cannot build dynamic prompt.")

        fields_str = ", ".join(invoice_data.get("fields_to_extract", []))

        def format_rules(rules_list):
            if not rules_list:
                return "    - N/A"
            return "\n".join([f"    - {rule}" for rule in rules_list])

        # Ekstrak info untuk setiap field
        po_info = invoice_data.get("po_number_info", {})
        invoice_info = invoice_data.get("invoice_number_info", {})
        date_info = invoice_data.get("invoice_date_info", {})

        self.rebind_prompt_variable(
            supplier_name=invoice_data.get("supplier_name", "N/A"),
            fields_to_extract=fields_str,

            po_number_hint=po_info.get("hint_text", "N/A"),
            po_number_location_keywords=", ".join(po_info.get("location_keywords", [])),
            po_number_rules=format_rules(po_info.get("rules", [])),

            invoice_number_hint=invoice_info.get("hint_text", "N/A"),
            invoice_number_location_keywords=", ".join(invoice_info.get("location_keywords", [])),
            invoice_number_rules=format_rules(invoice_info.get("rules", [])),

            invoice_date_hint=date_info.get("hint_text", "N/A"),
            invoice_date_location_keywords=", ".join(date_info.get("location_keywords", [])),
            invoice_date_format_hint=date_info.get("date_format_hint", "N/A"),
            invoice_date_rules=format_rules(date_info.get("rules", []))
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
        
        print(f"TypesenseHeaderAgentService Output: {parsed}")

        return {
            "header_data": parsed
        }
