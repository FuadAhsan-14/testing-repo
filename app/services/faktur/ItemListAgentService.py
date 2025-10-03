import datetime

from core.BaseAgent import BaseAgent
from langchain_core.messages import HumanMessage
from app.schemas.AgentItemListOutputSchema import AgentItemListOutput

### Gemini Model
PROMPT_TEMPLATE = """
# Role & Goal
You are an AI assistant specialized in analyzing invoice images to extract a list of items. Your goal is to accurately identify each item, its quantity, unit of measure (UOM), batch number, and expiration date.

# Rules & Constraints
1.  Extract all line items from the invoice table.
2.  For each item, you must extract: `item_name`, `quantity`, `UOM`, `batch_number`, and `expired_date`.
3.  If a value for a specific field (e.g., `batch_number`) is not present for an item, set it to `null`.
4.  Format the `expired_date` as `YYYY-MM-DD`. If the date is ambiguous or missing, set it to `null`.
5.  The final output must be a JSON object with a single key "item_list" containing a list of item objects.

# Process / Steps
1.  Scan the invoice image to locate the item table section.
2.  Iterate through each row of the table.
3.  For each row, identify and extract the required fields: `item_name`, `quantity`, `UOM`, `batch_number`, `expired_date`.
4.  Assemble the extracted data into a list of JSON objects.
5.  Wrap the list within a final JSON object under the "item_list" key.

# Examples
{fewshot_examples}
"""

class ItemListAgentService(BaseAgent):
    """An agent responsible for ..."""
    def __init__(self, llm, **kwargs):
        super().__init__(
            llm=llm,
            prompt_template=PROMPT_TEMPLATE,
            use_structured_output=True,
            output_model=AgentItemListOutput,
            # agent_name="ItemListAgentService",
            **kwargs
        )

    async def __call__(self, state):
        
        fewshot_examples = """
        ### Example 1: Complete Item
        -   **Invoice Image Row:** "Item A | 10 | BOX | BN123 | 2025-12-31"
        -   **Reasoning:** All fields are present and clearly identifiable.
        -   **Final Output (for one item):**
            {
                "item_name": "Item A",
                "quantity": 10,
                "UOM": "BOX",
                "batch_number": "BN123",
                "expired_date": "2025-12-31"
            }

        ### Example 2: Missing Batch Number
        -   **Invoice Image Row:** "Item B | 5 | PCS | | 2026-01-15"
        -   **Reasoning:** The batch number is missing from the invoice row.
        -   **Final Output (for one item):**
            {
                "item_name": "Item B",
                "quantity": 5,
                "UOM": "PCS",
                "batch_number": null,
                "expired_date": "2026-01-15"
            }
        """
        
        self.rebind_prompt_variable(
            fewshot_examples=fewshot_examples
        )
        
        image_url = state.get("image_url")
        if not image_url:
            raise ValueError("image_url not found in state for ItemListAgent.")

        message_content = [
            {
                "type": "text",
                "text": "Based on the provided invoice image, extract all line items according to the specified format."
            },
            {
                "type": "image_url",
                "image_url": {"url": image_url}
            }
        ]
        
        raw, parsed = await self.arun_chain(input=message_content)
        
        print(parsed)

        state['item_list_data'] = parsed.model_dump() if parsed else {}

        return {"item_list_data": parsed.model_dump() if parsed else {}}
