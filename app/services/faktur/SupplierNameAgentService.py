import datetime

from core.BaseAgent import BaseAgent
from langchain_core.messages import HumanMessage
from app.schemas.AgentSupplierNameOutputSchema import AgentSupplierNameOutput

### Gemini Model
PROMPT_TEMPLATE = """
# Role & Goal
You are an AI assistant specialized in analyzing invoice images. Your primary goal is to accurately extract the 'supplier_name' from the provided invoice image, focusing solely on the company that supplied the goods, not the buyer.

# Rules & Constraints
1. Extract only the supplier name, which typically appears prominently at the top of the invoice, often near the company logo or address.
2. Do not extract the buyer's name or any other information.
3. If the only company name you can identify as the supplier is "PT. SILOAM" or "PT. SILOAM International," set the supplier_name to null.
4. If no clear supplier name is identifiable, set it to null.

# Process / Steps
1. Analyze the invoice image for text elements, focusing on the top section.
2. Identify the company name associated with the supplier (e.g., near logo or address).
3. Verify it matches the criteria and is not the buyer.
4. Apply the special condition for "PT. SILOAM" or "PT. SILOAM International".
5. Output the supplier_name or null if conditions are not met.

# Examples
{fewshot_examples}
"""

class SupplierNameAgentService(BaseAgent):
    """An agent responsible for ..."""
    def __init__(self, llm, **kwargs):
        super().__init__(
            llm=llm,
            prompt_template=PROMPT_TEMPLATE,
            use_structured_output=True,
            output_model=AgentSupplierNameOutput,
            # agent_name="SupplierNameAgentService",
            **kwargs
        )
        
    async def __call__(self, state):
        
        fewshot_examples = """
        ### Example 1: Standard Supplier Extraction
        -   **Invoice Image:** An invoice with "ABC Pharmaceuticals" at the top.
        -   **Reasoning:** "ABC Pharmaceuticals" is prominently displayed at the top, associated with the supplier.
        -   **Final Output:** ABC Pharmaceuticals

        ### Example 2: Special Condition
        -   **Invoice Image:** The invoice with "PT. SILOAM International" is just the company name listed in the header.
        -   **Reasoning:** The name matches the special condition, so it must be set to null.
        -   **Final Output:** null

        ### Example 3: No Clear Supplier
        -   **Invoice Image:** An invoice with unclear or missing supplier details.
        -   **Reasoning:** No prominent supplier name identifiable.
        -   **Final Output:** null
        """
        
        self.rebind_prompt_variable(
            fewshot_examples = fewshot_examples
        )
        
        # 1. Ambil image_url dari state
        image_url = state.get("image_url")
        if not image_url:
            raise ValueError("image_url not found in state.")

        # 2. Buat konten multi-modal untuk HumanMessage
        message_content = [
            {
                "type": "text",
                "text": "Based on the provided invoice image, extract the supplier's name."
            },
            {
                "type": "image_url",
                "image_url": {"url": image_url}
            }
        ]

        raw, parsed = await self.arun_chain(input=message_content)
        
        print(parsed)

        state['supplier_name_extracted_from_llm'] = parsed.supplier_name

        return {'supplier_name_extracted_from_llm': parsed.supplier_name}
