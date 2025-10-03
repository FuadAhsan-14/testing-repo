import json
from langchain_core.messages import HumanMessage

from core.BaseAgent import BaseAgent
from app.schemas.AgentMatchingOutputSchema import MatchingSchema


### Gemini Model
PROMPT_TEMPLATE = """
# Role & Goal
You are an expert data matching and reordering assistant, specializing in pharmaceutical product names. You will be given two lists of items in JSON format: the first one is a `source_list` and second one is a `target_list`. Your primary goal is to reorder the target_list to align with the source_list based on the following critical rules.

# Rules & Constraints
1.  IF THE LISTS HAVE THE SAME NUMBER OF ITEMS:
    *   You **MUST** create a one-to-one pairing for every single item.
    *   The output lists must have the same number of items as the input lists. No items are to be excluded.
    *   First, match the items that are clearly similar.
    *   Then, pair the remaining dissimilar items based on what is left. Your job is to complete the mapping, even if the similarity is zero.
2.  IF THE LISTS HAVE DIFFERENT NUMBERS OF ITEMS:
    *   You must only match items that have a clear and plausible similarity based on their core name and specifications.
    *   Exclude any items from the **longer list** that do not have a good match in the shorter list. The final lists should have the length of the shorter original list.

# Process / Steps
1.  PUT THE LIST RESULT ON THIS CORRESPONDING FIELD:
    *   Put matched_source_list as PO_List.
    *   Put sorted_target_list as Faktur_List.

# Examples (Optional)
### Example 1: Match items from these list of items
-   **Item List:** Two item list of pharmaceutical products
-   **Final Output:**
    ```json
    {{
        "PO_List": [
            "FEMARA2,5MG FC TAB",
            "KRYXANA 200MG TAB",
            "CALCIUM FOLINATE 50MG/5ML INJ (FERRON)",
            "VINORELBIN 10MG INJ (FERRON)"
        ],
        "Faktur_List": [
            "FEMARA FCT 2.5MG (Box/30)",
            "KRYXANA FCT 200MG (Box/21)",
            "CALCIUM FOLINATE (Box/1vL)",
            "VINORELBINE 10MG/ML SOL. INJ (Box/1vL)"
        ]
    }}
    ```
"""

# class AgentMatching(BaseAgent):
#     """An agent responsible for matching item list."""
#     def __init__(self, llm, **kwargs):
#         super().__init__(
#             llm=llm,
#             prompt_template=PROMPT_TEMPLATE,
#             output_model=MatchingSchema,
#             **kwargs
#         )

#     async def __call__(self, state):
        
#         # self.rebind_prompt_variable(
#         #     time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         # )

#         raw, parsed = await self.arun_chain(state=state)

#         return parsed

class MatchingAgentService(BaseAgent):
    """
    Agent yang bertanggung jawab untuk mencocokkan item antara dua list.
    """
    def __init__(self, llm):
        # Panggil constructor BaseAgent dengan prompt dan model output Pydantic
        super().__init__(
            llm=llm,
            prompt_template=PROMPT_TEMPLATE,
            output_model=MatchingSchema
        )

    async def execute(self, po_items: list, faktur_items: list) -> dict:
        """
        Mengeksekusi logika pencocokan.
        """
        # 1. Siapkan input untuk model dalam format yang diharapkan
        source_list_json = json.dumps({"source_list": po_items}, indent=2)
        target_list_json = json.dumps({"target_list": faktur_items}, indent=2)

        human_content = f"""
Please match the items from the following two lists.

Source List (from PO):
```json
{source_list_json}
```

Target List (from Faktur):
```json
{target_list_json}
```
"""
        agent_state = {"messages": [HumanMessage(content=human_content)]}

        # 2. Jalankan chain yang sudah dibangun di BaseAgent
        _, parsed_output = await self.arun_chain(state=agent_state)
        
        # 3. Kembalikan hasil yang sudah di-parse
        return parsed_output