import datetime
import datetime

from core.BaseAgent import BaseAgent
from app.schemas.AgentExampleOutputSchema import AgentExampleOutput

### Gemini Model
PROMPT_TEMPLATE = """
# Role & Goal
You are a ... .Your primary goal is to ...

# Persona (Optional)
(Define the language to be used in responses to the user, including tone, style, and any other relevant characteristics)

# Rules & Constraints
1.
2.
3.

# Context & Resources
-   **Current Time:** The reference datetime for all relative date calculations.
    `{time}`

# Process / Steps
1.
2.
3.
4.

# Examples (Optional)
{examples_gemini}
"""

### Claude Model
PROMPT_TEMPLATE = """ 

You are a ...
Your primary goal is to ...
Your persona is ... (Optional)
You adhere strictly to the instructions and constraints provided below.

<constraints>
You must adhere to the following rules:
1. [Rule 1, e.g., Do not mention specific competitors by name.]
2. [Rule 2, e.g., The final output must be in JSON format.]
3. [Rule 3, e.g., The summary must be under 100 words.]
</constraints>

<context>
Here is the necessary context for this task.
- The current time is: {time}
- [Any other relevant document, data, or background info can be placed here, preferably within its own tags like <document_content> or <dataset>]
</context>

<steps>
Follow these steps precisely to complete the task:
1. [Step 1]
2. [Step 2]
3. [Step 3]
</steps>

<examples> (Optional)
{examples_claude}
</examples>
"""

### GPT Model
PROMPT_TEMPLATE = """ 
# Identity & Goal
You are a ... 
Your primary objective is to ...

---

# Persona (Opsional)
Define language, tone, and style for responses here.

---

# Rules & Constraints
<rules>
-   **Rules 1:** Define Rules 1 here.
-   **Rules 2:** Define Rules 2 here.
-   **Rules 3:** Define Rules 3 here.
</rules>

---

# Instructions & Workflow
Follow this structured workflow concisely.
<workflow_steps>
1.
2.
3.
</workflow_steps>

---

# Examples (Optional)
{examples_gpt}

---

# Context & Resources
-   **Current Time:** The reference datetime for all relative date calculations.
    `{time}`
    
"""

class AgentExample(BaseAgent):
    """An agent responsible for ..."""
    def __init__(self, llm, **kwargs):
        super().__init__(
            llm=llm,
            prompt_template=PROMPT_TEMPLATE,
            output_model=AgentExampleOutput,
            **kwargs
        )

    async def __call__(self, state):
        
        examples_gemini = """
        ### Example 1: (Define task or condition here)
        -   **(Input name or User Query or Image or Anything):** (Define Input here)
        -   **Reasoning:** (Define Reasoning here).
        -   **Final Output:**
            (Define Final Output here)

        ### Example 2: Count Medicine Boxes using the right side of the drug object
        -   **Image 1:** Image 1 shows the right side of the drug
        -   **Final Output:** 3
            
        ### Example 2: xxx
        -   **Doctor Name:** dr. Dian Alhusari Sp.JP
        -   **Reasoning:** Dr. Dian Alhusari holds the specialist degree Sp.JP, which is a cardiology specialization. Therefore, the medical field of expertise is `cardiology`.
        -   **Final Output:** cardiology
        """
        
        examples_claude = """
        Here are some examples of how to perform the task correctly. Follow these patterns closely.
        <example>
        <user_query or input or anything>
        [Define input here]
        </user_query>
        <thinking> (Optional)
        [Define thinking here] (Optional)
        </thinking> (Optional)
        <final_output>
        [Define Final Output Here]
        </final_output>
        </example>

        <example>
        <user_query>
        Doctor Name: dr. Dian Alhusari Sp.JP
        </user_query>
        <thinking>
        The user provided a doctor's name with a specialist degree. The degree is "Sp.JP", which stands for Spesialis Jantung dan Pembuluh Darah. This corresponds to the field of cardiology.
        </thinking>
        <final_output>
        cardiology
        </final_output>
        </example>

        <example>
        <user_query>
        [Image of the right side of a medicine box, showing 3 boxes]
        </user_query>
        <final_output>
        3
        </final_output>
        </example>
        """
        
        examples_gpt = """
        Here are examples demonstrating the desired behavior.
        <example1>
        <user_query>
        [Define Input Here]
        </user_query>
        <assistant_reasoning> (Optional)
        [Define reasoning here] (Optional)
        </assistant_reasoning> (Optional)
        <assistant_response>
        [Define response here]
        </assistant_response>
        </example1>

        <example2>
        <user_query>
        Doctor Name: dr. Dian Alhusari Sp.JP
        </user_query>
        <assistant_reasoning>
        The user provided a doctor's name with a specialist degree. The degree is "Sp.JP", which stands for Spesialis Jantung dan Pembuluh Darah. This corresponds to the field of cardiology.
        </assistant_reasoning>
        <assistant_response>
        cardiology
        </assistant_response>
        </example2>
        """
        
        self.rebind_prompt_variable(
            # Bisa pilih salah satu untuk fewshot examplesnya
            examples_gemini=examples_gemini,
            # examples_claude=examples_claude, 
            # examples_gpt=examples_gpt,
            time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        raw, parsed = await self.arun_chain(state=state)

        return parsed
