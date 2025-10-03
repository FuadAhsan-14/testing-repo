r"""
Descriptions    : This code defines a versatile BaseAgent class using LangChain. It serves as a foundational component for creating sophisticated language model agents that can be configured with specific prompts, a set of tools for action, and structured output models for predictable responses.

Objective       : The primary objective is to create a reusable, modular, and extensible agent framework. It abstracts the common patterns of initializing LangChain components, managing prompt templates, handling tools, and executing different agentic logics like simple chains and the ReAct framework. This allows developers to quickly build and customize various agents by focusing on high-level logic (prompts and tools) rather than low-level implementation details.

Functionallity  : This code defines a base agent class for a language model-based agent system. It includes methods for initializing the agent, creating a default prompt template, parsing LLM outputs, executing tools, and managing the agent's state. The agent can perform actions based on user input and intermediate steps, allowing for complex workflows.
"""

from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser, JsonOutputToolsParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain.output_parsers import OutputFixingParser
from langchain.agents import create_react_agent, AgentExecutor
from sqlalchemy.engine import Engine
from typing import List, Any, Optional
from pydantic import BaseModel
from langchain_community.callbacks import get_openai_callback
from app.utils.Helper import compute_usage

REACT_TEMPLATE = """You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Agent Scratchpad: {agent_scratchpad}
Question: {input}
Thought:"""

class BaseAgent:
    """
    A foundational agent class for both single-chain and LangGraph implementations.

    This agent can be configured with a language model, a set of tools, and a prompt
    template. It is designed to be extensible for more complex workflows.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        prompt_template: str = "",
        output_model: Optional[BaseModel] = None,
        tools: Optional[List[BaseTool]] = None,
        use_structured_output: bool = False,
        db: Engine = None,
        max_retries: int = 3, 
        retry_delay: float = 1.0
    ):
        """
        Initializes the BaseAgent.

        Args:
            llm: The language model to be used by the agent.
            tools: An optional list of tools that the agent can use.
            prompt_template: An optional prompt template for the agent. If not
                provided, a default template will be used.
        """
        self.llm = llm
        self.raw_prompt = prompt_template
        self.output_model = output_model
        self.tools = tools or []
        self.db = db
        self.parser = JsonOutputParser(pydantic_object=self.output_model) if self.output_model else StrOutputParser()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.use_structured_output = use_structured_output
        if self.use_structured_output and not self.output_model:
            raise ValueError("output_model must be provided when use_structured_output is True.")
        if self.use_structured_output and self.tools:
            raise ValueError("tools cannot be used when use_structured_output is True.")

        self._setup_prompt_template()
        self._rebuild_chains()

    def _rebuild_chains(self):
        """Rebuilds the chains after modifying tools or prompt."""
        self.chain = (
            self.prompt 
            | self.llm
            | RunnableParallel(
                raw=RunnablePassthrough(),
                parsed=OutputFixingParser.from_llm(self.llm, self.parser)
            )
        ) 
        if self.tools:
            self.chain = (
                self.prompt
                | self.llm.bind_tools(self.tools)
                | RunnableParallel(
                    raw=RunnablePassthrough(),
                    parsed=JsonOutputToolsParser()
                )
            )
        if self.use_structured_output:
            self.chain = self.prompt | self.llm.with_structured_output(self.output_model)

    def _setup_prompt_template(self) -> ChatPromptTemplate:
        additional_template = """\n{parser}"""
        system_prompt = "\n".join([self.raw_prompt, additional_template])
        self.prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder(variable_name="messages", optional=True),
                ]
        )
        if self.output_model and not self.use_structured_output:
            self.prompt = self.prompt.partial(parser=self.parser.get_format_instructions())
        else:
            self.prompt = self.prompt.partial(parser="")
        
        react_system_prompt = "\n".join([self.raw_prompt, REACT_TEMPLATE])
        self.react_prompt = ChatPromptTemplate.from_template(react_system_prompt)
        
    def _init_react_agent(self,  **kwargs):
        if not self.tools:
            raise ValueError("Initialize the agent with tools parameter when using react agent.")
        return AgentExecutor(
            agent=create_react_agent(self.llm, self.tools, self.react_prompt),
            tools=self.tools,
            handle_parsing_errors=True,
            **kwargs
        )
        
    def _prepare_inputs(
        self,
        input: str,
        **kwargs: Any
    ) -> tuple:
        """
        Private helper to validate inputs, select the correct chain, and prepare
        the arguments for invocation or streaming.
        """
        
        invoke_kwargs = kwargs.copy()
        if "state" in invoke_kwargs:
            if "messages" in invoke_kwargs.get("state"):
                msg_state = invoke_kwargs["state"]
        else:
            if not input:
                raise ValueError("Either 'input' or 'state messages' must be provided.")
            msg_state = {"messages": [HumanMessage(content=input)]}
            
        final_input = msg_state or input
        return final_input
    
    def rebind_prompt_variable(self, **variables: Any):
        self.prompt = self.prompt.partial(**variables)
        self._rebuild_chains()
        
    def rebind_react_prompt_variable(self, **variables: Any):
        self.react_prompt = self.react_prompt.partial(**variables)
        self._rebuild_chains()

    def add_tool(self, tool: BaseTool):
        """Dynamically add a tool to the agent"""
        self.tools.append(tool)
        self._rebuild_chains()
    
    def remove_tool(self, tool_name: str):
        """Remove a tool by name"""
        self.tools = [t for t in self.tools if t.name != tool_name]
        self._rebuild_chains()

    def filter_tools(self, tool_names: List[str]):
        """Filter tools by a list of names"""
        self.tools = [t for t in self.tools if t.name in tool_names]
        self._rebuild_chains()
    
    def run_chain(self, input: str = "", **kwargs: Any):
        """Invokes the chain synchronously and returns a single response."""
        invoke_kwargs = self._prepare_inputs(input, **kwargs)
        response = self.chain.invoke(invoke_kwargs)
        if self.use_structured_output:
            return AIMessage(content=response.model_dump_json()), response
        return response['raw'], response['parsed']
        
    async def arun_chain(self, input: str = "", **kwargs: Any):
        """Invokes the chain asynchronously and returns a single response."""
        invoke_kwargs = self._prepare_inputs(input, **kwargs)
        response = await self.chain.ainvoke(invoke_kwargs)

        if self.use_structured_output:
            return AIMessage(content=response.model_dump_json()), response
        return response['raw'], response['parsed']
        
    def run_react_agent(self, input: str = "",**kwargs):
        """Runs the react agent synchronously and returns a response."""
        response = self._init_react_agent(**kwargs).invoke({"input": input})
        return response, response.get("output", response)
    
    async def arun_react_agent(self, input: str = "",**kwargs): 
        """Runs the react agent asynchronously and returns a response."""
        response = await self._init_react_agent(**kwargs).ainvoke({"input": input})
        return response, response.get("output", response)
    
if __name__ == "__main__":
    from app.generative import manager
    from pydantic import BaseModel, Field
    class SQLQuery(BaseModel):
        query: str = Field(..., description="The SQL query to execute.")
        explanation: str = Field(..., description="A brief explanation of the query.")
    PROMPT = """You are an expert in converting natural language to SQL queries. {table_name}"""
    agent = BaseAgent(llm=manager.gemini_mini(), prompt_template=PROMPT, use_structured_output=True, output_model=SQLQuery)
    agent.rebind_prompt_variable(table_name="Table: users(id, name, email)")
    raw, parsed = agent.run_chain(input="Get me the names of all users.")
    raw.pretty_print()
    print(parsed)