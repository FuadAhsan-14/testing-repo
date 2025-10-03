import json
from functools import partial
from app.generative.engine import GenAI
from config.setting import env
from langchain_core.language_models.chat_models import BaseChatModel

CONFIG = {
    # "gemini_model_flash_2_exp": {
    #     "creator_method": "chatGgenai",
    #     "params": {"model": env.gemini_model},
    # },
    # "gemini_model_pro15": {
    #     "creator_method": "chatGgenai",
    #     "params": {"model": env.gemini_pro_model},
    # },
    "gemini_model_pro25": {
        "creator_method": "chatGgenai",
        "params": {"model": env.gemini_model_pro25},
    },
    "gemini_model_flash2": {
        "creator_method": "chatGgenai",
        "params": {"model": env.gemini_model_flash2},
    },
    "gemini_model_flash25": {
        "creator_method": "chatGgenai",
        "params": {"model": env.gemini_model_flash25},
    },
    "gemini_model_flash_lite25": {
        "creator_method": "chatGgenai",
        "params": {"model": env.gemini_model_flash_lite25},
    },
    "claude_model_sonnet_35": {
        "creator_method": "chatBedrock",
        "params": {"model": env.claude_model_sonnet_35},
    },
    "claude_model_sonnet_37": {
        "creator_method": "chatBedrock",
        "params": {"model": env.claude_model_sonnet_37},
    },
    "claude_model_sonnet_4": {
        "creator_method": "chatBedrock",
        "params": {"model": env.claude_model_sonnet_4},
    },
    "openai_regular": {
        "creator_method": "chatAzureOpenAi",
        "params": {"model": env.openai_regular_model},
    },
    "openai_mini": {
        "creator_method": "chatAzureOpenAi",
        "params": {"model": env.openai_mini_model, "deployment":"002"},
    },
    "openai_model_gpt_mini_41": {
        "creator_method": "chatAzureOpenAi",
        "params": {"model": "gpt-4.1-mini-003", "deployment": "003"}
    },
    "openai_model_gpt_mini_5": {
        "creator_method": "chatAzureOpenAi",
        "params": {"model": env.gpt_5_mini_model, "deployment": "dev"}
    },
    "openai_model_gpt_mini_4o": {
        "creator_method": "chatAzureOpenAi",
        "params": {"model": env.gpt_4o_mini_model, "deployment": "002"}
    },
    # "openai_model_gpt_chat_5": {
    #     "creator_method": "chatAzureOpenAi",
    #     "params": {"model": env.azure_deployment_name, "deployment": "dev"}
    # }
}

class LLMManager:
    with open('app/generative/default.json', 'r') as f:
        DEFAULTS = json.load(f)
        
    def __init__(self):
        self._llms = {}
        self.gen_ai = GenAI()
        
        self.llm_configs = CONFIG
        for name, config in self.llm_configs.items():
            if name in self.DEFAULTS and "default_params" in self.DEFAULTS[name]:
                config["default_params"] = self.DEFAULTS[name]["default_params"]

    def _get_llm(self, name: str, **override_params):
        if name in self._llms:
            return self._llms[name]

        config = self.llm_configs.get(name)
        if not config:
            raise AttributeError(f"No LLM named '{name}' is configured.")

        if config["params"].get("model"):
            base_params = config["params"].copy()
        else:
            base_params = config["default_params"].copy()

        base_params.update(override_params)
        final_params = base_params
        
        creator_method = getattr(self.gen_ai, config["creator_method"])
        
        llm_instance = creator_method(**final_params)
        self._llms[name] = llm_instance
        return llm_instance

    def __getattr__(self, name: str) -> BaseChatModel:
        if name in self.llm_configs:
            return partial(self._get_llm, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

manager = LLMManager()
