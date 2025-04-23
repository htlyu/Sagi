import logging
from typing import List, Optional

from langchain_openai import ChatOpenAI
from litellm import completion, embedding


class FixedChatOpenAI(ChatOpenAI):
    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        overall_token_usage: dict = {}
        system_fingerprint = None
        for output in llm_outputs:
            if output is None:
                # Happens in streaming
                continue
            token_usage = output["token_usage"]
            if token_usage is not None:
                for k, v in token_usage.items():
                    if k in overall_token_usage:
                        overall_token_usage[k] |= v
                    else:
                        overall_token_usage[k] = v
            if system_fingerprint is None:
                system_fingerprint = output.get("system_fingerprint")
        combined = {"token_usage": overall_token_usage, "model_name": self.model_name}
        if system_fingerprint:
            combined["system_fingerprint"] = system_fingerprint
        return combined


class LLMClient:
    """Manages communication with the LLM provider."""

    OPENAI_GPT_4O = "gpt-4o"
    OPENAI_GPT_4O_MINI = "gpt-4o-mini"
    OPENAI_GPT_3_5_TURBO = "gpt-3.5-turbo"
    OPENAI_GPT_4_TURBO = "gpt-4-turbo"
    OPENAI_GPT_4 = "gpt-4"
    DEEPSEEK_DEEPSEEK_CHAT = "deepseek/deepseek-chat"
    OLLAMA_LLAMA_2 = "ollama/llama2"
    OLLAMA_LLAMA_3 = "ollama/llama3"

    LOCAL_MODEL_LIST = [OLLAMA_LLAMA_2, OLLAMA_LLAMA_3]

    def __init__(
        self, base_url: str, api_key: str, model_name: str, embedding_name: str = None
    ) -> None:
        self.base_url: str = base_url
        self.api_key: str = api_key
        self.model_name: str = model_name
        self.embedding_name: str = embedding_name
        self.remote_model_list: list[str] = [
            self.OPENAI_GPT_4O,
            self.OPENAI_GPT_4O_MINI,
            self.OPENAI_GPT_3_5_TURBO,
            self.OPENAI_GPT_4_TURBO,
            self.OPENAI_GPT_4,
            self.DEEPSEEK_DEEPSEEK_CHAT,
        ]
        # self.local_model_list: list[str] = [self.OLLAMA_LLAMA_2, self.OLLAMA_LLAMA_3]
        self.available_model_list: list[str] = (
            self.remote_model_list + self.LOCAL_MODEL_LIST
        )

    def get_response(self, messages: list[dict[str, str]]) -> str:
        if self.model_name in self.available_model_list:
            if self.model_name in [self.OLLAMA_LLAMA_2, self.OLLAMA_LLAMA_3]:
                self.model_name = self.OPENAI_GPT_4O_MINI
            try:
                # TODO: add support for LLM custom_llm_provider config
                response = completion(
                    model=self.model_name,
                    messages=messages,
                    api_base=self.base_url,
                    api_key=self.api_key,
                    custom_llm_provider="openai",
                )
                return response.choices[0].message.content
            except Exception as e:
                logging.error("Error getting LLM response: %s", str(e))
                return (
                    f"I encountered an error: {str(e)}. "
                    "Please try again or rephrase your request."
                )
        raise NotImplementedError(f"Model {self.model_name} is not supported yet")

    def get_embedding(self, text: list[str]) -> str:
        try:
            response = embedding(
                model=self.embedding_name,  # add `openai/` prefix to model so litellm knows to route to OpenAI
                input=text,
                api_base=self.base_url,  # set API Base of your Custom OpenAI Endpoint
                api_key=self.api_key,
            )
            return response["data"][0]["embedding"]
        except Exception as e:
            logging.error("Error getting LLM embeddings: %s", str(e))
            return (
                f"I encountered an error: {str(e)}. "
                "Please try again or rephrase your request."
            )

    def get_langchain_llm_model(self):
        if self.model_name in self.remote_model_list:
            return FixedChatOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model_name,
            )
        raise NotImplementedError(f"Model {self.model_name} is not supported yet")
