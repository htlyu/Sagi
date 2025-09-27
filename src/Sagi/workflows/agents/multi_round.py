from typing import Any, Awaitable, Callable, Dict, List, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.models import ChatCompletionClient
from autogen_core.tools import BaseTool
from resources.model_client_wrapper import ModelClientWrapper

from Sagi.tools.pdf_extraction.pdf_extraction_tool import PDFExtractionTool
from Sagi.tools.web_search_agent import WebSearchAgent
from Sagi.utils.prompt import (
    get_multi_round_agent_system_prompt,
    get_web_search_summary_prompt,
)
from Sagi.workflows.sagi_memory import SagiMemory


class MultiRoundWebSearchAgentWorkflow:
    @classmethod
    def create(
        cls,
        model_name: str,
        mcp_tools: Dict[
            str,
            List[
                BaseTool[Any, Any] | Callable[..., Any] | Callable[..., Awaitable[Any]]
            ],
        ],
        language: str,
        web_search: bool = True,
        pdf_extraction: bool = True,
    ):
        self = cls()
        self.model_client = ModelClientWrapper(model_name, None)
        self.language = language
        self.participant_list = []

        # 1. Web search agent - responsible for searching web content and finding PDF links
        if web_search:
            web_search_agent: WebSearchAgent = WebSearchAgent(
                name="web_search_agent",
                model_client=self.model_client,
                tools=mcp_tools["web_search"],
                model_client_stream=True,
                max_retries=3,
            )
            self.participant_list.append(web_search_agent)

        # 2. PDF extraction agent - responsible for parsing PDF content
        if pdf_extraction:
            pdf_extraction_agent: AssistantAgent = AssistantAgent(
                name="pdf_extraction_agent",
                model_client=self.model_client,
                tools=[PDFExtractionTool()],
                model_client_stream=False,
                system_message="You are a PDF extraction agent. Use the pdf_extractor tool to extract content from PDF URLs provided by other agents.",
            )
            self.participant_list.append(pdf_extraction_agent)

        # 3. Summary agent - integrates all results and returns final answer
        web_search_summary_agent: AssistantAgent = AssistantAgent(
            name="web_search_summary_agent",
            model_client=self.model_client,
            model_client_stream=True,
            system_message=get_web_search_summary_prompt(language),
        )
        self.participant_list.append(web_search_summary_agent)

        return self

    def set_model(self, model_name: str) -> None:
        self.model_client.set_model_client(model_name=model_name)

    def reset(self):
        pass

    async def cleanup(self):
        if hasattr(self, "session_manager"):
            await self.session_manager.close_all()


class MultiRoundAgent:
    agent: AssistantAgent
    language: str
    memory: SagiMemory

    def __init__(
        self,
        model_client: ChatCompletionClient,
        memory: SagiMemory,
        language: str,
        model_client_stream: bool = True,
        markdown_output: bool = False,
        enable_web_search: bool = False,
        mcp_tools: Optional[Dict[str, List[Any]]] = None,
        model_name: Optional[str] = None,
    ):

        self.memory = memory
        self.language = language
        self.enable_web_search = enable_web_search
        self.mcp_tools = mcp_tools
        self.model_name = model_name

        system_prompt = self._get_system_prompt(markdown_output)
        self.agent = AssistantAgent(
            name="multi_round_agent",
            model_client=model_client,
            model_client_stream=model_client_stream,
            memory=[memory],
            system_message=system_prompt,
        )

    def _get_system_prompt(self, markdown_output=False):
        lang_prompt = {
            "en": "You are a helpful assistant that can answer questions and help with tasks. Please use English to answer.",
            "cn-s": "你是一个乐于助人的助手, 可以回答问题并帮助完成任务。请用简体中文回答",
            "cn-t": "你是一個樂於助人的助手, 可以回答問題並幫助完成任務。請用繁體中文回答",
        }
        if markdown_output:
            markdown_prompt = get_multi_round_agent_system_prompt()
            return markdown_prompt.get(self.language, markdown_prompt["en"])
        return lang_prompt.get(self.language, lang_prompt["en"])

    def run_workflow(
        self,
        user_input: str,
        experimental_attachments: Optional[List[Dict[str, str]]] = None,
    ):
        # TODO(klma): handle the case of experimental_attachments
        if self.enable_web_search and self.mcp_tools:
            if not self.model_name:
                raise ValueError("model_name is required for web search workflow")

            search_workflow = MultiRoundWebSearchAgentWorkflow.create(
                model_name=self.model_name,
                mcp_tools=self.mcp_tools,
                language=self.language,
                web_search=True,
                pdf_extraction=True,
            )

            search_workflow.participant_list.append(self.agent)

            team = RoundRobinGroupChat(
                participants=search_workflow.participant_list,
                termination_condition=TextMessageTermination(
                    source="multi_round_agent"
                ),
            )

            return team.run_stream(task=user_input)
        else:
            return self.agent.run_stream(task=user_input)

    async def cleanup(self):
        pass
