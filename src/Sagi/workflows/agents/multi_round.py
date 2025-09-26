import re
from typing import Any, Awaitable, Callable, Dict, List, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMessageTermination
from autogen_agentchat.messages import (
    ModelClientStreamingChunkEvent,
    TextMessage,
    ToolCallSummaryMessage,
)
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient
from autogen_core.tools import BaseTool

from Sagi.utils.prompt import (
    get_multi_round_agent_base_prompt,
    get_multi_round_agent_system_prompt,
    get_multi_round_agent_web_search_prompt,
)
from Sagi.workflows.agents.search_result_analysis_agent import SearchResultAnalysisAgent
from Sagi.workflows.sagi_memory import SagiMemory


class MultiRoundAgent:
    agent: AssistantAgent
    language: str
    memory: SagiMemory
    search_analyzer: Optional[SearchResultAnalysisAgent]
    team: RoundRobinGroupChat

    def __init__(
        self,
        model_client: ChatCompletionClient,
        memory: SagiMemory,
        language: str,
        model_client_stream: bool = True,
        markdown_output: bool = False,
        tools: Optional[
            List[
                BaseTool[Any, Any] | Callable[..., Any] | Callable[..., Awaitable[Any]]
            ]
        ] = None,
    ):
        self.memory = memory
        self.language = language

        has_web_search = self._has_web_search_tools(tools)

        if markdown_output:
            if has_web_search:
                system_prompt = self._get_markdown_web_search_system_prompt()
                self.search_analyzer = SearchResultAnalysisAgent(
                    name="search_result_analyzer",
                    model_client=model_client,
                    language=language,
                    model_client_stream=True,
                )
            else:
                system_prompt = self._get_markdown_system_prompt()
                self.search_analyzer = None
        else:
            if has_web_search:
                system_prompt = self._get_web_search_system_prompt(tools)
                self.search_analyzer = SearchResultAnalysisAgent(
                    name="search_result_analyzer",
                    model_client=model_client,
                    language=language,
                    model_client_stream=True,
                )
            else:
                system_prompt = self._get_system_prompt()
                self.search_analyzer = None
        self.agent = AssistantAgent(
            name="multi_round_agent",
            model_client=model_client,
            model_client_stream=model_client_stream,
            memory=[memory],
            system_message=system_prompt,
            tools=tools,
            tool_call_summary_format="{result}",
        )

        self.team = RoundRobinGroupChat(
            participants=[self.agent],
            termination_condition=TextMessageTermination("multi_round_agent"),
        )

    def _get_system_prompt(self):
        return get_multi_round_agent_base_prompt(self.language)

    def _get_markdown_system_prompt(self):
        markdown_prompt = get_multi_round_agent_system_prompt()
        return markdown_prompt.get(self.language, markdown_prompt["en"])

    def _get_web_search_system_prompt(self, tools):
        base_prompt = self._get_system_prompt()
        has_pdf_tools = self._has_pdf_tools(tools)
        web_search_addition = get_multi_round_agent_web_search_prompt(
            self.language, has_pdf_tools
        )
        return base_prompt + web_search_addition

    def _get_markdown_web_search_system_prompt(self):
        base_prompt = self._get_markdown_system_prompt()
        web_search_addition = {
            "en": " You have access to web search tools to find current information when needed.",
            "cn-s": " 你可以使用网络搜索工具来查找所需的最新信息。",
            "cn-t": " 你可以使用網路搜尋工具來查找所需的最新資訊。",
        }
        addition = web_search_addition.get(self.language, web_search_addition["en"])
        return base_prompt + addition

    def _has_web_search_tools(self, tools):
        if not tools:
            return False

        for tool in tools:
            tool_name = getattr(tool, "name", "") or getattr(tool, "__name__", "")
            if "search" in tool_name.lower() or "brave" in tool_name.lower():
                return True
        return False

    def _has_pdf_tools(self, tools):
        if not tools:
            return False

        for tool in tools:
            tool_name = getattr(tool, "name", "") or getattr(tool, "__name__", "")
            if "pdf" in tool_name.lower() or "extractor" in tool_name.lower():
                return True
        return False

    def run_workflow(
        self,
        user_input: str,
        experimental_attachments: Optional[List[Dict[str, str]]] = None,
    ):
        # TODO(klma): handle the case of experimental_attachments
        if self.search_analyzer is None:
            return self.team.run_stream(task=user_input)
        else:
            return self._run_workflow_with_search_analysis(user_input)

    async def _run_workflow_with_search_analysis(self, user_input: str):
        async for message in self.team.run_stream(task=user_input):
            is_search_result = self._is_web_search_result(message)

            if is_search_result:
                yield message

                search_results = self._extract_search_results(message)

                if search_results:
                    try:
                        display_text = self._format_search_results_for_display(
                            search_results
                        )

                        if display_text:
                            yield ModelClientStreamingChunkEvent(
                                content=display_text,
                                source="web_search_summary",
                            )

                        analysis_result = await self._analyze_search_results(
                            search_results
                        )

                        formatted_content = self._format_analysis_output(
                            analysis_result
                        )

                        if formatted_content:
                            yield ModelClientStreamingChunkEvent(
                                content=formatted_content,
                                source="search_result_analyzer",
                            )

                    except Exception as e:
                        yield ModelClientStreamingChunkEvent(
                            content=(
                                "Search result analysis failed: "
                                f"{str(e)}. Original results are preserved above."
                            ),
                            source="search_result_analyzer",
                        )
            else:
                yield message

    def _is_web_search_result(self, message) -> bool:
        if isinstance(message, ToolCallSummaryMessage):
            chat_message = message
        elif hasattr(message, "chat_message") and isinstance(
            message.chat_message, ToolCallSummaryMessage
        ):
            chat_message = message.chat_message
        else:
            return False

        source = getattr(chat_message, "source", "")

        if "web_search" in source.lower() or "search" in source.lower():
            return True

        content = getattr(chat_message, "content", "")

        if isinstance(content, str) and content.strip():
            url_pattern = r'https?://[^\s<>"\'{}|\\^`\[\]]+|www\.[^\s<>"\'{}|\\^`\[\]]+'
            urls_found = re.findall(url_pattern, content, re.IGNORECASE)

            if urls_found:
                structured_patterns = [
                    r"Title:\s*[^\n]+",
                    r"URL:\s*[^\n]+",
                    r"Description:\s*[^\n]+",
                    r"Snippet:\s*[^\n]+",
                ]
                pattern_matches = sum(
                    1
                    for pattern in structured_patterns
                    if re.search(pattern, content, re.IGNORECASE)
                )

                if pattern_matches >= 2:
                    return True

        return False

    def _extract_search_results(self, message) -> Optional[str]:
        if isinstance(message, ToolCallSummaryMessage):
            chat_message = message
        elif hasattr(message, "chat_message"):
            chat_message = message.chat_message
        else:
            return None

        if hasattr(chat_message, "content") and isinstance(chat_message.content, str):
            return chat_message.content

        return None

    async def _analyze_search_results(self, search_results: str) -> str:
        if self.search_analyzer is None:
            return "Analysis not available - SearchResultAnalysisAgent not initialized."

        try:
            search_message = TextMessage(
                content=search_results, source="web_search_results"
            )

            response = await self.search_analyzer.on_messages(
                [search_message], CancellationToken()
            )

            if hasattr(response, "chat_message") and hasattr(
                response.chat_message, "content"
            ):
                return response.chat_message.content
            else:
                return "Analysis could not be generated."

        except Exception as e:
            return f"Analysis failed: {str(e)}"

    def _format_analysis_output(self, analysis: str) -> str:
        separator = "\n" + "=" * 80 + "\n"
        analysis_header = "🔍 AI ANALYSIS OF SEARCH RESULTS"

        formatted_output = f"""
{separator}
{analysis_header}
{separator}

{analysis}

{separator}
NOTE: Original search results are shown above. This analysis provides additional insights and evaluation of the search findings.
{separator}
"""
        return formatted_output

    def _format_search_results_for_display(self, search_results: str) -> str:
        if not search_results:
            return ""

        entries: List[Dict[str, str]] = []
        current: Dict[str, str] = {}

        for raw_line in search_results.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            lower = line.lower()
            if lower.startswith("title:"):
                if current:
                    entries.append(current)
                    current = {}
                current["title"] = line.split(":", 1)[1].strip()
            elif lower.startswith("url:"):
                current["url"] = line.split(":", 1)[1].strip()
            elif lower.startswith("description:"):
                current["description"] = line.split(":", 1)[1].strip()
            elif lower.startswith("snippet:"):
                current["description"] = line.split(":", 1)[1].strip()

        if current:
            entries.append(current)

        if not entries:
            return search_results.strip()

        header_map = {
            "cn-s": "## 🔎 网络搜索结果",
            "cn-t": "## 🔎 網路搜尋結果",
        }
        header = header_map.get(self.language, "## 🔎 Web Search Results")

        lines: List[str] = [header]
        for index, item in enumerate(entries, 1):
            title = item.get("title") or "Untitled"
            url = item.get("url")
            description = item.get("description")

            if url:
                lines.append(f"{index}. [{title}]({url})")
            else:
                lines.append(f"{index}. {title}")

            if description:
                lines.append(f"   - {description}")

        return "\n".join(lines)

    async def cleanup(self):
        pass
