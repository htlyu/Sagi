import asyncio
from typing import Any, Callable, List, Mapping

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.base import ChatAgent, TerminationCondition
from autogen_agentchat.messages import (
    AgentEvent,
    ChatMessage,
)
from autogen_agentchat.state import TeamState
from autogen_agentchat.teams import BaseGroupChat
from autogen_agentchat.teams._group_chat._chat_agent_container import ChatAgentContainer
from autogen_agentchat.teams._group_chat._events import (
    GroupChatTermination,
)
from autogen_agentchat.teams._group_chat._magentic_one._prompts import (
    ORCHESTRATOR_FINAL_ANSWER_PROMPT,
)
from autogen_core import (
    AgentRuntime,
    AgentType,
    Component,
    ComponentModel,
    TypeSubscription,
)
from autogen_core.models import (
    ChatCompletionClient,
)
from pydantic import BaseModel, Field
from typing_extensions import Self

from workflows.planning_orchestrator import PlanningOrchestrator


class PlanningChatState(TeamState):
    """State for a team of agents."""

    agent_states: Mapping[str, Any] = Field(default_factory=dict)
    type: str = Field(default="PlanningChatState")


class PlanningGroupChatConfig(BaseModel):
    """The declarative configuration for a PlanningGroupChat."""

    participants: List[ComponentModel]
    model_client: ComponentModel
    termination_condition: ComponentModel | None = None
    max_turns: int | None = None
    max_stalls: int
    final_answer_prompt: str
    planning_model_client: ComponentModel  # for json plan output
    reflection_model_client: ComponentModel  # for reflection
    user_proxy: ComponentModel | None = None  # for new feat: human in the loop
    domain_specific_agent: ComponentModel | None = (
        None  # for new feat: domain specific prompt
    )
    group_chat_manager_topic_type: str = None


class PlanningGroupChat(BaseGroupChat, Component[PlanningGroupChatConfig]):
    component_config_schema = PlanningGroupChatConfig
    component_provider_override = "workflows.PlanningGroupChat"

    def __init__(
        self,
        participants: List[ChatAgent],
        model_client: ChatCompletionClient,
        *,
        termination_condition: TerminationCondition | None = None,
        max_turns: int | None = None,
        runtime: AgentRuntime | None = None,
        max_stalls: int = 100,  # avoiding modifying the original plan
        final_answer_prompt: str = ORCHESTRATOR_FINAL_ANSWER_PROMPT,
        planning_model_client: ChatCompletionClient,  # for json plan output
        reflection_model_client: ChatCompletionClient,  # for reflection
        user_proxy: UserProxyAgent | None = None,  # for new feat: human in the loop
        domain_specific_agent: (
            AssistantAgent | None
        ) = None,  # for new feat: domain specific prompt
    ):
        super().__init__(
            participants,
            group_chat_manager_name="PlanningOrchestrator",
            group_chat_manager_class=PlanningOrchestrator,
            termination_condition=termination_condition,
            max_turns=max_turns,
            runtime=runtime,
        )

        # Validate the participants.
        if len(participants) == 0:
            raise ValueError(
                "At least one participant is required for MagenticOneGroupChat."
            )
        self._model_client = model_client
        self._max_stalls = max_stalls
        self._final_answer_prompt = final_answer_prompt
        self._planning_model_client = planning_model_client  # for json plan output
        self._reflection_model_client = (
            reflection_model_client  # for new feat: reflection
        )
        self._user_proxy = user_proxy  # for new feat: human in the loop
        self._domain_specific_agent = (
            domain_specific_agent  # for new feat: domain specific prompt
        )

    async def _init(self, runtime: AgentRuntime) -> None:
        # Constants for the group chat manager.
        group_chat_manager_agent_type = AgentType(self._group_chat_manager_topic_type)

        # Register participants.
        # Use the participant topic type as the agent type.
        for participant, agent_type in zip(
            self._participants, self._participant_topic_types, strict=True
        ):
            # Register the participant factory.
            await ChatAgentContainer.register(
                runtime,
                type=agent_type,
                factory=self._create_participant_factory(
                    self._group_topic_type,
                    self._output_topic_type,
                    participant,
                    self._message_factory,
                ),
            )
            # Add subscriptions for the participant.
            # The participant should be able to receive messages from its own topic.
            await runtime.add_subscription(
                TypeSubscription(topic_type=agent_type, agent_type=agent_type)
            )
            # The participant should be able to receive messages from the group topic.
            await runtime.add_subscription(
                TypeSubscription(
                    topic_type=self._group_topic_type, agent_type=agent_type
                )
            )

        # Register the group chat manager.
        await self._base_group_chat_manager_class.register(
            runtime,
            type=group_chat_manager_agent_type.type,
            factory=self._create_group_chat_manager_factory(
                name=self._group_chat_manager_name,
                group_topic_type=self._group_topic_type,
                output_topic_type=self._output_topic_type,
                group_chat_manager_topic_type=self._group_chat_manager_topic_type,
                participant_names=self._participant_names,
                participant_topic_types=self._participant_topic_types,
                participant_descriptions=self._participant_descriptions,
                output_message_queue=self._output_message_queue,
                termination_condition=self._termination_condition,
                max_turns=self._max_turns,
            ),
        )
        # Add subscriptions for the group chat manager.
        # The group chat manager should be able to receive messages from the its own topic.
        await runtime.add_subscription(
            TypeSubscription(
                topic_type=self._group_chat_manager_topic_type,
                agent_type=group_chat_manager_agent_type.type,
            )
        )
        # The group chat manager should be able to receive messages from the group topic.
        await runtime.add_subscription(
            TypeSubscription(
                topic_type=self._group_topic_type,
                agent_type=group_chat_manager_agent_type.type,
            )
        )
        # The group chat manager will relay the messages from output topic to the output message queue.
        await runtime.add_subscription(
            TypeSubscription(
                topic_type=self._output_topic_type,
                agent_type=group_chat_manager_agent_type.type,
            )
        )

        self._initialized = True

    def _create_group_chat_manager_factory(
        self,
        name: str,
        group_topic_type: str,
        output_topic_type: str,
        group_chat_manager_topic_type: str,
        participant_topic_types: List[str],
        participant_names: List[str],
        participant_descriptions: List[str],
        output_message_queue: asyncio.Queue[
            AgentEvent | ChatMessage | GroupChatTermination
        ],
        termination_condition: TerminationCondition | None,
        max_turns: int | None,
    ) -> Callable[[], PlanningOrchestrator]:
        return lambda: PlanningOrchestrator(
            name,
            group_topic_type,
            output_topic_type,
            group_chat_manager_topic_type,
            participant_topic_types,
            participant_names,
            participant_descriptions,
            max_turns,
            self._message_factory,
            self._model_client,
            self._max_stalls,
            self._final_answer_prompt,
            output_message_queue,
            termination_condition,
            self._emit_team_events,
            self._planning_model_client,
            self._reflection_model_client,  # for new feat: reflection
            self._user_proxy,  # for new feat: human in the loop
            self._domain_specific_agent,  # for new feat: domain specific prompt
        )

    def _to_config(self) -> PlanningGroupChatConfig:
        participants = [
            participant.dump_component() for participant in self._participants
        ]
        termination_condition = (
            self._termination_condition.dump_component()
            if self._termination_condition
            else None
        )
        user_proxy = (
            self._user_proxy.dump_component() if self._user_proxy else None
        )  # for new feat: human in the loop
        domain_specific_agent = (
            self._domain_specific_agent.dump_component()
            if self._domain_specific_agent
            else None
        )  # for new feat: domain specific prompt
        planning_model_client = (
            self._planning_model_client.dump_component()
            if self._planning_model_client
            else None
        )
        reflection_model_client = (
            self._reflection_model_client.dump_component()
            if self._reflection_model_client
            else None
        )
        return PlanningGroupChatConfig(
            participants=participants,
            model_client=self._model_client.dump_component(),
            termination_condition=termination_condition,
            max_turns=self._max_turns,
            max_stalls=self._max_stalls,
            final_answer_prompt=self._final_answer_prompt,
            planning_model_client=planning_model_client,
            reflection_model_client=reflection_model_client,  # for new feat: reflection
            user_proxy=user_proxy,  # for new feat: human in the loop
            domain_specific_agent=domain_specific_agent,  # for new feat: domain specific prompt
            group_chat_manager_topic_type=self._group_chat_manager_topic_type,
        )

    @classmethod
    def _from_config(cls, config: PlanningGroupChatConfig) -> Self:
        participants = [
            ChatAgent.load_component(participant) for participant in config.participants
        ]
        model_client = ChatCompletionClient.load_component(config.model_client)
        termination_condition = (
            TerminationCondition.load_component(config.termination_condition)
            if config.termination_condition
            else None
        )
        planning_model_client = (
            ChatCompletionClient.load_component(config.planning_model_client)
            if config.planning_model_client
            else None
        )
        reflection_model_client = (
            ChatCompletionClient.load_component(config.reflection_model_client)
            if config.reflection_model_client
            else None
        )
        user_proxy = (
            UserProxyAgent.load_component(config.user_proxy)
            if config.user_proxy
            else None
        )
        domain_specific_agent = (
            AssistantAgent.load_component(config.domain_specific_agent)
            if config.domain_specific_agent
            else None
        )
        return cls(
            participants,
            model_client,
            termination_condition=termination_condition,
            max_turns=config.max_turns,
            max_stalls=config.max_stalls,
            final_answer_prompt=config.final_answer_prompt,
            planning_model_client=planning_model_client,
            reflection_model_client=reflection_model_client,
            user_proxy=user_proxy,
            domain_specific_agent=domain_specific_agent,
            group_chat_manager_topic_type=config.group_chat_manager_topic_type,
        )

    async def save_state(self) -> Mapping[str, Any]:
        base_state = await super().save_state()
        state = PlanningChatState(
            agent_states=base_state["agent_states"],
        )
        return state.model_dump()
