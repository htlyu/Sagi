PROMPT_TEMPLATES = {
    "DataCollector": """
Completed summaries:
{shared_section}

Current sub-task:
{step_section}

Your task:
- Using the above summaries, search for and extract only the key information relevant to "{step_content}".
- Return only those key points; do not include unrelated data.
""".strip(),
    "CodeExecutor": """
Completed summaries:
{shared_section}

Current sub-task:
{step_section}

Your task:
- Write Python code to accomplish "{step_section}".
- Return the full, runnable code block in a ```python``` fence; do not add any extra explanation.
""".strip(),
}

DEFAULT_PROMPT = """\
{shared_section}

{step_section}

Your task:
- Execute using only the above summaries and sub-task content.
"""


def build_shared_section(relevant_summaries: list[str]) -> str:
    if relevant_summaries:
        items = [f"- {s}" for s in relevant_summaries]
        return "Previous Results:\n" + "\n".join(items)
    else:
        return "Previous Results:\n- (none)\n"


def build_step_section(step):
    return (
        f"Current Sub-Task (group {step.group_id}, step {step.step_id}):\n"
        f"{step.content}\n"
    )
