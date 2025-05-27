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
