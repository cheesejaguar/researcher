"""Prompt templates for the CLI subagent path."""

SYSTEM_PROMPT = (
    "You are a fact-extraction research agent. You have access to WebSearch "
    "and WebFetch tools. For each candidate entity matching the user's "
    "request, return one Extraction per field, citing the URL you extracted "
    "it from. Confidence is a number between 0.0 and 1.0. Return JSON "
    "matching the schema exactly — no prose, no markdown, no commentary."
)


USER_PROMPT_TEMPLATE = """Goal: {goal}
Entity type: {entity_type}
Task: {task_description}
Fields required: {field_list}
Return up to {max_entities} entities.

Respond with JSON matching this schema:
{schema_json}
"""


def build_user_prompt(
    *,
    goal: str,
    entity_type: str,
    task_description: str,
    field_list: list[str],
    max_entities: int,
    schema_json: str,
) -> str:
    return USER_PROMPT_TEMPLATE.format(
        goal=goal,
        entity_type=entity_type,
        task_description=task_description,
        field_list=", ".join(field_list),
        max_entities=max_entities,
        schema_json=schema_json,
    )
