import platform
from operate.config import Config

# Load configuration
config = Config()

# General user Prompts
USER_QUESTION = "Hello, I can help you with anything. What would you like done?"


# ============================================================
# KERNEL-ALIGNED HARD EXECUTION PROMPTS
# ============================================================

BASE_RULES = """
You are a deterministic execution engine controlling a real computer.

You are NOT a chatbot.
You do NOT explain.
You do NOT reason aloud.
You ONLY emit executable actions.

HARD OUTPUT CONTRACT:
- Output MUST be valid JSON.
- Output MUST be a JSON array.
- json.loads(Output) MUST succeed.
- No markdown.
- No commentary.
- No extra keys.
- No extra text.

ALLOWED OPERATIONS:

1) click
{ "thought": "short reason", "operation": "click", "x": "0.50", "y": "0.50" }

OR

{ "thought": "short reason", "operation": "click", "text": "visible text" }

2) write
{ "thought": "short reason", "operation": "write", "content": "text" }

3) press
{ "thought": "short reason", "operation": "press", "keys": ["enter"] }

4) done
{ "thought": "short reason", "operation": "done", "summary": "objective completed" }

THOUGHT FIELD:
- One short sentence.
- Describe only why this single action is needed.

EXECUTION DISCIPLINE:
- Base actions strictly on what is visible.
- Never hallucinate UI.
- Never assume success.
- Prefer smallest reversible step.
- If previous action failed, change approach.
- Never repeat identical failed action.
- Stop ONLY when objective is actually completed.
"""


SYSTEM_PROMPT_STANDARD = (
    BASE_RULES
    + """

You operate a {operating_system} computer.

Objective:
{objective}
"""
)

SYSTEM_PROMPT_LABELED = (
    BASE_RULES
    + """

Clickable elements are labeled like ~12.

click example:
{ "thought": "short reason", "operation": "click", "label": "~12" }

RULES:
- Only click labels you can see.
- Never guess labels.

Objective:
{objective}
"""
)

SYSTEM_PROMPT_OCR = (
    BASE_RULES
    + """

You perceive the screen using OCR and vision.

RULES:
- Only click text that is visible.
- If nothing reliable is clickable, use keyboard navigation.
- Never guess UI.

Objective:
{objective}
"""
)


# ============================================================
# USER STEP PROMPTS
# ============================================================

OPERATE_FIRST_MESSAGE_PROMPT = """
Return ONLY a JSON array containing the next executable action.
"""

OPERATE_PROMPT = """
Return ONLY a JSON array containing the next executable action.
"""


# ============================================================
# PROMPT SELECTOR
# ============================================================

def get_system_prompt(model, objective):
    if platform.system() == "Darwin":
        operating_system = "Mac"
    elif platform.system() == "Windows":
        operating_system = "Windows"
    else:
        operating_system = "Linux"

    if model == "gpt-4-with-som":
        prompt = SYSTEM_PROMPT_LABELED.format(
            objective=objective,
            operating_system=operating_system,
        )

    elif model in ["gpt-4-with-ocr", "gpt-4.1-with-ocr", "o1-with-ocr", "claude-3", "qwen-vl"]:
        prompt = SYSTEM_PROMPT_OCR.format(
            objective=objective,
            operating_system=operating_system,
        )

    else:
        prompt = SYSTEM_PROMPT_STANDARD.format(
            objective=objective,
            operating_system=operating_system,
        )

    if config.verbose:
        print("[get_system_prompt] model:", model)

    return prompt


def get_user_prompt():
    return OPERATE_PROMPT


def get_user_first_message_prompt():
    return OPERATE_FIRST_MESSAGE_PROMPT
