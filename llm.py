import json
from google import genai
from configuration import GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use ONLY the provided excerpts to answer "
    "thoroughly and conversationally. If the excerpts don’t cover everything, "
    "say so and offer to elaborate."
)
REWRITE_PROMPT = (
    "Extract up to 5 concise keyword phrases from the user question "
    "that would best help search the document. Return them as a JSON list."
)
LIST_TABLE_INSTR = (
    "\n\nIf your answer is a list, use bullets:\n"
    "• Item A\n• Item B\n\n"
    "If a table fits, use ASCII tables:\n"
    "| Col1 | Col2 |\n|------|------|\n| X    | Y    |\n"
)

def rewrite_query(question: str) -> list[str]:
    prompt = f"SYSTEM: {REWRITE_PROMPT}\nQUESTION: {question}"
    resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    text = resp.text.strip()
    try:
        return json.loads(text)
    except:
        return question.lower().split()[:3]

def ask_gemini_with_history(history: list[dict], context: str, question: str) -> str:
    lines = [f"SYSTEM: {SYSTEM_PROMPT}"]
    for turn in history:
        role = turn.get("role", "user").upper()
        lines.append(f"{role}: {turn['content']}")
    lines.append(f"ASSISTANT: Here are the relevant excerpts:\n{context}")
    lines.append(f"USER: {question}")
    lines.append(f"ASSISTANT:{LIST_TABLE_INSTR}\nAnswer:")
    prompt = "\n".join(lines)

    resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return resp.text.strip()
