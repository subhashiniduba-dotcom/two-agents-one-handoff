"""
agents.py

Two plain Python classes, each wrapping ONE call to the Gemini API:

    ResearcherAgent  -> topic          -> research notes
    SummarizerAgent  -> research notes -> exactly 3 bullet points

No agent frameworks are used. An "agent" here is just a class with a
role (its prompt) that makes one LLM call.
"""

import os

from google import genai
from google.genai import errors

# Used only if GEMINI_MODEL is not set in the .env file.
DEFAULT_MODEL = "gemini-3.5-flash-lite"


def call_gemini(prompt):
    """Send one prompt to Gemini and return the text of the answer."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found. Create a .env file (copy .env.example) "
            "and add your key."
        )

    model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(model=model_name, contents=prompt)
    except errors.APIError as error:
        # Turn common API errors into short, friendly messages.
        if error.code == 429:
            raise RuntimeError("Quota/rate limit reached. Wait a minute and retry.")
        if error.code == 404:
            raise RuntimeError(
                f"Model '{model_name}' not found. Change GEMINI_MODEL in .env."
            )
        if error.code in (400, 401, 403):
            raise RuntimeError(f"Request rejected (check your API key): {error.message}")
        raise RuntimeError(f"Gemini API error {error.code}: {error.message}")

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")
    return response.text.strip()


class ResearcherAgent:
    """Agent 1: turns a topic into a few paragraphs of research notes."""

    name = "ResearcherAgent"
    role = "Researches the user's topic and produces research notes."

    def research(self, topic):
        prompt = (
            "You are a research assistant. Write research notes about the "
            f"topic below in 3 to 4 short paragraphs of plain text "
            "(no headings, no bullet points).\n\n"
            f"Topic: {topic}"
        )
        return call_gemini(prompt)


class SummarizerAgent:
    """Agent 2: turns research notes into exactly 3 bullet points."""

    name = "SummarizerAgent"
    role = "Converts research notes into a concise 3-bullet summary."

    def summarize(self, research_notes):
        prompt = (
            "Summarize the research notes below into EXACTLY 3 bullet points. "
            "Start each line with '- '. Output only the 3 bullet lines, "
            "nothing else.\n\n"
            f"Research notes:\n{research_notes}"
        )
        answer = call_gemini(prompt)
        return self._keep_three_bullets(answer)

    def _keep_three_bullets(self, answer):
        """Make sure the result is exactly 3 lines, each starting with a bullet."""
        bullets = []
        for line in answer.splitlines():
            line = line.strip().lstrip("-*• ").strip()
            if line:
                bullets.append("• " + line)

        if len(bullets) < 3:
            raise RuntimeError("The model returned fewer than 3 bullet points.")
        return "\n".join(bullets[:3])
