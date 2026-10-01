"""
main.py

Runs the workflow:
    User topic -> ResearcherAgent -> HAND-OFF -> SummarizerAgent -> 3 bullets
"""

import sys

from dotenv import load_dotenv

from agents import ResearcherAgent, SummarizerAgent

THICK_LINE = "=" * 50
THIN_LINE = "-" * 50


def print_agent_header(number, agent):
    print(f"\n{THIN_LINE}")
    print(f"AGENT {number}: {agent.name.upper()}")
    print(THIN_LINE)
    print(f"\nRole:\n{agent.role}")


def print_handoff():
    print(f"\n{THICK_LINE}")
    print("HAND-OFF")
    print(THICK_LINE)
    print("\nResearcherAgent output is being passed to")
    print("SummarizerAgent as its input.")
    print(f"\n{THICK_LINE}")


def main():
    # Make sure bullet characters print correctly on Windows consoles.
    sys.stdout.reconfigure(encoding="utf-8")

    # Read GEMINI_API_KEY / GEMINI_MODEL from the .env file.
    load_dotenv()

    print(THICK_LINE)
    print("TWO AGENTS, ONE HAND-OFF")
    print(THICK_LINE)

    topic = input("\nEnter a topic: ").strip()
    if not topic:
        print("No topic entered. Exiting.")
        return

    researcher = ResearcherAgent()
    summarizer = SummarizerAgent()

    try:
        # ---- Agent 1 ----
        print_agent_header(1, researcher)
        print(f"\nReceived:\n{topic}")
        research_notes = researcher.research(topic)
        print(f"\nProduced:\n{research_notes}")

        # ---- Hand-off ----
        print_handoff()

        # ---- Agent 2 ---- (its input IS the researcher's output)
        print_agent_header(2, summarizer)
        print(f"\nReceived:\n{research_notes}")
        summary = summarizer.summarize(research_notes)
        print(f"\nProduced:\n{summary}")
    except RuntimeError as error:
        print(f"\nERROR: {error}")
        return

    print(f"\n{THICK_LINE}")
    print("WORKFLOW COMPLETED")
    print(THICK_LINE)


if __name__ == "__main__":
    main()
