"""
Installation Guide Generation Agent (IGGA)
Optimized for Vertex AI Gemini 2.5 Pro
Date: 2025-10-25
"""


from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import json


# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------


# Ensure Vertex AI environment variables are correctly set before running:
# export GOOGLE_GENAI_USE_VERTEXAI=true
# export GOOGLE_CLOUD_PROJECT="your-project-id"
# export GOOGLE_CLOUD_LOCATION="us-central1"


client = genai.Client()


# ----------------------------------------------------------------------
# STRUCTURED OUTPUT DEFINITIONS
# ----------------------------------------------------------------------


class RepairStep(BaseModel):
    """A single actionable step in a repair guide."""
    id: int = Field(..., description="Sequential step number.")
    action: str = Field(..., description="Specific, detailed user action (e.g., 'Loosen the bolt').")
    tool: str = Field(..., description="Required tool for the step, like '10mm wrench' or 'Phillips screwdriver'.")


class RepairGuide(BaseModel):
    """Complete structured installation guide."""
    steps: list[RepairStep] = Field(..., description="Ordered list of all steps, each described by action and tool.")


# ----------------------------------------------------------------------
# TOOL FUNCTION — KNOWLEDGE RETRIEVAL MOCK
# ----------------------------------------------------------------------


def general_web_search(query: str) -> str:
    """
    Simulated retrieval tool for repair instructions.
    Replace with a real Vertex Search or Google Custom Search integration in production.
    """
    print(f"⚙️ IGGA is looking for documentation about: {query}")
   
    if "battery terminal" in query.lower():
        return (
            "Step 1: Turn off ignition and disconnect the negative cable using a 10mm wrench. "
            "Step 2: Loosen the bolt securing the terminal. Step 3: Remove the old terminal. "
            "Step 4: Clean the battery post with a wire brush. Step 5: Attach and tighten the new terminal."
        )
    return "No specific guide found. Follow generic safety and installation procedures."


# ----------------------------------------------------------------------
# CORE AGENT FUNCTION
# ----------------------------------------------------------------------


def run_igga(part_name: str, model_year_vehicle: str = "Generic Vehicle") -> list[dict]:
    """
    Generates a structured, step-by-step repair guide using Gemini 2.5 Pro.
   
    Args:
        part_name: The automotive part requiring installation.
        model_year_vehicle: Contextual vehicle info for guide personalization.


    Returns:
        A list of structured repair steps.
    """


    # Provide contextual and grounding instructions for Gemini.
    system_prompt = f"""
    You are IGGA (Installation Guide Generation Agent).
    Your goal is to produce a structured installation guide for automotive parts.


    1. Analyze the provided query, determine the probable installation method.
    2. Use the "general_web_search" tool when necessary to fetch detailed instructions.
    3. Deconstruct the found directions into atomic, numbered steps.
    4. Ensure every step includes an 'action' and a 'tool', even 'none' if no tool is needed.
    5. Always include a first step about safety (e.g., 'disconnect power' or 'wear gloves').
    6. Output must match the RepairGuide schema exactly in JSON format.
    """


    # Combine query content and system directions
    user_prompt = f"Generate an installation guide for {part_name} on a {model_year_vehicle}."


    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[system_prompt, user_prompt],
        config=types.GenerateContentConfig(
            tools=[general_web_search],
            response_mime_type="application/json",
            response_schema=RepairGuide,  # Vertex-enforced schema validation
            temperature=0.1,              # Near-deterministic reasoning
            max_output_tokens=1024,       # Prevent truncation for long guides
            tool_choice="auto"            # Let Gemini invoke `general_web_search` intelligently
        ),
    )


    # Parse structured output safely
    try:
        parsed = json.loads(response.text)
        return parsed.get("steps", [])
    except json.JSONDecodeError:
        print("⚠️ Schema mismatch: returning empty results.")
        return []


# ----------------------------------------------------------------------
# EXAMPLE RUN (FOR TESTING)
# ----------------------------------------------------------------------


if __name__ == "__main__":
    example_part = "Automotive Battery Terminal Clamp"
    print(f"\n🔧 Generating guide for: {example_part}")


    steps = run_igga(example_part)
    if steps:
        print("\n✅ Generated Structured Repair Guide:\n")
        for s in steps:
            print(f"Step {s['id']}: {s['action']} (Tool: {s['tool']})")
    else:
        print("⚠️ No steps generated. Check model logs or network config.")