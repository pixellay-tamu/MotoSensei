# -------------------------------------------------------------------
# app.py — Combined PISA + IGGA System
# -------------------------------------------------------------------

import os
import json
import tempfile
from flask import Flask, request, jsonify
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
os.environ["GOOGLE_CLOUD_PROJECT"] = "tamu-hackathon25cll-545"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

client = genai.Client()


# -------------------------------------------------------------------
# PISA: PART IDENTIFICATION & SOURCING AGENT
# -------------------------------------------------------------------

class SourcingResult(BaseModel):
    """Structured output schema for the sourcing agent."""
    part_name: str = Field(..., description="The formal name or SKU of the identified part.")
    best_price: float = Field(..., description="Lowest price found for the part.")
    vendor: str = Field(..., description="The vendor offering the best price.")
    purchase_url: str = Field(..., description="Direct product link for purchase.")


def search_shopping_data(query: str) -> str:
    """Simulated real-time product sourcing lookup."""
    print(f"🔎 Searching for price data on: {query}")
    mock_results = {
        "battery terminal": [
            {"vendor": "Amazon", "price": 12.99, "link": "https://amazon.com/terminal-cheap"},
            {"vendor": "AutoParts Pro", "price": 18.50, "link": "https://autoparts.com/terminal"},
            {"vendor": "Local Hardware", "price": 10.50, "link": "https://localhardware.com/best-price-terminal"},
        ]
    }
    return json.dumps(mock_results.get(query.lower(), [
        {"vendor": "Ebay", "price": 99.99, "link": "https://ebay.com/default-part"}
    ]))


def run_pisa(image_path: str) -> SourcingResult:
    """Identifies a part from an image, finds best source, returns structured data."""
    img = types.Part.from_uri(uri=image_path, mime_type="image/jpeg")

    system_instruction = (
        "You are PISA (Part Identification and Sourcing Assistant). "
        "Identify the car part in the provided image, then find the cheapest vendor online. "
        "Return results following the SourcingResult schema."
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[img, system_instruction],
        config=types.GenerateContentConfig(
            tools=[search_shopping_data],
            response_mime_type="application/json",
            response_schema=SourcingResult,
            temperature=0.0,
            thinking_budget=0
        ),
    )

    try:
        parsed = json.loads(response.text)
        return SourcingResult(**parsed)
    except Exception as e:
        print(f"Error parsing PISA output: {e}")
        print(response.text)
        raise


# -------------------------------------------------------------------
# IGGA: INSTALLATION GUIDE GENERATION AGENT
# -------------------------------------------------------------------

class RepairStep(BaseModel):
    id: int
    action: str
    tool: str


class RepairGuide(BaseModel):
    steps: list[RepairStep]


def general_web_search(query: str) -> str:
    """Mock web retrieval for repair guides."""
    print(f"⚙️ IGGA is researching installation procedure for: {query}")
    if "battery terminal" in query.lower():
        return (
            "Step 1: Turn off ignition and disconnect the negative cable using a 10mm wrench. "
            "Step 2: Loosen the bolt securing the terminal. Step 3: Remove the old terminal. "
            "Step 4: Clean the battery post with a wire brush. Step 5: Attach and tighten the new terminal."
        )
    return "No detailed guide found. Follow general safety protocols."


def run_igga(part_name: str, model_year_vehicle: str = "Generic Vehicle") -> list[dict]:
    """Generates a structured guide for installing or replacing a car part."""
    system_prompt = f"""
    You are IGGA (Installation Guide Generation Agent).
    Generate a professional, structured step-by-step guide for installing or replacing '{part_name}'.
    Use the 'general_web_search' tool for external reference knowledge.
    Each step must include an action and an associated tool (or 'none' if not required).
    """
    user_prompt = f"Provide an installation guide for {part_name} on a {model_year_vehicle}."

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[system_prompt, user_prompt],
        config=types.GenerateContentConfig(
            tools=[general_web_search],
            response_mime_type="application/json",
            response_schema=RepairGuide,
            temperature=0.1,
            tool_choice="auto"
        ),
    )

    try:
        guide = json.loads(response.text)
        return guide.get("steps", [])
    except json.JSONDecodeError:
        print("⚠️ Schema issue: returning empty guide.")
        return []


# -------------------------------------------------------------------
# FLASK API — UNIFIED ENDPOINT
# -------------------------------------------------------------------

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_and_guide():
    """Handles image upload and returns sourcing data + installation guide."""
    if 'file' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['file']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        file.save(tmp.name)
        pisa_result = run_pisa(tmp.name)
        igga_steps = run_igga(pisa_result.part_name)

    response = {
        "part_name": pisa_result.part_name,
        "best_price": pisa_result.best_price,
        "vendor": pisa_result.vendor,
        "purchase_url": pisa_result.purchase_url,
        "repair_guide": igga_steps
    }

    return jsonify(response)


# -------------------------------------------------------------------
# RUN SERVER
# -------------------------------------------------------------------

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
