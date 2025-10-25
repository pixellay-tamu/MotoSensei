# -------------------------------------------------------------------
# app.py — PISA Web Service Integration
# -------------------------------------------------------------------

import os
import json
import tempfile
from flask import Flask, request, jsonify
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# -------------------------------------------------------------------
# CONFIGURATION & CLIENT INITIALIZATION
# -------------------------------------------------------------------

# Load environment configuration securely
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
os.environ["GOOGLE_CLOUD_PROJECT"] = "tamu-hackathon25cll-545"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

client = genai.Client()


# -------------------------------------------------------------------
# STRUCTURED RESPONSE MODEL
# -------------------------------------------------------------------

class SourcingResult(BaseModel):
    """Structured output schema for the sourcing agent."""
    part_name: str = Field(..., description="The formal name or SKU of the identified part.")
    best_price: float = Field(..., description="Lowest price found for the part.")
    vendor: str = Field(..., description="The vendor offering the best price.")
    purchase_url: str = Field(..., description="Direct product link for purchase.")


# -------------------------------------------------------------------
# TOOL FUNCTION (Mock Product Search for Prototype)
# -------------------------------------------------------------------

def search_shopping_data(query: str) -> str:
    """Simulated product sourcing lookup."""
    print(f"🔎 Searching for real-time price data on: {query}")
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


# -------------------------------------------------------------------
# CORE FUNCTION: Run PISA Agent
# -------------------------------------------------------------------

def run_pisa(image_path: str) -> SourcingResult:
    """Identifies car part from image and returns sourcing info."""
    img = types.Part.from_uri(uri=image_path, mime_type="image/jpeg")

    system_instruction = (
        "You are PISA, the Part Identification and Sourcing Assistant. "
        "Analyze the given image to identify the part name, type, or SKU. "
        "Then use the available product search tool to find the cheapest vendor. "
        "Return the results according to the SourcingResult schema."
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

    # Attempt to parse structured response
    try:
        parsed_json = json.loads(response.text)
        return SourcingResult(**parsed_json)
    except Exception as e:
        print(f"Parsing error: {e}")
        print(f"Response text:\n{response.text}")
        raise


# -------------------------------------------------------------------
# FLASK SERVER FOR CHATBOT INTEGRATION
# -------------------------------------------------------------------

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """API endpoint to receive image uploads and return sourcing results."""
    if 'file' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['file']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        file.save(tmp.name)
        result = run_pisa(tmp.name)

    return jsonify(result.dict())


# -------------------------------------------------------------------
# RUN SERVER
# -------------------------------------------------------------------

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
