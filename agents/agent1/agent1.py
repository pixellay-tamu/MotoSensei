"""
Optimized Part Identification & Sourcing Agent (PISA)
Powered by Vertex AI Gemini 2.5 Flash
Version: 2025-10-25
"""


import os
import json
import requests
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# -------------------------------------------------------------------
# CONFIGURATION & CLIENT INITIALIZATION
# -------------------------------------------------------------------


# Safely load environment credentials instead of embedding keys directly
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
os.environ["GOOGLE_CLOUD_PROJECT"] = "your-project-id"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"


client = genai.Client()  # Auto-detects Vertex AI configuration




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
    """
    Simulates an online product sourcing lookup.
    Replace this later with Vertex Search, SerpAPI, or Shopping API integration.
    """
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
    """
    Identifies a car part from an image, finds cheapest vendor, and returns structured results.
    """


    img = types.Part.from_uri(uri=image_path, mime_type="image/jpeg")


    system_instruction = (
        "You are PISA, the Part Identification and Sourcing Assistant. "
        "Analyze the given image to identify the part name, type, or SKU. "
        "Then use the available product search tool to find the cheapest online vendor. "
        "Return your findings strictly following the SourcingResult schema."
    )


    # Use Gemini 2.5 Flash thinking budget to optimize speed vs context comprehension
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[img, system_instruction],
        config=types.GenerateContentConfig(
            tools=[search_shopping_data],               # Registered tool
            response_mime_type="application/json",      # Ensures valid JSON output
            response_schema=SourcingResult,             # Enforces output validation
            temperature=0.0,                            # Deterministic cost-efficient inference
            thinking_budget=0                           # Prioritize faster responses
        ),
    )


    # Parse the model’s structured response into Python object
    try:
        parsed_json = json.loads(response.text)
        return SourcingResult(**parsed_json)
    except Exception as e:
        print(f"⚠️ Parsing error: {e}")
        print(f"Response text:\n{response.text}")
        raise




# -------------------------------------------------------------------
# EXAMPLE EXECUTION (Simulated Run)
# -------------------------------------------------------------------


if __name__ == "__main__":
    mock_image = "path/to/part_image.jpg"
    print(f"🧠 Starting PISA for image: {mock_image}")


    result = run_pisa(mock_image)


    print("\n✅ Best Sourcing Option Found:")
    print(f"Part: {result.part_name}")
    print(f"Vendor: {result.vendor}")
    print(f"Price: ${result.best_price}")
    print(f"Purchase Link: {result.purchase_url}\n")