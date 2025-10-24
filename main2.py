from fastapi import FastAPI, Depends, HTTPException, File, UploadFile, Request
from fastapi.responses import JSONResponse
import os
import base64
import requests
import json
from typing import Dict, Any
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import dotenv

dotenv.load_dotenv()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://www.agilityaiinvoicely.com"],
    allow_origin_regex=r"^https?://localhost:\d{4}$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def create_enhanced_invoice_prompt() -> str:
    """Create a compact enhanced prompt used to ask the model to extract invoice text.
    This is a simplified version of the prompt used in the main project but preserves
    the same intent (include user context and extraction instructions).
    """

    prompt = f"""You are an invoice extraction assistant. Use the context below to classify the invoice
and extract structured details. Return readable extracted text (not JSON yet).

Extract invoice number, date, due date, items, taxes, totals, and parties. If uncertain, mark as "Unclear".
"""
    return prompt


def create_json_conversion_prompt(invoice_type: str) -> str:
    # Re-use the same high-level JSON schema idea from the main code (simplified)
    if invoice_type.lower() == "income":
        schema = {
            "invoiceType": "income",
            "invoiceNumber": "string",
            "date": "YYYY-MM-DD",
            "dueDate": "YYYY-MM-DD or null",
            "currency": "INR",
            "status": "draft",
            "billTo": {"name": "string", "email": "string or null", "address": "string", "state": "string or null", "gst": "string or null", "pan": "string or null", "phone": "string or null"},
            "items": [{"description": "string", "hsn": "string or null", "quantity": 0, "unitPrice": 0.0, "gst": 0.0, "discount": 0.0, "amount": 0.0}],
            "subtotal": 0.0,
            "cgst": 0.0,
            "sgst": 0.0,
            "igst": 0.0,
            "total": 0.0
        }
    else:
        schema = {
            "invoiceType": "expense",
            "invoiceNumber": "string",
            "date": "YYYY-MM-DD",
            "dueDate": "YYYY-MM-DD or null",
            "currency": "INR",
            "status": "draft",
            "billFrom": {"name": "string or null", "address": "string or null", "state": "string or null", "gst": "string or null", "pan": "string or null", "phone": "string or null", "email": "string or null"},
            "billTo": {"name": "string", "email": "string or null", "address": "string", "state": "string or null", "gst": "string or null", "pan": "string or null", "phone": "string or null"},
            "items": [{"description": "string or null", "hsn": "string or null", "quantity": 0, "price": 0.0, "gst": 0.0, "discount": 0.0, "total": 0.0}],
            "subtotal": 0.0,
            "total": 0.0
        }

    prompt = (
        "Convert the extracted invoice information below into a valid JSON object that strictly matches the schema.\n"
        "Return ONLY the JSON object, nothing else. Use null for missing fields.\n"
        f"Schema: {json.dumps(schema)}\n\n"
    )
    return prompt


@app.get("/")
def root():
    return {"status": "ok", "message": "running - endpoints: GET / and POST /api/scan-invoice"}


@app.post("/api/scan-invoice")
def scan_invoice(file: UploadFile = File(...)):
    """Endpoint that accepts an invoice image and uses the OpenAI API to extract structured JSON."""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set in environment")

    OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

    # Read file bytes and encode
    try:
        image_bytes = file.file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    try:
        image_b64 = base64.b64encode(image_bytes).decode()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to base64-encode file: {e}")

    enhanced_prompt = create_enhanced_invoice_prompt()

    # Use OpenAI's GPT-4 Vision model for image extraction
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": enhanced_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{file.content_type};base64,{image_b64}"}}
                ]
            }
        ],
        "max_tokens": 1024
    }

    try:
        response = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=90)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Network error calling OpenAI API: {e}")

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {response.status_code} {response.text}")

    openai_response = response.json()
    choices = openai_response.get("choices") or []
    if not choices:
        raise HTTPException(status_code=500, detail=f"OpenAI API returned no choices. Raw response: {openai_response}")

    extracted_text = choices[0]["message"]["content"].strip()
    if not extracted_text:
        raise HTTPException(status_code=500, detail="No text extracted from the invoice by the model")

    # Quick invoice type detection
    invoice_type = "income"
    up = extracted_text.upper()
    if "EXPENSE" in up or "INVOICE TYPE: EXPENSE" in up:
        invoice_type = "expense"
    elif "INCOME" in up or "INVOICE TYPE: INCOME" in up:
        invoice_type = "income"

    # Ask OpenAI to convert to strict JSON matching a known schema
    json_prompt = create_json_conversion_prompt(invoice_type)
    db_prompt = json_prompt + "\n" + extracted_text

    db_payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": db_prompt}
        ],
        "max_tokens": 1024
    }

    try:
        db_response = requests.post(OPENAI_API_URL, headers=headers, json=db_payload, timeout=90)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Network error calling OpenAI API for JSON conversion: {e}")

    if db_response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"OpenAI JSON conversion error: {db_response.status_code} {db_response.text}")

    db_response_json = db_response.json()
    db_text = db_response_json.get("choices", [{}])[0].get("message", {}).get("content", "")

    # Extract JSON object from the model reply (first { .. last })
    start_index = db_text.find("{")
    end_index = db_text.rfind("}")
    if start_index == -1 or end_index == -1 or end_index <= start_index:
        raise HTTPException(status_code=500, detail=f"Failed to find JSON object in model response. Raw text: {db_text}")

    json_text = db_text[start_index:end_index+1]

    try:
        invoice_data = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse JSON from model response: {e}. Raw JSON text: {json_text}")

    return JSONResponse({"success": True, "invoice": invoice_data})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)