from fastapi import FastAPI, Depends, HTTPException, status, Request, File, UploadFile, Body, Query, APIRouter
from fastapi.responses import FileResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta, date
import motor.motor_asyncio
import os
import io
import base64
import requests
import jwt
from passlib.context import CryptContext
from bson import ObjectId
import random
import httpx
import json

from dotenv import load_dotenv
import calendar
from enum import Enum
import pdfkit
import pandas as pd

load_dotenv()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","https://www.agilityaiinvoicely.com"],
    allow_origin_regex=r"^https?://localhost:\d{4}$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MONGODB_URI = os.getenv("MONGODB_URI")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
db = client["test"]

SECRET_KEY = os.getenv("JWT_SECRET", "your-super-secret-jwt-key")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class FileImportBody(BaseModel):
    fileContent: str  
    fileName: str

class InvoiceItem(BaseModel):
    description: str
    hsn: Optional[str]
    quantity: int
    unitPrice: float
    gst: Optional[float] = 0
    discount: Optional[float] = 0
    amount: Optional[float]

class BillTo(BaseModel):
    name: str
    email: Optional[str]
    address: str
    state: Optional[str]
    gst: Optional[str]
    pan: Optional[str]
    phone: Optional[str]

class ShipTo(BaseModel):
    name: Optional[str]
    address: Optional[str]
    gst: Optional[str]
    pan: Optional[str]
    phone: Optional[str]
    email: Optional[str]

class Invoice(BaseModel):
    user: Optional[str] = None
    invoiceNumber: str
    date: datetime
    dueDate: Optional[datetime]
    billTo: BillTo
    shipTo: Optional[ShipTo]
    items: List[InvoiceItem]
    notes: Optional[str]
    currency: Optional[str] = "INR"
    status: Optional[str] = "draft"
    subtotal: Optional[float]
    cgst: Optional[float]
    sgst: Optional[float]
    igst: Optional[float]
    total: Optional[float]
    termsAndConditions: Optional[str]

class ExpenseInvoiceItem(BaseModel):
    description: Optional[str]
    hsn: Optional[str]
    quantity: Optional[int]
    price: Optional[float]
    gst: Optional[float]
    discount: Optional[float]
    total: Optional[float]

class BillFrom(BaseModel):
    name: Optional[str]
    address: Optional[str]
    state: Optional[str]
    gst: Optional[str]
    pan: Optional[str]
    phone: Optional[str]
    email: Optional[str]

class ExpenseInvoice(BaseModel):
    invoiceNumber: str
    date: datetime
    dueDate: Optional[datetime]
    currency: Optional[str] = "INR"
    status: Optional[str] = "draft"
    billFrom: BillFrom
    billTo: BillTo
    shipTo: Optional[ShipTo]
    items: List[ExpenseInvoiceItem]
    termsAndConditions: Optional[str]
    subtotal: Optional[float]
    cgst: Optional[float]
    sgst: Optional[float]
    igst: Optional[float]
    total: Optional[float]

class ExpenseStep1(BaseModel):
    expenseNumber: str = Field(..., alias="expenseNumber")
    expenseDate: datetime = Field(..., alias="expenseDate")
    dueDate: Optional[datetime]
    currency: Optional[str] = "INR"
    status: Optional[str] = "draft"
    notes: Optional[str]
    paymentMethod: Optional[str]

class ExpenseStep2(BaseModel):
    vendorName: str
    businessName: Optional[str]
    billingAddress: str
    shippingAddress: Optional[str]
    email: Optional[str]

class ExpenseItemFromStep3(BaseModel):
    id: Union[int, str]
    name: str
    hsn: Optional[str]
    qty: int
    price: float

class ExpenseStep3(BaseModel):
    items: List[ExpenseItemFromStep3]

class ExpenseStep4(BaseModel):
    cgst: Optional[float] = 0
    sgst: Optional[float] = 0
    igst: Optional[float] = 0
    discount: Optional[float] = 0
    shipping: Optional[float] = 0

class NewExpenseFromSteps(BaseModel):
    step1: ExpenseStep1
    step2: ExpenseStep2
    step3: ExpenseStep3
    step4: ExpenseStep4


class User(BaseModel):
    email: str
    password: str
    name: str
    company: str
    address: str
    gstNumber: Optional[str] = None
    panNumber: Optional[str] = None
    phone: str
    website: Optional[str] = None
    state: str
    isGstRegistered: Optional[bool] = False
    businessLogo: Optional[str] = None
    createdAt: Optional[datetime] = None
    theme: Optional[str] = "light"

class Address(BaseModel):
    address1: str
    address2: Optional[str] = None
    city: str
    state: str
    pincode: str
    country: str

class Customers(BaseModel):
    customerType: str
    fullName: str
    email: str
    phone: str
    companyName: Optional[str] = None
    website: Optional[str] = None
    billingAddress: Address
    shippingAddress: Address
    sameAsBilling: bool
    panNumber: Optional[str] = None
    isGstRegistered: bool
    gstNumber: Optional[str] = None
    placeOfSupply: str
    currency: str
    paymentTerms: str
    notes: Optional[str] = None
    tags: List[str]


class TeamMemberRole(str, Enum):
    admin = "admin"
    manager = "manager"
    accountant = "accountant"
    viewer = "viewer"
    sales = "sales"

class TeamMemberStatus(str, Enum):
    active = "active"
    inactive = "inactive"

class TeamMemberBase(BaseModel):
    name: str
    role: TeamMemberRole
    email: str
    phone: str
    status: TeamMemberStatus

class TeamMemberCredentials(BaseModel):
    username: str
    password: str

class TeamMemberCreate(TeamMemberBase):
    joiningDate: date
    credentials: TeamMemberCredentials


class TeamMemberUpdate(BaseModel):
    name: Optional[str] = None
    role: Optional[TeamMemberRole] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    status: Optional[TeamMemberStatus] = None

class TeamMemberInDB(TeamMemberBase):
    id: str = Field(..., alias="_id")
    userId: str
    dateJoined: date
    lastActive: Optional[date] = None
    avatar: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        allow_population_by_field_name = True

class ExpenseSteps(BaseModel):
    step1: dict
    step2: dict
    step3: dict
    step4: dict

class BillToPayload(BaseModel):
    name: str
    email: Optional[str]
    address: Optional[str]
    state: Optional[str]
    gst: Optional[str]
    pan: Optional[str]
    phone: Optional[str]
    companyName: Optional[str]

class BillFromPayload(BaseModel):
    businessName: Optional[str]
    address: Optional[str]
    state: Optional[str]
    phone: Optional[str]
    email: Optional[str]
    gst: Optional[str]

class InvoiceItemPayload(BaseModel):
    description: str
    hsn: Optional[str]
    quantity: int
    unitPrice: float
    gst: Optional[float] = 0
    discount: Optional[float] = 0

class NewInvoicePayload(BaseModel):
    invoiceNumber: str
    date: datetime
    dueDate: Optional[datetime]
    billTo: BillToPayload
    shipTo: Optional[dict]
    items: List[InvoiceItemPayload]
    notes: Optional[str]
    currency: Optional[str] = "INR"
    status: Optional[str] = "draft"
    termsAndConditions: Optional[str]
    paymentTerms: Optional[str]
    billFrom: Optional[BillFromPayload]
    discount: Optional[float] = 0
    shipping: Optional[float] = 0


def convert_objids(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, ObjectId):
                    obj[k] = str(v)
                elif isinstance(v, (datetime, date)):
                    obj[k] = v.isoformat()
                elif isinstance(v, dict):
                    obj[k] = convert_objids(v)
                elif isinstance(v, list):
                    obj[k] = [convert_objids(i) for i in v]
        elif isinstance(obj, list):
            obj = [convert_objids(i) for i in obj]
        return obj

async def get_current_user(request: Request):
    auth_header = request.headers.get("Authorization")
    token = None
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1].strip()
    if not token:
        token = request.cookies.get("authToken")

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id = payload.get("_id") or payload.get("id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")

        try:
            user_obj_id = ObjectId(user_id)
        except Exception:
            user_obj_id = user_id

        user = await db["users"].find_one({"_id": user_obj_id})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        return {"_id": str(user["_id"]), "email": user.get("email")}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

GEMINI_API_KEY = str(os.getenv("GEMINI_API_KEY"))
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=" + GEMINI_API_KEY

def create_enhanced_invoice_prompt(user_profile):
    user_company = user_profile.get("company", "")
    user_name = user_profile.get("name", "")
    user_gst = user_profile.get("gstNumber", "")
    user_pan = user_profile.get("panNumber", "")
    user_address = user_profile.get("address", "")
    user_state = user_profile.get("state", "")
    user_phone = user_profile.get("phone", "")
    user_email = user_profile.get("email", "")
    
    prompt = f"""You are an expert invoice analyzer with advanced OCR capabilities. Analyze this invoice image and extract all information in a structured format.

**USER CONTEXT FOR INVOICE CLASSIFICATION:**
- User Company: {user_company}
- User Name: {user_name}
- User GST Number: {user_gst}
- User PAN Number: {user_pan}
- User Address: {user_address}
- User State: {user_state}
- User Phone: {user_phone}
- User Email: {user_email}

**INVOICE TYPE DETECTION:**
First, determine if this is an INCOME or EXPENSE invoice by comparing the user's business details with the invoice parties:

- INCOME Invoice: If the user's company/details appear as the "From/Seller/Service Provider" or if the invoice is being sent TO another party
- EXPENSE Invoice: If the user's company/details appear as the "To/Buyer/Client" or if the invoice is being received FROM another party

Look for exact matches in company names, GST numbers, PAN numbers, addresses, phone numbers, or email addresses.

**COMPREHENSIVE DATA EXTRACTION:**

1. **Invoice Classification & Basic Details:**
   - Invoice Type: [INCOME/EXPENSE] (based on user context analysis)
   - Invoice Number
   - Date of Issue (format: YYYY-MM-DD)
   - Due Date (format: YYYY-MM-DD)
   - Currency (default INR if not specified)
   - Status (draft/pending/paid/overdue)

2. **Party Information (extract ALL available details):**
   
   For INCOME Invoices:
   - Bill To (Client/Customer):
     * Company/Individual Name
     * Complete Address with Pin Code
     * State
     * Email Address
     * Phone Number
     * GST Number (if available)
     * PAN Number (if available)
   
   - Ship To (if different from Bill To):
     * Company/Individual Name
     * Complete Address with Pin Code
     * State
     * Email Address
     * Phone Number
     * GST Number (if available)
     * PAN Number (if available)
   
   For EXPENSE Invoices:
   - Bill From (Vendor/Supplier):
     * Company/Individual Name
     * Complete Address with Pin Code
     * State
     * Email Address
     * Phone Number
     * GST Number (if available)
     * PAN Number (if available)
   
   - Bill To (User's Company - should match user context):
     * Verify against user's company details

3. **Line Items (extract ALL items with complete details):**
   For each product/service:
   - Description (detailed)
   - HSN/SAC Code (if available)
   - Quantity
   - Unit of Measurement (pcs, kg, hours, etc.)
   - Unit Price/Rate
   - GST Rate (percentage)
   - Discount Percentage (if any)
   - Discount Amount (if any)
   - Line Total Amount

4. **Tax Calculations (extract exact amounts and rates):**
   - Subtotal (before taxes)
   - CGST: Amount and Rate percentage
   - SGST: Amount and Rate percentage  
   - IGST: Amount and Rate percentage
   - UTGST: Amount and Rate percentage (if applicable)
   - CESS: Amount and Rate percentage (if applicable)
   - TDS Deducted (if applicable)
   - Rounding Off Amount
   - **Final Total Amount**

5. **Payment & Terms:**
   - Payment Terms (Net 30, Cash on Delivery, etc.)
   - Payment Method Accepted
   - Bank Details (if provided)
   - Terms and Conditions
   - Additional Notes or Instructions
   - Late Payment Penalties (if mentioned)

6. **Quality Assurance Checks:**
   - Verify mathematical calculations
   - Cross-check GST rates with HSN codes
   - Ensure address components are complete
   - Validate date formats
   - Check for any watermarks or stamps

**IMPORTANT EXTRACTION RULES:**
- Extract EXACT text as it appears, don't paraphrase
- For numerical values, provide numbers without currency symbols
- If information is partially visible, note it as "Partially Visible: [text]"
- If information is unclear, mark as "Unclear: [best guess]"
- If information is completely missing, use "Not Found"
- Pay special attention to faded, rotated, or poorly scanned text
- Look for information in headers, footers, watermarks, and stamps
- Consider various invoice formats: traditional, modern, GST-compliant, service invoices, etc.

**OUTPUT FORMAT:**
Provide a clear, structured response with all extracted information organized by categories. Use consistent formatting and ensure all monetary values are accurate."""

    return prompt

def create_json_conversion_prompt(invoice_type):
    if invoice_type.lower() == "income":
        json_schema = """{
    "invoiceType": "income",
    "invoiceNumber": "string",
    "date": "YYYY-MM-DD",
    "dueDate": "YYYY-MM-DD or null",
    "currency": "INR",
    "status": "draft",
    "billTo": {
        "name": "string",
        "email": "string or null",
        "address": "string",
        "state": "string or null",
        "gst": "string or null",
        "pan": "string or null",
        "phone": "string or null"
    },
    "shipTo": {
        "name": "string or null",
        "address": "string or null",
        "state": "string or null",
        "gst": "string or null",
        "pan": "string or null",
        "phone": "string or null",
        "email": "string or null"
    },
    "items": [
        {
            "description": "string",
            "hsn": "string or null",
            "quantity": number,
            "unitPrice": number,
            "gst": number,
            "discount": number,
            "amount": number
        }
    ],
    "notes": "string or null",
    "subtotal": number,
    "cgst": number,
    "sgst": number,
    "igst": number,
    "total": number,
    "termsAndConditions": "string or null"
}"""
    else:
        json_schema = """{
    "invoiceType": "expense",
    "invoiceNumber": "string",
    "date": "YYYY-MM-DD",
    "dueDate": "YYYY-MM-DD or null",
    "currency": "INR",
    "status": "draft",
    "billFrom": {
        "name": "string or null",
        "address": "string or null",
        "state": "string or null",
        "gst": "string or null",
        "pan": "string or null",
        "phone": "string or null",
        "email": "string or null"
    },
    "billTo": {
        "name": "string",
        "email": "string or null",
        "address": "string",
        "state": "string or null",
        "gst": "string or null",
        "pan": "string or null",
        "phone": "string or null"
    },
    "shipTo": {
        "name": "string or null",
        "address": "string or null",
        "gst": "string or null",
        "pan": "string or null",
        "phone": "string or null",
        "email": "string or null"
    },
    "items": [
        {
            "description": "string or null",
            "hsn": "string or null",
            "quantity": number or null,
            "price": number or null,
            "gst": number or null,
            "discount": number or null,
            "total": number or null
        }
    ],
    "termsAndConditions": "string or null",
    "subtotal": number or null,
    "cgst": number or null,
    "sgst": number or null,
    "igst": number or null,
    "total": number or null
}"""

    return f"""Convert the following invoice information into a valid JSON object that strictly matches this schema. 

**REQUIREMENTS:**
- Return ONLY valid JSON, no additional text or formatting
- Use null for missing fields (not "Not Found" or empty strings)
- Ensure all numeric values are actual numbers, not strings
- Date format must be YYYY-MM-DD
- GST rates should be decimal (e.g., 18 for 18%, not 0.18)
- Maintain data accuracy from the extracted text

**JSON Schema:**
{json_schema}

**Important Notes:**
- If invoiceType is detected as "expense", use billFrom for vendor details
- If invoiceType is detected as "income", use billTo for client details  
- shipTo is optional and only used if shipping address differs from billing
- All monetary calculations should be mathematically correct
- Include all line items as separate objects in the items array

Please convert the extracted invoice information below into the JSON format:

"""

def get_date_range(period: str):
    today = date.today()
    if period == "this-month":
        start_date = today.replace(day=1)
        end_date = (start_date + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    elif period == "last-month":
        end_date = today.replace(day=1) - timedelta(days=1)
        start_date = end_date.replace(day=1)
    elif period == "this-year":
        start_date = date(today.year, 1, 1)
        end_date = date(today.year, 12, 31)
    elif period == "last-year":
        start_date = date(today.year - 1, 1, 1)
        end_date = date(today.year - 1, 12, 31)
    elif period == "7-days":
        start_date = today - timedelta(days=7)
        end_date = today
    elif period == "6-months":
        start_date = today - timedelta(days=180)
        end_date = today
    else: # Default to 30 days
        start_date = today - timedelta(days=30)
        end_date = today

    return datetime.combine(start_date, datetime.min.time()), datetime.combine(end_date, datetime.max.time())

def get_previous_period_range(start_date, end_date):
    duration = end_date - start_date
    prev_end_date = start_date - timedelta(seconds=1)
    prev_start_date = prev_end_date - duration
    return prev_start_date, prev_end_date
    
def calculate_trend(current, previous):
    if previous > 0:
        change = ((current - previous) / previous) * 100
    elif current > 0:
        change = 100.0
    else:
        change = 0.0
    
    trend = "up" if change > 0 else "down" if change < 0 else "neutral"
    return {"change": round(change, 2), "trend": trend}

def generate_invoice_html(invoice_data, user_profile):
    items_html = ""
    for item in invoice_data.get("items", []):
        items_html += f"""
        <tr>
            <td>{item.get('description', '')}</td>
            <td>{item.get('hsn', '')}</td>
            <td>{item.get('quantity', 0)}</td>
            <td>{item.get('unitPrice', 0.0):.2f}</td>
            <td>{item.get('gst', 0)}%</td>
            <td>{item.get('amount', 0.0):.2f}</td>
        </tr>
        """
    html_template = f"""
    <html>
    <head>
        <style>
            body {{ font-family: sans-serif; color: #333; }}
            .invoice-box {{ max-width: 800px; margin: auto; padding: 30px; border: 1px solid #eee; box-shadow: 0 0 10px rgba(0, 0, 0, .15); font-size: 16px; line-height: 24px; }}
            .header {{ text-align: right; }}
            .header h1 {{ color: #333; }}
            .details {{ display: flex; justify-content: space-between; margin-top: 50px; margin-bottom: 50px; }}
            table {{ width: 100%; text-align: left; border-collapse: collapse; }}
            table td, table th {{ padding: 8px; border-bottom: 1px solid #eee; }}
            table th {{ background-color: #f2f2f2; font-weight: bold; }}
            .totals {{ text-align: right; margin-top: 20px; }}
            .totals p {{ margin: 5px 0; }}
        </style>
    </head>
    <body>
        <div class="invoice-box">
            <div class="header">
                <h1>INVOICE</h1>
                <p><b>Invoice #:</b> {invoice_data.get('invoiceNumber', '')}</p>
                <p><b>Date:</b> {invoice_data.get('date').strftime('%Y-%m-%d')}</p>
                <p><b>Due Date:</b> {invoice_data.get('dueDate').strftime('%Y-%m-%d') if invoice_data.get('dueDate') else 'N/A'}</p>
            </div>
            <div class="details">
                <div>
                    <b>From:</b><br>
                    {user_profile.get('company', '')}<br>
                    {user_profile.get('address', '')}<br>
                    GST: {user_profile.get('gstNumber', '')}
                </div>
                <div>
                    <b>Bill To:</b><br>
                    {invoice_data.get('billTo', {}).get('name', '')}<br>
                    {invoice_data.get('billTo', {}).get('address', '')}<br>
                    GST: {invoice_data.get('billTo', {}).get('gst', '')}
                </div>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Description</th>
                        <th>HSN</th>
                        <th>Qty</th>
                        <th>Unit Price</th>
                        <th>GST</th>
                        <th>Amount</th>
                    </tr>
                </thead>
                <tbody>
                    {items_html}
                </tbody>
            </table>
            <div class="totals">
                <p><b>Subtotal:</b> {invoice_data.get('subtotal', 0.0):.2f}</p>
                <p><b>CGST:</b> {invoice_data.get('cgst', 0.0):.2f}</p>
                <p><b>SGST:</b> {invoice_data.get('sgst', 0.0):.2f}</p>
                <p><b>IGST:</b> {invoice_data.get('igst', 0.0):.2f}</p>
                <h3><b>Total: {invoice_data.get('currency', 'INR')} {invoice_data.get('total', 0.0):.2f}</b></h3>
            </div>
        </div>
    </body>
    </html>
    """
    return html_template

@app.post("/api/invoices", status_code=status.HTTP_201_CREATED)
async def create_invoice(payload: NewInvoicePayload, user=Depends(get_current_user)):
    user_profile = await db["users"].find_one({"_id": ObjectId(user["_id"])})
    if not user_profile:
        raise HTTPException(status_code=404, detail="User profile not found")

    user_state = user_profile.get("state")
    customer_state = payload.billTo.state

    subtotal = 0.0
    total_cgst = 0.0
    total_sgst = 0.0
    total_igst = 0.0
    transformed_items = []

    for item in payload.items:
        line_amount = item.quantity * item.unitPrice
        discount_amount = line_amount * (item.discount / 100) if item.discount else 0
        taxable_amount = line_amount - discount_amount
        
        gst_amount = taxable_amount * (item.gst / 100) if item.gst else 0

        if user_state and customer_state and user_state.lower() == customer_state.lower():
            total_cgst += gst_amount / 2
            total_sgst += gst_amount / 2
        else:
            total_igst += gst_amount

        transformed_items.append({
            "description": item.description,
            "hsn": item.hsn,
            "quantity": item.quantity,
            "unitPrice": item.unitPrice,
            "gst": item.gst,
            "discount": item.discount,
            "amount": taxable_amount
        })
        subtotal += taxable_amount

    total_tax = total_cgst + total_sgst + total_igst
    final_total = (subtotal - (payload.discount or 0)) + (payload.shipping or 0) + total_tax

    invoice_doc = {
        "user": ObjectId(user["_id"]),
        "invoiceNumber": payload.invoiceNumber,
        "date": payload.date,
        "dueDate": payload.dueDate,
        "billTo": {
            "name": payload.billTo.companyName or payload.billTo.name,
            "email": payload.billTo.email,
            "address": payload.billTo.address,
            "state": payload.billTo.state,
            "gst": payload.billTo.gst,
            "pan": payload.billTo.pan,
            "phone": payload.billTo.phone,
        },
        "shipTo": payload.shipTo,
        "items": transformed_items,
        "notes": payload.notes,
        "currency": payload.currency,
        "status": payload.status,
        "subtotal": subtotal,
        "discount": payload.discount,
        "shipping": payload.shipping,
        "cgst": round(total_cgst, 2),
        "sgst": round(total_sgst, 2),
        "igst": round(total_igst, 2),
        "total": round(final_total, 2),
        "termsAndConditions": payload.termsAndConditions,
        "paymentTerms": payload.paymentTerms,
        "createdAt": datetime.utcnow()
    }

    result = await db["invoices"].insert_one(invoice_doc)
    created_invoice = await db["invoices"].find_one({"_id": result.inserted_id})
    
    return convert_objids(created_invoice)

@app.get("/api/invoices")
async def get_invoices(
    status: Optional[str] = None,
    month: Optional[int] = Query(None, ge=1, le=12),
    year: Optional[int] = Query(None, ge=2000, le=2100),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    user=Depends(get_current_user)
):
    query = {}
    query["user"] = ObjectId(user["_id"])
    if status and status.lower() != "all":
        query["status"] = status.lower()
    if year:
        start_month = month or 1
        end_month = month or 12
        start_date = datetime(year, start_month, 1)
        days_in_month = calendar.monthrange(year, end_month)[1]
        end_date = datetime(year, end_month, days_in_month, 23, 59, 59)
        query["date"] = {"$gte": start_date, "$lte": end_date}

    skip = (page - 1) * limit
    total_invoices = await db["invoices"].count_documents(query)
    invoices_cursor = db["invoices"].find(query).sort("date", -1).skip(skip).limit(limit)
    invoices = await invoices_cursor.to_list(limit)
    
    return {
        "data": [convert_objids(inv) for inv in invoices],
        "total": total_invoices,
        "page": page,
        "totalPages": (total_invoices + limit - 1) // limit if limit > 0 else 0
    }

@app.get("/api/invoices/metrics")
async def get_invoice_metrics(year: Optional[int] = None, user=Depends(get_current_user)):
    query = {}
    query ["user"]= ObjectId(user["_id"])
    if year:
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31, 23, 59, 59)
        query["date"] = {"$gte": start_date, "$lte": end_date}
    
    total_invoices = await db.invoices.count_documents(query)
    paid_invoices_count = await db.invoices.count_documents({**query, "status": "paid"})
    pending_invoices_count = await db.invoices.count_documents({**query, "status": "pending"})
    
    receivables_pipeline = [{"$match": query}, {"$group": {"_id": None, "total": {"$sum": "$total"}}}]
    total_receivables_result = await db.invoices.aggregate(receivables_pipeline).to_list(1)
    total_receivables = total_receivables_result[0]["total"] if total_receivables_result else 0
    
    overdue_pipeline = [{"$match": {**query, "status": "overdue"}}, {"$group": {"_id": None, "total": {"$sum": "$total"}}}]
    overdue_amount_result = await db.invoices.aggregate(overdue_pipeline).to_list(1)
    overdue_amount = overdue_amount_result[0]["total"] if overdue_amount_result else 0
    
    exp_query = {"userId": user["_id"]}
    if year:
        exp_query["date"] = {"$gte": start_date, "$lte": end_date}
    outgoing_pipeline = [{"$match": exp_query}, {"$group": {"_id": None, "total": {"$sum": "$total"}}}]
    outgoing_result = await db.expenseinvoices.aggregate(outgoing_pipeline).to_list(1)
    outgoing = outgoing_result[0]["total"] if outgoing_result else 0

    return {
        "totalInvoices": total_invoices,
        "paidInvoices": paid_invoices_count,
        "pendingInvoices": pending_invoices_count,
        "totalReceivables": total_receivables,
        "unpaidInvoices": pending_invoices_count,
        "overdueAmount": overdue_amount,
        "cashAmount": total_receivables,
        "incoming": total_receivables,
        "outgoing": outgoing,
        "changePercentage": random.uniform(1.0, 5.0)
    }

@app.get("/api/invoices/{invoice_id}")
async def get_invoice(invoice_id: str, user=Depends(get_current_user)):
    try:
        obj_id = ObjectId(invoice_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid invoice ID format")
    
    invoice = await db["invoices"].find_one({"_id": obj_id, "user": ObjectId(user["_id"])})
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")
    return convert_objids(invoice)

@app.put("/api/invoices/{invoice_id}")
async def update_invoice(invoice_id: str, invoice_update: Invoice, user=Depends(get_current_user)):
    try:
        obj_id = ObjectId(invoice_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid invoice ID format")

    update_data = invoice_update.dict(exclude_unset=True)
    if "user" in update_data:
        del update_data["user"]
    if "_id" in update_data:
        del update_data["_id"]

    result = await db["invoices"].update_one(
        {"_id": obj_id, "user": ObjectId(user["_id"])},
        {"$set": update_data}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Invoice not found or you do not have permission to edit it")
    
    updated_invoice = await db["invoices"].find_one({"_id": obj_id})
    return {"message": "Invoice updated successfully", "invoice": convert_objids(updated_invoice)}


@app.delete("/api/invoices/{invoice_id}")
async def delete_invoice(invoice_id: str, user=Depends(get_current_user)):
    try:
        obj_id = ObjectId(invoice_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid invoice ID format")
    result = await db["invoices"].delete_one({"_id": obj_id, "user": ObjectId(user["_id"])})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Invoice not found")
    return {"message": "Invoice deleted successfully"}

@app.get("/api/invoices/{invoice_id}/download")
async def download_invoice(invoice_id: str, user=Depends(get_current_user)):
    try:
        obj_id = ObjectId(invoice_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid invoice ID format")

    invoice = await db["invoices"].find_one({"_id": obj_id, "user": ObjectId(user["_id"])})
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")
        
    user_profile = await db["users"].find_one({"_id": ObjectId(user["_id"])})
    if not user_profile:
        raise HTTPException(status_code=404, detail="User profile not found")

    html_content = generate_invoice_html(invoice, user_profile)
    pdf_bytes = pdfkit.from_string(html_content, False)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=invoice-{invoice.get('invoiceNumber', 'download')}.pdf"}
    )

@app.get("/api/invoices/export")
async def export_invoices(
    status: Optional[str] = None,
    month: Optional[int] = Query(None, ge=1, le=12),
    year: Optional[int] = Query(None, ge=2000, le=2100),
    user=Depends(get_current_user)
):
    query = {}
    query["user"] = ObjectId(user["_id"])
    if status and status.lower() != "all":
        query["status"] = status.lower()
    if year:
        start_month = month or 1
        end_month = month or 12
        start_date = datetime(year, start_month, 1)
        days_in_month = calendar.monthrange(year, end_month)[1]
        end_date = datetime(year, end_month, days_in_month, 23, 59, 59)
        query["date"] = {"$gte": start_date, "$lte": end_date}

    invoices = await db["invoices"].find(query).to_list(None)
    
    if not invoices:
        return Response(status_code=204)
        
    export_data = []
    for inv in invoices:
        export_data.append({
            "Invoice #": inv.get("invoiceNumber"),
            "Date": inv.get("date").strftime('%Y-%m-%d') if inv.get("date") else None,
            "Customer": inv.get("billTo", {}).get("name"),
            "Amount": inv.get("total"),
            "Status": inv.get("status")
        })

    df = pd.DataFrame(export_data)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    return StreamingResponse(iter([stream.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=invoices.csv"})


@app.get("/api/expenses")
async def get_expense_invoices(user=Depends(get_current_user)):
    invoices = await db["expenseinvoices"].find({"userId": user["_id"]}).to_list(100)
    invoices = [convert_objids(invoice) for invoice in invoices]
    return {
        "success": True,
        "data": {
            "expenses": invoices,
            "pagination": {
                "total": len(invoices),
                "page": 1,
                "limit": 100
            }
        }
    }

@app.get("/api/expenses/metrics")
async def get_expense_metrics(user=Depends(get_current_user)):
    '''Expected Response -> {
  "totalExpenses": { "amount": 50000, "trend": "up", "percentageChange": "11.11" },
  "paidExpenses": { "amount": 30000, "trend": "up", "percentageChange": "20.00" },
  "pendingExpenses": { "amount": 15000, "trend": "up", "percentageChange": "0.00" },
  "overdueExpenses": { "amount": 5000, "trend": "up", "percentageChange": "0.00" }
}'''
    query = {"userId": user["_id"]}
    total_expense_this_month = await db["expenseinvoices"].aggregate([
        {"$match": {**query, "date": {"$gte": datetime.utcnow().replace(day=1, hour=0, minute=0, second=0)}}},
        {"$group": {"_id": None, "total": {"$sum": "$total"}}}
    ]).to_list(1)
    total_expense_previous_month = await db["expenseinvoices"].aggregate([
        {"$match": {**query, "date": {"$gte": (datetime.utcnow().replace(day=1, hour=0, minute=0, second=0) - timedelta(days=30))}}},
        {"$group": {"_id": None, "total": {"$sum": "$total"}}}
    ]).to_list(1)
    total_expense_this_month = total_expense_this_month[0]["total"] if total_expense_this_month else 0
    total_expense_previous_month = total_expense_previous_month[0]["total"] if total_expense_previous_month else 0
    expense_trend = calculate_trend(total_expense_this_month, total_expense_previous_month)
    total_expenses = {
        "amount": total_expense_this_month,
        "trend": expense_trend["trend"],
        "percentageChange": str(expense_trend["change"])
    }
    paid_expenses_this_month = await db["expenseinvoices"].aggregate([
        {"$match": {**query, "status": "paid", "date": {"$gte": datetime.utcnow().replace(day=1, hour=0, minute=0, second=0)}}},
        {"$group": {"_id": None, "total": {"$sum": "$total"}}}
    ]).to_list(1)
    paid_expenses_previous_month = await db["expenseinvoices"].aggregate([
        {"$match": {**query, "status": "paid", "date": {"$gte": (datetime.utcnow().replace(day=1, hour=0, minute=0, second=0) - timedelta(days=30))}}},
        {"$group": {"_id": None, "total": {"$sum": "$total"}}}
    ]).to_list(1)
    paid_expenses_this_month = paid_expenses_this_month[0]["total"] if paid_expenses_this_month else 0
    paid_expenses_previous_month = paid_expenses_previous_month[0]["total"] if paid_expenses_previous_month else 0
    paid_expenses_trend = calculate_trend(paid_expenses_this_month, paid_expenses_previous_month)
    paid_expenses = {
        "amount": paid_expenses_this_month,
        "trend": paid_expenses_trend["trend"],
        "percentageChange": str(paid_expenses_trend["change"])
    }
    pending_expenses_this_month = await db["expenseinvoices"].aggregate([
        {"$match": {**query, "status": "pending", "date": {"$gte": datetime.utcnow().replace(day=1, hour=0, minute=0, second=0)}}},
        {"$group": {"_id": None, "total": {"$sum": "$total"}}}
    ]).to_list(1)
    pending_expenses_previous_month = await db["expenseinvoices"].aggregate([
        {"$match": {**query, "status": "pending", "date": {"$gte": (datetime.utcnow().replace(day=1, hour=0, minute=0, second=0) - timedelta(days=30))}}},
        {"$group": {"_id": None, "total": {"$sum": "$total"}}}
    ]).to_list(1)
    pending_expenses_this_month = pending_expenses_this_month[0]["total"] if pending_expenses_this_month else 0
    pending_expenses_previous_month = pending_expenses_previous_month[0]["total"] if pending_expenses_previous_month else 0
    pending_expenses_trend = calculate_trend(pending_expenses_this_month, pending_expenses_previous_month)
    pending_expenses = {
        "amount": pending_expenses_this_month,
        "trend": pending_expenses_trend["trend"],
        "percentageChange": str(pending_expenses_trend["change"])
    }
    overdue_expenses_this_month = await db["expenseinvoices"].aggregate([
        {"$match": {**query, "status": "overdue", "date": {"$gte": datetime.utcnow().replace(day=1, hour=0, minute=0, second=0)}}},
        {"$group": {"_id": None, "total": {"$sum": "$total"}}}
    ]).to_list(1)
    overdue_expenses_previous_month = await db["expenseinvoices"].aggregate([
        {"$match": {**query, "status": "overdue", "date": {"$gte": (datetime.utcnow().replace(day=1, hour=0, minute=0, second=0) - timedelta(days=30))}}},
        {"$group": {"_id": None, "total": {"$sum": "$total"}}}
    ]).to_list(1)
    overdue_expenses_this_month = overdue_expenses_this_month[0]["total"] if overdue_expenses_this_month else 0
    overdue_expenses_previous_month = overdue_expenses_previous_month[0]["total"] if overdue_expenses_previous_month else 0
    overdue_expenses_trend = calculate_trend(overdue_expenses_this_month, overdue_expenses_previous_month)
    overdue_expenses = {
        "amount": overdue_expenses_this_month,
        "trend": overdue_expenses_trend["trend"],
        "percentageChange": str(overdue_expenses_trend["change"])
    }
    return {
        "totalExpenses": total_expenses,
        "paidExpenses": paid_expenses,
        "pendingExpenses": pending_expenses,
        "overdueExpenses": overdue_expenses
    }
    


@app.post("/api/expenses", status_code=status.HTTP_201_CREATED)
async def create_expense_from_steps(expense_data: NewExpenseFromSteps, user=Depends(get_current_user)):
    try:
        step1 = expense_data.step1
        step2 = expense_data.step2
        step3 = expense_data.step3
        step4 = expense_data.step4

        transformed_items = []
        subtotal = 0.0
        for item in step3.items:
            item_total = item.qty * item.price
            subtotal += item_total
            transformed_items.append({
                "description": item.name,
                "hsn": item.hsn,
                "quantity": item.qty,
                "price": item.price,
                "total": item_total,
            })
        
        if step4.shipping and step4.shipping > 0:
            transformed_items.append({
                "description": "Shipping Charges",
                "quantity": 1,
                "price": step4.shipping,
                "total": step4.shipping
            })

        total_after_discount = (subtotal + (step4.shipping or 0)) - (step4.discount or 0)
        final_total = total_after_discount + (step4.cgst or 0) + (step4.sgst or 0) + (step4.igst or 0)
        
        expense_doc = {
            "userId": user["_id"],
            "invoiceNumber": step1.expenseNumber,
            "date": step1.expenseDate,
            "dueDate": step1.dueDate,
            "currency": step1.currency,
            "status": step1.status,
            "notes": step1.notes,
            "paymentMethod": step1.paymentMethod,
            "billFrom": {
                "name": step2.vendorName,
                "address": step2.billingAddress,
                "email": step2.email
            },
            "shipTo": {
                "address": step2.shippingAddress
            },
            "items": transformed_items,
            "subtotal": subtotal,
            "discount": step4.discount,
            "shipping": step4.shipping,
            "cgst": step4.cgst,
            "sgst": step4.sgst,
            "igst": step4.igst,
            "total": final_total,
            "createdAt": datetime.utcnow()
        }

        result = await db["expenseinvoices"].insert_one(expense_doc)
        created_expense = await db["expenseinvoices"].find_one({"_id": result.inserted_id})

        return convert_objids(created_expense)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create expense: {str(e)}")

@app.post("/api/invoices/{invoice_id}/duplicate")
async def duplicate_invoice(invoice_id: str, user=Depends(get_current_user)):
    invoice = await db["invoices"].find_one({"_id": ObjectId(invoice_id), "user": ObjectId(user["_id"])})
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")
    del invoice["_id"]
    invoice["invoiceNumber"] += "-COPY"
    invoice["date"] = datetime.now()
    result = await db["invoices"].insert_one(invoice)
    invoice["_id"] = str(result.inserted_id)
    return convert_objids(invoice)

@app.post("/api/expenses/{invoice_id}/duplicate")
async def duplicate_expense_invoice(invoice_id: str, user=Depends(get_current_user)):
    invoice = await db["expenseinvoices"].find_one({"_id": ObjectId(invoice_id), "userId": user["_id"]})
    if not invoice:
        raise HTTPException(status_code=404, detail="Expense invoice not found")
    del invoice["_id"]
    invoice["invoiceNumber"] += "-COPY"
    invoice["date"] = datetime.now()
    result = await db["expenseinvoices"].insert_one(invoice)
    invoice["_id"] = str(result.inserted_id)
    return convert_objids(invoice)

@app.delete("/api/expenses/{id}")
async def delete_expense_invoice(id: str, user=Depends(get_current_user)):
    try:
        obj_id = ObjectId(id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid expense ID format")

    result = await db["expenseinvoices"].delete_one({"_id": obj_id, "userId": user["_id"]})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Expense not found")
    return {"success": True, "message": "Expense deleted successfully"}

@app.post("/api/scan-invoice")
async def scan_invoice(file: UploadFile = File(...), user=Depends(get_current_user)):
    user_profile = await db["users"].find_one({"_id": ObjectId(user["_id"])})
    if not user_profile:
        raise HTTPException(status_code=404, detail="User profile not found")
    
    user_profile_dict = convert_objids(user_profile)
    user_profile_dict.pop("password", None)
    
    try:
        image_bytes = await file.read()
        image_b64 = base64.b64encode(image_bytes).decode()
        
        enhanced_prompt = create_enhanced_invoice_prompt(user_profile_dict)
        
        payload = {
            "contents": [{"parts": [{"text": enhanced_prompt}, {"inlineData": {"data": image_b64, "mimeType": file.content_type}}]}],
            "generationConfig": {"temperature": 0.1, "topP": 0.8, "maxOutputTokens": 4096}
        }
        
        response = requests.post(GEMINI_API_URL, json=payload, timeout=30)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Gemini API error: {response.text}")
        
        gemini_response = response.json()
        extracted_text = gemini_response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        
        if not extracted_text:
            raise HTTPException(status_code=500, detail="No text extracted from invoice")
        
        invoice_type = "income"  
        if "EXPENSE" in extracted_text.upper() or "Invoice Type: EXPENSE" in extracted_text:
            invoice_type = "expense"
        elif "INCOME" in extracted_text.upper() or "Invoice Type: INCOME" in extracted_text:
            invoice_type = "income"
        
        json_prompt = create_json_conversion_prompt(invoice_type)
        db_prompt = json_prompt + extracted_text
        
        db_payload = {
            "contents": [{"parts": [{"text": db_prompt}]}],
            "generationConfig": {"temperature": 0, "topP": 0.9, "maxOutputTokens": 2048}
        }
        
        db_response = requests.post(GEMINI_API_URL, json=db_payload, timeout=30)
        if db_response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Gemini JSON conversion error: {db_response.text}")
        
        db_response_json = db_response.json()
        db_text = db_response_json.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        
        json_text = db_text.replace("```json", "").replace("```", "").strip()
        
        invoice_data = json.loads(json_text)
        
        invoice_data["status"] = "draft"
        detected_type = invoice_data.get("invoiceType", invoice_type)
        
        if detected_type == "income":
            invoice_data["user"] = user["_id"]
            invoice_data.pop("invoiceType", None)
            result = await db["invoices"].insert_one(invoice_data)
        else:
            invoice_data["userId"] = user["_id"]
            invoice_data.pop("invoiceType", None)
            result = await db["expenseinvoices"].insert_one(invoice_data)
        
        return {"success": True, "message": f"{detected_type.title()} invoice saved.", "invoiceId": str(result.inserted_id), "invoiceType": detected_type}
        
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invoice processing failed: {str(e)}")

class Period(str, Enum):
    this_month = "this-month"
    last_month = "last-month"
    this_year = "this-year"
    last_year = "last-year"
    thirty_days = "30-days"
    seven_days = "7-days"
    six_months = "6-months"

class TopProductsBy(str, Enum):
    sales = "sales"
    units = "units"

@app.get("/api/dashboard/metrics")
async def get_dashboard_metrics(dateFrom: Optional[date] = None, dateTo: Optional[date] = None, user=Depends(get_current_user)):
    today = datetime.now()
    start_dt = datetime.combine(dateFrom or today.replace(day=1), datetime.min.time())
    end_dt = datetime.combine(dateTo or today, datetime.max.time())
    
    duration = end_dt - start_dt
    prev_start_dt = start_dt - duration
    prev_end_dt = start_dt - timedelta(seconds=1)

    user_id_obj = ObjectId(user["_id"])

    async def get_sum(coll, filter):
        pipeline = [{"$match": filter}, {"$group": {"_id": None, "total": {"$sum": "$total"}}}]
        res = await db[coll].aggregate(pipeline).to_list(1)
        return res[0]["total"] if res else 0

    total_revenue = await get_sum("invoices", {"user": user_id_obj, "date": {"$gte": start_dt, "$lte": end_dt}})
    paid_invoices = await get_sum("invoices", {"user": user_id_obj, "status": "paid", "date": {"$gte": start_dt, "$lte": end_dt}})
    pending_invoices = await get_sum("invoices", {"user": user_id_obj, "status": "pending", "date": {"$gte": start_dt, "$lte": end_dt}})
    total_expenses = await get_sum("expenseinvoices", {"userId": user["_id"], "date": {"$gte": start_dt, "$lte": end_dt}})

    prev_total_revenue = await get_sum("invoices", {"user": user_id_obj, "date": {"$gte": prev_start_dt, "$lte": prev_end_dt}})
    prev_paid_invoices = await get_sum("invoices", {"user": user_id_obj, "status": "paid", "date": {"$gte": prev_start_dt, "$lte": prev_end_dt}})
    prev_pending_invoices = await get_sum("invoices", {"user": user_id_obj, "status": "pending", "date": {"$gte": prev_start_dt, "$lte": prev_end_dt}})
    prev_total_expenses = await get_sum("expenseinvoices", {"userId": user["_id"], "date": {"$gte": prev_start_dt, "$lte": prev_end_dt}})

    def format_metric(current, previous):
        trend_data = calculate_trend(current, previous)
        return {"amount": current, "trend": trend_data["trend"], "trendPercentage": trend_data["change"]}

    return {
        "totalRevenue": format_metric(total_revenue, prev_total_revenue),
        "paidInvoices": format_metric(paid_invoices, prev_paid_invoices),
        "pendingInvoices": format_metric(pending_invoices, prev_pending_invoices),
        "totalExpenses": format_metric(total_expenses, prev_total_expenses),
    }

@app.get("/api/dashboard/revenue-chart")
async def get_revenue_chart(
    granularity: str = Query("monthly", enum=["daily", "weekly", "monthly"]),
    # FIX: Change type hints from 'date' to 'datetime' to match frontend data
    dateFrom: datetime = Query(default_factory=lambda: datetime.now().replace(day=1, month=1)),
    dateTo: datetime = Query(default_factory=datetime.now),
    user=Depends(get_current_user)
):
    # No need to use datetime.combine anymore as we receive full datetime objects
    start_date = dateFrom
    end_date = dateTo
    
    if granularity == "monthly":
        group_id = {"$dateToString": {"format": "%Y-%m", "date": "$date"}}
        period_format = "%b"
    elif granularity == "weekly":
        group_id = {"$dateToString": {"format": "%Y-%U", "date": "$date"}}
        period_format = "Week %U"
    else:
        group_id = {"$dateToString": {"format": "%Y-%m-%d", "date": "$date"}}
        period_format = "%Y-%m-%d"

    pipeline = [
        {"$match": {"user": ObjectId(user["_id"]), "date": {"$gte": start_date, "$lte": end_date}}},
        {"$group": {
            "_id": {"period_group": group_id, "date": {"$min": "$date"}},
            "revenueAccrued": {"$sum": "$total"},
            "revenueRealised": {"$sum": {"$cond": [{"$eq": ["$status", "paid"]}, "$total", 0]}}
        }},
        {"$sort": {"_id.date": 1}},
        {"$project": {
            "period": {"$dateToString": {"format": period_format, "date": "$_id.date"}}, 
            "revenueAccrued": 1, 
            "revenueRealised": 1, 
            "_id": 0
        }}
    ]
    data = await db["invoices"].aggregate(pipeline).to_list(None)
    return {"data": data}

@app.get("/api/dashboard/cash-flow")
async def get_cash_flow(asOfDate: date = Query(None, alias="date"), user=Depends(get_current_user)):
    if asOfDate is None:
        asOfDate = date.today()

    start_of_month = asOfDate.replace(day=1)
    start_date = datetime.combine(start_of_month, datetime.min.time())
    end_date = datetime.combine(asOfDate, datetime.max.time())
    user_id_obj = ObjectId(user["_id"])

    async def get_paid_sum(collection, user_field, user_id):
        pipeline = [
            {"$match": {user_field: user_id, "status": "paid", "date": {"$gte": start_date, "$lte": end_date}}},
            {"$group": {"_id": None, "total": {"$sum": "$total"}}}
        ]
        result = await db[collection].aggregate(pipeline).to_list(1)
        return result[0]["total"] if result else 0

    incoming = await get_paid_sum("invoices", "user", user_id_obj)
    outgoing = await get_paid_sum("expenseinvoices", "userId", user["_id"])
    
    return {
        "cashPosition": incoming - outgoing,
        "incoming": incoming,
        "outgoing": outgoing,
        "asOfDate": end_date.isoformat()
    }

@app.get("/api/dashboard/stats")
async def get_dashboard_stats(period: Period = Period.this_month, user=Depends(get_current_user)):
    user_id = user["_id"]
    start_date, end_date = get_date_range(period.value)
    prev_start_date, prev_end_date = get_previous_period_range(start_date, end_date)

    async def get_total(collection, field, date_filter):
        pipeline = [
            {"$match": date_filter},
            {"$group": {"_id": None, "total": {"$sum": "$total"}}}
        ]
        result = await db[collection].aggregate(pipeline).to_list(1)
        return result[0]["total"] if result else 0

    current_sales = await get_total("invoices", "user", {"user": ObjectId(user_id), "date": {"$gte": start_date, "$lte": end_date}})
    previous_sales = await get_total("invoices", "user", {"user": ObjectId(user_id), "date": {"$gte": prev_start_date, "$lte": prev_end_date}})

    customer_pipeline = [
        {"$match": {"user": ObjectId(user_id), "date": {"$gte": start_date, "$lte": end_date}}},
        {"$group": {"_id": "$billTo.name"}},
        {"$count": "total"}
    ]
    customer_result = await db.invoices.aggregate(customer_pipeline).to_list(1)
    new_customers = customer_result[0]["total"] if customer_result else 0

    total_sales_invoice_count = await db.invoices.count_documents({"user": ObjectId(user_id)})
    total_expense_invoice_count = await db.expenseinvoices.count_documents({"userId": user_id})
    total_orders = total_sales_invoice_count + total_expense_invoice_count
    refund_requests = await db.invoices.count_documents({
        "user": ObjectId(user_id), 
        "status": "refunded",
        "date": {"$gte": start_date, "$lte": end_date}
    })

    sales_trend = calculate_trend(current_sales, previous_sales)

    return [
        {
            "title": "Total Sales",
            "value": current_sales,
            "change": sales_trend["change"],
            "changeLabel": f"Since last {period.value.split('-')[-1]}",
            "trend": sales_trend["trend"]
        },
        {
            "title": "New Customers",
            "value": new_customers,
            "change": 0,
            "changeLabel": f"In this {period.value.split('-')[-1]}",
            "trend": "neutral"
        },
        {
            "title": "Refund Requests",  
            "value": refund_requests,
            "change": 0,
            "changeLabel": f"In this {period.value.split('-')[-1]}",
            "trend": "neutral"
        },
        {
            "title": "Total Orders",
            "value": total_orders,
            "change": 0,
            "changeLabel": "All time",
            "trend": "neutral"
        }
    ]

@app.get("/api/dashboard/sales-report")
async def get_sales_report(period: Period = Period.this_year, user=Depends(get_current_user)):
    start_date, end_date = get_date_range(period.value)
    year = start_date.year
    user_id = user["_id"]

    async def get_monthly_data(collection, user_filter):
        pipeline = [
            {"$match": {**user_filter, "date": {"$gte": start_date, "$lte": end_date}}},
            {"$group": {
                "_id": {"$month": "$date"},
                "total": {"$sum": "$total"}
            }},
            {"$sort": {"_id": 1}}
        ]
        return await db[collection].aggregate(pipeline).to_list(None)

    sales_data = await get_monthly_data("invoices", {"user": ObjectId(user_id)})
    expenses_data = await get_monthly_data("expenseinvoices", {"userId": user_id})

    sales_map = {item["_id"]: item["total"] for item in sales_data}
    expenses_map = {item["_id"]: item["total"] for item in expenses_data}

    labels = [calendar.month_abbr[i] for i in range(1, 13)]
    final_sales = [sales_map.get(i, 0) for i in range(1, 13)]
    final_expenses = [expenses_map.get(i, 0) for i in range(1, 13)]

    return {
        "labels": labels,
        "datasets": [
            {
                "label": "Total Sales",
                "data": final_sales,
                "backgroundColor": "#22c55e",
                "borderColor": "#22c55e"
            },
            {
                "label": "Total Expenses",
                "data": final_expenses,
                "backgroundColor": "#f87171",
                "borderColor": "#f87171"
            }
        ]
    }

@app.get("/api/dashboard/recent-activity")
async def get_recent_activity(limit: int = 20, user=Depends(get_current_user)):
    user_id = user["_id"]
    
    recent_invoices = await db.invoices.find(
        {"user": ObjectId(user_id)}
    ).sort("date", -1).limit(limit).to_list(limit)

    recent_expenses = await db.expenseinvoices.find(
        {"userId": user_id}
    ).sort("date", -1).limit(limit).to_list(limit)

    def transform_to_activity(doc, type):
        if type == "Income":
            desc = f"Invoice {doc.get('invoiceNumber', 'N/A')} to {doc.get('billTo', {}).get('name', 'N/A')}"
        else:
            desc = f"Expense from {doc.get('billFrom', {}).get('name', 'N/A')} ({doc.get('invoiceNumber', 'N/A')})"
        
        return {
            "description": desc,
            "type": "Invoice" if type == "Income" else "Expense",
            "user": "System",
            "date": doc.get("date"),
            "amount": doc.get("total", 0),
            "status": doc.get("status", "draft").capitalize()
        }

    activities = [transform_to_activity(doc, "Income") for doc in recent_invoices]
    activities.extend([transform_to_activity(doc, "Expense") for doc in recent_expenses])
    
    activities.sort(key=lambda x: x["date"], reverse=True)
    
    return activities[:limit]

@app.get("/api/dashboard/top-products")
async def get_top_products(sortBy: TopProductsBy = Query(TopProductsBy.sales, alias="sortBy"), period: Period = Period.thirty_days, user=Depends(get_current_user)):
    start_date, end_date = get_date_range(period.value)
    user_id = user["_id"]
    
    value_field = "$items.amount" if sortBy == TopProductsBy.sales else "$items.quantity"

    pipeline = [
        {"$match": {"user": ObjectId(user_id), "date": {"$gte": start_date, "$lte": end_date}}},
        {"$unwind": "$items"},
        {"$group": {
            "_id": "$items.description",
            "value": {"$sum": value_field}
        }},
        {"$sort": {"value": -1}}
    ]
    
    all_results = await db.invoices.aggregate(pipeline).to_list(None)
    
    top_products = all_results[:4]
    
    labels = [r["_id"] for r in top_products]
    data = [r["value"] for r in top_products]
    
    if len(all_results) > 4:
        others_total = sum(r["value"] for r in all_results[4:])
        labels.append("Others")
        data.append(others_total)
    
    colors = ["#6366f1", "#34d399", "#60a5fa", "#f87171", "#fbbf24"]

    return {
        "labels": labels,
        "datasets": [{
            "data": data,
            "backgroundColor": colors[:len(data)]
        }]
    }

@app.get("/api/dashboard/top-customers")
async def get_top_customers(period: Period = Period.thirty_days, user=Depends(get_current_user)):
    start_date, end_date = get_date_range(period.value)
    user_id = user["_id"]

    pipeline = [
        {"$match": {"user": ObjectId(user_id), "date": {"$gte": start_date, "$lte": end_date}}},
        {"$group": {
            "_id": "$billTo.name",
            "total": {"$sum": "$total"}
        }},
        {"$sort": {"total": -1}},
        {"$limit": 10}
    ]
    
    results = await db.invoices.aggregate(pipeline).to_list(None)
    
    labels = [r["_id"] for r in results]
    data = [r["total"] for r in results]
    
    colors = ["#6366f1", "#22c55e", "#3b82f6", "#ef4444", "#eab308", "#a855f7", "#14b8a6", "#f97316", "#8b5cf6", "#d946ef"]

    return {
        "labels": labels,
        "datasets": [{
            "label": "Top Customers",
            "data": data,
            "backgroundColor": colors[:len(data)]
        }]
    }

@app.post("/api/login")
async def login(data: dict):
    email = data.get("email")
    password = data.get("password")

    if email:
        email = email.lower()

    user = await db["users"].find_one({"email": email})
    hashed = user.get("password") if user else None
    if not user or not hashed or not password or not pwd_context.verify(password, hashed):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = jwt.encode({"_id": str(user["_id"]), "email": user["email"]}, SECRET_KEY, algorithm="HS256")
    
    # response.set_cookie(
    #     key="authToken",
    #     value=token,
    #     httponly=True,
    #     secure=True,
    #     samesite="none",
    #     max_age=86400,
    #     path="/"
    # )
    return {"token": token}

@app.post("/api/register")
async def register(user: User):
    if user.email:
        user.email = user.email.lower()
        existing = await db["users"].find_one({"email": user.email})
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")

    password = user.password
    if not password:
        raise HTTPException(status_code=400, detail="Password is required")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters long")
    if not any(char.isupper() for char in password):
        raise HTTPException(status_code=400, detail="Password must contain at least one uppercase letter")
    if not any(char.islower() for char in password):
        raise HTTPException(status_code=400, detail="Password must contain at least one lowercase letter")

    hashed_password = pwd_context.hash(user.password)
    user_dict = user.dict()
    user_dict["password"] = hashed_password
    user_dict["createdAt"] = datetime.utcnow()
    result = await db["users"].insert_one(user_dict)
    return {"success": True, "userId": str(result.inserted_id)}



class CustomerCreateRequest(BaseModel):
    customerType: str
    fullName: str
    email: Optional[str] = ""
    phone: Optional[str] = ""
    companyName: Optional[str] = ""
    website: Optional[str] = ""
    billingAddress: Optional[str] = ""
    billingCity: Optional[str] = ""
    billingState: Optional[str] = ""
    billingZip: Optional[str] = ""
    shippingAddress: Optional[str] = ""
    shippingCity: Optional[str] = ""
    shippingState: Optional[str] = ""
    shippingZip: Optional[str] = ""
    pan: Optional[str] = ""
    documents: Optional[List[str]] = []
    gstRegistered: Optional[str] = ""
    gstNumber: Optional[str] = ""
    supplyPlace: Optional[str] = ""
    currency: Optional[str] = ""
    paymentTerms: Optional[str] = ""
    logo: Optional[str] = None
    notes: Optional[str] = ""
    tags: Optional[str] = ""
    name: Optional[str] = ""
    billingAddressLine1: Optional[str] = ""
    billingCountry: Optional[str] = ""
    billingAddressLine2: Optional[str] = ""
    shippingAddressLine1: Optional[str] = ""
    shippingCountry: Optional[str] = ""
    shippingAddressLine2: Optional[str] = ""

@app.post("/api/customers", status_code=status.HTTP_201_CREATED)
async def add_customer(customer: CustomerCreateRequest, user=Depends(get_current_user)):
    try:
        customer_data = customer.dict()
        customer_data['userId'] = ObjectId(user["_id"])
        customer_data['createdAt'] = datetime.utcnow()
        customer_data['status'] = 'active'
        customer_data['balance'] = 0.0
        customer_data['lastInvoice'] = None
        if not customer_data.get('companyName'):
            customer_data['companyName'] = customer_data.get('fullName', "")
        result = await db["customers"].insert_one(customer_data)
        created_customer = await db["customers"].find_one({"_id": result.inserted_id})
        return {"success": True, "customer": convert_objids(created_customer)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding customer: {str(e)}")

@app.get("/api/get_customer")
async def get_customer_pages(
    page: int = Query(1, alias="page"),
    perPage: int = Query(10, alias="limit"),
    user=Depends(get_current_user)
):
    try:
        skip = (page - 1) * perPage
        query = {"userId": ObjectId(user["_id"])}
        #print(f"Query: {query}, Page: {page}, PerPage: {perPage}, Skip: {skip}")
        total = await db["customers"].count_documents(query)
        total_pages = (total + perPage - 1) // perPage
        customers = await db["customers"].find(query).skip(skip).limit(perPage).to_list(perPage)
        #print(customers)
        customers = [convert_objids(customer) for customer in customers]
        return {
            "customers": customers,
            "pagination": {
                "total": total,
                "perPage": perPage,
                "currentPage": page,
                "totalPages": total_pages
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid pagination parameters: {str(e)}")
    
@app.get("/api/pagination")
async def get_pagination(user=Depends(get_current_user)):
    try:
        page = 1
        per_page = 10
        total = await db["customers"].count_documents({"userId": user["_id"]})
        total_pages = (total + per_page - 1) // per_page
        return {
            "pagination": {
                "total": total,
                "perPage": per_page,
                "currentPage": page,
                "totalPages": total_pages
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid Request {str(e)}")

class UpdateInventoryItemBody(BaseModel):
    productName: Optional[str] = None
    category: Optional[str] = None
    unitPrice: Optional[float] = Field(None, gt=0)
    inStock: Optional[int] = Field(None, ge=0)
    discount: Optional[float] = Field(None, ge=0)
    image: Optional[str] = None

class BulkUpdateItem(BaseModel):
    id: str
    unitPrice: Optional[float] = Field(None, gt=0)
    inStock: Optional[int] = Field(None, ge=0)
    discount: Optional[float] = Field(None, ge=0)

class BulkUpdateBody(BaseModel):
    items: List[BulkUpdateItem]

class BulkDeleteBody(BaseModel):
    itemIds: List[str]

class CreateInventoryItemBody(BaseModel):
    productName: str
    category: Optional[str] = None
    unitPrice: float = Field(..., gt=0)
    inStock: int = Field(..., ge=0)
    discount: Optional[float] = Field(None, ge=0)
    image: Optional[str] = None
    note: Optional[str] = None
    vendor: Optional[str] = None
    vendorProductCode: Optional[str] = None
    id: Optional[str] = None

def process_inventory_item(item: Dict[str, Any]) -> Dict[str, Any]:
    inStock = item.get("inStock", 0)
    unitPrice = item.get("unitPrice", 0)
    discount = item.get("discount", 0)
    
    if inStock == 0:
        status = "Out of Stock"
    elif inStock <= 10:
        status = "Low in Stock"
    else:
        status = "In Stock"
        
    item["_id"] = str(item["_id"])
    item["productName"] = item.get("productName")
    item["unitPrice"] = unitPrice
    item["inStock"] = inStock
    item["status"] = status
    item["totalValue"] = (unitPrice * inStock) - discount
    return item

@app.get("/api/inventory/items")
async def get_inventory_items(
    page: int = 1,
    limit: int = 10,
    search: Optional[str] = None,
    category: Optional[str] = None,
    status: Optional[str] = None,
    sortBy: Optional[str] = None,
    sortOrder: str = "asc",
    user=Depends(get_current_user)
):
    query = {"userId": user["_id"]}

    if search:
        query["productName"] = {"$regex": search, "$options": "i"}
    if category:
        query["category"] = category
    if status:
        if status == "In Stock":
            query["inStock"] = {"$gt": 10}
        elif status == "Low in Stock":
            query["inStock"] = {"$gt": 0, "$lte": 10}
        elif status == "Out of Stock":
            query["inStock"] = 0

    sort_options = {}
    if sortBy:
        sort_direction = 1 if sortOrder == "asc" else -1
        sort_options[sortBy] = sort_direction

    skip = (page - 1) * limit
    total_items = await db["inventory"].count_documents(query)
    cursor = db["inventory"].find(query).skip(skip).limit(limit)
    if sort_options:
        cursor = cursor.sort(sort_options)

    items = await cursor.to_list(length=limit)
    
    processed_items = [process_inventory_item(item) for item in items]

    return {
        "success": True,
        "data": processed_items,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total_items,
            "totalPages": (total_items + limit - 1) // limit
        }
    }

@app.post("/api/inventory/items", status_code=status.HTTP_201_CREATED)
async def create_inventory_item(item_data: CreateInventoryItemBody, user=Depends(get_current_user)):
    item_doc = item_data.dict(exclude_unset=True)
    item_doc["userId"] = user["_id"]
    
    if "id" not in item_doc or not item_doc["id"]:
        item_doc["id"] = str(ObjectId())

    existing_item = await db["inventory"].find_one({"id": item_doc["id"], "userId": user["_id"]})
    if existing_item:
        raise HTTPException(status_code=409, detail=f"Inventory item with id {item_doc['id']} already exists.")
        
    result = await db["inventory"].insert_one(item_doc)
    created_item = await db["inventory"].find_one({"_id": result.inserted_id})
    
    if not created_item:
         raise HTTPException(status_code=500, detail="Failed to create inventory item.")

    return {
        "success": True,
        "data": process_inventory_item(created_item)
    }

@app.get("/api/inventory/items/{item_id}")
async def get_inventory_item(item_id: str, user=Depends(get_current_user)):
    item = await db["inventory"].find_one({"id": item_id, "userId": user["_id"]})
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    return {
        "success": True,
        "data": process_inventory_item(item)
    }

@app.put("/api/inventory/items/{item_id}")
async def update_inventory_item(item_id: str, item_update: UpdateInventoryItemBody, user=Depends(get_current_user)):
    update_data = item_update.dict(exclude_unset=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No update data provided")

    result = await db["inventory"].update_one(
        {"id": item_id, "userId": user["_id"]},
        {"$set": update_data}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Item not found")
    
    updated_item = await db["inventory"].find_one({"id": item_id, "userId": user["_id"]})
    if not updated_item:
        raise HTTPException(status_code=500, detail="Error updating item")

    return {
        "success": True,
        "data": process_inventory_item(updated_item)
    }

@app.delete("/api/inventory/items/{item_id}")
async def delete_inventory_item(item_id: str, user=Depends(get_current_user)):
    result = await db["inventory"].delete_one({"id": item_id, "userId": user["_id"]})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"success": True, "message": "Item deleted successfully"}

@app.get("/api/inventory/summary")
async def get_inventory_summary(user=Depends(get_current_user)):
    try:
        query = {"userId": user["_id"]}
        
        all_products = await db["inventory"].count_documents(query)
        if all_products == 0:
             return {
                "success": True,
                "data": {
                    "totals": {"allProducts": 0, "activeProducts": 0, "totalInventoryValue": 0},
                    "stockDistribution": {
                        "inStock": {"count": 0, "percentage": 0, "segments": []},
                        "lowStock": {"count": 0, "percentage": 0, "segments": []},
                        "outOfStock": {"count": 0, "percentage": 0, "segments": []}
                    }
                }
            }

        active_products = await db["inventory"].count_documents({**query, "inStock": {"$gt": 0}})
        
        total_value_cursor = db["inventory"].aggregate([
            {"$match": query},
            {"$group": {"_id": None, "totalValue": {"$sum": {"$subtract": [{"$multiply": ["$inStock", "$unitPrice"]}, {"$ifNull": ["$discount", 0]}] }}}}
        ])
        total_value_result = await total_value_cursor.to_list(1)
        total_inventory_value = total_value_result[0]["totalValue"] if total_value_result else 0

        in_stock_count = await db["inventory"].count_documents({**query, "inStock": {"$gt": 10}})
        low_stock_count = await db["inventory"].count_documents({**query, "inStock": {"$gt": 0, "$lte": 10}})
        out_of_stock_count = await db["inventory"].count_documents({**query, "inStock": 0})
        
        return {
            "success": True,
            "data": {
                "totals": {
                    "allProducts": all_products,
                    "activeProducts": active_products,
                    "totalInventoryValue": total_inventory_value
                },
                "stockDistribution": {
                    "inStock": { "count": in_stock_count, "percentage": (in_stock_count / all_products) * 100 if all_products > 0 else 0, "segments": []},
                    "lowStock": { "count": low_stock_count, "percentage": (low_stock_count / all_products) * 100 if all_products > 0 else 0, "segments": []},
                    "outOfStock": { "count": out_of_stock_count, "percentage": (out_of_stock_count / all_products) * 100 if all_products > 0 else 0, "segments": []}
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting inventory summary: {str(e)}")

@app.put("/api/inventory/items/bulk-update")
async def bulk_update_inventory_items(body: BulkUpdateBody, user=Depends(get_current_user)):
    updated_ids = []
    for item in body.items:
        update_data = item.dict(exclude={'id'}, exclude_unset=True)
        if not update_data:
            continue
        result = await db["inventory"].update_one(
            {"id": item.id, "userId": user["_id"]},
            {"$set": update_data}
        )
        if result.modified_count > 0:
            updated_ids.append(item.id)
    
    return {
        "success": True, 
        "message": f"{len(updated_ids)} items updated successfully",
        "updatedItems": updated_ids
    }
    
@app.delete("/api/inventory/items/bulk-delete")
async def bulk_delete_inventory_items(body: BulkDeleteBody, user=Depends(get_current_user)):
    if not body.itemIds:
        raise HTTPException(status_code=400, detail="No item IDs provided for bulk delete")
        
    result = await db["inventory"].delete_many({"id": {"$in": body.itemIds}, "userId": user["_id"]})
    
    return {
        "success": True, 
        "message": f"{result.deleted_count} items deleted successfully",
        "deletedItems": body.itemIds
    }

@app.get("/api/inventory/export")
async def export_inventory(
    format: str = "csv",
    search: Optional[str] = None,
    category: Optional[str] = None,
    status: Optional[str] = None,
    user=Depends(get_current_user)
):
    query = {"userId": user["_id"]}
    if search:
        query["productName"] = {"$regex": search, "$options": "i"}
    if category:
        query["category"] = category
    if status:
        if status == "In Stock":
            query["inStock"] = {"$gt": 10}
        elif status == "Low in Stock":
            query["inStock"] = {"$gt": 0, "$lte": 10}
        elif status == "Out of Stock":
            query["inStock"] = 0

    items_cursor = db["inventory"].find(query)
    items = await items_cursor.to_list(None)
    
    if not items:
        return Response(status_code=204)
        
    processed_items = [process_inventory_item(item) for item in items]
    df = pd.DataFrame(processed_items)
    df.drop(columns=['_id', 'userId'], inplace=True, errors='ignore')

    if format == "csv":
        output = io.StringIO()
        df.to_csv(output, index=False)
        return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=inventory.csv"})
    elif format == "excel":
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Inventory')
        output.seek(0)
        return StreamingResponse(output, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": "attachment; filename=inventory.xlsx"})
    elif format == "pdf":
        html = df.to_html(index=False)
        pdf = pdfkit.from_string(html, False)
        return Response(content=pdf, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=inventory.pdf"})
    else:
        raise HTTPException(status_code=400, detail="Invalid format specified. Use 'csv', 'excel', or 'pdf'.")


@app.post("/api/inventory/import")
async def import_inventory(file: UploadFile = File(...), user=Depends(get_current_user)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    if not file.filename.endswith(('.csv', '.xlsx')):
        raise HTTPException(status_code=400, detail="Invalid file format. Only CSV and Excel files are supported.")
    try:
        contents = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(contents.decode()))
        else:
            df = pd.read_excel(io.BytesIO(contents))

        if df.empty:
            raise HTTPException(status_code=400, detail="File is empty or invalid format")
            
        items = df.to_dict(orient="records")
        imported_count = 0
        skipped_count = 0
        errors = []

        for record in items:
            record["userId"] = user["_id"]
            if "id" not in record or pd.isna(record["id"]):
                 record["id"] = str(ObjectId())

            if 'productName' not in record or 'unitPrice' not in record or 'inStock' not in record:
                errors.append(f"Skipping record due to missing required fields: {record}")
                skipped_count += 1
                continue

            await db["inventory"].update_one(
                {"id": record["id"], "userId": user["_id"]},
                {"$set": record},
                upsert=True
            )
            imported_count += 1

        return {
            "success": True, 
            "message": f"{imported_count} items imported successfully",
            "errors": errors,
            "importedCount": imported_count,
            "skippedCount": skipped_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error importing inventory: {str(e)}")
    
@app.get("/api/inventory/categories")
async def get_inventory_categories(user=Depends(get_current_user)):
    try:
        categories = await db["inventory"].distinct("category", {"userId": user["_id"], "category": {"$ne": None}})
        return {"success": True, "data": categories}
    except Exception as e: 
        raise HTTPException(status_code=500, detail=f"Error fetching inventory categories: {str(e)}")

@app.get("/")
def root():
    return {"message": "Ok"}

@app.get("/api/invoices/clients")
async def get_invoice_clients(user=Depends(get_current_user)):
    clients = await db["invoices"].distinct("billTo.name", {"user": ObjectId(user["_id"]), "billTo.name": {"$ne": None}})
    return clients

@app.get("/api/invoices/clients/{name}")
async def get_invoice_client_details(name: str, user=Depends(get_current_user)):
    invoice = await db["invoices"].find_one({"user": ObjectId(user["_id"]), "billTo.name": name}, sort=[("date", -1)])
    if invoice and "billTo" in invoice:
        return invoice["billTo"]
    return {}

@app.get("/api/sales")
async def get_sales(user=Depends(get_current_user)):
    sales_cursor = db.invoices.find({"user": ObjectId(user["_id"])})
    sales_records = []
    for inv in await sales_cursor.to_list(None):
        first_item = inv.get("items")[0] if inv.get("items") else {}
        sales_records.append({
            "id": str(inv["_id"]),
            "invoiceNumber": inv.get("invoiceNumber"),
            "customerName": inv.get("billTo", {}).get("name"),
            "product": first_item.get("description"),
            "quantity": first_item.get("quantity"),
            "unitPrice": first_item.get("unitPrice"),
            "totalAmount": inv.get("total"),
            "dateOfSale": inv.get("date").strftime("%d %B %Y"),
            "paymentStatus": inv.get("status", "Paid").capitalize()
        })
    return sales_records

@app.get("/api/sales/stats")
async def get_sales_stats(from_: str = Query(None, alias="from"), to: str = Query(None), user=Depends(get_current_user)):
    query = {}
    query["user"] = ObjectId(user["_id"])
    if from_ and to:
        query["date"] = {"$gte": datetime.fromisoformat(from_), "$lte": datetime.fromisoformat(to)}
    total_sales = await db["invoices"].aggregate([
        {"$match": query},
        {"$group": {"_id": None, "total": {"$sum": "$total"}, "count": {"$sum": 1}}}
    ]).to_list(1)
    current_month = datetime.now().month
    current_year = datetime.now().year
    month_query = {"user": ObjectId(user["_id"]), "date": {"$gte": datetime(current_year, current_month, 1), "$lte": datetime.now()}}
    month_sales = await db["invoices"].aggregate([
        {"$match": month_query},
        {"$group": {"_id": None, "total": {"$sum": "$total"}, "count": {"$sum": 1}}}
    ]).to_list(1)
    total = total_sales[0]["total"] if total_sales else 0
    count = total_sales[0]["count"] if total_sales else 0
    month_total = month_sales[0]["total"] if month_sales else 0
    aov = total/count if count else 0
    return {"totalSales": total, "currentMonthSales": month_total, "averageOrderValue": aov}

@app.get("/api/sales/performance")
async def get_sales_performance(from_: str = Query(None, alias="from"), to: str = Query(None), interval: str = Query("day"), user=Depends(get_current_user)):
    match = {}
    match["user"] = ObjectId(user["_id"])
    if from_ and to:
        match["date"] = {"$gte": datetime.fromisoformat(from_), "$lte": datetime.fromisoformat(to)}
    if interval == "month":
        group_id = {"$dateToString": {"format": "%Y-%m", "date": "$date"}}
    elif interval == "week":
        group_id = {"$dateToString": {"format": "%Y-%U", "date": "$date"}}
    else:
        group_id = {"$dateToString": {"format": "%Y-%m-%d", "date": "$date"}}
    pipeline = [
        {"$match": match},
        {"$group": {"_id": group_id, "sales": {"$sum": "$total"}}},
        {"$sort": {"_id": 1}}
    ]
    data = await db["invoices"].aggregate(pipeline).to_list(None)
    return {"series": [{"date": d["_id"], "sales": d["sales"]} for d in data]}

@app.get("/api/sales/regions")
async def get_sales_regions(from_: str = Query(None, alias="from"), to: str = Query(None), user=Depends(get_current_user)):
    match = {}
    match["user"] = ObjectId(user["_id"])
    if from_ and to:
        match["date"] = {"$gte": datetime.fromisoformat(from_), "$lte": datetime.fromisoformat(to)}
    pipeline = [
        {"$match": match},
        {"$group": {"_id": "$billTo.state", "sales": {"$sum": "$total"}}},
        {"$sort": {"sales": -1}}
    ]
    data = await db["invoices"].aggregate(pipeline).to_list(None)
    return {"regions": [{"name": d["_id"] or "Unknown", "sales": d["sales"]} for d in data]}

@app.post("/api/sales/import")
async def import_sales(body: FileImportBody, user=Depends(get_current_user)):
    try:
        decoded_content = base64.b64decode(body.fileContent)
        
        if body.fileName.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(decoded_content))
        elif body.fileName.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(decoded_content))
        else:
            raise HTTPException(status_code=400, detail="Invalid file format. Only CSV and Excel files are supported.")

        if df.empty:
            raise HTTPException(status_code=400, detail="File is empty or invalid format")
            
        items = df.to_dict(orient="records")
        imported = 0
        skipped = 0
        for record in items:
            record["user"] = ObjectId(user["_id"])
            if "invoiceNumber" not in record or "total" not in record:
                skipped += 1
                continue
            if 'date' in record and record['date']:
                record['date'] = pd.to_datetime(record['date'])
            await db["invoices"].insert_one(record)
            imported += 1
        return {"imported": imported, "skipped": skipped}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error importing sales data: {str(e)}")

@app.get("/api/sales/export")
async def export_sales(user=Depends(get_current_user)):
    invoices = await db["invoices"].find({"user": ObjectId(user["_id"])}).to_list(None)
    if not invoices:
        return Response(status_code=204)
    export_data = []
    for inv in invoices:
        export_data.append({
            "Invoice #": inv.get("invoiceNumber"),
            "Date": inv.get("date").strftime('%Y-%m-%d') if inv.get("date") else None,
            "Customer": inv.get("billTo", {}).get("name"),
            "Amount": inv.get("total"),
            "Status": inv.get("status")
        })
    df = pd.DataFrame(export_data)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    return StreamingResponse(iter([stream.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=sales.csv"})

@app.get("/api/expenses/last")
async def get_last_expense_invoice(user=Depends(get_current_user)):
    invoice = await db["expenseinvoices"].find_one({"userId": user["_id"]}, sort=[("date", -1)])
    if invoice:
        return convert_objids(invoice)
    return None

@app.post("/api/expenses/import")
async def import_expenses(body: FileImportBody, user=Depends(get_current_user)):
    try:
        decoded_content = base64.b64decode(body.fileContent)
        
        if body.fileName.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(decoded_content))
        elif body.fileName.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(decoded_content))
        else:
            raise HTTPException(status_code=400, detail="Invalid file format. Only CSV and Excel files are supported.")

        if df.empty:
            raise HTTPException(status_code=400, detail="File is empty or invalid format")
            
        items = df.to_dict(orient="records")
        imported = 0
        skipped = 0
        for record in items:
            record["userId"] = user["_id"]
            if "invoiceNumber" not in record or "total" not in record:
                skipped += 1
                continue
            if 'date' in record and record['date']:
                record['date'] = pd.to_datetime(record['date'])
            await db["expenseinvoices"].insert_one(record)
            imported += 1
        return {"imported": imported, "skipped": skipped}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error importing expense data: {str(e)}")

@app.get("/api/expenses/export")
async def export_expenses(user=Depends(get_current_user)):
    expenses = await db["expenseinvoices"].find({"userId": user["_id"]}).to_list(None)
    if not expenses:
        return Response(status_code=204)
    export_data = []
    for exp in expenses:
        export_data.append({
            "Invoice #": exp.get("invoiceNumber"),
            "Date": exp.get("date").strftime('%Y-%m-%d') if exp.get("date") else None,
            "Vendor": exp.get("billFrom", {}).get("name"),
            "Amount": exp.get("total"),
            "Status": exp.get("status")
        })
    df = pd.DataFrame(export_data)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    return StreamingResponse(iter([stream.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=expenses.csv"})

@app.get("/api/tax/metrics")
async def get_tax_metrics(date: str = Query(None), from_: str = Query(None, alias="from"), to: str = Query(None), user=Depends(get_current_user)):
    match = {}
    match["user"] = ObjectId(user["_id"])
    if date:
        dt = datetime.fromisoformat(date)
        match["date"] = {"$gte": dt.replace(hour=0, minute=0, second=0, microsecond=0), "$lte": dt.replace(hour=23, minute=59, second=59, microsecond=999999)}
    elif from_ and to:
        match["date"] = {"$gte": datetime.fromisoformat(from_), "$lte": datetime.fromisoformat(to)}
    pipeline = [
        {"$match": match},
        {"$group": {
            "_id": None,
            "taxCollected": {"$sum": {"$add": ["$cgst", "$sgst", "$igst"]}},
            "taxableSales": {"$sum": "$subtotal"},
        }}
    ]
    result = await db["invoices"].aggregate(pipeline).to_list(1)
    tax_collected = result[0]["taxCollected"] if result else 0
    taxable_sales = result[0]["taxableSales"] if result else 0
    pipeline_exp = [
        {"$match": {"userId": user["_id"]}},
        {"$group": {"_id": None, "taxPaid": {"$sum": {"$add": ["$cgst", "$sgst", "$igst"]}}}}
    ]
    result_exp = await db["expenseinvoices"].aggregate(pipeline_exp).to_list(1)
    tax_paid = result_exp[0]["taxPaid"] if result_exp else 0
    net_tax_liability = tax_collected - tax_paid
    return {"taxCollected": tax_collected, "taxPaid": tax_paid, "netTaxLiability": net_tax_liability, "taxableSales": taxable_sales}

@app.get("/api/tax/collected-timeseries")
async def get_tax_collected_timeseries(from_: str = Query(None, alias="from"), to: str = Query(None), interval: str = Query("day"), user=Depends(get_current_user)):
    match = {}
    match["user"] = ObjectId(user["_id"])
    if from_ and to:
        # FIX: Make date parsing robust by handling the 'Z' suffix
        from_dt = datetime.fromisoformat(from_.replace('Z', '+00:00'))
        to_dt = datetime.fromisoformat(to.replace('Z', '+00:00'))
        match["date"] = {"$gte": from_dt, "$lte": to_dt}

    if interval == "month":
        group_id = {"$dateToString": {"format": "%Y-%m", "date": "$date"}}
    elif interval == "week":
        group_id = {"$dateToString": {"format": "%Y-%U", "date": "$date"}}
    else:
        group_id = {"$dateToString": {"format": "%Y-%m-%d", "date": "$date"}}
    
    pipeline = [
        {"$match": match},
        {"$group": {"_id": group_id, "collected": {"$sum": {"$add": ["$cgst", "$sgst", "$igst"]}}}},
        {"$sort": {"_id": 1}}
    ]
    collected = await db["invoices"].aggregate(pipeline).to_list(None)
    
    match_exp = {"userId": user["_id"]}
    if from_ and to:
        # FIX: Apply the same robust date parsing here
        from_dt = datetime.fromisoformat(from_.replace('Z', '+00:00'))
        to_dt = datetime.fromisoformat(to.replace('Z', '+00:00'))
        match_exp["date"] = {"$gte": from_dt, "$lte": to_dt}

    pipeline_exp = [
        {"$match": match_exp},
        {"$group": {"_id": group_id, "paid": {"$sum": {"$add": ["$cgst", "$sgst", "$igst"]}}}},
        {"$sort": {"_id": 1}}
    ]
    paid = await db["expenseinvoices"].aggregate(pipeline_exp).to_list(None)
    paid_map = {d["_id"]: d["paid"] for d in paid}
    series = []
    for c in collected:
        series.append({"date": c["_id"], "collected": c["collected"], "paid": paid_map.get(c["_id"], 0)})
    
    return {"series": series}



@app.get("/api/tax/summary/export")
async def export_tax_summary(from_: str = Query(None, alias="from"), to: str = Query(None), groupBy: str = Query(None), user=Depends(get_current_user)):
    summary_data = await get_tax_summary(user=user) # Simplified call for this example
    if not summary_data:
        return Response(status_code=204)
    df = pd.DataFrame(summary_data)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    return StreamingResponse(iter([stream.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=tax_summary.csv"})

@app.post("/api/send-otp-register")
async def send_otp_register(data: dict = Body(...)):
    phonenumber = data.get("phonenumber")
    email = data.get("email")
    password = data.get("password")
    if not password:
        return {'message': 'Error Sending OTP'}
    name = data.get("name")
    website = data.get("website")
    if not all([phonenumber, password, email, name]):
        raise HTTPException(status_code=400, detail="Phone number, email, password and name are required")
    otp = str(random.randint(100000, 999999))
    otp_expiration = datetime.utcnow() + timedelta(minutes=10)
    await db["pendinguser"].delete_many({"phonenumber": phonenumber})
    await db["pendinguser"].insert_one({
        "phonenumber": phonenumber,
        "email": email,
        "password": pwd_context.hash(password),
        "name": name,
        "website": website,
        "otp": otp,
        "otpExpiration": otp_expiration
    })
    if not phonenumber:
        raise HTTPException(status_code=400, detail="Phone number is required")
    api_key = os.getenv("EDUMARC_SMS_API_KEY", "")
    result = await send_otp_helper(str(phonenumber), otp, db, api_key)
    return {"message": "OTP sent for registration", "otp": otp, **result}

@app.post("/api/verify-otp-register")
async def verify_otp_register(data: dict = Body(...)):
    phonenumber = data.get("phonenumber")
    otp = data.get("otp")
    if not phonenumber or not otp:
        raise HTTPException(status_code=400, detail="Phone and OTP required")
    otp_doc = await db["otpverification"].find_one({"phonenumber": phonenumber, "otp": otp})
    if not otp_doc or otp_doc["otpExpiration"] < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")
    pending = await db["pendinguser"].find_one({"phonenumber": phonenumber, "otp": otp})
    if not pending:
        raise HTTPException(status_code=400, detail="No pending registration found")
    user_data = {
        "email": pending["email"],
        "password": pending["password"],
        "name": pending["name"],
        "phonenumber": pending["phonenumber"],
        "website": pending["website"],
        "createdAt": datetime.utcnow()
    }
    await db["users"].insert_one(user_data)
    await db["pendinguser"].delete_many({"phonenumber": phonenumber})
    await db["otpverification"].delete_many({"phonenumber": phonenumber})
    return {"message": "OTP verified and user registered"}

@app.post("/api/send-otp-login")
async def send_otp_login(data: dict = Body(...)):
    phonenumber = data.get("phonenumber")
    if not phonenumber:
        raise HTTPException(status_code=400, detail="Phone required")
    user = await db["users"].find_one({"phonenumber": phonenumber})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    otp = str(random.randint(100000, 999999))
    api_key = os.getenv("EDUMARC_SMS_API_KEY", "")
    result = await send_otp_helper(str(phonenumber), otp, db, api_key)
    return {"message": "OTP sent for login", "otp": otp, **result}

@app.post("/api/verify-otp-login")
async def verify_otp_login(data: dict = Body(...)):
    phonenumber = data.get("phonenumber")
    otp = data.get("otp")
    if not phonenumber or not otp:
        raise HTTPException(status_code=400, detail="Phone and OTP required")
    otp_doc = await db["otpverification"].find_one({"phonenumber": phonenumber, "otp": otp})
    if not otp_doc or otp_doc["otpExpiration"] < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")
    user = await db["users"].find_one({"phonenumber": phonenumber})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    token = jwt.encode({"_id": str(user["_id"]), "email": user["email"]}, SECRET_KEY, algorithm="HS256")
    await db["otpverification"].delete_many({"phonenumber": phonenumber})
    return {"message": "OTP verified and user logged in", "token": token}

@app.get("/api/me")
async def get_me(user=Depends(get_current_user)):
    profile = await db["users"].find_one({"_id": ObjectId(user["_id"])})
    if not profile:
        raise HTTPException(status_code=404, detail="User not found")
    if profile is not None:
        profile.pop("password", None)
    return convert_objids(profile)

@app.get("/api/test")
async def test():
    return {"message": "Test endpoint working"}

def normalize_phone(phonenumber: str) -> str:
    return phonenumber.strip().replace(" ", "")

async def send_otp_helper(phonenumber: str, otp: str, db, api_key: str):
    phone = normalize_phone(phonenumber)
    otp_expiration = datetime.utcnow() + timedelta(minutes=5)

    await db["otpverification"].update_one(
        {"phonenumber": phone},
        {"$set": {"otp": otp, "otpExpiration": otp_expiration}},
        upsert=True
    )

    message = f"Your Invoicely OTP for verification is: {otp}. OTP is confidential, refrain from sharing it with anyone. By Edumarc Technologies"

    payload = {
        "number": [phone],
        "message": message,
        "senderId": "EDUMRC",
        "templateId": "1707168926925165526",
    }
    headers = {
        "apikey": api_key,
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                "https://smsapi.edumarcsms.com/api/v1/sendsms",
                json=payload,
                headers=headers,
            )
        data = response.json()
        if data.get("success"):
            return {"message": "OTP sent successfully"}
        else:
            raise HTTPException(status_code=500, detail={"message": "Failed to send OTP", "details": data})
    except Exception as err:
        raise HTTPException(status_code=500, detail={"message": "Internal server error", "error": str(err)})

# @app.get("/api/get_customer")
# async def get_customers(page: int = 1, limit: int = 10, user=Depends(get_current_user)):
#     skip = (page - 1) * limit
#     query = {"userId": ObjectId(user["_id"])}
#     total_customers = await db.customers.count_documents(query)
#     customers_cursor = db.customers.find(query).sort("createdAt", -1).skip(skip).limit(limit)
#     customers = await customers_cursor.to_list(limit)

#     response_data = []
#     for customer in customers:
#         response_data.append({
#             "id": str(customer["_id"]),
#             "company": {
#                 "name": customer.get("companyName"),
#                 "email": customer.get("email"),
#                 "logo": customer.get("logo")
#             },
#             "customer": {
#                 "name": customer.get("fullName"),
#                 "avatar": customer.get("logo") or "/avatars/user_default.png"
#             },
#             "phone": customer.get("phone"),
#             "status": customer.get("status", "Active"),
#             "lastInvoice": customer.get("lastInvoice").strftime("%Y-%m-%d") if customer.get("lastInvoice") else None,
#             "balance": customer.get('balance', 0.0)
#         })
    
#     return {
#         "data": response_data,
#         "pagination": {
#             "total": total_customers,
#             "perPage": limit,
#             "currentPage": page,
#             "totalPages": (total_customers + limit - 1) // limit if limit > 0 else 0
#         }
#     }

@app.get("/api/expenses")
async def get_expenses(user=Depends(get_current_user)):
    expenses_cursor = db.expenseinvoices.find({"userId": user["_id"]})
    expense_records = []
    for exp in await expenses_cursor.to_list(None):
        first_item = exp.get("items")[0] if exp.get("items") else {}
        vendor_name = exp.get("billFrom", {}).get("name", "Vendor Name")
        expense_records.append({
            "id": f"expense-{str(exp['_id'])}",
            "expenseId": exp.get("invoiceNumber"),
            "title": first_item.get("description", "Expense Item"),
            "vendorName": vendor_name,
            "vendorAvatar": vendor_name[0] if vendor_name else "V",
            "paymentMethod": "Cash",
            "amount": exp.get("total"),
            "status": exp.get("status", "Paid").capitalize(),
            "date": exp.get("date").strftime("%d %B %Y")
        })
    return expense_records

@app.get("/api/purchases")
async def get_purchases(page: int = 1, limit: int = 10, filters: Optional[str] = None, user=Depends(get_current_user)):
    skip = (page - 1) * limit
    query = {"userId": user["_id"]}
    if filters:
        try:
            filter_obj = json.loads(filters)
            if "status" in filter_obj:
                query["status"] = filter_obj["status"]
        except json.JSONDecodeError:
            pass

    total_purchases = await db.expenseinvoices.count_documents(query)
    purchases_cursor = db.expenseinvoices.find(query).skip(skip).limit(limit)
    purchase_records = []
    for pur in await purchases_cursor.to_list(limit):
        first_item = pur.get("items")[0] if pur.get("items") else {}
        purchase_records.append({
            "id": str(pur["_id"]),
            "purchaseId": pur.get("invoiceNumber"),
            "supplierName": pur.get("billFrom", {}).get("name", "Supplier Name"),
            "supplierAvatar": "/placeholder.svg?height=32&width=32",
            "product": first_item.get("description", "Purchase Item"),
            "quantity": first_item.get("quantity"),
            "balance": random.randint(1000, 5000),
            "purchaseDate": pur.get("date").strftime("%d %B %Y"),
            "totalAmount": pur.get("total"),
            "paymentStatus": pur.get("status", "Paid").capitalize()
        })
    return {
        "data": purchase_records,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total_purchases,
            "totalPages": (total_purchases + limit - 1) // limit
        }
    }

@app.get("/api/tax/summary")
async def get_tax_summary(user=Depends(get_current_user)):
    pipeline = [
        {"$match": {"user": ObjectId(user["_id"])}},
        {"$project": {
            "_id": 1,
            "cgst": {"$ifNull": ["$cgst", 0]},
            "sgst": {"$ifNull": ["$sgst", 0]},
            "igst": {"$ifNull": ["$igst", 0]},
            "subtotal": {"$ifNull": ["$subtotal", 0]},
            "date": "$date",
        }}
    ]
    invoices = await db.invoices.aggregate(pipeline).to_list(None)
    
    tax_summary = {
        "CGST": {"taxable": 0, "collected": 0, "invoices": set()},
        "SGST": {"taxable": 0, "collected": 0, "invoices": set()},
        "IGST": {"taxable": 0, "collected": 0, "invoices": set()},
    }

    for inv in invoices:
        if inv["cgst"] > 0:
            tax_summary["CGST"]["collected"] += inv["cgst"]
            tax_summary["CGST"]["taxable"] += inv["subtotal"]
            tax_summary["CGST"]["invoices"].add(str(inv["_id"]))
        if inv["sgst"] > 0:
            tax_summary["SGST"]["collected"] += inv["sgst"]
            tax_summary["SGST"]["taxable"] += inv["subtotal"]
            tax_summary["SGST"]["invoices"].add(str(inv["_id"]))
        if inv["igst"] > 0:
            tax_summary["IGST"]["collected"] += inv["igst"]
            tax_summary["IGST"]["taxable"] += inv["subtotal"]
            tax_summary["IGST"]["invoices"].add(str(inv["_id"]))

    result = []
    i = 1
    for tax_type, data in tax_summary.items():
        if data["collected"] > 0:
            result.append({
                "id": str(i),
                "taxType": tax_type,
                "taxRate": "18%", # This is a mock value as per docs
                "taxableAmount": f"₹{data['taxable']:.2f}",
                "taxCollected": f"₹{data['collected']:.2f}",
                "taxPaid": "₹2000.00", # This is a mock value
                "netTaxLiability": "₹5000.00", # This is a mock value
                "period": "29 July 2024", # This is a mock value
                "noOfInvoices": len(data["invoices"]),
                "expanded": False,
                "children": [],
                "isParent": False
            })
            i += 1
    
    return result

@app.get("/api/team-members") 
async def get_team_members(search: Optional[str] = None, page: int = 1, limit: int = 10, status: Optional[TeamMemberStatus] = None, role: Optional[TeamMemberRole] = None, user=Depends(get_current_user)):
    query = {}
    query ["userId"] = ObjectId(user["_id"])
    if search:
        query["$or"] = [
            {"name": {"$regex": search, "$options": "i"}},
            {"email": {"$regex": search, "$options": "i"}},
            {"role": {"$regex": search, "$options": "i"}}
        ]
    if status:
        query["status"] = status.value
    if role:
        query["role"] = role.value
    
    skip = (page - 1) * limit
    total_items = await db.teammembers.count_documents(query)
    cursor = db.teammembers.find(query).skip(skip).limit(limit)
    members = await cursor.to_list(limit)

    def format_member(member):
        date_joined_obj = member.get("joiningDate")
        date_joined_str = date_joined_obj.strftime("%Y-%m-%d") if isinstance(date_joined_obj, (datetime, date)) else "N/A"

        last_active_obj = member.get("lastActive")
        last_active_str = last_active_obj.strftime("%Y-%m-%d") if isinstance(last_active_obj, (datetime, date)) else date.today().strftime("%Y-%m-%d")

        return {
            "id": str(member["_id"]),
            "name": member.get("name"),
            "role": member.get("role", "").capitalize(),
            "email": member.get("email"),
            "phone": member.get("phone"),
            "dateJoined": date_joined_str,
            "lastActive": last_active_str,
            "status": member.get("status", "").capitalize(),
            "avatar": member.get("avatar")
        }

    return {
        "success": True,
        "data": [format_member(m) for m in members],
        "pagination": {
            "currentPage": page,
            "totalPages": (total_items + limit - 1) // limit,
            "totalItems": total_items
        }
    }

@app.post("/api/team-members") 
async def add_team_member(member_data: TeamMemberCreate, user=Depends(get_current_user)):
    hashed_password = pwd_context.hash(member_data.credentials.password)
    
    member_doc = {
        "name": member_data.name,
        "role": member_data.role.value,
        "email": member_data.email,
        "phone": member_data.phone,
        "status": member_data.status.value,
        "joiningDate": datetime.combine(member_data.joiningDate, datetime.min.time()),
        "userId": ObjectId(user["_id"]), 
        "lastActive": None,
        "avatar": f"/uploads/{member_data.name.split(' ')[0].lower()}.png",
        "username": member_data.credentials.username,
        "password": hashed_password
    }

    result = await db.teammembers.insert_one(member_doc)
    created_member = await db.teammembers.find_one({"_id": result.inserted_id})

    def format_created(member):
        date_joined_obj = member.get("joiningDate") 
        date_joined_str = date_joined_obj.strftime("%Y-%m-%d") if isinstance(date_joined_obj, (datetime, date)) else "N/A"

        return {
            "id": str(member["_id"]),
            "name": member.get("name"),
            "role": member.get("role", "").capitalize(),
            "email": member.get("email"),
            "phone": member.get("phone"),
            "dateJoined": date_joined_str,
            "lastActive": None,
            "status": member.get("status", "").capitalize(),
            "avatar": member.get("avatar"),
            "username": member.get("username")
        }

    return {
        "success": True,
        "message": "Team member added successfully",
        "data": format_created(created_member)
    }


@app.put("/api/team-members/{id}")
async def update_team_member(id: str, member_update: TeamMemberUpdate, user=Depends(get_current_user)):
    update_data = member_update.dict(exclude_unset=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No update data provided")

    result = await db.teammembers.update_one(
        {"_id": ObjectId(id), "userId": ObjectId(user["_id"])}, 
        {"$set": update_data}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Team member not found")
    
    updated_member = await db.teammembers.find_one({"_id": ObjectId(id)})
    return {
        "success": True,
        "message": "Team member updated successfully",
        "data": convert_objids(updated_member)
    }

@app.delete("/api/team-members/{id}")
async def delete_team_member(id: str, user=Depends(get_current_user)):
    try:
        obj_id = ObjectId(id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid team member ID format")
    result = await db.teammembers.delete_one({"_id": obj_id, "userId": ObjectId(user["_id"])}) # CORRECTED: Query by ObjectId
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Team member not found")
    return {"success": True, "message": "Team member deleted successfully"}

@app.post("/api/team-members/import")
async def import_team_members(file: UploadFile = File(...), user=Depends(get_current_user)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    if not file.filename.endswith(('.csv', '.xlsx')):
        raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported")
    
    contents = await file.read()
    if file.filename.endswith('.csv'):
        df = pd.read_csv(io.StringIO(contents.decode()))
    else:
        df = pd.read_excel(io.BytesIO(contents))
    
    imported_count = 0
    for _, row in df.iterrows():
        member_data = row.to_dict()
        member_data['userId'] = ObjectId(user['_id']) # CORRECTED: Store as ObjectId
        if 'joiningDate' in member_data:
             member_data['joiningDate'] = pd.to_datetime(member_data['joiningDate'])
        await db.teammembers.insert_one(member_data)
        imported_count += 1
    
    return {"success": True, "message": f"{imported_count} team members imported successfully"}

@app.get("/api/team-members/export")
async def export_team_members(format: str, user=Depends(get_current_user)):
    members = await db.teammembers.find({"userId": ObjectId(user["_id"])}).to_list(None) # CORRECTED: Query by ObjectId
    if not members:
        return Response(status_code=204)
        
    df = pd.DataFrame(members)
    df.drop(columns=['_id', 'userId'], inplace=True, errors='ignore')

    if format == "csv":
        output = io.StringIO()
        df.to_csv(output, index=False)
        return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=team_members.csv"})
    elif format == "excel":
        output = io.BytesIO()
        df.to_excel(output, index=False, sheet_name='Team Members')
        output.seek(0)
        return StreamingResponse(output, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": "attachment; filename=team_members.xlsx"})
    elif format == "pdf":
        html = df.to_html(index=False)
        pdf = pdfkit.from_string(html, False)
        return Response(content=pdf, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=team_members.pdf"})
    else:
        raise HTTPException(status_code=400, detail="Invalid format specified. Use 'csv', 'excel', or 'pdf'.")

@app.get("/api/profile")
async def get_profile(user=Depends(get_current_user)):
    profile = await db["users"].find_one({"_id": ObjectId(user["_id"])})
    if not profile:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "success": True,
        "data": {
            "name": profile.get("name"),
            "email": profile.get("email"),
            "phone": profile.get("phone"),
            "businessName": profile.get("company"),
            "address": profile.get("address"),
            "state": profile.get("state"),
            "website": profile.get("website"),
            "pan": profile.get("panNumber"),
            "gst": profile.get("gstNumber"),
            "dateFormat": "DD/MM/YYYY",
            "logoUrl": profile.get("businessLogo"),
            "plan": "Premium"
        }
    }

@app.put("/api/profile")
async def update_profile(profile_data: dict, user=Depends(get_current_user)):
    update_fields = {
        "name": profile_data.get("name"),
        "company": profile_data.get("businessName"),
        "address": profile_data.get("address"),
        "state": profile_data.get("state"),
        "website": profile_data.get("website"),
        "panNumber": profile_data.get("pan"),
        "gstNumber": profile_data.get("gst"),
        "phone": profile_data.get("phone")
    }
    
    update_data_cleaned = {k: v for k, v in update_fields.items() if v is not None}

    result = await db["users"].update_one(
        {"_id": ObjectId(user["_id"])},
        {"$set": update_data_cleaned}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
        
    return {"success": True, "message": "Profile updated successfully."}

@app.put("/api/profile/password")
async def change_password(data: dict = Body(...), user=Depends(get_current_user)):
    current_password = data.get("currentPassword")
    new_password = data.get("newPassword")

    user_doc = await db["users"].find_one({"_id": ObjectId(user["_id"])})
    if not current_password:
        raise HTTPException(status_code=500, detail="No password provided for verification")
    if not user_doc or not pwd_context.verify(current_password, user_doc["password"]):
        raise HTTPException(status_code=401, detail="Current password is incorrect")

    if not new_password or len(new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters")

    new_hashed_password = pwd_context.hash(new_password)
    await db["users"].update_one(
        {"_id": ObjectId(user["_id"])},
        {"$set": {"password": new_hashed_password}}
    )

    return {"success": True, "message": "Password updated successfully."}

@app.post("/api/profile/upload")
async def upload_logo(file: UploadFile = File(...), user=Depends(get_current_user)):
    if not file.content_type:
        raise HTTPException(status_code=400, detail="No file provided.")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed.")
    
    file_path = f"/uploads/logos/{user['_id']}_{file.filename}"
    
    await db.users.update_one(
        {"_id": ObjectId(user["_id"])},
        {"$set": {"businessLogo": file_path}}
    )
    
    return {
        "success": True,
        "message": "File uploaded successfully.",
        "fileUrl": file_path
    }

@app.put("/api/profile/theme")
async def update_theme(data: dict = Body(...), user=Depends(get_current_user)):
    theme = data.get("theme")
    if theme not in ["light", "dark"]:
        raise HTTPException(status_code=400, detail="Invalid theme value.")

    await db.users.update_one(
        {"_id": ObjectId(user["_id"])},
        {"$set": {"theme": theme}}
    )
    
    return {"success": True, "message": "Theme updated."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))