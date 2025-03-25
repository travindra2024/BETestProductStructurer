
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import openai
import pdfplumber
import requests
from bs4 import BeautifulSoup
import io
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_text_from_pdf(file: UploadFile):
    text = ""
    with pdfplumber.open(io.BytesIO(file.file.read())) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

def extract_text_from_url(url: str):
    headers = {'User-Agent': 'Mozilla/5.0'}
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, 'html.parser')
    visible_text = " ".join(t.strip() for t in soup.find_all(text=True) if t.parent.name not in ['style', 'script', 'head', 'meta'])
    return visible_text[:10000]

def structure_product_data(text: str, openai_key: str):
    openai.api_key = openai_key
    system_prompt = """
You are a product data extractor AI. Given product information from a PDF or website, extract structured product details in this JSON format:

{
  "product_name": "",
  "brand": "",
  "price": "",
  "materials": "",
  "finish_options": [],
  "dimensions": {},
  "bulb_info": {},
  "features": [],
  "assembly": "",
  "care": "",
  "delivery": {},
  "related_products": [],
  "reviews": [],
  "product_url": ""
}
Fill out all the fields as best you can. If data is missing, leave the field empty.
"""
    res = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
    )
    return res.choices[0].message.content

@app.post("/extract")
async def extract_data(
    pdf_files: Optional[List[UploadFile]] = File(None),
    url: Optional[str] = Form(None),
    api_key: str = Form(...)
):
    if not pdf_files and not url:
        return {"error": "Provide either PDF(s) or a URL"}

    raw_text = ""
    if pdf_files:
        for file in pdf_files:
            raw_text += extract_text_from_pdf(file) + "\n"
    if url:
        raw_text += extract_text_from_url(url)

    try:
        structured = structure_product_data(raw_text, api_key)
        parsed = json.loads(structured)
        return parsed
    except json.JSONDecodeError:
        return {"raw_output": structured, "warning": "Could not parse as JSON."}
