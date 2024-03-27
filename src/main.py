# establishing api endpoints (RESTFUL)

from fastapi import FastAPI, UploadFile, File, HTTPException
from ocr import extract_invoice_single
from llm import parse_to_df

app = FastAPI()

# GET APIs
@app.get("/invoice/{invoice_id}/items") # get invoice items details
async def get_invoice_items(invoice_id: str):
    return {"message": "Hello World"} # fix return value

@app.get("/invoice/{invoice_id}/date") # get invoice date
async def get_invoice_date(invoice_id: str):
    return {"message": "Hello World"} # fix return value

@app.get("/invoice/{invoice_id}/summary") # get invoice summary - i.e. address, number, date
async def get_invoice_details(invoice_id: str):
    return {"message": "Hello World"} # fix return value

# POST APIs
@app.post("/process_image_inputs/")
async def process_image_inputs(file: UploadFile = File(...)):
    contents = await file.read()

    if not file:
        return {"message": "No upload file sent"}
    else:
        txt = extract_invoice_single(contents)
        summary = parse_to_df(txt)

        return {"filename": file.filename}
    