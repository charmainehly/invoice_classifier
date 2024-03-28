# establishing api endpoints (RESTFUL)
from fastapi import FastAPI, UploadFile, File, HTTPException
from ocr import extract_invoice_single
from llm import parse_to_df
from db_connector import query_db, connect_db, close_db
from tags import Tag

app = FastAPI()

# GET APIs
@app.get("/invoice/{invoice_id}/items", status_code=200) # get invoice items details
async def get_invoice_items(invoice_id: str):
    con, cur = connect_db()
    query_db(con, invoice_id, Tag.ITEMS)
    close_db()
    return {"message": "Hello World"} # fix return value

@app.get("/invoice/{invoice_id}/date", status_code=200) # get invoice date
async def get_invoice_date(invoice_id: str):
    con, cur = connect_db()
    query_db(con, invoice_id, Tag.DATE)
    close_db()
    return {"message": "Hello World"} # fix return value

@app.get("/invoice/{invoice_id}/summary", status_code=200) # get invoice summary - i.e. address, number, date
async def get_invoice_details(invoice_id: str):
    con, cur = connect_db()
    query_db(con, invoice_id, Tag.SUMMARY)
    close_db()
    return {"message": "Hello World"} # fix return value

# POST APIs
@app.post("/process_image_inputs/", status_code=201)
async def process_image_inputs(file: UploadFile = File(...)):
    contents = await file.read()

    if not file:
        raise HTTPException(status_code=400, detail="Bad Request")
    else:
        txt = extract_invoice_single(contents)
        summary = parse_to_df(txt)

        return {"invoice_id": summary['Store Name'][0],
                "detail": "Created Successfully."}
    