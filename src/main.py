# establishing api endpoints (RESTFUL)
from fastapi import FastAPI, UploadFile, File, HTTPException, Path, status
from fastapi.responses import JSONResponse
from typing import Annotated
from contextlib import asynccontextmanager
from ocr import extract_invoice_single
from llm import parse_to_df
from db_connector import query_db, connect_db, close_db, insert_db, query_column, query_category_items
from tags import Tag
from fastapi import FastAPI, status
from ml_model import predict

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    app.state.db_connection = connect_db()
    yield 
    # Clean up the ML models and release the resources
    close_db(app.state.db_connection)

app = FastAPI(lifespan=lifespan)

# GET APIs
@app.get("/invoice/{invoice_id}/items", status_code=status.HTTP_200_OK) # get invoice items details
async def get_invoice_items(invoice_id: Annotated[str, Path(title="The ID of the invoice to get")]):
    if invoice_id not in query_column(app.state.db_connection, Tag.INVOICE):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invoice not found")

    con = app.state.db_connection
    res = query_db(con, invoice_id, Tag.ITEMS)

    return res

@app.get("/invoice/{invoice_id}/date", status_code=status.HTTP_200_OK) # get invoice date
async def get_invoice_date(invoice_id: Annotated[str, Path(title="The ID of the invoice to get")]):
    if invoice_id not in query_column(app.state.db_connection, Tag.INVOICE):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invoice not found")

    con = app.state.db_connection
    res = query_db(con, invoice_id, Tag.DATE)

    return res

@app.get("/invoice/{invoice_id}/summary", status_code=status.HTTP_200_OK) # get invoice summary - i.e. address, number, date
async def get_invoice_details(invoice_id: Annotated[str, Path(title="The ID of the invoice to get")]):
    if invoice_id not in query_column(app.state.db_connection, Tag.INVOICE):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invoice not found")

    con = app.state.db_connection
    res = query_db(con, invoice_id, Tag.SUMMARY)

    return res

@app.get("/categories/{category_id}/items", status_code=status.HTTP_200_OK) # get all items within a category (ids from 0-10)
async def get_category_items(category_id: Annotated[int, Path(title="The ID of the category to get")]):
    if int(category_id) not in range(0, 11):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Category not found")

    con = app.state.db_connection
    res = query_category_items(con, category_id)

    return res


# POST APIs
@app.post("/process_image_inputs/", status_code=status.HTTP_201_CREATED)
async def process_image_inputs(file: UploadFile = File(...)):
    contents = await file.read()

    if not file:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Bad Request")
    else:
        txt = extract_invoice_single(contents)
        summary = parse_to_df(txt)
        complete = predict(summary)
        insert_db(app.state.db_connection, complete)

        return {"invoice_id": str(complete['Invoice No.'][0]),
                "detail": "Created Successfully."}
    