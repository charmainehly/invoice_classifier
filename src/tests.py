import pytest
import sqlite3
import os
import requests
import numpy as np
from fastapi.testclient import TestClient
from main import app  # Import FastAPI app instance

def create_dummy_database():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS invoices (
                    store_name TEXT,
                    address TEXT,
                    contact TEXT,
                    invoice_no TEXT,
                    date TEXT,
                    item_description TEXT,
                    count REAL,
                    total_cost REAL
            )''')

    dummy_data = [
        ("Store A", "123 Main St", "123-456-7890",
         "INV001", "2024-03-28", "Product A", 5, 100.00),
        ("Store B", "456 Elm St", "987-654-3210",
         "INV001", "2024-03-27", "Product B", 3, 150.00),
        ("Store C", "789 Oak St", "111-222-3333",
         "INV003", "2024-03-26", "Product C", 2, 75.00)
    ]

    c.executemany('INSERT INTO invoices VALUES (?,?,?,?,?,?,?,?)', dummy_data)

    conn.commit()
    return conn

@pytest.fixture(scope="session", autouse=True)
def initialize_app():
    with TestClient(app) as client:
        conn = create_dummy_database()
        yield client, conn

    conn.close()


# GET apis - success
def test_get_invoice_items_200(initialize_app):
    client, conn = initialize_app
    app.state.db_connection = conn

    response = client.get("/invoice/INV001/items")

    assert response.status_code == 200
    assert "item_description" in response.json()
    assert "count" in response.json()
    assert "total_cost" in response.json()

# GET apis - fail (404)
def test_get_invoice_item_404(initialize_app):
    client, conn = initialize_app
    app.state.db_connection = conn

    response = client.get("/invoice/404/items")

    assert response.status_code == 404

# GET apis - success
def test_get_invoice_date_200(initialize_app):
    client, conn = initialize_app
    app.state.db_connection = conn

    response = client.get("/invoice/INV001/date")

    assert response.status_code == 200
    assert "date" in response.json()


# GET apis - success
def test_get_invoice_summary_200(initialize_app):
    client, conn = initialize_app
    app.state.db_connection = conn

    response = client.get("/invoice/INV001/summary")

    assert response.status_code == 200
    assert "store_name" in response.json()
    assert "address" in response.json()
    assert "contact" in response.json()

# POST api - success


def test_process_image_inputs_with_file(initialize_app):
    # Prepare a sample image file
    client, conn = initialize_app
    image_path = os.path.join("../datasets", "sample_images", "sample_5.png")

    with open(image_path, "rb") as f:
        files = {'file': ("sample_5.png", f)}

        # Make a request to the API
        response = client.post("/process_image_inputs/", files=files)

    assert response.status_code == 201
    assert "invoice_id" in response.json()


if __name__ == "__main__":
    pytest_args = [__file__, '-s']
    pytest.main(pytest_args)
