import pytest
import sqlite3
import os
import requests
import numpy as np
from fastapi.testclient import TestClient
from io import BytesIO
from main import app  # Import FastAPI app instance

client = TestClient(app)


# Function to create the dummy SQLite3 database
def create_dummy_database():
    conn = sqlite3.connect(":memory:")  # Creating an in-memory SQLite3 database
    c = conn.cursor()

    # Creating the invoices table
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

    # Inserting some dummy data
    dummy_data = [
        ("Store A", "123 Main St", "123-456-7890", "INV001", "2024-03-28", "Product A", 5, 100.00),
        ("Store B", "456 Elm St", "987-654-3210", "INV002", "2024-03-27", "Product B", 3, 150.00),
        ("Store C", "789 Oak St", "111-222-3333", "INV003", "2024-03-26", "Product C", 2, 75.00)
    ]

    c.executemany('INSERT INTO invoices VALUES (?,?,?,?,?,?,?,?)', dummy_data)

    conn.commit()
    conn.close()

# Fixture to initialize the dummy database before tests
@pytest.fixture(scope="session", autouse=True)
def initialize_database():
    create_dummy_database()

# # GET apis - successes
# def test_get_invoice_items():
#     # invoice_id =

#     response = client.get("/invoices/{}/items")

#     assert response.status_code == 200

# POST api - success
def test_process_image_inputs_with_file():
    # Prepare a sample image file
    image_path = os.path.join("../datasets", "sample_images", "sample_5.png")

    with open(image_path, "rb") as f:
        files = {'file': ("sample_5.png", f)}

        # Make a request to the API
        response = client.post("/process_image_inputs/", files=files)

    assert response.status_code == 201
    assert "invoice_id" in response.json()
    

if __name__ == "__main__":
    pytest.main([__file__])
