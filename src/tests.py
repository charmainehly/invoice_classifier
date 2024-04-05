import pytest
import pandas as pd
import sqlite3
import json
import os
from fastapi.testclient import TestClient
from main import app  # Import FastAPI app instance

# MOCKED METHODS WITH PYTEST-MOCK
@pytest.fixture
def mock_extract_invoice_single(mocker):
    return mocker.patch("ocr.extract_invoice_single")

@pytest.fixture
def mock_parse_to_df(mocker):
    return mocker.patch("llm.parse_to_df")

@pytest.fixture
def mock_predict(mocker):
    return mocker.patch("ml_model.predict")

def create_dummy_database():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    c = conn.cursor()

    c.execute('DROP TABLE IF EXISTS invoices')

    c.execute('''CREATE TABLE IF NOT EXISTS invoices (
                    store_name TEXT,
                    address TEXT,
                    contact TEXT,
                    invoice_no TEXT,
                    date TEXT,
                    item_description TEXT,
                    count REAL,
                    total_cost REAL,
                    category INTEGER
            )''')

    dummy_data = [
        ("Store A", "123 Main St", "123-456-7890",
         "INV001", "2024-03-28", "Product A", 5, 100.00, 0),
        ("Store B", "456 Elm St", "987-654-3210",
         "INV001", "2024-03-27", "Product B", 3, 150.00, 0),
        ("Store C", "789 Oak St", "111-222-3333",
         "INV003", "2024-03-26", "Product C", 2, 75.00, 0)
    ]

    c.executemany(
        'INSERT INTO invoices VALUES (?,?,?,?,?,?,?,?,?)', dummy_data)

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

    response_json = json.loads(response.json())
    expected_values = [{"item_description": "Product A", "count": 5.0, "total_cost": 100.0, "category": 0},
                       {"item_description": "Product B", "count": 3.0, "total_cost": 150.0, "category": 0}]
    expected_keys = ["item_description", "count", "total_cost", "category"]

    for index, item in enumerate(response_json):
        for key in expected_keys:
            assert key in response_json[index]
            assert item[key] == expected_values[index][key]

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

# GET apis - fail (404)
def test_get_invoice_date_404(initialize_app):
    client, conn = initialize_app
    app.state.db_connection = conn

    response = client.get("/invoice/404/date")

    assert response.status_code == 404

# GET apis - success
def test_get_invoice_summary_200(initialize_app):
    client, conn = initialize_app
    app.state.db_connection = conn

    response = client.get("/invoice/INV001/summary")

    assert response.status_code == 200
    assert "store_name" in response.json()
    assert "address" in response.json()
    assert "contact" in response.json()

# GET apis - fail (404)
def test_get_invoice_summary_404(initialize_app):
    client, conn = initialize_app
    app.state.db_connection = conn

    response = client.get("/invoice/404/summary")

    assert response.status_code == 404

# GET apis - success
def test_get_category_items(initialize_app):
    client, conn = initialize_app
    app.state.db_connection = conn

    response = client.get("/categories/0/items")

    assert response.status_code == 200
    assert "item_description" in response.json()
    assert "count" in response.json()
    assert "total_cost" in response.json()

# GET apis - fail (404)
def test_get_category_items_404(initialize_app):
    client, conn = initialize_app
    app.state.db_connection = conn

    response = client.get("/category/404/items")

    assert response.status_code == 404

# POST api - success
def test_process_image_inputs_with_file(initialize_app):
    mock_extract_invoice_single.return_value = {"mocked_data": '''
                                                                Tech Treasures

                                                                ‘94th Ave, Code Cove, TT

                                                                108-109-110
                                                                Receipt: 55478 Date: 07/01/2021

                                                                ‘S.No tem Description Items Cost.

                                                                1 Legal Consultation 1 300.00

                                                                2 ‘AccountingServices 1 250.00

                                                                3 Web Development 1 1000.00

                                                                5.15
                                                                115
                                                                115

                                                                1299.10
                                                                '''}
    mock_parse_to_df.return_value = {"mocked_data": pd.read_csv('../datasets/validation/mocked_output_parse_to_df')}
    mock_predict.return_value = {"mocked_data": pd.read_csv('../datasets/validation/mocked_output_predict')}

    # Prepare a sample image file
    client, conn = initialize_app
    image_path = os.path.join("../datasets", "sample_images", "sample_5.png")

    with open(image_path, "rb") as f:
        files = {'file': ("sample_5.png", f)}

        # Make a request to the API
        response = client.post("/process_image_inputs/", files=files)

    assert response.status_code == 201
    assert "invoice_id" in response.json()

# POST apis - fail (422)
def test_get_category_items_422(initialize_app):
    client, conn = initialize_app

    response = client.post("/process_image_inputs/", files=None)

    json_response = response.json()

    assert response.status_code == 422
    assert json_response['detail'][0]['msg'] == "Field required"


if __name__ == "__main__":
    pytest_args = [__file__, '-s']
    pytest.main(pytest_args)
