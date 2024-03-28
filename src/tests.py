import pytest
import os
import requests
import numpy as np
from fastapi.testclient import TestClient
from main import app  # Import your FastAPI app instance
from io import BytesIO

client = TestClient(app)


def test_process_image_inputs_with_file():
    # Prepare a sample image file
    image_path = os.path.join("../datasets", "sample_images", "sample_5.png")

    with open(image_path, "rb") as f:
        files = {'file': ("sample_5.png", f)}

        # Make a request to the API
        response = client.post("/process_image_inputs/", files=files)

    # Assert the response status code is 200
    assert response.status_code == 200

    # Assert response body contains expected keys
    assert "filename" in response.json()

# def test_process_image_inputs_no_file():
#     # Make a request to the API without attaching a file
#     response = client.post("/process_image_inputs/")

#     # Assert the response status code is 422 (unprocessable entity) as no file is sent
#     assert response.status_code == 422
#     assert response.json()["detail"][0]["msg"] == "field required"


if __name__ == "__main__":
    test_process_image_inputs_with_file()
