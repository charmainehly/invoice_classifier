import numpy as np
import sys
import requests
import gradio as gr
from PIL import Image as PILImage
from io import BytesIO
import json

def format_dict(data):
    formatted_output = ""
    for key, value in data.items():
        formatted_output = formatted_output + f"{key}: {value}\n"

    return formatted_output


def upload_image(image):
    url = "http://127.0.0.1:8000/process_image_inputs/"  # FastAPI endpoint URL
    pil_image = PILImage.fromarray(image.astype('uint8'))
    image_bytes = BytesIO()
    pil_image.save(image_bytes, format='PNG')  # Save PIL Image to BytesIO object
    files = {'file': (f'image.jpg', image_bytes.getvalue(), 'image/png')}
    
    response = requests.post(url, files=files)  # Send POST request to FastAPI server

    if response.status_code == 201:
        return "Invoice " + str(response.json()["invoice_id"]) + " " + str(response.json()["detail"])
    else:
        return "Error connecting to FastAPI"

def make_api_request(number, request_type):
    if request_type == "Summary":
        url = f"http://127.0.0.1:8000/invoice/{number}/summary"
    elif request_type == "Items":
        url = f"http://127.0.0.1:8000/invoice/{number}/items"
    elif request_type == "Date":
        url = f"http://127.0.0.1:8000/invoice/{number}/date"
    elif request_type == "Category Items":
        url = f"http://127.0.0.1:8000/categories/{number}/items"
    else:
        return "Invalid request type"

    response = requests.get(url)
    json_response = json.loads(response.json())
    formatted_response = ""
    for each in json_response:
        formatted_response = formatted_response + "\n" + format_dict(each)
    
    return formatted_response


with gr.Blocks() as demo:
    gr.Markdown(
        """
    # Invoice Classifier!
    Upload an invoice image to store its contents to the database and retrieve information.
    """)
    inp = gr.Interface(
        fn=upload_image,
        inputs=gr.Image(),
        outputs="text"
    )
    out = gr.Interface(
        fn=make_api_request,
        inputs=["text", gr.Dropdown(
            ["Summary", "Items", "Date", "Category Items"], label="Select Request Type")],
        outputs="text",
        title="Send Requests to Retrieve Invoice Information"
    )

if __name__ == "__main__":
    demo.launch(share=False)