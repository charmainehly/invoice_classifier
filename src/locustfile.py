import subprocess
import os
import sqlite3
from locust import HttpUser, task
import signal

class UnitTestAPIs(HttpUser):
    host = "http://127.0.0.1:8000"  # Set the base URL here

    def on_start(self):
        image_path = os.path.join("../datasets", "sample_images", "sample_5.png")

        with open(image_path, "rb") as f:
            files = {'file': ("sample_5.png", f)}
            self.client.post("/process_image_inputs/", files=files)

    @task
    def run_get_api_invoice_items(self):        
        self.client.get("/invoice/55478/items")

    @task
    def run_get_api_invoice_summary(self):
        self.client.get("/invoice/55478/date")

    @task
    def run_get_api_invoice_date(self):
        self.client.get("/invoice/55478/summary")

    @task
    def run_get_api_category_items(self):
        self.client.get("/categories/0/items")

    @task
    def run_post_api(self):
        image_path = os.path.join("../datasets", "sample_images", "sample_5.png")

        with open(image_path, "rb") as f:
            files = {'file': ("sample_5.png", f)}
            self.client.post("/process_image_inputs/", files=files)

    def on_stop(self):
        conn = sqlite3.connect('invoices.db', check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM invoices")
        conn.commit()
        conn.close()
        print("Database cleaned up")

if __name__ == "__main__":
    # Start uvicorn process
    uvicorn_process = subprocess.Popen(["uvicorn", "main:app", "--reload"])

    # Run unit tests
    unit_test_runner = UnitTestAPIs()
    unit_test_runner.run()

    # Terminate uvicorn process
    uvicorn_process.send_signal(signal.SIGINT)