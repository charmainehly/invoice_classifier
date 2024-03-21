'''
https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
'''
import os
import openai
import pandas as pd
from dotenv import load_dotenv
from io import StringIO

load_dotenv()
openai.api_key = os.getenv('API_KEY')


def format_raw_text(prompt: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content":
                '''
                System Instruction: Converting OCR Text to CSV for Invoice Line Items

                Input: Provide OCR text extracted from an invoice.
                Output: Return CSV-formatted text containing the extracted information for line items, ready to be parsed into a dataframe and saved into a CSV file.
                CSV Formatting Rules:

                Each line item in the invoice should be represented as a row in the CSV.
                If any text contains commas, they should be formatted with open inverted commas.
                The CSV should contain the following columns:
                Store Name
                Address
                Contact
                Invoice No.
                Date
                Item Description
                Count
                Total Cost

                -------------------------------------------

                Example Input 1:
                """
                Gadget Galleria

                5 Oak St, Binaryburg, TT

                104-105-106
                Receipt: 93670 Date: 09/01/2020

                ‘S.No tem Description Items Cost.

                1 Chocolate Cake(1 Kg) 1 895.00

                2 Flower Bookie 1 500.50

                3 Rat Poisen(S00m!) 1 50.50

                5.15
                115
                115

                1299.10
                """

                Example Output 1:
                """
                Store Name,Address,Contact,Invoice No.,Date,Item Description,Count,Total Cost,,
                Gadget Galleria,"5 Oak St, Binaryburg, TT",104-105-106,93670,09/01/2020,Chocolate Cake(1 Kg),1,895.00,,
                Gadget Galleria,"5 Oak St, Binaryburg, TT",104-105-106,93670,09/01/2020,Flower Bookie,1,500.50,,
                Gadget Galleria,"5 Oak St, Binaryburg, TT",104-105-106,93670,09/01/2020,Rat Poisen(S00m!),1,50.50,,
                """
                
                -------------------------------------------
                '''
             },
            {"role": "user",
             "content": prompt
             }
        ]
    )

    reply = response['choices'][0]['message']['content']
    return reply


def process_to_df(csv_text: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(csv_text))
    validate_csv_format(df)
    return df

def validate_csv_format(df: pd.DataFrame) -> None:
    '''
    check that the dataframe has all the necessary columns. Convert the necessary columns to numerical data.
    '''
    return ######

def save_df(df: pd.DataFrame, filename: str) -> None:
    df.to_csv("./datasets/db/"+filename)
    return

if __name__ == "__main__":
    file_path = "./datasets/ocr/"
    files = os.listdir(file_path)
    index = 0
    
    while index < len(files):
        filename = files[index]
        
        if filename.endswith('.txt'):
            with open(file_path+filename, 'r') as file:
                txt = file.read()
            csv_txt = format_raw_text(txt)
            df = process_to_df(csv_txt)
            save_df(df, filename+".csv")

        index += 1
    
