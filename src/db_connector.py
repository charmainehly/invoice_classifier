import json
import sqlite3
import pandas as pd
from tags import Tag

def connect_db():
    con = sqlite3.connect("invoices.db")
    cur = con.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS invoices (
                    store_name TEXT,
                    address TEXT,
                    contact TEXT,
                    invoice_no TEXT,
                    date TEXT,
                    item_description TEXT,
                    count REAL,
                    total_cost REAL
            )''')
    con.commit()
    return con

def close_db(con) -> None:
    con.close()
    return 

def query_db(con, invoice_id: str, tag: Tag):
    cur = con.cursor()
    match tag:
        case Tag.ITEMS:
            cur.execute('''SELECT item_description, count, total_cost 
                        FROM invoices 
                        WHERE invoice_no = ?''', (invoice_id,))
        case Tag.DATE:
            cur.execute('''SELECT date 
                        FROM invoices 
                        WHERE invoice_no = ?''', (invoice_id,))
        case Tag.SUMMARY:
            cur.execute('''SELECT store_name, address, contact
                        FROM invoices 
                        WHERE invoice_no = ?''', (invoice_id,))
    
    return convert_to_json(cur.fetchall(), cur)

def insert_db(con, df: pd.DataFrame) -> None:
    cur = con.cursor()
    for index, row in df.iterrows():
        cur.execute('''INSERT INTO invoices (store_name, address, contact, invoice_no, date, item_description, count, total_cost)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', 
                    (row['Store Name'], row['Address'], row['Contact'], row['Invoice No.'], 
                    row['Date'], row['Item Description'], row['Count'], row['Total Cost']))

    con.commit()
    return

def convert_to_json(res, cur):

    columns = [desc[0] for desc in cur.description]

    # Convert rows into dictionary format
    results = []
    for row in res:
        result = {}
        for i, column in enumerate(columns):
            result[column] = row[i]
        results.append(result)

    # Serialize the results into JSON
    return json.dumps(results)