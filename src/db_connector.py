import sqlite3

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
    return con, cur

def close_db(con) -> None:
    con.close()
    return 

def query_db():
    return

def insert_db(con, cur, df) -> None:
    for index, row in df.iterrows():
        cur.execute('''INSERT INTO invoices (store_name, address, contact, invoice_no, date, item_description, count, total_cost)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', 
                    (row['Store Name'], row['Address'], row['Contact'], row['Invoice No.'], 
                    row['Date'], row['Item Description'], row['Count'], row['Total Cost']))

    con.commit()
    return