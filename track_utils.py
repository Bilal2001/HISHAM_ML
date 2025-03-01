import sqlite3
import pytz
from datetime import datetime

# Load Database Packages
conn = sqlite3.connect('./data/data.db', check_same_thread=False)
c = conn.cursor()

IST = pytz.timezone('Asia/Kolkata')  # Indian Standard Time

# Function to create emotion classifier table
def create_emotionclf_table():
    c.execute('CREATE TABLE IF NOT EXISTS emotionclfTable(age NUMBER, name TEXT, primary_hobby TEXT, rawtext TEXT, prediction TEXT, probability NUMBER, timeOfvisit TIMESTAMP)')
    c.execute('CREATE TABLE IF NOT EXISTS emotionclfImgTable(name TEXT, prediction TEXT, timeOfvisit TIMESTAMP)')

# Function to add prediction details
def add_prediction_details(age, name, primary_hobby, rawtext, prediction, probability, timeOfvisit=None):
    if timeOfvisit is None:
        timeOfvisit = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    else:
        timeOfvisit = timeOfvisit.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S")
    c.execute('INSERT INTO emotionclfTable(age, name, primary_hobby, rawtext, prediction, probability, timeOfvisit) VALUES (?, ?, ?, ?, ?, ?, ?)', (age, name, primary_hobby, rawtext, prediction, probability, timeOfvisit))
    conn.commit()

def add_image_prediction_details(name, prediction, timeOfvisit=None):
    if timeOfvisit is None:
        timeOfvisit = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    else:
        timeOfvisit = timeOfvisit.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S")
    c.execute('INSERT INTO emotionclfImgTable(name, prediction, timeOfvisit) VALUES (?, ?, ?)', (name, prediction, timeOfvisit))
    conn.commit()

# Function to view prediction details
def get_names():
    c.execute('SELECT DISTINCT name FROM emotionclfTable UNION SELECT DISTINCT name FROM emotionclfImgTable')
    data = c.fetchall()
    data = [i[0] for i in data]
    data = list(set(data))
    return data

def view_prediction_details(given_name):
    query = "SELECT * FROM emotionclfTable"

    if given_name != "all":
        query += f' WHERE name = "{given_name}"'
    c.execute(query)
    data = c.fetchall()
    return data

def view_prediction_details_images(given_name):
    query = "SELECT * FROM emotionclfImgTable"

    if given_name != "all":
        query += f' WHERE name = "{given_name}"'
    c.execute(query)
    data = c.fetchall()
    return data
