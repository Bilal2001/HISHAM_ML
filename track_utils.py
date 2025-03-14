import hashlib
import sqlite3
import pytz
from datetime import datetime

# Load Database Packages
conn = sqlite3.connect('./data/data.db', check_same_thread=False)
c = conn.cursor()

IST = pytz.timezone('Asia/Kolkata')  # Indian Standard Time

# Function to create emotion classifier table
def create_emotionclf_table():
    c.execute('CREATE TABLE IF NOT EXISTS emotionclfTable(age NUMBER, name TEXT, rawtext TEXT, prediction TEXT, probability NUMBER, timeOfvisit TIMESTAMP)')
    c.execute('CREATE TABLE IF NOT EXISTS emotionclfImgTable(name TEXT, age INT, prediction TEXT, timeOfvisit TIMESTAMP)')
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    username TEXT UNIQUE, 
                    password TEXT)''')
    conn.commit()

# Fuctions for Login and Registrations
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_login(username, password):
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hash_password(password)))
    return c.fetchone()

def add_user(username, password):
    c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
    conn.commit()


# Function to add prediction details
def add_prediction_details(age, name, rawtext, prediction, probability, timeOfvisit=None):
    if timeOfvisit is None:
        timeOfvisit = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    else:
        timeOfvisit = timeOfvisit.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S")
    c.execute('INSERT INTO emotionclfTable(age, name, rawtext, prediction, probability, timeOfvisit) VALUES (?, ?, ?, ?, ?, ?)', (age, name, rawtext, prediction, probability, timeOfvisit))
    conn.commit()

def add_image_prediction_details(name, age, prediction, timeOfvisit=None):
    if timeOfvisit is None:
        timeOfvisit = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    else:
        timeOfvisit = timeOfvisit.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S")
    c.execute('INSERT INTO emotionclfImgTable(name, age, prediction, timeOfvisit) VALUES (?, ?, ?, ?)', (name, age, prediction, timeOfvisit))
    conn.commit()

# Function to view prediction details
def get_names():
    c.execute('SELECT DISTINCT name FROM emotionclfTable UNION SELECT DISTINCT name FROM emotionclfImgTable')
    data = c.fetchall()
    data = [i[0] for i in data]
    data = list(set(data))
    return data

def get_ages():
    c.execute('SELECT DISTINCT age FROM emotionclfTable UNION SELECT DISTINCT age FROM emotionclfImgTable')
    data = c.fetchall()
    data = [i[0] for i in data]
    data = list(set(data))
    return data

def view_prediction_details(given_name, age):
    query = "SELECT * FROM emotionclfTable"

    flag = 0
    if given_name != "all":
        query += f' WHERE name = "{given_name}"'
        if age != "all":
            query += f" AND age = {age}"
            flag = 1
    if flag == 0 and age != "all":
        query += f" WHERE age = {age}"
    c.execute(query)
    data = c.fetchall()
    return data

def view_prediction_details_images(given_name, age):
    query = "SELECT * FROM emotionclfImgTable"

    flag = 0
    if given_name != "all":
        query += f' WHERE name = "{given_name}"'
        if age != "all":
            query += f" AND age = {age}"
            flag = 1
    if flag == 0 and age != "all":
        query += f" WHERE age = {age}"  
    c.execute(query)
    data = c.fetchall()
    return data
