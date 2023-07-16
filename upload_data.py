from pymongo.mongo_client import MongoClient
import pandas as pd
import json

# uniform resource indentifier
uri = "mongodb+srv://milanbeherazyx:VKF4Yq7WqLwGIpkG@cluster0.wui6u7y.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri)

# create database name and collection name
DATABASE_NAME="Project"
COLLECTION_NAME="Credit_Card_Fault"

# read the data as a dataframe
df = pd.read_csv(r"/home/milan/Data Science/Project/Credit_Card_Default_Prediction/notebooks/data/UCI_Credit_Card_modified.csv")
df = df.drop(columns='ID')

# Convert the data into json
json_record=list(json.loads(df.T.to_json()).values())

#now dump the data into the database
client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)