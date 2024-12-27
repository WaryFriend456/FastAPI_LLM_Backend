from pymongo import MongoClient
import datetime

mongo_client = None
db = None
chats_collection = None
user_collection = None

def init_db():
    global mongo_client, db, chats_collection, user_collection
    mongo_client = MongoClient("mongodb://localhost:27017/")
    db = mongo_client["chat_db"]
    chats_collection = db["chat_sessions"]
    user_collection = db["user_session"]

def delete_empty_sessions():
    try:
        result = chats_collection.delete_many({"messages": []})
        return result.deleted_count
    except Exception as e:
        print(f"Failed to delete empty sessions: {str(e)}")
        return 0

def get_chats_collection():
    global chats_collection
    return chats_collection

def get_user_collection():
    global user_collection
    return user_collection