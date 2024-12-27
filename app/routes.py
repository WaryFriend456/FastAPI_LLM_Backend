from fastapi import FastAPI, Request, HTTPException
import datetime
import uuid
from .database import get_chats_collection, get_user_collection, delete_empty_sessions
from .models import Retrival_Augmentation, generate_answer

def register_routes(app: FastAPI):
    @app.post("/start_session")
    async def start_session(request: Request):
        data = await request.json()
        UID = data.get("UID")

        deleted_count = delete_empty_sessions()
        print(f"Deleted {deleted_count} empty sessions before starting a new session")

        session_id = str(uuid.uuid4())
        chats_collection = get_chats_collection()
        chats_collection.insert_one({
            "UID": UID,
            "session_id": session_id,
            "messages": [],
            "created_at": datetime.datetime.now()
        })
        return {"session_id": session_id}

    @app.post("/registration")
    async def register_user(request: Request):
        data = await request.json()
        name = data.get("name")
        email = data.get("email")
        password = data.get("password")

        if not name or not email or not password:
            raise HTTPException(status_code=400, detail="Name, email, and password are required")

        UID = str(uuid.uuid4())

        user = {
            "UID": UID,
            "name": name,
            "email": email,
            "password": password,
            "created_at": datetime.datetime.now()
        }
        user_collection = get_user_collection()
        user_collection.insert_one(user)
        return {"message": "User registered successfully"}

    @app.post("/login")
    async def login_user(request: Request):
        data = await request.json()
        username = data.get("name")
        password = data.get("password")
        print(username)

        if not username or not password:
            raise HTTPException(status_code=400, detail="Email and password are required")

        user_collection = get_user_collection()
        user = user_collection.find_one({"name": username, "password": password})
        if user:
            return {"message": "Login successful", "UID": user["UID"]}
        else:
            raise HTTPException(status_code=401, detail="Invalid credentials")

    @app.post("/query")
    async def receive_query(request: Request):
        data = await request.json()
        query = data.get("query")
        session_id = data.get("session_id")
        print(session_id)

        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")

        print(f"Received query: {query}")
        context = Retrival_Augmentation(query)
        answer = generate_answer(context, query)
        print(answer)

        chat = {
            "user": query,
            "context": context,
            "chatbot": answer,
            "timestamp": datetime.datetime.now()
        }

        chats_collection = get_chats_collection()
        chats_collection.update_one(
            {"session_id": session_id},
            {"$push": {"messages": chat}}
        )

        return {"response": answer}

    @app.post("/chats")
    async def get_chats(request: Request):
        try:
            data = await request.json()
            UID = data.get("UID")
            chats_collection = get_chats_collection()
            chats = list(chats_collection.find({"UID": UID}))
            for chat in chats:
                chat["_id"] = str(chat["_id"])
            return chats[::-1]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch chats: {str(e)}")

    @app.get("/query")
    async def get_query():
        return {"response": "Hello"}

    @app.get("/hello")
    async def get_name():
        return {"message": "Hello"}

    @app.get("/chats/{session_id}")
    async def get_chats(session_id: str):
        chats_collection = get_chats_collection()
        session = chats_collection.find_one({"session_id": session_id})
        session["_id"] = str(session["_id"])
        if session:
            return session
        else:
            raise HTTPException(status_code=404, detail="Session not found")
