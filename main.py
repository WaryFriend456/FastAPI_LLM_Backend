from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
import pickle
from langchain_community.docstore.in_memory import InMemoryDocstore
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from fastapi.templating import Jinja2Templates
import datetime
import uuid
from pymongo import MongoClient


force_download = True

torch.random.manual_seed(0)
app = FastAPI()

KNOWLEDGE_VECTOR_DATABASE = None
RAG_PROMPT_TEMPLATE = None
pipe = None
generation_args = None
mongo_client = None
db = None
chats_collection = None
RAG_PROMPT_TEMPLATE1 = None
RAG_PROMPT_TEMPLATE2 = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing models and vector database... from app.main")
    init()
    yield
    print("Cleaning up resources...")


app = FastAPI(lifespan=lifespan)


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def init():
    global KNOWLEDGE_VECTOR_DATABASE, RAG_PROMPT_TEMPLATE, pipe, generation_args, mongo_client, db, chats_collection, RAG_PROMPT_TEMPLATE1, RAG_PROMPT_TEMPLATE2

    mongo_client = MongoClient("mongodb://localhost:27017/")
    db = mongo_client["chat_db"]
    chats_collection = db["chat_sessions"]

    embedding_model = HuggingFaceEmbeddings(
        model_name="thenlper/gte-small",
        multi_process=True,
        model_kwargs={"device": "cuda:0"},
        encode_kwargs={"normalize_embeddings": True},
    )

    index = faiss.read_index("models/faiss_index_LOL.bin")

    with open("models/faiss_metadata_LOL.pkl", "rb") as f:
        metadata = pickle.load(f)

    docstore = InMemoryDocstore(metadata['docstore'])
    index_to_docstore_id = metadata['index_to_docstore_id']

    KNOWLEDGE_VECTOR_DATABASE = FAISS(
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        embedding_function=embedding_model
    )

    print("Models and vector database initialized")

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-128k-instruct",
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }

    # prompt_chat = [
    #     {
    #         "role": "system",
    #         "content": """Your are an helpful AI assistant, Using the information contained in the context,
    #         greet the user and Give a comprehensive answer to the Question.
    #         Respond only to the Question asked, response should be concise and relevant to the question.""",
    #     },
    #     {
    #         "role": "user",
    #         "content": """Context:
    #     {context}
    #     ---
    #     Now here is the Question you need to answer.
    #     Question:{question}
    #             """,
    #     },
    # ]
    #
    # RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
    #     prompt_chat, tokenize=False, add_generation_prompt=True,
    # )

    prompt_chat1 = [
        {
            "role": "system",
            "content": """Your are an helpful AI assistant, Using the information contained in the context,
                greet the user and Give a comprehensive answer to the Question.
                Respond only to the Question asked, response should be concise and relevant to the question.""",
        },
        {
            "role": "user",
            "content": """Context:
            {context}
            ---
            Now here is the Question you need to answer.
            Question:{question}
                    """,
        },
    ]
    prompt_chat2 = [
        {
            "role": "system",
            "content": """You are helpful AI assistant, 
                assistant is unable to answer the question given by user. inform the user that you cannot answer the question politely and inform the user to ask questions regarding only the transport services only.""",

        },
        {
            "role": "user",
            "content": """
                Now here is the Question.
                Question:{question}
                """,
        },
    ]

    RAG_PROMPT_TEMPLATE1 = tokenizer.apply_chat_template(
        prompt_chat1, tokenize=False, add_generation_prompt=True,
    )
    RAG_PROMPT_TEMPLATE2 = tokenizer.apply_chat_template(
        prompt_chat2, tokenize=False, add_generation_prompt=True,
    )

    print("Microsoft Phi-3 model initialized")


def Retrival_Augmentation(query):
    global KNOWLEDGE_VECTOR_DATABASE

    # user_query = query
    # retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=1)
    #
    # print("======================================\n")
    # print(retrieved_docs[0].page_content)
    # print("======================================\n")
    #
    # return retrieved_docs[0].page_content

    user_query = query
    args = {'score_threshold': 0.70}

    retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search_with_relevance_scores(user_query, k=3, **args)
    if len(retrieved_docs) == 0:
        print("no docs retrived")
        return ""

    print("======================================\n")
    print(retrieved_docs[:][:])
    print("======================================\n")

    result = ""

    for i, j in retrieved_docs:

        result += i.page_content + "\n"

    return result


def generate_answer(context, question):
    global RAG_PROMPT_TEMPLATE, pipe, generation_args, RAG_PROMPT_TEMPLATE1, RAG_PROMPT_TEMPLATE2

    # final_prompt = RAG_PROMPT_TEMPLATE.format(
    #     question="greet me by saying hello and answer the question." + question, context=context
    # )
    #
    # output = pipe(final_prompt, **generation_args)
    # return output[0]['generated_text']

    if (context == ""):
        final_prompt = RAG_PROMPT_TEMPLATE2.format(
            question="greet me by saying hello and answer the question." + question
        )
    else:
        final_prompt = RAG_PROMPT_TEMPLATE1.format(
            question="greet me by saying hello and answer the question." + question, context=context
        )

    output = pipe(final_prompt, **generation_args)
    return output[0]['generated_text']


def delete_empty_sessions():
    try:
        result = chats_collection.delete_many({"messages": []})
        return result.deleted_count
    except Exception as e:
        print(f"Failed to delete empty sessions: {str(e)}")
        return 0


@app.post("/start_session")
async def start_session():
    deleted_count = delete_empty_sessions()
    print(f"Deleted {deleted_count} empty sessions before starting a new session")

    session_id = str(uuid.uuid4())
    chats_collection.insert_one({
        "session_id": session_id,
        "messages": [],
        "created_at": datetime.datetime.now()
    })
    return {"session_id": session_id}


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

    chats_collection.update_one(
        {"session_id": session_id},
        {"$push": {"messages": chat}}
    )

    return {"response": answer}


@app.get("/chats")
async def get_chats():
    try:
        chats = list(chats_collection.find())
        for chat in chats:
            chat["_id"] = str(chat["_id"])
        return chats[::-1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch chats: {str(e)}")


@app.get("/query")
async def get_query():
    # return 200
    return {"response": "Hello"}


@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/hello")
async def get_name():
    return {"message": "Hello"}

@app.get("/chats/{session_id}")
async def get_chats(session_id: str):
    session = chats_collection.find_one({"session_id": session_id})
    session["_id"] = str(session["_id"])
    if session:
        return session
    else:
        raise HTTPException(status_code=404, detail="Session not found")


# if _name_ == "_main_":
#     uvicorn.run(app, host="0.0.0.0", port=8000)