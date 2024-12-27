import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
import pickle
from langchain_community.docstore.in_memory import InMemoryDocstore

KNOWLEDGE_VECTOR_DATABASE = None
RAG_PROMPT_TEMPLATE1 = None
RAG_PROMPT_TEMPLATE2 = None
pipe = None
generation_args = None

def init_models():
    global KNOWLEDGE_VECTOR_DATABASE, RAG_PROMPT_TEMPLATE1, RAG_PROMPT_TEMPLATE2, pipe, generation_args

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
        "microsoft/Phi-3.5-mini-instruct",
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

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
                assistant is unable to answer the question given by user. inform the user that you cannot answer the question politely and inform the user to ask questions regarding the transport services only.""",

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
    global RAG_PROMPT_TEMPLATE1, RAG_PROMPT_TEMPLATE2, pipe, generation_args

    if context == "":
        final_prompt = RAG_PROMPT_TEMPLATE2.format(
            question="greet me by saying hello and answer the question." + question
        )
    else:
        final_prompt = RAG_PROMPT_TEMPLATE1.format(
            question="greet me by saying hello and answer the question." + question, context=context
        )

    output = pipe(final_prompt, **generation_args)

    # Free up GPU memory
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return output[0]['generated_text']
