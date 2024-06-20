from fastapi import FastAPI, File, UploadFile
from utils import variables, set_variable, relative_path
from utils.logger import Logger
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.model import Model, Chatbot
from PIL import Image
from services import db as database


db = database(variables["database_file"])

model = Model(variables["t5_model_name"])
chatbot = Chatbot(
    model_name=variables["model_name"],
    csv_file=variables["model_embedding_csv_file"],
    embedding_file=variables["model_embedding_file"],
    chunks_file=variables["model_chunks_file"],
    fass_index_file=variables["model_faiss_index_file"],
)

logger = Logger(log_dir=relative_path("/logs"))
logger.info("Imported the required modules")
logger.info("Starting the application")

medicalsearch = FastAPI()


class SearchPrompt(BaseModel):
    input: str


class ConversationPrompt(BaseModel):
    input: str

class Feedback(BaseModel):
    feedback: str
    uuid: str


# Configure CORS settings
origins = [
    "http://127.0.0.1:5173",
]

medicalsearch.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@medicalsearch.get("/")
async def root():
    return {"message": "Hello World"}


# @medicalsearch.post("/set_account_details")
# async def get_account_details(account: GetAccountDetail):
#     set_variable("bucket_url", account.bucket_url)
#     set_variable("bucket_key", account.bucket_key)
#     set_variable("bucket_secret", account.bucket_secret)
#     set_variable("bucket_name", account.bucket_name)
#     return {"message": "Account details updated"}


@medicalsearch.get("/simplify_data")
def search_image(simplify: SearchPrompt):
    uuid = db.insert_input(simplify.input)
    response = model.generate_response(simplify.input)
    db.insert_output(uuid, response, "text")
    return response


@medicalsearch.get("/simplify_text_llm")
def conversation(conv: ConversationPrompt):
    uuid = db.insert_input(conv.input)
    response = chatbot.get_chatbot_answer(conv.input)
    db.insert_output(uuid, response, "text")
    return response


@medicalsearch.get("/simplify_image_report_llm")
def simplify_image_report(simplify: SearchPrompt ):
    # uuid = db.insert_input(simplify.input)
    response = chatbot.get_chatbot_answer(simplify.input)
    # db.insert_output(uuid, response, "ocr")
    return response

@medicalsearch.get("/simplify_text_llm_context")
def simplify_text_llm(simplify: SearchPrompt):
    uuid = db.insert_input(simplify.input)
    response = chatbot.get_chatbot_answer_with_context(simplify.input)
    db.insert_output(uuid, response, "text")
    return response

@medicalsearch.get("/simplify_reset_context")
def reset_context():
    chatbot.reset_conversation()
    return {"message": "Context reset"}

@medicalsearch.get("/feedback_random")
def feedback():
    random_input = db.get_random_input()
    output = db.get_output_from_input(random_input[0])
    return {"input": random_input[1], "output": output[1], "uuid": output[0]}

@medicalsearch.get("/get_feedback")
def get_feedback(feed: Feedback):
    db.insert_feedback(feed.uuid, feed.feedback, "doctor")
    return {"message": "Feedback received"}
    
    