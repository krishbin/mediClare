from fastapi import FastAPI, File, UploadFile
from utils import variables, set_variable, relative_path
from utils.logger import Logger
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.model import Model, Chatbot
from PIL import Image

model = Model("nadika/medical_jargons_simplifier")
chatbot = Chatbot(
    model_name=variables["model_name"],
    csv_file=variables["model_embedding_csv_file"],
    embedding_file=variables["model_embedding_file"],
    chunks_file=variables["model_chunks_file"],
)

logger = Logger(log_dir=relative_path("/logs"))
logger.info("Imported the required modules")
logger.info("Starting the application")

medicalsearch = FastAPI()


class SearchPrompt(BaseModel):
    input: str


class ConversationPrompt(BaseModel):
    input: str


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
    print(simplify)
    return model.generate_response(simplify.input)


@medicalsearch.get("/simplify_text_llm")
def conversation(conv: ConversationPrompt):
    return chatbot.get_chatbot_answer(conv.input)


@medicalsearch.get("/simplify_image_report_llm")
def simplify_image_report(simplify: SearchPrompt ):
    chatbot.reset_conversation()
    return chatbot.get_chatbot_answer(simplify.input)

@medicalsearch.get("/simplify_text_llm_context")
def simplify_text_llm(simplify: SearchPrompt):
    return chatbot.get_chatbot_answer_with_context(simplify.input)

@medicalsearch.get("/simplify_reset_context")
def reset_context():
    chatbot.reset_conversation()
    return {"message": "Context reset"}
    