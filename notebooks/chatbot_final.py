from langchain.document_loaders import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import TokenTextSplitter,RecursiveCharacterTextSplitter
import pandas as pd
import csv
from langchain_community.llms import Ollama
import os
import pickle
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage


def load_faiss_index(index_file):
    """
    Load the FAISS index from disk.

    Args:
    - index_file (str): Path to the FAISS index file.

    Returns:
    - FAISS: Loaded FAISS index.
    """
    with open(index_file, 'rb') as f:
        db = pickle.load(f)
    return db

def initialize_chatbot():
    """
    Initialize the Ollama model and load and chunk documents.

    Returns:
    - tuple: Tuple containing Ollama model instance and FAISS index.
    """
    csv_file = 'wiki_medical_terms.csv'  # Replace with your CSV file path
    index_file = 'faiss_index.pkl'  # File to store the FAISS index

    # Initialize Ollama model
    llm = Ollama(model="llama3")
    
    db = load_faiss_index(index_file)
    retriever=db.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return llm, retriever,rag_chain

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_chatbot_answer(llm, retriever, question,chat_history,rag_chain):
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt)
    qa_system_prompt = """
    You are a clinical bot that explains medical jargons in simple words. Answer the given questions based on your knowledge and given context.\
    {context} 
    You are allowed to rephrase the answer based on the context. Explain it so that the normal person can understand it.
    """
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    

    ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
    chat_history.extend([HumanMessage(content=question), ai_msg_1["answer"]])
    return ai_msg_1["answer"]



def main():
    llm, retriever, rag_chain = initialize_chatbot()

    chat_history = []

    while True:
        user_input = input("User: ")
        
        if user_input.lower() == 'exit':
            print("Exiting chatbot...")
            break
        
        response = get_chatbot_answer(llm, retriever, user_input, chat_history,rag_chain)
        print("Bot:", response)
        print()

main()