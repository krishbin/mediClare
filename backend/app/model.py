from transformers import AutoTokenizer, T5ForConditionalGeneration
from utils import relative_path, Logger
from joblib import dump, load
import os
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import TokenTextSplitter,RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA,LLMChain, create_history_aware_retriever
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage


logger = Logger()
logger.info("Model module loaded")

class Model:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_path = relative_path(f"/models/{model_name}/model")
        self.tokenizer_path = relative_path(f"/models/{model_name}/tokenizer")
        if not os.path.exists(self.model_path):
            self.download_model()
            logger.info(f"Model {model_name} downloaded successfully")
        elif not os.path.exists(self.tokenizer_path) or os.path.exists(self.model_path):
            self.download_model()
            logger.info(f"Model {model_name} downloaded successfully")
        else:
            self.load_model()
            logger.info(f"Model {model_name} loaded successfully")
    
    def __str__(self) -> str:
        return f"Model Name: {self.model_name}\nModel Path: {self.model_path}\nTokenizer Path: {self.tokenizer_path}"
    
    def __repr__(self) -> str:
        self.__str__()

    def refresh_model(self):
        self.download_model()

    def download_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.tokenizer.save_pretrained(self.tokenizer_path)
        self.model.save_pretrained(self.model_path)

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_path)

    def generate_response(self, input_text: str):
        input_text = input_text
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(inputs.input_ids, max_length=1080, num_beams=4, early_stopping=True)
        ret = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Response generated for input: {input_text}")
        logger.info(f"Response: {ret}")
        return ret

class Chatbot:
    def __init__(self, model_name: str, csv_file: str, embedding_file: str, chunks_file: str,fass_index_file:str):
        self.model_name = model_name
        self._llm = None
        self.csv_file = relative_path(csv_file)
        self.embedding_file = relative_path(embedding_file)
        self.chunks_file = relative_path(chunks_file)
        self.fass_index_file = relative_path(fass_index_file)
        self.embeddings_saved = False
        self.chunks_saved = False
        self.faiss_index_saved = False
        self._db = None
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.PROMPT_TEMPLATE = """You are a clinical bot that explains medical jargons in simple words. Answer the given questions based on your knowledge and given context.
        {context}
        You are allowed to rephrase the answer based on the context. Explain it so that the normal person can understand it.
        Question: {question}
        """
        self.PROMPT = PromptTemplate.from_template(self.PROMPT_TEMPLATE)
        self._qa_chain = None
        self.chat_history = []


    @property
    def llm(self):
        if self._llm is None:
            logger.info("Loading LLM")
            self._llm = Ollama(model=self.model_name)
            logger.info("LLM loaded successfully")
        return self._llm
    
    @property
    def db(self):
        if self._db is None:
            logger.info("Loading and chunking documents")
            self._db = self.load_and_chunk_documents()
            logger.info("Documents loaded and chunked successfully")
        return self._db
    
    @property
    def qa_chain(self):
        if self._qa_chain is None:
            self._qa_chain = RetrievalQA.from_chain_type(
                self.llm,
                retriever=self.db.as_retriever(k=2),
                chain_type_kwargs={"prompt": self.PROMPT},
                return_source_documents=True,
            )
        return self._qa_chain
    
    def load_and_chunk_documents(self):
        """
        Load documents from CSV file, chunk them, and create FAISS index.

        Args:
        - csv_file (str): Path to the CSV file containing documents.

        Returns:
        - FAISS: FAISS index containing chunked documents.
        """
        # Load documents from CSV
        if os.path.exists(self.embedding_file):
            logger.info(f"Loading embeddings from pickle file: {self.embedding_file}")
            with open(self.embedding_file, "rb") as f:
                embeddings = load(f)
            self.embeddings_saved = True
            logger.info(f"Embeddings loaded successfully")
        
        if os.path.exists(self.chunks_file):
            logger.info(f"Loading chunks from pickle file: {self.chunks_file}")
            with open(self.chunks_file, "rb") as f:
                chunks = load(f)
            self.chunks_saved = True
            logger.info(f"Chunks loaded successfully")
        
        if not self.embeddings_saved or not self.chunks_saved:
            logger.info(f"Loading documents from CSV file: {self.csv_file}")
            loader = CSVLoader(file_path=self.csv_file)
            data = loader.load()

            # Chunk documents
            if not self.chunks_saved:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=50)
                chunks = text_splitter.split_documents(data)
                logger.info(f"Documents chunked successfully")
                with open(self.chunks_file, "wb") as f:
                    dump(chunks, f)
                logger.info(f"Chunks saved to file: {self.chunks_file}")

            # Create embeddings
            if not self.embeddings_saved:
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                logger.info("Creating embeddings")
                with open (self.embedding_file, "wb") as f:
                    dump(embeddings, f)
                logger.info(f"Embeddings saved to file: {self.embedding_file}")
        # Build FAISS index
        if os.path.exists(self.fass_index_file):
            logger.info(f"Loading FAISS index from file: {self.fass_index_file}")
            db = FAISS.load_local(self.fass_index_file,embeddings=embeddings,allow_dangerous_deserialization=True)
            self.faiss_index_saved = True
            logger.info(f"FAISS index loaded successfully")
        else:
            logger.info("Building FAISS index")
            db = FAISS.from_documents(chunks, embeddings)
            db.save_local(self.fass_index_file)
            logger.info(f"FAISS index saved to file: {self.fass_index_file}")
        return db
    
    def get_answer(self,data: dict):
        result = self.qa_chain(data)
        return result['result']
    
    def reset_conversation(self):
        self.chat_history = []

    def get_chatbot_answer(self, user_input: str,require_context=False):
        """
        Retrieve answer from the chatbot for a given query.

        Args:
        - llm (Ollama): Initialized Ollama model instance.
        - db (FAISS): FAISS index containing chunked documents.
        - query (str): Query for which the chatbot should provide an answer.

        Returns:
        - dict: Dictionary containing the chatbot's answer.
        """
        if require_context:
            history = self.memory.load_memory_variables({})
            if history['chat_history']:
                history = history['chat_history']
            else:
                history = ""
            data = {"query": user_input, "context": history}
            result = self.get_answer(data)
            self.memory.save_context({"chat_history": result})
        result = self.get_answer({"query": user_input})
        return result
    
    def get_chatbot_answer_with_context(self,question):
        retriever = self.db.as_retriever()
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
        self.llm, retriever, contextualize_q_prompt)
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
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        

        ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": self.chat_history})
        self.chat_history.extend([HumanMessage(content=question), ai_msg_1["answer"]])
        return ai_msg_1["answer"]
    
