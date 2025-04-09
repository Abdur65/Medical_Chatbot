from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Step 1 - Load file
DATA_PATH = "data/"

def load_pdf_files(data):
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    
    documents = loader.load()
    return documents

documments = load_pdf_files(data=DATA_PATH)
# print(f"Length of documents: {len(documments)}")

# Step 2 - Create chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    
    chunks = text_splitter.split_documents(extracted_data)
    
    return chunks

chunks = create_chunks(extracted_data=documments)
# print(f"Length of chunks: {len(chunks)}")

# Step 3 - Create vector embeddings

def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model=get_embedding_model()

# Step 4: Store embeddings in FAISS
DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(chunks, embedding_model)
db.save_local(DB_FAISS_PATH)