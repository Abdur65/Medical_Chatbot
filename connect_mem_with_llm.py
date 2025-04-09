# import os

# from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
# from langchain_community.vectorstores import FAISS
# from dotenv import load_dotenv, find_dotenv
# # from huggingface_hub import InferenceClient

# # Initialize the client with the correct task type
# # client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.3", task="text-generation")
# load_dotenv(find_dotenv())

# # Step 1: Setup LLM (Mistral with HuggingFace)
# HF_TOKEN = os.environ.get("HF_TOKEN")
# HUGGINFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"


# def load_llm(huggingface_repo_id):
#     llm= HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         task="text-generation",
#         temperature=0.5,
#         model_kwargs={"token":HF_TOKEN,
#                       "max_length":"512"}
#     )
#     return llm

# # Step 2: Connect LLM with FAISS and Create chain
# custom_prompt = """
# You are a helpful assistant. You will be provided with a question and a context. Your task is to answer the question based on the context provided.

# Context: {context}
# Question: {question}

# Be concise and accurate, no need to add any extra information. Avoid unecessary details. Do not drag on the answer.

# If the context does not provide enough information, say "I don't know". Do not make up answers.
# """

# def set_custom_prompt(custom_prompt):
#     prompt = PromptTemplate(template=custom_prompt, input_variables=["context", "question"])
    
#     return prompt

# # Load Database
# DB_FAISS_PATH = "vectorstore/db_faiss"
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# # QA Chain
# qa_chain = RetrievalQA.from_chain_type(
#     llm = load_llm(HUGGINFACE_REPO_ID),
#     chain_type = "stuff",
#     # Where info is retrieved from. k denotes the top k results to be retrieved
#     retriever = db.as_retriever(search_kwargs={"k": 3}),
#     # where the results are retrieved from
#     return_source_documents = True,
#     chain_type_kwargs={"prompt": set_custom_prompt(custom_prompt)}
    
# )

# # Invoke the chain with a single query
# USER_QUERY = input("Enter your question: ")
# result = qa_chain.invoke({"query": USER_QUERY})
# print("Result: ", result["result"])
# # print("Source Documents: ", result["source_documents"])


import os
from dotenv import load_dotenv, find_dotenv

from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
from langchain.llms.base import LLM
from typing import Optional, List

# Load environment variables
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

# Step 1: Setup HuggingFace InferenceClient
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
client = InferenceClient(model=HUGGINGFACE_REPO_ID, token=HF_TOKEN)

# Step 2: Define custom call wrapper (replaces HuggingFaceEndpoint)
def generate_answer(prompt):
    response = client.text_generation(prompt=prompt, max_new_tokens=512, temperature=0.5)
    return response  # Directly return the response as it is already a string

# Step 3: Prompt template
custom_prompt_template = """
You are a helpful assistant. You will be provided with a question and a context. Your task is to answer the question based on the context provided.

Context: {context}
Question: {question}

Be concise and accurate, no need to add any extra information. Avoid unecessary details. Do not drag on the answer.

If the context does not provide enough information, say "I am not equipped to answer this question". Do not make up answers.
"""

def set_custom_prompt(custom_prompt):
    return PromptTemplate(template=custom_prompt, input_variables=["context", "question"])

# Step 4: Load FAISS vector DB
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 5: Define a custom LLM wrapper

class HFClientLLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return generate_answer(prompt)
    
    @property
    def _identifying_params(self):
        return {"model": HUGGINGFACE_REPO_ID}

    @property
    def _llm_type(self):
        return "huggingface-inference-client"

# Step 6: Use LangChain's RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=HFClientLLM(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(custom_prompt_template)}
)

# Step 7: Get query from user
USER_QUERY = input("Enter your question: ")
result = qa_chain.invoke({"query": USER_QUERY})
print("Result:", result["result"])
# print("Source Documents:", result["source_documents"])
