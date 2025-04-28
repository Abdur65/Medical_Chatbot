# import streamlit as st
# import os
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from huggingface_hub import InferenceClient
# from langchain.llms.base import LLM
# from typing import Optional, List

# HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
# HF_TOKEN = os.environ.get("HF_TOKEN")
# client = InferenceClient(model=HUGGINGFACE_REPO_ID, token=HF_TOKEN)

# DB_FAISS_PATH = "vectorstore/db_faiss"


# @st.cache_resource
# def get_vectorstore():
#     embedding_model = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )
#     db = FAISS.load_local(
#         DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True
#     )

#     return db


# def set_custom_prompt(custom_prompt):
#     return PromptTemplate(
#         template=custom_prompt, input_variables=["context", "question"]
#     )


# def main():
#     st.title("🩺 Medical Chatbot 🏥")

#     # Keeping track of messages in session state
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     for message in st.session_state.messages:
#         st.chat_message(message["role"]).markdown(message["content"])

#     prompt = st.chat_input("Ask a medical question")

#     if prompt:
#         st.chat_message("user").markdown(prompt)
#         # Keeping history of messages
#         st.session_state.messages.append({"role": "user", "content": prompt})

#         custom_prompt_template = """
# You are a medical helpful assistant. You will be provided with a question and a context. Your task is to answer the question based on the context provided.

# Context: {context}
# Question: {question}


# Write your answer is paragraphs. Provide accurate answers.

# First explain what was asked of you. Then if the question is about any type of sickness or disease or any type of medical condition, tell the symptoms of said disease. Finally, recommend ways to avoid the mentioned sickness or diesease. Avoid unecessary details. Do not drag on the answer.

# If the context does not provide enough information, say "I am not equipped to answer this question". Do not make up answers.
# """
#         HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
#         HF_TOKEN = os.environ.get("HF_TOKEN")
#         client = InferenceClient(model=HUGGINGFACE_REPO_ID, token=HF_TOKEN)

#         def generate_answer(prompt):
#             response = client.text_generation(
#                 prompt=prompt, max_new_tokens=512, temperature=0.5
#             )
#             return response  # Directly return the response as it is already a string

#         class HFClientLLM(LLM):
#             def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
#                 return generate_answer(prompt)

#             @property
#             def _identifying_params(self):
#                 return {"model": HUGGINGFACE_REPO_ID}

#             @property
#             def _llm_type(self):
#                 return "huggingface-inference-client"

#         try:
#             vectorstore = get_vectorstore()
#             if vectorstore is None:
#                 st.error("Error loading the vector store.")

#             qa_chain = RetrievalQA.from_chain_type(
#                 llm=HFClientLLM(),
#                 chain_type="stuff",
#                 retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
#                 return_source_documents=True,
#                 chain_type_kwargs={"prompt": set_custom_prompt(custom_prompt_template)},
#             )

#             response = qa_chain.invoke({"query": prompt})

#             result = response["result"]
#             # source_documents = response["source_documents"]
#             result_to_show = result

#             # response = "Hi! I am a medical chatbot. I can help you with your medical queries. Please ask me anything."
#             st.chat_message("assistant").markdown(result_to_show)
#             # Keeping history of messages
#             st.session_state.messages.append(
#                 {"role": "assistant", "content": result_to_show}
#             )

#         except Exception as e:
#             st.error(f"An error occurred: {str(e)}")


# if __name__ == "__main__":
#     main()
import streamlit as st
import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from huggingface_hub import InferenceClient
from langchain.llms.base import LLM
from typing import Optional, List
import time

# Set Streamlit page configuration
st.set_page_config(page_title="Medical Chatbot", page_icon="🩺", layout="wide")

# Advanced custom CSS
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #e0f7fa, #e1f5fe);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .chat-container {
            max-width: 800px;
            margin: auto;
            padding: 30px 15px;
        }
        .message {
            display: flex;
            align-items: flex-start;
            margin-bottom: 20px;
        }
        .user-icon, .bot-icon {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background-size: cover;
            margin-right: 15px;
        }
        .user-icon {
            background-image: url('https://cdn-icons-png.flaticon.com/512/1946/1946429.png');
        }
        .bot-icon {
            background-image: url('https://cdn-icons-png.flaticon.com/512/4712/4712038.png');
        }
        .message-content {
            background-color: white;
            padding: 15px 20px;
            border-radius: 20px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
            max-width: 80%;
            word-wrap: break-word;
        }
        .user .message-content {
            background-color: #d1e7dd;
            margin-left: auto;
            margin-right: 0;
        }
        .assistant .message-content {
            background-color: #fde2e2;
            margin-right: auto;
            margin-left: 0;
        }
        .stChatInputContainer {
            background: transparent;
        }
    </style>
""", unsafe_allow_html=True)

HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = os.environ.get("HF_TOKEN")
client = InferenceClient(model=HUGGINGFACE_REPO_ID, token=HF_TOKEN)

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(
        DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True
    )
    return db

def set_custom_prompt(custom_prompt):
    return PromptTemplate(template=custom_prompt, input_variables=["context", "question"])

def main():
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    st.markdown("## 🩺 Welcome to Your Personal Medical Assistant 🏥")
    st.markdown("Feel free to ask any medical-related questions!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the chat history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]

        if role == "user":
            st.markdown(f"""
                <div class="message user">
                    <div class="user-icon"></div>
                    <div class="message-content">{content}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="message assistant">
                    <div class="bot-icon"></div>
                    <div class="message-content">{content}</div>
                </div>
            """, unsafe_allow_html=True)

    prompt = st.chat_input("Type your medical question...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f"""
            <div class="message user">
                <div class="user-icon"></div>
                <div class="message-content">{prompt}</div>
            </div>
        """, unsafe_allow_html=True)

        custom_prompt_template = """
You are a medical helpful assistant. You will be provided with a question and a context. Your task is to answer the question based on the context provided.

Context: {context}
Question: {question}

Write your answer in paragraphs. Provide accurate answers.

First explain what was asked of you. Then if the question is about any type of sickness or disease or any type of medical condition, tell the symptoms of said disease. Finally, recommend ways to avoid the mentioned sickness or disease. Avoid unnecessary details. Do not drag on the answer.

If the context does not provide enough information, say "I am not equipped to answer this question." Do not make up answers.
"""

        def generate_answer(prompt):
            response = client.text_generation(prompt=prompt, max_new_tokens=512, temperature=0.5)
            return response

        class HFClientLLM(LLM):
            def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
                return generate_answer(prompt)

            @property
            def _identifying_params(self):
                return {"model": HUGGINGFACE_REPO_ID}

            @property
            def _llm_type(self):
                return "huggingface-inference-client"

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Error loading the vector store.")

            qa_chain = RetrievalQA.from_chain_type(
                llm=HFClientLLM(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt(custom_prompt_template)},
            )

            response = qa_chain.invoke({"query": prompt})

            result = response["result"]

            # Simulate typing animation
            placeholder = st.empty()
            full_response = ""
            for char in result:
                full_response += char
                placeholder.markdown(f"""
                    <div class="message assistant">
                        <div class="bot-icon"></div>
                        <div class="message-content">{full_response}</div>
                    </div>
                """, unsafe_allow_html=True)
                time.sleep(0.01)  # Typing speed

            st.session_state.messages.append({"role": "assistant", "content": result})

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
