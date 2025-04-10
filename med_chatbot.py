import streamlit as st
import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from huggingface_hub import InferenceClient
from langchain.llms.base import LLM
from typing import Optional, List

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
    return PromptTemplate(
        template=custom_prompt, input_variables=["context", "question"]
    )


def main():
    st.title("🩺 Medical Chatbot 🏥")

    # Keeping track of messages in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    prompt = st.chat_input("Ask a medical question")

    if prompt:
        st.chat_message("user").markdown(prompt)
        # Keeping history of messages
        st.session_state.messages.append({"role": "user", "content": prompt})

        custom_prompt_template = """
You are a helpful assistant. You will be provided with a question and a context. Your task is to answer the question based on the context provided.

Context: {context}
Question: {question}


Be concise and accurate, no need to add any extra information. Write your answer is paragraphs. 

First explain what was asked of you. Then if the question is about any type of sickness or disease of medical condition, tell the symptoms of said disease. Finally, recommend ways to avoid the mentioned sickness or diesease. Avoid unecessary details. Do not drag on the answer.

If the context does not provide enough information, say "I am not equipped to answer this question". Do not make up answers.
"""
        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")
        client = InferenceClient(model=HUGGINGFACE_REPO_ID, token=HF_TOKEN)

        def generate_answer(prompt):
            response = client.text_generation(
                prompt=prompt, max_new_tokens=512, temperature=0.5
            )
            return response  # Directly return the response as it is already a string

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
            # source_documents = response["source_documents"]
            result_to_show = result

            # response = "Hi! I am a medical chatbot. I can help you with your medical queries. Please ask me anything."
            st.chat_message("assistant").markdown(result_to_show)
            # Keeping history of messages
            st.session_state.messages.append(
                {"role": "assistant", "content": result_to_show}
            )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
