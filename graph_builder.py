# graph_builder.py
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

def create_rag_chain(retriever, google_api_key):
    """
    Tạo một chuỗi xử lý RAG (Retrieval-Augmented Generation).
    """
    if not retriever:
        return None
    
    # Mẫu prompt để hướng dẫn LLM trả lời dựa trên ngữ cảnh được cung cấp
    prompt_template = """
    You are Khoa Dang Le, a highly professional and experienced Data Leader.
    Your personality is helpful, concise, and confident.
    You are speaking to a recruiter. 
    Answer the following question based ONLY on the provided context.
    If the context does not contain the answer, say "I don't have enough information about that in my CV, but I'd be happy to discuss it further."
    Answer ONLY in English.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    YOUR ANSWER (as Khoa Dang Le):
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=google_api_key)
    
    # Xây dựng RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain
