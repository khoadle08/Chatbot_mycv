# main_app.py
import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, AIMessage
import time
from datetime import datetime
import os

# Nhập các hàm mới từ các file đã được cấu trúc lại
from vector_store_builder import get_retriever
from graph_builder import create_rag_chain

# --- PHẦN 0: CẤU HÌNH API KEY VÀ GIỚI HẠN ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("Vui lòng thiết lập GOOGLE_API_KEY trong secrets của Streamlit.")
    GOOGLE_API_KEY = None

REQUESTS_PER_MINUTE = 15
REQUESTS_PER_DAY = 50 

# --- PHẦN 1: KHỞI TẠO ỨNG DỤNG VÀ RAG CHAIN ---

st.set_page_config(page_title="Khoa Dang Le's AI Assistant", page_icon="🤖")
st.title("🚀 Khoa Dang Le's AI Recruiter Assistant (v.RAG)")
st.markdown("""
Welcome, Recruiter! This is a high-speed assistant powered by RAG (Retrieval-Augmented Generation).
Ask me anything about Khoa's CV, and I'll get you the most relevant information instantly.
If you intend to call me, please email me first because during business hours, I will not be able to take calls from you unless by appointment. Thank you (mail: khoa.d.le08@gmail.com)
""")

# Tạo retriever và RAG chain chỉ một lần
@st.cache_resource
def initialize_rag_chain():
    if not GOOGLE_API_KEY:
        return None
    retriever = get_retriever(GOOGLE_API_KEY)
    rag_chain = create_rag_chain(retriever, GOOGLE_API_KEY)
    return rag_chain

rag_chain = initialize_rag_chain()

# --- PHẦN 2: QUẢN LÝ SESSION VÀ LỊCH SỬ CHAT ---

if "messages" not in st.session_state:
    st.session_state.messages = []
if "request_timestamps" not in st.session_state:
    st.session_state.request_timestamps = []
if "daily_request_count" not in st.session_state:
    st.session_state.daily_request_count = 0
if "last_request_date" not in st.session_state:
    st.session_state.last_request_date = datetime.now().date().isoformat()

# Hiển thị các tin nhắn cũ
for message in st.session_state.messages:
    role = "assistant" if isinstance(message, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(message.content)

# --- PHẦN 3: HÀM KIỂM TRA GIỚI HẠN YÊU CẦU ---

def check_rate_limits():
    current_time = time.time()
    current_date = datetime.now().date()
    if st.session_state.last_request_date != current_date.isoformat():
        st.session_state.daily_request_count = 0
        st.session_state.last_request_date = current_date.isoformat()
    if st.session_state.daily_request_count >= REQUESTS_PER_DAY:
        return False, f"Daily request limit ({REQUESTS_PER_DAY}) reached."
    st.session_state.request_timestamps = [ts for ts in st.session_state.request_timestamps if current_time - ts < 60]
    if len(st.session_state.request_timestamps) >= REQUESTS_PER_MINUTE:
        return False, f"Per-minute request limit ({REQUESTS_PER_MINUTE}) reached."
    return True, ""

# --- PHẦN 4: XỬ LÝ INPUT CỦA NGƯỜI DÙNG ---

if not rag_chain:
    st.warning("Hệ thống chưa sẵn sàng. Vui lòng kiểm tra cấu hình API key.")
elif prompt := st.chat_input("Ask me about Khoa's profile..."):
    is_allowed, message = check_rate_limits()
    if not is_allowed:
        st.warning(message)
    else:
        st.session_state.request_timestamps.append(time.time())
        st.session_state.daily_request_count += 1
        st.session_state.messages.append(HumanMessage(content=prompt))
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.status("Searching CV...", expanded=False) as status:
                full_response = ""
                try:
                    # Gọi RAG chain và stream kết quả
                    for chunk in rag_chain.stream(prompt):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                    status.update(label="Done!", state="complete")
                except Exception as e:
                    full_response = f"Sorry, an error occurred: {e}"
                    st.error(full_response)
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append(AIMessage(content=full_response))
