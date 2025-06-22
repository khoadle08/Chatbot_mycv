# main_app.py
import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import time
from datetime import datetime

# Nhập hàm tạo graph từ graph_builder.py
from graph_builder import create_recruitment_graph

# --- PHẦN 0: CẤU HÌNH GIỚI HẠN YÊU CẦU (RATE LIMITING) ---
REQUESTS_PER_MINUTE = 15
REQUESTS_PER_DAY = 50 

# --- PHẦN 1: KHỞI TẠO ỨNG DỤNG VÀ GRAPH ---

st.set_page_config(page_title="Khoa Dang Le's AI Assistant", page_icon="🤖")

st.title("🤖 Khoa Dang Le's AI Recruiter Assistant")
st.markdown("""
Welcome, Recruiter! I am an AI assistant representing Khoa Dang Le.
You can ask me anything about his CV, from his experience and projects to his technical skills.
I'm here to provide you with accurate information quickly. Let's chat!
""")

# Tạo graph chỉ một lần và lưu vào cache của Streamlit để tăng hiệu suất
@st.cache_resource
def get_graph():
    return create_recruitment_graph()

recruitment_app = get_graph()

# --- PHẦN 2: QUẢN LÝ SESSION VÀ LỊCH SỬ CHAT ---

# Thiết lập session_state để lưu trữ tin nhắn, thread_id và thông tin rate limit
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "request_timestamps" not in st.session_state:
    st.session_state.request_timestamps = []
if "daily_request_count" not in st.session_state:
    st.session_state.daily_request_count = 0
if "last_request_date" not in st.session_state:
    st.session_state.last_request_date = datetime.now().date().isoformat()

# Hiển thị các tin nhắn đã có trong lịch sử chat
for message in st.session_state.messages:
    if not isinstance(message, ToolMessage):
        role = "assistant" if isinstance(message, AIMessage) else "user"
        with st.chat_message(role):
            st.markdown(message.content)

# --- PHẦN 3: HÀM KIỂM TRA GIỚI HẠN YÊU CẦU ---

def check_rate_limits():
    """Kiểm tra xem người dùng có vượt quá giới hạn yêu cầu hay không."""
    current_time = time.time()
    current_date = datetime.now().date()

    if st.session_state.last_request_date != current_date.isoformat():
        st.session_state.daily_request_count = 0
        st.session_state.last_request_date = current_date.isoformat()
        st.session_state.request_timestamps = []

    if st.session_state.daily_request_count >= REQUESTS_PER_DAY:
        return False, f"You have reached the daily limit of {REQUESTS_PER_DAY} requests. Please try again tomorrow."

    st.session_state.request_timestamps = [
        ts for ts in st.session_state.request_timestamps if current_time - ts < 60
    ]
    if len(st.session_state.request_timestamps) >= REQUESTS_PER_MINUTE:
        return False, f"You have reached the limit of {REQUESTS_PER_MINUTE} requests per minute. Please wait a moment."

    return True, ""

# --- PHẦN 4: XỬ LÝ INPUT CỦA NGƯỜI DÙNG ---

if prompt := st.chat_input("Ask me about Khoa's profile..."):
    
    is_allowed, message = check_rate_limits()
    
    if not is_allowed:
        st.warning(message)
    else:
        st.session_state.request_timestamps.append(time.time())
        st.session_state.daily_request_count += 1

        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        # CẬP NHẬT: Hiển thị trạng thái bot đang làm gì
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Sử dụng st.status để hiển thị các bước xử lý của bot
            with st.status("Assistant is thinking...", expanded=False) as status:
                try:
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}
                    inputs = {"messages": [HumanMessage(content=prompt)]}
                    
                    # Sử dụng stream_mode="events" để nhận các sự kiện chi tiết
                    for event in recruitment_app.stream(inputs, config=config, stream_mode="events"):
                        kind = event["event"]
                        name = event.get("name", "")

                        # Khi một công cụ bắt đầu chạy, cập nhật trạng thái
                        if kind == "on_tool_start":
                            tool_name = event['name']
                            tool_input = event['data'].get('input', {})
                            status.update(label=f"Calling tool: `{tool_name}`...")
                        
                        # Khi agent đang tạo câu trả lời cuối cùng, stream nó ra màn hình
                        if name == "agent" and kind == "on_chat_model_stream":
                            content = event["data"]["chunk"].content
                            if content:
                                status.update(label="Generating final answer...")
                                full_response += content
                                message_placeholder.markdown(full_response + "▌")
                    
                    status.update(label="Done!", state="complete", expanded=False)

                except Exception as e:
                    full_response = f"Sorry, I encountered an error: {e}"
                    st.error(full_response)
            
            # Hiển thị câu trả lời cuối cùng và cập nhật lịch sử chat
            message_placeholder.markdown(full_response)
            if "error" not in full_response:
                 final_state = recruitment_app.get_state(config)
                 st.session_state.messages = final_state.values()["messages"]
            else:
                 st.session_state.messages.append(AIMessage(content=full_response))
