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
# Lưu ý: Giới hạn 1,000,000 token/phút là của API Gemini và được quản lý ở phía Google,
# không được thực thi trực tiếp trong mã nguồn của ứng dụng này.

# --- PHẦN 1: KHỞI TẠO ỨNG DỤNG VÀ GRAPH ---

st.set_page_config(page_title="Khoa Dang Le's AI Assistant", page_icon="�")

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
    # Không hiển thị các tin nhắn của ToolMessage cho người dùng
    if not isinstance(message, ToolMessage):
        # SỬA LỖI: Đối tượng message của LangChain có thuộc tính 'type' thay vì 'role' cho các phiên bản mới hơn, 
        # nhưng thuộc tính 'role' vẫn tồn tại cho mục đích tương thích.
        # Streamlit cần 'user' hoặc 'assistant'.
        role = "assistant" if isinstance(message, AIMessage) else "user"
        with st.chat_message(role):
            st.markdown(message.content)

# --- PHẦN 3: HÀM KIỂM TRA GIỚI HẠN YÊU CẦU ---

def check_rate_limits():
    """Kiểm tra xem người dùng có vượt quá giới hạn yêu cầu hay không."""
    current_time = time.time()
    current_date = datetime.now().date()

    # 1. Kiểm tra và đặt lại giới hạn hàng ngày
    if st.session_state.last_request_date != current_date.isoformat():
        st.session_state.daily_request_count = 0
        st.session_state.last_request_date = current_date.isoformat()
        st.session_state.request_timestamps = [] # Đặt lại cả yêu cầu mỗi phút khi sang ngày mới

    if st.session_state.daily_request_count >= REQUESTS_PER_DAY:
        return False, f"You have reached the daily limit of {REQUESTS_PER_DAY} requests. Please try again tomorrow."

    # 2. Kiểm tra giới hạn mỗi phút
    # Loại bỏ các timestamp cũ hơn 60 giây
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
        # Cập nhật bộ đếm giới hạn
        st.session_state.request_timestamps.append(time.time())
        st.session_state.daily_request_count += 1

        # SỬA LỖI: Tạo HumanMessage không cần tham số `role`
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        # Hiển thị "thinking..." trong khi chatbot đang xử lý
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking... ▌")

            # Cấu hình cho graph, quan trọng nhất là `thread_id`
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            # Tạo input cho graph
            inputs = {"messages": [HumanMessage(content=prompt)]}
            
            # Gọi graph và stream câu trả lời
            full_response = ""
            try:
                for chunk in recruitment_app.stream(inputs, config=config, stream_mode="values"):
                    # Lấy tin nhắn cuối cùng từ state của graph
                    last_message = chunk["messages"][-1]
                    # Cập nhật UI ngay khi có nội dung mới
                    if last_message.content:
                        full_response = last_message.content
                        message_placeholder.markdown(full_response + " ▌")
                
                # Hoàn tất và hiển thị câu trả lời cuối cùng
                message_placeholder.markdown(full_response)
                
                # Lấy toàn bộ lịch sử từ graph và cập nhật session_state
                # Điều này đảm bảo state (bao gồm ToolMessage) được đồng bộ
                final_state = recruitment_app.get_state(config)
                st.session_state.messages = final_state.values()["messages"]

            except Exception as e:
                error_message = f"Sorry, I encountered an error. Please ensure your API keys are correctly configured in Streamlit secrets. Error: {e}"
                st.error(error_message)
                # SỬA LỖI: Sử dụng AIMessage cho tin nhắn lỗi của trợ lý
                st.session_state.messages.append(AIMessage(content=error_message))
