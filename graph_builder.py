# graph_builder.py
import os
import streamlit as st
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# Nhập danh sách các công cụ chuyên biệt từ file tools.py
from tools import all_tools

# --- PHẦN 1: CẤU HÌNH LLM VÀ TRẠNG THÁI ---

# Cố gắng lấy API key từ secrets của Streamlit
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    # Bật LangSmith tracing nếu có key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
except (FileNotFoundError, KeyError):
    st.warning("API keys not found in Streamlit secrets. The app may not function correctly.")
    GOOGLE_API_KEY = ""
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Khởi tạo mô hình LLM và gắn các công cụ vào nó
# Bằng cách này, LLM sẽ biết khi nào cần sử dụng công cụ nào
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2, google_api_key=GOOGLE_API_KEY)
llm_with_tools = llm.bind_tools(all_tools)

# Định nghĩa trạng thái của Graph
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]

# --- PHẦN 2: ĐỊNH NGHĨA CÁC NODE CỦA GRAPH ---

# Định nghĩa node Agent chính: quyết định hành động tiếp theo
def agent_node(state: AgentState):
    """
    Node này quyết định xem nên gọi công cụ hay trả lời trực tiếp.
    Nó là bộ não của chatbot.
    """
    print("---Executing Agent Node---")
    # Tạo prompt với vai trò và hướng dẫn
    system_prompt = """
    You are Khoa Dang Le, a highly professional and experienced Data Leader. 
    Your personality is helpful, concise, and confident.
    You are speaking to a recruiter. Your task is to answer their questions based on your knowledge and the available tools.
    The available tools can retrieve specific parts of your CV.
    - First, analyze the user's question.
    - If the question can be answered by calling one or more tools, call them.
    - If you have already called tools and have the results, use that information to formulate a final, comprehensive answer to the user.
    - If the question is a simple greeting or doesn't require CV data, respond directly.
    - Answer ONLY in English. Be friendly and professional.
    """
    
    # Tạo danh sách tin nhắn mới để gửi đến LLM, bao gồm cả system prompt
    messages_for_llm = state["messages"]
    if messages_for_llm[0].role != "system":
        messages_for_llm = [AIMessage(content=system_prompt, role="system")] + messages_for_llm

    # Gọi LLM đã được gắn công cụ
    response = llm_with_tools.invoke(messages_for_llm)
    return {"messages": [response]}

# Node thực thi công cụ
# ToolNode là một node dựng sẵn của LangGraph giúp đơn giản hóa việc gọi công cụ.
tool_node = ToolNode(all_tools)

# Định nghĩa hàm quyết định định tuyến (conditional edge)
def should_continue(state: AgentState) -> str:
    """
    Hàm này kiểm tra xem agent có quyết định gọi công cụ hay không,
    từ đó quyết định luồng đi tiếp theo của graph.
    """
    print("---Executing Conditional Edge---")
    last_message = state['messages'][-1]
    # Nếu không có tool_calls, kết thúc vòng lặp
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        print("Decision: End Graph")
        return "end"
    # Ngược lại, tiếp tục gọi công cụ
    else:
        print("Decision: Continue to Tool Node")
        return "continue"

# --- PHẦN 3: XÂY DỰNG VÀ BIÊN DỊCH GRAPH ---

def create_recruitment_graph():
    """
    Tạo và biên dịch graph LangGraph cho chatbot tuyển dụng.
    Kiến trúc này sử dụng một agent có khả năng gọi nhiều công cụ chuyên biệt.
    """
    workflow = StateGraph(AgentState)

    # Thêm các node vào graph
    workflow.add_node("agent", agent_node)
    workflow.add_node("action", tool_node)

    # Đặt điểm bắt đầu là node agent
    workflow.set_entry_point("agent")

    # Thêm cạnh điều kiện: sau khi agent chạy, quyết định xem nên
    # tiếp tục gọi công cụ (action) hay kết thúc (end).
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",
            "end": END,
        },
    )

    # Thêm cạnh thông thường: sau khi thực thi công cụ (action),
    # quay trở lại node agent để nó xử lý kết quả và quyết định tiếp.
    workflow.add_edge("action", "agent")

    # Tạo bộ nhớ để lưu trữ lịch sử chat
    memory = MemorySaver()

    # Biên dịch graph
    recruitment_app = workflow.compile(checkpointer=memory)
    
    return recruitment_app
