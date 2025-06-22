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
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2, google_api_key=GOOGLE_API_KEY)
llm_with_tools = llm.bind_tools(all_tools)

# Định nghĩa trạng thái của Graph
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]

# --- PHẦN 2: ĐỊNH NGHĨA CÁC NODE CỦA GRAPH ---

def agent_node(state: AgentState):
    """
    Node này quyết định xem nên gọi công cụ hay trả lời trực tiếp.
    Nó là bộ não của chatbot.
    """
    print("---Executing Agent Node---")
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
    
    messages_for_llm = state["messages"]

    # SỬA LỖI: Kiểm tra loại đối tượng trước khi truy cập thuộc tính 'role'.
    # HumanMessage không có thuộc tính 'role', vì vậy chúng ta cần một cách kiểm tra an toàn hơn.
    is_system_prompt_present = False
    if messages_for_llm and isinstance(messages_for_llm[0], AIMessage):
        # Chỉ những tin nhắn AIMessage mới có khả năng là system prompt
        if hasattr(messages_for_llm[0], 'role') and messages_for_llm[0].role == 'system':
            is_system_prompt_present = True

    # Nếu system prompt chưa có, thêm nó vào đầu danh sách tin nhắn
    if not is_system_prompt_present:
        messages_for_llm = [AIMessage(content=system_prompt, role="system")] + messages_for_llm

    # Gọi LLM với danh sách tin nhắn đã được chuẩn bị
    response = llm_with_tools.invoke(messages_for_llm)
    return {"messages": [response]}

# Node thực thi công cụ
tool_node = ToolNode(all_tools)

def should_continue(state: AgentState) -> str:
    """
    Hàm này kiểm tra xem agent có quyết định gọi công cụ hay không,
    từ đó quyết định luồng đi tiếp theo của graph.
    """
    print("---Executing Conditional Edge---")
    last_message = state['messages'][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        print("Decision: End Graph")
        return "end"
    else:
        print("Decision: Continue to Tool Node")
        return "continue"

# --- PHẦN 3: XÂY DỰNG VÀ BIÊN DỊCH GRAPH ---

def create_recruitment_graph():
    """
    Tạo và biên dịch graph LangGraph cho chatbot tuyển dụng.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("action", tool_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",
            "end": END,
        },
    )

    workflow.add_edge("action", "agent")

    memory = MemorySaver()
    recruitment_app = workflow.compile(checkpointer=memory)
    
    return recruitment_app
