# vector_store_builder.py
import streamlit as st
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

def load_and_chunk_cv(file_path: str = "mycv.json") -> list[Document]:
    """
    Tải dữ liệu CV, chuyển đổi thành các document và chia nhỏ chúng.
    Mỗi phần chính của CV (kinh nghiệm, dự án) sẽ là một document riêng
    với metadata để dễ dàng truy xuất sau này.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return []

    docs = []
    # Ghép nối các phần của CV thành các chuỗi văn bản dài
    # để embedding có nhiều ngữ cảnh hơn.
    
    # Giới thiệu
    if "introduction" in data:
        docs.append(Document(page_content=data["introduction"], metadata={"source": "introduction"}))
        
    # Kinh nghiệm
    if "experience" in data:
        exp_text = "\n\n".join([f"Title: {exp['title']} at {exp['company']}\nDates: {exp['dates']}\nResponsibilities:\n- " + "\n- ".join(exp['responsibilities']) for exp in data['experience']])
        docs.append(Document(page_content=exp_text, metadata={"source": "experience"}))
        
    # Kỹ năng
    if "technical_skills" in data:
        skills_text = json.dumps(data["technical_skills"], indent=2)
        docs.append(Document(page_content=f"Technical Skills:\n{skills_text}", metadata={"source": "technical_skills"}))

    # Dự án tóm tắt
    if "projects" in data:
        proj_summary_text = "\n\n".join([f"Company: {p['company']}\n" + "\n".join([f"- {item['title']}: {item.get('key_achievements', '')}" for item in p['project_list']]) for p in data['projects']])
        docs.append(Document(page_content=f"Projects Summary:\n{proj_summary_text}", metadata={"source": "projects_summary"}))
        
    # Dự án chi tiết
    if "detail_project" in data:
        for project in data["detail_project"]:
            proj_detail_text = f"Project Name: {project['project_name']}\nCompany: {project['company']}\nStatus: {project['status']}\nGoal: {project['project_goal']}\nProblem: {project['problem_to_solve']}\nRole: {project['role_and_responsibilities']}\nAchievements: {', '.join(project.get('achievements', []))}"
            docs.append(Document(page_content=proj_detail_text, metadata={"source": f"detail_project_{project['project_name']}"}))

    # Chia nhỏ các document thành các chunk
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunked_docs = text_splitter.split_documents(docs)
    return chunked_docs

@st.cache_resource
def create_vector_store(_docs, _google_api_key):
    """
    Tạo vector store từ các document đã được chia nhỏ.
    Sử dụng cache của Streamlit để không phải tạo lại mỗi lần.
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=_google_api_key)
        vector_store = FAISS.from_documents(_docs, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Lỗi khi tạo vector store: {e}")
        return None

def get_retriever(google_api_key):
    """
    Hàm chính để lấy retriever, bao gồm cả việc tải và tạo vector store.
    """
    docs = load_and_chunk_cv()
    if not docs:
        st.error("Không thể tải dữ liệu từ mycv.json. Vui lòng kiểm tra lại file.")
        return None
        
    vector_store = create_vector_store(docs, google_api_key)
    if vector_store:
        return vector_store.as_retriever(search_kwargs={"k": 3}) # Lấy 3 kết quả liên quan nhất
    return None
