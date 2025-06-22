# vector_store_builder.py
import streamlit as st
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from datetime import datetime

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
    
    # Giới thiệu
    if "introduction" in data:
        docs.append(Document(page_content=data["introduction"], metadata={"source": "introduction"}))
        
    # Kinh nghiệm
    if "experience" in data:
        # SỬA LỖI DỨT ĐIỂM: Tạo một tài liệu tóm tắt kinh nghiệm
        experience_summary_lines = ["Here is a summary of my professional experience in reverse chronological order:"]
        
        # Đồng thời, tạo các tài liệu chi tiết cho từng công việc
        for exp in data['experience']:
            dates_str = exp.get('dates', 'N/A')
            is_present_job = 'Present' in dates_str
            if is_present_job:
                current_date = datetime.now().strftime('%B %Y')
                dates_str = dates_str.replace('Present', current_date)

            company_name = exp.get('company', 'N/A')
            title = exp.get('title', 'N/A')
            
            # Thêm vào danh sách tóm tắt
            experience_summary_lines.append(f"- {title} at {company_name} ({dates_str})")
            
            # Tạo tài liệu chi tiết cho từng công việc
            exp_detail_text = (
                f"Detailed work experience at {company_name}:\n"
                f"Title: {title}\n"
                f"Dates: {dates_str}\n"
                "Responsibilities:\n- " + "\n- ".join(exp.get('responsibilities', []))
            )
            docs.append(Document(page_content=exp_detail_text, metadata={"source": f"experience_{company_name}"}))

        # Thêm tài liệu tóm tắt vào danh sách docs
        docs.append(Document(page_content="\n".join(experience_summary_lines), metadata={"source": "experience_summary"}))
        
    # Kỹ năng
    if "technical_skills" in data:
        skills_text = json.dumps(data["technical_skills"], indent=2)
        docs.append(Document(page_content=f"Technical Skills:\n{skills_text}", metadata={"source": "technical_skills"}))

    # Dự án tóm tắt
    if "projects" in data:
        proj_summary_text = "\n\n".join([f"Company: {p.get('company', 'N/A')}\n" + "\n".join([f"- {item.get('title', 'N/A')}: {item.get('key_achievements', '')}" for item in p.get('project_list', [])]) for p in data['projects']])
        docs.append(Document(page_content=f"Projects Summary:\n{proj_summary_text}", metadata={"source": "projects_summary"}))
        
    # Dự án chi tiết
    if "detail_project" in data:
        for project in data["detail_project"]:
            achievements_list = project.get('achievements', [])
            achievements_text = "\n- ".join(achievements_list) if achievements_list else "Not specified."
            methodology = project.get('methodology_and_solution', 'Not specified.')
            methodology_text = ""
            if isinstance(methodology, dict):
                if 'layers' in methodology:
                     methodology_text = "\n".join([f"  - {layer}" for layer in methodology.get('layers', [])])
                elif 'phase_1' in methodology:
                     methodology_text = f"  - Phase 1: {methodology.get('phase_1', 'N/A')}\n  - Phase 2: {methodology.get('phase_2', 'N/A')}"
            elif isinstance(methodology, str):
                methodology_text = methodology
            
            proj_detail_text = (
                f"--- DETAILED PROJECT REPORT ---\n\n"
                f"**Project Name:** {project.get('project_name', 'N/A')}\n"
                f"**Company:** {project.get('company', 'N/A')}\n"
                f"**Project Status:** {project.get('status', 'N/A')}\n\n"
                f"**1. Project Goal:**\n{project.get('project_goal', 'N/A')}\n\n"
                f"**2. The Problem It Solved:**\n{project.get('problem_to_solve', 'N/A')}\n\n"
                f"**3. My Role and Responsibilities:**\n{project.get('role_and_responsibilities', 'N/A')}\n\n"
                f"**4. Technical Solution & Methodology:**\n{methodology_text}\n\n"
                f"**5. Key Achievements:**\n- {achievements_text}\n\n"
                f"**6. Technologies Used:**\n{project.get('technologies_used', 'N/A')}\n"
                f"---------------------------------"
            )
            docs.append(Document(page_content=proj_detail_text, metadata={"source": f"detail_project_{project.get('project_name', 'unknown')}"}))

    # Chia nhỏ các document thành các chunk
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
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
        return vector_store.as_retriever(search_kwargs={"k": 10}) 
    return None
