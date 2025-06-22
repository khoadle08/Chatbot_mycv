# tools.py
import json
from langchain_core.tools import tool
from typing import Optional

def load_cv_data(file_path: str = "mycv.json") -> dict:
    """
    Tải và phân tích cú pháp dữ liệu CV từ một file JSON.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {file_path}.")
        return {"error": f"File {file_path} không tồn tại."}
    except json.JSONDecodeError:
        print(f"Lỗi: Không thể giải mã JSON từ file {file_path}.")
        return {"error": f"Nội dung file {file_path} không phải là JSON hợp lệ."}

# Tải dữ liệu CV một lần khi module được import
CV_DATA = load_cv_data()

@tool
def get_personal_info() -> str:
    """Sử dụng công cụ này để lấy thông tin cá nhân cơ bản như tên, email, số điện thoại, LinkedIn."""
    print("🛠️ Đang chạy công cụ: get_personal_info")
    if "error" in CV_DATA:
        return json.dumps(CV_DATA)
    info = CV_DATA.get("personal_info", {})
    return json.dumps(info, indent=2, ensure_ascii=False)

@tool
def get_introduction() -> str:
    """Sử dụng công cụ này để lấy đoạn giới thiệu tổng quan về bản thân."""
    print("🛠️ Đang chạy công cụ: get_introduction")
    if "error" in CV_DATA:
        return json.dumps(CV_DATA)
    return CV_DATA.get("introduction", "Không có đoạn giới thiệu.")

@tool
def get_work_experience() -> str:
    """Sử dụng công cụ này để lấy toàn bộ thông tin về kinh nghiệm làm việc."""
    print("🛠️ Đang chạy công cụ: get_work_experience")
    if "error" in CV_DATA:
        return json.dumps(CV_DATA)
    experience = CV_DATA.get("experience", [])
    return json.dumps(experience, indent=2, ensure_ascii=False)

@tool
def get_technical_skills() -> str:
    """Sử dụng công cụ này để lấy danh sách các kỹ năng kỹ thuật, bao gồm ngôn ngữ lập trình, cloud, machine learning, v.v."""
    print("🛠️ Đang chạy công cụ: get_technical_skills")
    if "error" in CV_DATA:
        return json.dumps(CV_DATA)
    skills = CV_DATA.get("technical_skills", {})
    return json.dumps(skills, indent=2, ensure_ascii=False)

@tool
def get_projects_summary() -> str:
    """Sử dụng công cụ này để lấy danh sách tóm tắt tất cả các dự án đã thực hiện."""
    print("🛠️ Đang chạy công cụ: get_projects_summary")
    if "error" in CV_DATA:
        return json.dumps(CV_DATA)
    projects = CV_DATA.get("projects", [])
    return json.dumps(projects, indent=2, ensure_ascii=False)

@tool
def get_detail_project_info(project_name: str) -> str:
    """Sử dụng công cụ này để lấy thông tin CHI TIẾT về một dự án cụ thể. Cần cung cấp tên dự án gần đúng."""
    print(f"🛠️ Đang chạy công cụ: get_detail_project_info với tên dự án: {project_name}")
    if "error" in CV_DATA:
        return json.dumps(CV_DATA)
    
    detailed_projects = CV_DATA.get("detail_project", [])
    if not detailed_projects:
        return "Không có thông tin chi tiết dự án nào trong CV."

    # Tìm kiếm dự án gần đúng (không phân biệt chữ hoa/thường)
    for project in detailed_projects:
        if project_name.lower() in project.get("project_name", "").lower():
            return json.dumps(project, indent=2, ensure_ascii=False)
            
    return f"Không tìm thấy dự án nào có tên giống '{project_name}'. Hãy thử hỏi về các dự án: SABA InsightAI, Fraud Detection, Formation Tactics, Customer Information Storage, ETL Migration, Market Analysis."

# Tập hợp tất cả các công cụ vào một danh sách để agent sử dụng
all_tools = [
    get_personal_info,
    get_introduction,
    get_work_experience,
    get_technical_skills,
    get_projects_summary,
    get_detail_project_info
]
