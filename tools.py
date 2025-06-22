# tools.py

import firebase_admin
from langchain.tools import tool
import json

# Connection to Firestore is handled in the main app file
# to support both local (credentials.json) and deployed (st.secrets) environments.

@tool
def get_personal_info() -> str:
    """
    Use this tool to get basic personal information such as name, email, phone number, and LinkedIn profile.
    """
    print("🛠️ Executing tool: get_personal_info")
    try:
        db = firestore.client()
        doc_ref = db.collection('cvs').document('khoa-dang-le')
        doc = doc_ref.get()
        if doc.exists:
            info = doc.to_dict().get("personal_info", {})
            return json.dumps(info, indent=2, ensure_ascii=False)
        return "Personal information not found."
    except Exception as e:
        return f"Error fetching personal info: {e}"

@tool
def get_introduction() -> str:
    """
    Use this tool to get the overall professional summary or introduction from the CV.
    """
    print("🛠️ Executing tool: get_introduction")
    try:
        db = firestore.client()
        doc_ref = db.collection('cvs').document('khoa-dang-le')
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict().get("introduction", "No introduction found.")
        return "CV document not found."
    except Exception as e:
        return f"Error fetching introduction: {e}"

@tool
def get_work_experience() -> str:
    """
    Use this tool to retrieve all information about professional work experience.
    """
    print("🛠️ Executing tool: get_work_experience")
    try:
        db = firestore.client()
        doc_ref = db.collection('cvs').document('khoa-dang-le')
        doc = doc_ref.get()
        if doc.exists:
            experience = doc.to_dict().get("experience", [])
            return json.dumps(experience, indent=2, ensure_ascii=False)
        return "No work experience information found."
    except Exception as e:
        return f"Error fetching work experience: {e}"

@tool
def get_technical_skills() -> str:
    """
    Use this tool to get a list of technical skills, including programming languages, cloud platforms, machine learning, etc.
    """
    print("🛠️ Executing tool: get_technical_skills")
    try:
        db = firestore.client()
        doc_ref = db.collection('cvs').document('khoa-dang-le')
        doc = doc_ref.get()
        if doc.exists:
            # Based on mycv.json, this field is named 'technical_skills_and_tools'
            skills = doc.to_dict().get("technical_skills_and_tools", {})
            return json.dumps(skills, indent=2, ensure_ascii=False)
        return "No technical skills information found."
    except Exception as e:
        return f"Error fetching technical skills: {e}"

@tool
def get_projects() -> str:
    """
    Use this tool to get a detailed list of all completed and ongoing projects.
    """
    print("🛠️ Executing tool: get_projects")
    try:
        db = firestore.client()
        doc_ref = db.collection('cvs').document('khoa-dang-le')
        doc = doc_ref.get()
        if doc.exists:
            # Based on mycv.json, this field is named 'projects'
            projects = doc.to_dict().get("projects", [])
            return json.dumps(projects, indent=2, ensure_ascii=False)
        return "No project information found."
    except Exception as e:
        return f"Error fetching projects: {e}"


# A list that contains all the available tools for the agent
all_tools = [
    get_personal_info,
    get_introduction,
    get_work_experience,
    get_technical_skills,
    get_projects
]
