"""
StudyMate AI - Personalized Smart Study Generator
A multi-agent AI-powered web application for college students.
Uses Flask, Grok API, RAG (FAISS + Sentence Transformers), PyMuPDF, and SQLite.
"""

import os
import json
import re
import sqlite3
import traceback
import uuid
from datetime import datetime, date, timedelta
from pathlib import Path
import tempfile

from flask import (
    Flask, request, jsonify, session, render_template_string, redirect, url_for
)
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# --- Optional heavy imports with graceful fallbacks ---
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from openai import OpenAI  # Grok API is OpenAI-compatible
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False

import requests  # fallback HTTP client

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "studymate-secret-key-change-in-production")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
app.config["DATABASE"] = os.path.join(os.path.dirname(__file__), "studymate.db")

ALLOWED_EXTENSIONS = {"pdf", "txt"}
GROK_API_KEY = os.getenv("GROK_API_KEY", "")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-3-mini")
GROK_BASE_URL = os.getenv("GROK_BASE_URL", "https://api.x.ai/v1")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# Ensure upload directory exists
Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# In-memory vector store (FAISS + Sentence Transformers)
# ─────────────────────────────────────────────
_embedding_model = None
_faiss_index = None
_doc_chunks = []  # list of {"text": ..., "doc_id": ..., "source": ...}

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _embedding_model = None
    return _embedding_model

def add_to_vector_store(chunks, doc_id, source):
    """Add document chunks to the FAISS in-memory vector store."""
    global _faiss_index, _doc_chunks
    model = get_embedding_model()
    if model is None or not FAISS_AVAILABLE:
        # Fallback: just store chunks for keyword search
        for chunk in chunks:
            _doc_chunks.append({"text": chunk, "doc_id": doc_id, "source": source})
        return

    texts = [c for c in chunks if c.strip()]
    if not texts:
        return
    embeddings = model.encode(texts, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")

    if _faiss_index is None:
        dim = embeddings.shape[1]
        _faiss_index = faiss.IndexFlatL2(dim)

    _faiss_index.add(embeddings)
    for chunk in texts:
        _doc_chunks.append({"text": chunk, "doc_id": doc_id, "source": source})

def retrieve_relevant_context(query, top_k=4):
    """Retrieve top-k relevant document chunks for a query."""
    global _faiss_index, _doc_chunks
    if not _doc_chunks:
        return []

    model = get_embedding_model()
    if model is None or not FAISS_AVAILABLE or _faiss_index is None:
        # Keyword fallback
        q_lower = query.lower()
        scored = []
        for chunk in _doc_chunks:
            score = sum(1 for w in q_lower.split() if w in chunk["text"].lower())
            scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c["text"] for _, c in scored[:top_k] if c["text"].strip()]

    query_emb = model.encode([query], show_progress_bar=False)
    query_emb = np.array(query_emb, dtype="float32")
    distances, indices = _faiss_index.search(query_emb, min(top_k, len(_doc_chunks)))
    results = []
    for idx in indices[0]:
        if 0 <= idx < len(_doc_chunks):
            results.append(_doc_chunks[idx]["text"])
    return results

# ─────────────────────────────────────────────
# Database helpers
# ─────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(app.config["DATABASE"])
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create all required tables."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS student_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            name TEXT,
            course TEXT,
            branch TEXT,
            university TEXT,
            semester INTEGER,
            subjects TEXT,
            exam_date TEXT,
            study_hours REAL DEFAULT 4.0,
            prep_level TEXT DEFAULT 'beginner',
            difficult_subjects TEXT,
            target_sgpa REAL DEFAULT 8.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS study_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            subject TEXT NOT NULL,
            topic TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            priority INTEGER DEFAULT 2,
            scheduled_date TEXT,
            completed_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS quiz_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            subject TEXT,
            topic TEXT,
            difficulty TEXT,
            score INTEGER,
            total INTEGER,
            percentage REAL,
            weak_topics TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS uploaded_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            original_name TEXT,
            doc_type TEXT,
            text_length INTEGER DEFAULT 0,
            processed INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS activity_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            activity_type TEXT,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()

def get_session_id():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]

def log_activity(session_id, activity_type, description):
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO activity_log (session_id, activity_type, description) VALUES (?,?,?)",
            (session_id, activity_type, description)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

def calculate_student_progress(session_id):
    """Calculate overall and subject-wise progress for a student."""
    conn = get_db()
    tasks = conn.execute(
        "SELECT subject, topic, status FROM study_tasks WHERE session_id=?",
        (session_id,)
    ).fetchall()
    quizzes = conn.execute(
        "SELECT subject, topic, percentage FROM quiz_results WHERE session_id=? ORDER BY created_at DESC",
        (session_id,)
    ).fetchall()
    conn.close()

    total = len(tasks)
    completed = sum(1 for t in tasks if t["status"] == "completed")
    overall_pct = round((completed / total * 100) if total > 0 else 0, 1)

    subject_data = {}
    for t in tasks:
        subj = t["subject"]
        subject_data.setdefault(subj, {"total": 0, "completed": 0})
        subject_data[subj]["total"] += 1
        if t["status"] == "completed":
            subject_data[subj]["completed"] += 1

    quiz_scores = {}
    for q in quizzes:
        subj = q["subject"] or "General"
        quiz_scores.setdefault(subj, []).append(q["percentage"])

    avg_quiz = {}
    for subj, scores in quiz_scores.items():
        avg_quiz[subj] = round(sum(scores) / len(scores), 1)

    return {
        "overall_pct": overall_pct,
        "total_tasks": total,
        "completed_tasks": completed,
        "pending_tasks": total - completed,
        "subject_data": subject_data,
        "avg_quiz_scores": avg_quiz,
    }

# ─────────────────────────────────────────────
# Grok API
# ─────────────────────────────────────────────
def generate_response(prompt, system_prompt=None, temperature=None, max_tokens=None):
    """
    Call Grok API and return the text response.
    Supports both the openai SDK (preferred) and raw HTTP fallback.
    """
    if not GROK_API_KEY:
        return "⚠️ GROK_API_KEY is not configured. Please set it in your .env file."

    temp = temperature if temperature is not None else TEMPERATURE
    tokens = max_tokens if max_tokens is not None else MAX_TOKENS
    sys_msg = system_prompt or (
        "You are StudyMate AI, an intelligent academic assistant for college students. "
        "Provide clear, accurate, and student-friendly explanations. "
        "Format responses with proper headings, bullet points, and examples."
    )

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": prompt},
    ]

    # Try OpenAI-compatible SDK first
    if OPENAI_SDK_AVAILABLE:
        try:
            client = OpenAI(api_key=GROK_API_KEY, base_url=GROK_BASE_URL)
            response = client.chat.completions.create(
                model=GROK_MODEL,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            err = str(e)
            if "401" in err or "authentication" in err.lower():
                return "⚠️ Authentication failed. Please check your GROK_API_KEY."
            if "429" in err:
                return "⚠️ Rate limit reached. Please try again in a moment."
            if "timeout" in err.lower():
                return "⚠️ Request timed out. Please try again."
            # Fall through to HTTP fallback

    # HTTP fallback
    try:
        headers = {
            "Authorization": f"Bearer {GROK_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": GROK_MODEL,
            "messages": messages,
            "temperature": temp,
            "max_tokens": tokens,
        }
        resp = requests.post(
            f"{GROK_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        return "⚠️ Request timed out. The AI server took too long to respond."
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response else 0
        if code == 401:
            return "⚠️ Invalid API key. Please check your GROK_API_KEY."
        if code == 429:
            return "⚠️ Rate limit exceeded. Please wait and try again."
        return f"⚠️ API error (HTTP {code}). Please try again later."
    except Exception as e:
        return f"⚠️ Unexpected error: {str(e)}"

# ─────────────────────────────────────────────
# Agent 1: Study Knowledge Agent
# ─────────────────────────────────────────────
def study_knowledge_agent(question, subject="", topic="", mark_type="", use_docs=True):
    """Answer academic questions, optionally using uploaded document context (RAG)."""
    doc_context = ""
    if use_docs:
        chunks = retrieve_relevant_context(question, top_k=4)
        if chunks:
            doc_context = "\n\n".join(chunks)

    mark_instruction = ""
    if mark_type:
        mark_map = {
            "2mark": "Write a concise 2-mark exam answer (2-3 sentences, key definition and one point).",
            "5mark": "Write a detailed 5-mark exam answer with definition, explanation, and 2-3 key points.",
            "8mark": "Write a comprehensive 8-mark exam answer with introduction, detailed explanation, examples, and conclusion.",
            "10mark": "Write an elaborate 10-mark exam answer with introduction, multiple sections, diagrams descriptions, examples, and conclusion.",
        }
        mark_instruction = mark_map.get(mark_type, "")

    subject_ctx = f"Subject: {subject}" if subject else ""
    topic_ctx = f"Topic: {topic}" if topic else ""

    if doc_context:
        prompt = f"""
{subject_ctx}
{topic_ctx}
{mark_instruction}

Student's Question: {question}

Relevant information from uploaded study materials:
---
{doc_context[:2000]}
---

Please answer the question using the above document context where relevant.
Clearly indicate if information comes from the uploaded documents vs general knowledge.
Provide a clear, structured, student-friendly answer.
"""
        system = (
            "You are StudyMate AI, an expert academic tutor. "
            "Use the provided document context to give accurate, exam-ready answers. "
            "Label document-sourced info as '[From your notes]' and general knowledge as '[General Knowledge]'."
        )
    else:
        prompt = f"""
{subject_ctx}
{topic_ctx}
{mark_instruction}

Student's Question: {question}

Please provide a clear, structured, student-friendly answer suitable for college-level study.
Include definitions, explanations, examples, and key points as appropriate.
"""
        system = (
            "You are StudyMate AI, an expert academic tutor for college students. "
            "Provide accurate, clear, well-structured answers that help students understand and remember concepts."
        )

    return generate_response(prompt, system_prompt=system)

# ─────────────────────────────────────────────
# Agent 2: Study Planner Agent
# ─────────────────────────────────────────────
def study_planner_agent(profile, weak_topics=None):
    """Generate a personalized study plan based on student profile."""
    subjects = profile.get("subjects", "")
    exam_date = profile.get("exam_date", "")
    hours = profile.get("study_hours", 4)
    prep_level = profile.get("prep_level", "beginner")
    difficult = profile.get("difficult_subjects", "")
    target = profile.get("target_sgpa", 8.0)
    semester = profile.get("semester", 1)

    days_left = ""
    if exam_date:
        try:
            ed = datetime.strptime(exam_date, "%Y-%m-%d").date()
            delta = (ed - date.today()).days
            days_left = f"{delta} days until examination" if delta >= 0 else "Examination has passed"
        except Exception:
            days_left = exam_date

    weak_info = f"\nWeak topics from quiz performance: {', '.join(weak_topics)}" if weak_topics else ""

    prompt = f"""
Create a personalized study plan for a college student with these details:

- Semester: {semester}
- Subjects: {subjects}
- Examination date: {exam_date} ({days_left})
- Available study hours per day: {hours}
- Current preparation level: {prep_level}
- Difficult subjects/topics: {difficult}
- Target SGPA: {target}
{weak_info}

Generate:
1. **Daily Study Timetable** - Hour-by-hour schedule for one full study day
2. **Weekly Study Plan** - Day-wise subject allocation for the next 7 days
3. **Subject Priority List** - Which subjects need most attention and why
4. **Topic-wise Breakdown** - Key topics to cover for each subject
5. **Revision Schedule** - When to revise completed topics
6. **Exam Week Strategy** - How to spend the last 3-5 days before exams
7. **Study Tips** - 3-5 personalized tips based on preparation level

Format with clear headings, bullet points, and tables where useful.
Be realistic and considerate of student well-being (breaks, sleep, meals).
"""
    return generate_response(prompt, max_tokens=2500)

def generate_study_plan(session_id, profile):
    """Generate and save study tasks to DB after creating a plan."""
    plan_text = study_planner_agent(profile)

    # Parse and create basic tasks if subjects provided
    subjects = [s.strip() for s in profile.get("subjects", "").split(",") if s.strip()]
    exam_date_str = profile.get("exam_date", "")
    try:
        exam_date_obj = datetime.strptime(exam_date_str, "%Y-%m-%d").date()
    except Exception:
        exam_date_obj = date.today() + timedelta(days=30)

    days_left = max((exam_date_obj - date.today()).days, 1)
    conn = get_db()
    # Remove old tasks for this session
    conn.execute("DELETE FROM study_tasks WHERE session_id=?", (session_id,))

    for i, subj in enumerate(subjects):
        for day_offset in range(min(days_left, 30)):
            sched_date = (date.today() + timedelta(days=day_offset)).isoformat()
            priority = 1 if subj in profile.get("difficult_subjects", "") else 2
            conn.execute(
                "INSERT INTO study_tasks (session_id, subject, topic, status, priority, scheduled_date) VALUES (?,?,?,?,?,?)",
                (session_id, subj, f"Day {day_offset+1} – {subj} study session", "pending", priority, sched_date)
            )
    conn.commit()
    conn.close()
    return plan_text

# ─────────────────────────────────────────────
# Agent 3: Quiz Agent
# ─────────────────────────────────────────────
def quiz_agent(subject, topic, difficulty, num_questions, context_text=""):
    """Generate MCQ quiz questions using Grok API."""
    doc_ctx = ""
    if not context_text:
        chunks = retrieve_relevant_context(f"{subject} {topic}", top_k=3)
        if chunks:
            doc_ctx = "\n".join(chunks)[:1500]
    else:
        doc_ctx = context_text[:1500]

    context_section = f"\nUse this study material as context:\n---\n{doc_ctx}\n---\n" if doc_ctx else ""

    prompt = f"""
Generate exactly {num_questions} multiple-choice questions for:
- Subject: {subject}
- Topic: {topic}
- Difficulty: {difficulty}
{context_section}

Return ONLY a valid JSON array. Each object must have exactly these keys:
{{
  "question": "question text",
  "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
  "correct": "A",
  "explanation": "brief explanation of the correct answer"
}}

Rules:
- correct must be "A", "B", "C", or "D"
- All 4 options must be plausible
- Explanation must be 1-2 sentences
- No markdown, no extra text — pure JSON array only
"""
    raw = generate_response(prompt, temperature=0.5, max_tokens=2000)

    # Extract JSON from response
    try:
        # Find JSON array in response
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            questions = json.loads(match.group())
            return {"success": True, "questions": questions[:num_questions]}
    except Exception:
        pass

    # Fallback: return error
    return {
        "success": False,
        "error": "Could not parse quiz questions. Please try again.",
        "raw": raw[:500]
    }

# ─────────────────────────────────────────────
# Agent 4: Question Paper Analyzer
# ─────────────────────────────────────────────
def question_paper_analyzer_agent(extracted_text, subject=""):
    """Analyze previous-year question papers for patterns and important topics."""
    prompt = f"""
Analyze this previous-year question paper text and provide a comprehensive analysis.
Subject: {subject if subject else "Unknown"}

Question Paper Text:
---
{extracted_text[:3000]}
---

Provide the following analysis in structured JSON format:
{{
  "topics": [
    {{"name": "topic name", "frequency": number, "importance": "high/medium/low", "question_types": ["short", "long", "numerical"]}}
  ],
  "repeated_questions": ["question 1", "question 2"],
  "important_topics": ["topic1", "topic2", "topic3"],
  "topic_distribution": {{"Topic A": percentage, "Topic B": percentage}},
  "recommendations": ["recommendation 1", "recommendation 2"],
  "exam_pattern": "brief description of exam pattern",
  "total_questions_found": number
}}

Important: Base analysis only on the provided text. Do not guarantee future exam questions.
Return ONLY valid JSON, no extra text.
"""
    raw = generate_response(prompt, temperature=0.3, max_tokens=1500)
    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            result = json.loads(match.group())
            return {"success": True, "analysis": result}
    except Exception:
        pass

    # Fallback analysis
    return {
        "success": True,
        "analysis": {
            "topics": [],
            "repeated_questions": [],
            "important_topics": [],
            "topic_distribution": {},
            "recommendations": ["Please review the extracted text manually for topic patterns."],
            "exam_pattern": "Could not automatically detect pattern.",
            "total_questions_found": 0,
        },
        "raw_analysis": raw[:1000]
    }

def generate_practice_questions(subject, important_topics):
    """Generate practice questions based on important topics from PYQ analysis."""
    topics_str = ", ".join(important_topics[:5]) if important_topics else subject
    prompt = f"""
Generate 5 practice exam questions for the following important topics:
Subject: {subject}
Topics: {topics_str}

Include a mix of:
- Short answer questions (2 marks)
- Medium questions (5 marks)  
- Long answer questions (10 marks)

Format as a numbered list with mark weightage.
Note: These are practice questions for study purposes only.
"""
    return generate_response(prompt, max_tokens=800)

# ─────────────────────────────────────────────
# Agent Orchestrator
# ─────────────────────────────────────────────
def agent_orchestrator(user_input, context=None):
    """
    Identify user intent and route to the appropriate agent.
    Returns: {"agent": agent_name, "response": response_text}
    """
    context = context or {}
    lower = user_input.lower()

    # Route based on keywords
    if any(kw in lower for kw in ["timetable", "schedule", "study plan", "planner", "plan my"]):
        return {"agent": "planner", "response": "Routing to Study Planner Agent..."}

    if any(kw in lower for kw in ["quiz", "mcq", "test me", "question", "assess"]):
        return {"agent": "quiz", "response": "Routing to Quiz Agent..."}

    if any(kw in lower for kw in ["previous year", "pyq", "past paper", "old paper", "analyze paper"]):
        return {"agent": "analyzer", "response": "Routing to Question Paper Analyzer..."}

    if any(kw in lower for kw in ["explain", "define", "what is", "how does", "describe", "summarize", "notes"]):
        response = study_knowledge_agent(user_input, use_docs=True)
        return {"agent": "knowledge", "response": response}

    # Default: knowledge agent
    response = study_knowledge_agent(user_input, use_docs=True)
    return {"agent": "knowledge", "response": response}

# ─────────────────────────────────────────────
# PDF / Document Processing
# ─────────────────────────────────────────────
def extract_pdf_text(filepath):
    """Extract text from a PDF file using PyMuPDF."""
    if not PYMUPDF_AVAILABLE:
        return None, "PyMuPDF (fitz) is not installed. Install it with: pip install PyMuPDF"

    try:
        doc = fitz.open(filepath)
        text = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            text += page_text
        doc.close()

        if not text.strip():
            return None, (
                "This PDF appears to be a scanned image. "
                "Text extraction requires OCR software (e.g., Tesseract). "
                "Please use a text-based PDF or paste the content manually."
            )
        return text, None
    except Exception as e:
        return None, f"Error reading PDF: {str(e)}"

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks for vector storage."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def process_uploaded_document(filepath, filename, doc_id, session_id):
    """Extract text, chunk it, and store in vector database."""
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext == "pdf":
        text, error = extract_pdf_text(filepath)
        if error:
            return False, error
    elif ext == "txt":
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            return False, f"Error reading text file: {str(e)}"
    else:
        return False, "Unsupported file type. Please upload PDF or TXT files."

    if not text or len(text.strip()) < 50:
        return False, "Document appears to be empty or contains very little text."

    chunks = chunk_text(text)
    add_to_vector_store(chunks, doc_id, filename)

    # Update DB
    conn = get_db()
    conn.execute(
        "UPDATE uploaded_documents SET text_length=?, processed=1 WHERE id=?",
        (len(text), doc_id)
    )
    conn.commit()
    conn.close()

    return True, text[:5000]  # Return first 5000 chars for preview

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ─────────────────────────────────────────────
# HTML Templates
# ─────────────────────────────────────────────
BASE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>StudyMate AI – {% block title %}Smart Study Generator{% endblock %}</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --primary: #4361ee;
    --secondary: #7c5cd8;
    --accent: #3bc9db;
    --success: #40c057;
    --sidebar-width: 260px;
    --sidebar-bg: #1a1d2e;
    --sidebar-text: #c8cde4;
  }
  body { background: #f0f2ff; font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; }
  .sidebar {
    width: var(--sidebar-width); background: var(--sidebar-bg);
    height: 100vh; position: fixed; top: 0; left: 0; z-index: 1000;
    overflow-y: auto; transition: transform 0.3s ease;
  }
  .sidebar .logo { padding: 20px; border-bottom: 1px solid rgba(255,255,255,0.08); }
  .sidebar .logo h5 { color: #fff; margin: 0; font-weight: 700; }
  .sidebar .logo small { color: var(--accent); font-size: 0.75rem; }
  .sidebar .nav-link {
    color: var(--sidebar-text); padding: 12px 20px; display: flex; align-items: center;
    gap: 10px; text-decoration: none; transition: all 0.2s; border-left: 3px solid transparent;
  }
  .sidebar .nav-link:hover, .sidebar .nav-link.active {
    background: rgba(255,255,255,0.08); color: #fff; border-left-color: var(--primary);
  }
  .sidebar .nav-link i { width: 20px; text-align: center; font-size: 1.1rem; }
  .main-content { margin-left: var(--sidebar-width); min-height: 100vh; }
  .topbar {
    background: #fff; border-bottom: 1px solid #e5e7eb;
    padding: 12px 24px; display: flex; align-items: center; justify-content: space-between;
    position: sticky; top: 0; z-index: 100;
  }
  .topbar h4 { margin: 0; font-weight: 600; color: #1a1d2e; }
  .content-area { padding: 24px; }
  .card { border: none; box-shadow: 0 2px 8px rgba(0,0,0,0.06); border-radius: 12px; }
  .card-header { border-radius: 12px 12px 0 0 !important; }
  .stat-card { background: #fff; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
  .stat-icon { width: 48px; height: 48px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.4rem; }
  .btn-primary { background: var(--primary); border-color: var(--primary); }
  .btn-primary:hover { background: #3451d1; border-color: #3451d1; }
  .chat-container { height: 400px; overflow-y: auto; background: #f8f9ff; border-radius: 8px; padding: 16px; }
  .chat-msg { margin-bottom: 12px; }
  .chat-msg.user .bubble { background: var(--primary); color: #fff; margin-left: auto; max-width: 80%; }
  .chat-msg.ai .bubble { background: #fff; border: 1px solid #e5e7eb; max-width: 85%; }
  .bubble { padding: 10px 14px; border-radius: 12px; line-height: 1.5; display: inline-block; }
  .bubble pre { white-space: pre-wrap; margin: 0; font-family: inherit; }
  .loading-spinner { display: none; }
  .loading-spinner.show { display: inline-block; }
  .progress { height: 8px; border-radius: 4px; }
  .badge-subject { background: #e8edff; color: var(--primary); padding: 4px 10px; border-radius: 20px; font-size: 0.8rem; }
  .quiz-option { cursor: pointer; padding: 10px 16px; border: 2px solid #e5e7eb; border-radius: 8px; margin-bottom: 8px; transition: all 0.2s; }
  .quiz-option:hover { border-color: var(--primary); background: #f0f4ff; }
  .quiz-option.selected { border-color: var(--primary); background: #e8edff; }
  .quiz-option.correct { border-color: var(--success); background: #f0fff4; }
  .quiz-option.wrong { border-color: #fa5252; background: #fff5f5; }
  @media (max-width: 768px) {
    .sidebar { transform: translateX(-100%); }
    .sidebar.open { transform: translateX(0); }
    .main-content { margin-left: 0; }
  }
</style>
</head>
<body>
<!-- Sidebar -->
<nav class="sidebar" id="sidebar">
  <div class="logo">
    <h5><i class="bi bi-mortarboard-fill text-warning me-2"></i>StudyMate AI</h5>
    <small>Personalized Study Companion</small>
  </div>
  <div class="mt-2">
    <a href="/" class="nav-link {% if active_page=='home' %}active{% endif %}"><i class="bi bi-house-fill"></i> Dashboard</a>
    <a href="/study-assistant" class="nav-link {% if active_page=='assistant' %}active{% endif %}"><i class="bi bi-chat-dots-fill"></i> Study Assistant</a>
    <a href="/study-material" class="nav-link {% if active_page=='material' %}active{% endif %}"><i class="bi bi-file-earmark-text-fill"></i> Study Material</a>
    <a href="/study-planner" class="nav-link {% if active_page=='planner' %}active{% endif %}"><i class="bi bi-calendar3"></i> Study Planner</a>
    <a href="/question-analyzer" class="nav-link {% if active_page=='analyzer' %}active{% endif %}"><i class="bi bi-file-earmark-bar-graph-fill"></i> Paper Analyzer</a>
    <a href="/quiz" class="nav-link {% if active_page=='quiz' %}active{% endif %}"><i class="bi bi-patch-question-fill"></i> AI Quiz</a>
    <a href="/progress" class="nav-link {% if active_page=='progress' %}active{% endif %}"><i class="bi bi-graph-up-arrow"></i> Progress</a>
    <a href="/about" class="nav-link {% if active_page=='about' %}active{% endif %}"><i class="bi bi-info-circle-fill"></i> About</a>
  </div>
  <div class="mt-auto p-3" style="position:absolute;bottom:0;width:100%">
    <small class="text-secondary">Educational Use Only</small>
  </div>
</nav>

<!-- Main Content -->
<div class="main-content">
  <div class="topbar">
    <div class="d-flex align-items-center gap-3">
      <button class="btn btn-sm btn-outline-secondary d-md-none" onclick="document.getElementById('sidebar').classList.toggle('open')">
        <i class="bi bi-list"></i>
      </button>
      <h4>{% block page_title %}StudyMate AI{% endblock %}</h4>
    </div>
    <div class="d-flex align-items-center gap-2">
      <span class="badge bg-primary"><i class="bi bi-robot me-1"></i>Grok AI</span>
      <span class="badge bg-secondary">v1.0</span>
    </div>
  </div>
  <div class="content-area">
    {% with messages = get_flashed_messages(with_categories=true) %}
      {% if messages %}
        {% for category, message in messages %}
          <div class="alert alert-{{ 'success' if category == 'success' else 'danger' }} alert-dismissible fade show" role="alert">
            {{ message }}<button type="button" class="btn-close" data-bs-dismiss="alert"></button>
          </div>
        {% endfor %}
      {% endif %}
    {% endwith %}
    {% block content %}{% endblock %}
  </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
{% block scripts %}{% endblock %}
</body>
</html>
"""

# ──────────────────────────────────────────────────────────
# Page Templates
# ──────────────────────────────────────────────────────────
HOME_TEMPLATE = BASE_TEMPLATE.replace("{% block content %}{% endblock %}", """{% block content %}
<div class="row g-4 mb-4">
  <!-- Welcome Card -->
  <div class="col-12">
    <div class="card" style="background: linear-gradient(135deg,#4361ee,#7c5cd8); color:#fff;">
      <div class="card-body p-4">
        <div class="row align-items-center">
          <div class="col">
            <h3 class="mb-1">Welcome back, {{ profile.name or 'Student' }}! 👋</h3>
            <p class="mb-2 opacity-75">{{ profile.course or 'Set up your profile to get started' }} {{ ('– Sem ' + profile.semester|string) if profile.semester else '' }}</p>
            {% if profile.exam_date %}
            <span class="badge bg-warning text-dark"><i class="bi bi-alarm me-1"></i>Exam: {{ profile.exam_date }}</span>
            {% endif %}
          </div>
          <div class="col-auto d-none d-md-block">
            <i class="bi bi-mortarboard" style="font-size:4rem; opacity:0.3"></i>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Stat Cards -->
  <div class="col-6 col-md-3">
    <div class="stat-card text-center">
      <div class="stat-icon bg-primary bg-opacity-10 text-primary mx-auto mb-2"><i class="bi bi-book"></i></div>
      <h3 class="mb-0 text-primary">{{ progress.total_tasks }}</h3>
      <small class="text-muted">Total Tasks</small>
    </div>
  </div>
  <div class="col-6 col-md-3">
    <div class="stat-card text-center">
      <div class="stat-icon bg-success bg-opacity-10 text-success mx-auto mb-2"><i class="bi bi-check-circle"></i></div>
      <h3 class="mb-0 text-success">{{ progress.completed_tasks }}</h3>
      <small class="text-muted">Completed</small>
    </div>
  </div>
  <div class="col-6 col-md-3">
    <div class="stat-card text-center">
      <div class="stat-icon bg-warning bg-opacity-10 text-warning mx-auto mb-2"><i class="bi bi-clock"></i></div>
      <h3 class="mb-0 text-warning">{{ progress.pending_tasks }}</h3>
      <small class="text-muted">Pending</small>
    </div>
  </div>
  <div class="col-6 col-md-3">
    <div class="stat-card text-center">
      <div class="stat-icon bg-info bg-opacity-10 text-info mx-auto mb-2"><i class="bi bi-percent"></i></div>
      <h3 class="mb-0 text-info">{{ progress.overall_pct }}%</h3>
      <small class="text-muted">Progress</small>
    </div>
  </div>

  <!-- Overall Progress -->
  <div class="col-md-8">
    <div class="card h-100">
      <div class="card-header bg-white"><h6 class="mb-0"><i class="bi bi-graph-up text-primary me-2"></i>Overall Study Progress</h6></div>
      <div class="card-body">
        <div class="mb-3">
          <div class="d-flex justify-content-between mb-1">
            <span>Overall Completion</span><strong>{{ progress.overall_pct }}%</strong>
          </div>
          <div class="progress"><div class="progress-bar bg-primary" style="width:{{ progress.overall_pct }}%"></div></div>
        </div>
        {% for subj, data in progress.subject_data.items() %}
        <div class="mb-2">
          <div class="d-flex justify-content-between mb-1">
            <small>{{ subj }}</small>
            <small>{{ ((data.completed/data.total*100)|round(0)|int) if data.total > 0 else 0 }}%</small>
          </div>
          <div class="progress" style="height:6px">
            <div class="progress-bar" style="width:{{ ((data.completed/data.total*100)|round(0)|int) if data.total > 0 else 0 }}%; background: var(--secondary)"></div>
          </div>
        </div>
        {% endfor %}
        {% if not progress.subject_data %}
        <div class="text-center text-muted py-3">
          <i class="bi bi-bar-chart" style="font-size:2rem"></i>
          <p class="mt-2">Set up your study plan to see progress</p>
          <a href="/study-planner" class="btn btn-sm btn-primary">Create Study Plan</a>
        </div>
        {% endif %}
      </div>
    </div>
  </div>

  <!-- Quick Actions -->
  <div class="col-md-4">
    <div class="card h-100">
      <div class="card-header bg-white"><h6 class="mb-0"><i class="bi bi-lightning text-warning me-2"></i>Quick Actions</h6></div>
      <div class="card-body d-flex flex-column gap-2">
        <a href="/study-assistant" class="btn btn-outline-primary text-start"><i class="bi bi-chat-dots me-2"></i>Ask a Question</a>
        <a href="/quiz" class="btn btn-outline-secondary text-start"><i class="bi bi-patch-question me-2"></i>Take a Quiz</a>
        <a href="/study-material" class="btn btn-outline-success text-start"><i class="bi bi-upload me-2"></i>Upload Notes</a>
        <a href="/study-planner" class="btn btn-outline-info text-start"><i class="bi bi-calendar-plus me-2"></i>Create Study Plan</a>
        <a href="/question-analyzer" class="btn btn-outline-warning text-start"><i class="bi bi-file-earmark-bar-graph me-2"></i>Analyze PYQs</a>
      </div>
    </div>
  </div>

  <!-- Recent Activity -->
  <div class="col-12">
    <div class="card">
      <div class="card-header bg-white"><h6 class="mb-0"><i class="bi bi-activity text-primary me-2"></i>Recent Activity</h6></div>
      <div class="card-body">
        {% if activities %}
        <div class="table-responsive">
          <table class="table table-hover">
            <thead><tr><th>Activity</th><th>Description</th><th>Time</th></tr></thead>
            <tbody>
              {% for a in activities %}
              <tr>
                <td><span class="badge badge-subject">{{ a.activity_type }}</span></td>
                <td>{{ a.description[:80] }}</td>
                <td><small class="text-muted">{{ a.created_at }}</small></td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
        {% else %}
        <div class="text-center text-muted py-4">
          <i class="bi bi-clock-history" style="font-size:2.5rem"></i>
          <p class="mt-2">No recent activity. Start studying!</p>
        </div>
        {% endif %}
      </div>
    </div>
  </div>
</div>
{% endblock %}""").replace("{% block title %}Smart Study Generator{% endblock %}", "{% block title %}Dashboard{% endblock %}").replace("{% block page_title %}StudyMate AI{% endblock %}", "{% block page_title %}Dashboard{% endblock %}")

# ──────────────────────────────────────────────────────────
# Flask Routes
# ──────────────────────────────────────────────────────────

@app.route("/")
def home():
    sid = get_session_id()
    conn = get_db()
    profile_row = conn.execute(
        "SELECT * FROM student_profiles WHERE session_id=?", (sid,)
    ).fetchone()
    activities = conn.execute(
        "SELECT * FROM activity_log WHERE session_id=? ORDER BY created_at DESC LIMIT 10", (sid,)
    ).fetchall()
    conn.close()

    profile = dict(profile_row) if profile_row else {}
    progress = calculate_student_progress(sid)

    return render_template_string(
        HOME_TEMPLATE,
        active_page="home",
        profile=profile,
        progress=progress,
        activities=activities,
    )

@app.route("/study-assistant")
def study_assistant_page():
    return render_template_string(STUDY_ASSISTANT_TEMPLATE, active_page="assistant")

@app.route("/study-material")
def study_material_page():
    sid = get_session_id()
    conn = get_db()
    docs = conn.execute(
        "SELECT * FROM uploaded_documents WHERE session_id=? ORDER BY created_at DESC", (sid,)
    ).fetchall()
    conn.close()
    return render_template_string(STUDY_MATERIAL_TEMPLATE, active_page="material", docs=docs)

@app.route("/study-planner")
def study_planner_page():
    sid = get_session_id()
    conn = get_db()
    profile_row = conn.execute(
        "SELECT * FROM student_profiles WHERE session_id=?", (sid,)
    ).fetchone()
    tasks = conn.execute(
        "SELECT * FROM study_tasks WHERE session_id=? ORDER BY scheduled_date, priority", (sid,)
    ).fetchall()
    conn.close()
    profile = dict(profile_row) if profile_row else {}
    return render_template_string(STUDY_PLANNER_TEMPLATE, active_page="planner", profile=profile, tasks=tasks)

@app.route("/question-analyzer")
def question_analyzer_page():
    return render_template_string(QUESTION_ANALYZER_TEMPLATE, active_page="analyzer")

@app.route("/quiz")
def quiz_page():
    return render_template_string(QUIZ_TEMPLATE, active_page="quiz")

@app.route("/progress")
def progress_page():
    sid = get_session_id()
    progress = calculate_student_progress(sid)
    conn = get_db()
    quizzes = conn.execute(
        "SELECT * FROM quiz_results WHERE session_id=? ORDER BY created_at DESC LIMIT 20", (sid,)
    ).fetchall()
    conn.close()
    return render_template_string(PROGRESS_TEMPLATE, active_page="progress", progress=progress, quizzes=quizzes)

@app.route("/about")
def about_page():
    return render_template_string(ABOUT_TEMPLATE, active_page="about")

# ─── API Endpoints ───

@app.route("/api/ask", methods=["POST"])
def api_ask():
    sid = get_session_id()
    data = request.get_json()
    question = (data.get("question") or "").strip()
    subject = data.get("subject", "")
    topic = data.get("topic", "")
    mark_type = data.get("mark_type", "")

    if not question:
        return jsonify({"error": "Question is required."}), 400

    response = study_knowledge_agent(question, subject=subject, topic=topic, mark_type=mark_type)
    log_activity(sid, "Study Assistant", f"Asked: {question[:60]}")
    return jsonify({"response": response})

@app.route("/api/orchestrate", methods=["POST"])
def api_orchestrate():
    sid = get_session_id()
    data = request.get_json()
    user_input = (data.get("input") or "").strip()
    if not user_input:
        return jsonify({"error": "Input is required."}), 400
    result = agent_orchestrator(user_input)
    log_activity(sid, "Orchestrator", f"Input: {user_input[:60]}")
    return jsonify(result)

@app.route("/api/save-profile", methods=["POST"])
def api_save_profile():
    sid = get_session_id()
    data = request.get_json()
    conn = get_db()
    existing = conn.execute(
        "SELECT id FROM student_profiles WHERE session_id=?", (sid,)
    ).fetchone()

    if existing:
        conn.execute("""
            UPDATE student_profiles SET name=?,course=?,branch=?,university=?,semester=?,
            subjects=?,exam_date=?,study_hours=?,prep_level=?,difficult_subjects=?,target_sgpa=?,updated_at=CURRENT_TIMESTAMP
            WHERE session_id=?
        """, (
            data.get("name",""), data.get("course",""), data.get("branch",""),
            data.get("university",""), data.get("semester",1), data.get("subjects",""),
            data.get("exam_date",""), data.get("study_hours",4), data.get("prep_level","beginner"),
            data.get("difficult_subjects",""), data.get("target_sgpa",8.0), sid
        ))
    else:
        conn.execute("""
            INSERT INTO student_profiles (session_id,name,course,branch,university,semester,subjects,
            exam_date,study_hours,prep_level,difficult_subjects,target_sgpa)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            sid, data.get("name",""), data.get("course",""), data.get("branch",""),
            data.get("university",""), data.get("semester",1), data.get("subjects",""),
            data.get("exam_date",""), data.get("study_hours",4), data.get("prep_level","beginner"),
            data.get("difficult_subjects",""), data.get("target_sgpa",8.0)
        ))
    conn.commit()
    conn.close()
    return jsonify({"success": True})

@app.route("/api/generate-plan", methods=["POST"])
def api_generate_plan():
    sid = get_session_id()
    data = request.get_json()

    # Save profile first
    conn = get_db()
    existing = conn.execute("SELECT id FROM student_profiles WHERE session_id=?", (sid,)).fetchone()
    if existing:
        conn.execute("""
            UPDATE student_profiles SET name=?,course=?,branch=?,subjects=?,exam_date=?,
            study_hours=?,prep_level=?,difficult_subjects=?,target_sgpa=?,semester=?,updated_at=CURRENT_TIMESTAMP
            WHERE session_id=?
        """, (
            data.get("name",""), data.get("course",""), data.get("branch",""),
            data.get("subjects",""), data.get("exam_date",""), data.get("study_hours",4),
            data.get("prep_level","beginner"), data.get("difficult_subjects",""),
            data.get("target_sgpa",8.0), data.get("semester",1), sid
        ))
    else:
        conn.execute("""
            INSERT INTO student_profiles (session_id,name,course,branch,subjects,exam_date,
            study_hours,prep_level,difficult_subjects,target_sgpa,semester)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            sid, data.get("name",""), data.get("course",""), data.get("branch",""),
            data.get("subjects",""), data.get("exam_date",""), data.get("study_hours",4),
            data.get("prep_level","beginner"), data.get("difficult_subjects",""),
            data.get("target_sgpa",8.0), data.get("semester",1)
        ))
    conn.commit()
    conn.close()

    plan = generate_study_plan(sid, data)
    log_activity(sid, "Study Planner", "Generated personalized study plan")
    return jsonify({"plan": plan})

@app.route("/api/generate-quiz", methods=["POST"])
def api_generate_quiz():
    sid = get_session_id()
    data = request.get_json()
    subject = data.get("subject", "General")
    topic = data.get("topic", "")
    difficulty = data.get("difficulty", "medium")
    num_q = min(int(data.get("num_questions", 5)), 10)

    result = quiz_agent(subject, topic, difficulty, num_q)
    log_activity(sid, "Quiz", f"Generated {num_q} {difficulty} questions for {subject}")
    return jsonify(result)

@app.route("/api/submit-quiz", methods=["POST"])
def api_submit_quiz():
    sid = get_session_id()
    data = request.get_json()
    questions = data.get("questions", [])
    answers = data.get("answers", {})
    subject = data.get("subject", "General")
    topic = data.get("topic", "")
    difficulty = data.get("difficulty", "medium")

    if not questions:
        return jsonify({"error": "No questions provided."}), 400

    score = 0
    results = []
    weak_topics = []

    for i, q in enumerate(questions):
        user_ans = answers.get(str(i), "")
        correct = q.get("correct", "")
        is_correct = user_ans.strip().upper() == correct.strip().upper()
        if is_correct:
            score += 1
        else:
            weak_topics.append(topic or subject)
        results.append({
            "question": q.get("question"),
            "options": q.get("options"),
            "user_answer": user_ans,
            "correct_answer": correct,
            "is_correct": is_correct,
            "explanation": q.get("explanation", ""),
        })

    total = len(questions)
    percentage = round(score / total * 100, 1) if total > 0 else 0

    conn = get_db()
    conn.execute("""
        INSERT INTO quiz_results (session_id,subject,topic,difficulty,score,total,percentage,weak_topics)
        VALUES (?,?,?,?,?,?,?,?)
    """, (sid, subject, topic, difficulty, score, total, percentage, ",".join(set(weak_topics))))
    conn.commit()
    conn.close()

    feedback = ""
    if percentage >= 80:
        feedback = "🎉 Excellent! You have a strong understanding of this topic."
    elif percentage >= 60:
        feedback = "👍 Good effort! Review the incorrect answers for improvement."
    else:
        feedback = "📚 Keep practicing! Focus on the weak areas identified below."

    log_activity(sid, "Quiz Result", f"{subject} – {score}/{total} ({percentage}%)")
    return jsonify({
        "score": score, "total": total, "percentage": percentage,
        "results": results, "weak_topics": list(set(weak_topics)), "feedback": feedback
    })

@app.route("/api/upload-document", methods=["POST"])
def api_upload_document():
    sid = get_session_id()
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": "Only PDF and TXT files are allowed."}), 400

    filename = secure_filename(f.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    f.save(filepath)

    conn = get_db()
    cursor = conn.execute(
        "INSERT INTO uploaded_documents (session_id, filename, original_name, doc_type) VALUES (?,?,?,?)",
        (sid, unique_name, filename, filename.rsplit(".",1)[-1].upper())
    )
    doc_id = cursor.lastrowid
    conn.commit()
    conn.close()

    success, result = process_uploaded_document(filepath, filename, doc_id, sid)
    if not success:
        return jsonify({"error": result}), 400

    log_activity(sid, "Upload", f"Uploaded: {filename}")
    return jsonify({
        "success": True,
        "doc_id": doc_id,
        "filename": filename,
        "preview": result[:500] if isinstance(result, str) else ""
    })

@app.route("/api/upload-notes", methods=["POST"])
def api_upload_notes():
    sid = get_session_id()
    data = request.get_json()
    notes_text = (data.get("text") or "").strip()
    title = (data.get("title") or "Pasted Notes").strip()

    if not notes_text or len(notes_text) < 20:
        return jsonify({"error": "Please enter at least 20 characters of notes."}), 400

    doc_id = f"notes_{uuid.uuid4().hex[:8]}"
    chunks = chunk_text(notes_text)
    add_to_vector_store(chunks, doc_id, title)

    conn = get_db()
    conn.execute(
        "INSERT INTO uploaded_documents (session_id, filename, original_name, doc_type, text_length, processed) VALUES (?,?,?,?,?,?)",
        (sid, doc_id, title, "TXT", len(notes_text), 1)
    )
    conn.commit()
    conn.close()

    log_activity(sid, "Notes", f"Added notes: {title}")
    return jsonify({"success": True, "message": f"Notes '{title}' added successfully."})

@app.route("/api/summarize-document", methods=["POST"])
def api_summarize_document():
    sid = get_session_id()
    data = request.get_json()
    doc_id = data.get("doc_id")

    # Get document text from vector store chunks
    doc_chunks = [c["text"] for c in _doc_chunks if str(c["doc_id"]) == str(doc_id)]
    if not doc_chunks:
        # Try retrieving with a broad query
        doc_chunks = retrieve_relevant_context("summary overview main concepts", top_k=5)

    if not doc_chunks:
        return jsonify({"error": "Document content not found. Please re-upload the document."}), 404

    combined = " ".join(doc_chunks)[:3000]
    prompt = f"""
Summarize the following study material in a structured format:

{combined}

Provide:
1. **Main Topic**: One sentence overview
2. **Key Concepts**: 5-8 bullet points of the most important concepts
3. **Important Definitions**: Any key terms and their definitions
4. **Important Formulas/Theorems**: (if applicable)
5. **Summary**: 2-3 paragraph summary
6. **Exam Tips**: 3 likely exam questions from this material
"""
    response = generate_response(prompt, max_tokens=1500)
    log_activity(sid, "Summarize", f"Summarized document {doc_id}")
    return jsonify({"summary": response})

@app.route("/api/analyze-paper", methods=["POST"])
def api_analyze_paper():
    sid = get_session_id()
    if "file" not in request.files and not request.get_json():
        return jsonify({"error": "No file or text provided."}), 400

    subject = ""
    extracted_text = ""

    if "file" in request.files:
        f = request.files["file"]
        subject = request.form.get("subject", "")
        if not f.filename or not allowed_file(f.filename):
            return jsonify({"error": "Please upload a PDF or TXT file."}), 400
        filename = secure_filename(f.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{uuid.uuid4().hex}_{filename}")
        f.save(filepath)

        if filename.lower().endswith(".pdf"):
            text, err = extract_pdf_text(filepath)
            if err:
                return jsonify({"error": err}), 400
            extracted_text = text
        else:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
                extracted_text = fh.read()
        os.remove(filepath)
    else:
        data = request.get_json()
        extracted_text = data.get("text", "")
        subject = data.get("subject", "")

    if not extracted_text.strip():
        return jsonify({"error": "No text could be extracted from the document."}), 400

    result = question_paper_analyzer_agent(extracted_text, subject)
    log_activity(sid, "PYQ Analyzer", f"Analyzed paper for {subject}")
    return jsonify(result)

@app.route("/api/generate-practice-questions", methods=["POST"])
def api_generate_practice_questions():
    sid = get_session_id()
    data = request.get_json()
    subject = data.get("subject", "")
    topics = data.get("topics", [])
    questions = generate_practice_questions(subject, topics)
    log_activity(sid, "Practice Questions", f"Generated for {subject}")
    return jsonify({"questions": questions})

@app.route("/api/complete-task", methods=["POST"])
def api_complete_task():
    sid = get_session_id()
    data = request.get_json()
    task_id = data.get("task_id")
    conn = get_db()
    conn.execute(
        "UPDATE study_tasks SET status='completed', completed_at=CURRENT_TIMESTAMP WHERE id=? AND session_id=?",
        (task_id, sid)
    )
    conn.commit()
    conn.close()
    return jsonify({"success": True})

@app.route("/api/progress")
def api_progress():
    sid = get_session_id()
    progress = calculate_student_progress(sid)
    return jsonify(progress)

# ─────────────────────────────────────────────
# Page Template Strings (Study Assistant)
# ─────────────────────────────────────────────
STUDY_ASSISTANT_TEMPLATE = BASE_TEMPLATE.replace(
    "{% block title %}Smart Study Generator{% endblock %}", "{% block title %}Study Assistant{% endblock %}"
).replace(
    "{% block page_title %}StudyMate AI{% endblock %}", "{% block page_title %}Study Assistant{% endblock %}"
).replace("{% block content %}{% endblock %}", """{% block content %}
<div class="row g-4">
  <div class="col-lg-4">
    <div class="card">
      <div class="card-header bg-primary text-white"><h6 class="mb-0"><i class="bi bi-sliders me-2"></i>Settings</h6></div>
      <div class="card-body">
        <div class="mb-3">
          <label class="form-label fw-semibold">Subject</label>
          <input type="text" id="subject" class="form-control" placeholder="e.g. Data Structures">
        </div>
        <div class="mb-3">
          <label class="form-label fw-semibold">Topic</label>
          <input type="text" id="topic" class="form-control" placeholder="e.g. Binary Trees">
        </div>
        <div class="mb-3">
          <label class="form-label fw-semibold">Answer Type</label>
          <select id="mark_type" class="form-select">
            <option value="">Detailed Explanation</option>
            <option value="2mark">2-Mark Answer</option>
            <option value="5mark">5-Mark Answer</option>
            <option value="8mark">8-Mark Answer</option>
            <option value="10mark">10-Mark Answer</option>
          </select>
        </div>
        <div class="alert alert-info p-2 small">
          <i class="bi bi-info-circle me-1"></i>Uploaded documents are automatically used as context.
        </div>
      </div>
    </div>
  </div>

  <div class="col-lg-8">
    <div class="card">
      <div class="card-header bg-white d-flex justify-content-between align-items-center">
        <h6 class="mb-0"><i class="bi bi-chat-dots text-primary me-2"></i>AI Study Assistant</h6>
        <button class="btn btn-sm btn-outline-secondary" onclick="clearChat()"><i class="bi bi-trash me-1"></i>Clear</button>
      </div>
      <div class="card-body p-0">
        <div class="chat-container p-3" id="chatBox">
          <div class="chat-msg ai">
            <div class="bubble">
              <p class="mb-1"><strong>Hello! I'm StudyMate AI 👋</strong></p>
              <p class="mb-0">Ask me any academic question. I'll provide clear explanations, exam-ready answers, and use your uploaded study materials as context.</p>
            </div>
          </div>
        </div>
        <div class="p-3 border-top">
          <div class="input-group">
            <textarea id="questionInput" class="form-control" rows="2" placeholder="Ask any academic question... (e.g. Explain Binary Search Trees)" style="resize:none"></textarea>
            <button class="btn btn-primary" id="sendBtn" onclick="sendQuestion()">
              <span class="loading-spinner spinner-border spinner-border-sm me-1" id="sendSpinner"></span>
              <i class="bi bi-send-fill"></i>
            </button>
          </div>
          <div class="mt-2 d-flex gap-2 flex-wrap">
            <small class="text-muted me-2">Quick:</small>
            <span class="badge badge-subject" style="cursor:pointer" onclick="quickAsk('Explain the concept of polymorphism in OOP')">Polymorphism</span>
            <span class="badge badge-subject" style="cursor:pointer" onclick="quickAsk('What is time complexity? Explain with examples')">Time Complexity</span>
            <span class="badge badge-subject" style="cursor:pointer" onclick="quickAsk('Explain TCP/IP model layers')">TCP/IP</span>
            <span class="badge badge-subject" style="cursor:pointer" onclick="quickAsk('What is normalization in DBMS?')">DBMS</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
{% endblock %}
{% block scripts %}
<script>
function quickAsk(q) { document.getElementById('questionInput').value = q; sendQuestion(); }
function clearChat() {
  const box = document.getElementById('chatBox');
  box.innerHTML = '<div class="chat-msg ai"><div class="bubble"><p class="mb-0">Chat cleared. Ask me anything!</p></div></div>';
}
function sendQuestion() {
  const q = document.getElementById('questionInput').value.trim();
  if (!q) return;
  appendMsg(q, 'user');
  document.getElementById('questionInput').value = '';
  document.getElementById('sendBtn').disabled = true;
  document.getElementById('sendSpinner').classList.add('show');

  fetch('/api/ask', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      question: q,
      subject: document.getElementById('subject').value,
      topic: document.getElementById('topic').value,
      mark_type: document.getElementById('mark_type').value
    })
  })
  .then(r => r.json())
  .then(d => {
    appendMsg(d.response || d.error, 'ai');
    document.getElementById('sendBtn').disabled = false;
    document.getElementById('sendSpinner').classList.remove('show');
  })
  .catch(e => {
    appendMsg('Error: ' + e.message, 'ai');
    document.getElementById('sendBtn').disabled = false;
    document.getElementById('sendSpinner').classList.remove('show');
  });
}
function appendMsg(text, role) {
  const box = document.getElementById('chatBox');
  const div = document.createElement('div');
  div.className = 'chat-msg ' + role;
  div.innerHTML = '<div class="bubble"><pre>' + escHtml(text) + '</pre></div>';
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}
function escHtml(t) {
  return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
document.getElementById('questionInput').addEventListener('keydown', function(e) {
  if (e.ctrlKey && e.key === 'Enter') sendQuestion();
});
</script>
{% endblock %}
""")

STUDY_MATERIAL_TEMPLATE = BASE_TEMPLATE.replace(
    "{% block title %}Smart Study Generator{% endblock %}", "{% block title %}Study Material{% endblock %}"
).replace(
    "{% block page_title %}StudyMate AI{% endblock %}", "{% block page_title %}Study Material{% endblock %}"
).replace("{% block content %}{% endblock %}", """{% block content %}
<div class="row g-4">
  <!-- Upload Section -->
  <div class="col-lg-5">
    <div class="card mb-4">
      <div class="card-header bg-success text-white"><h6 class="mb-0"><i class="bi bi-cloud-upload me-2"></i>Upload Document</h6></div>
      <div class="card-body">
        <div id="uploadZone" class="border-2 border-dashed rounded-3 p-4 text-center mb-3" style="border:2px dashed #dee2e6; cursor:pointer" onclick="document.getElementById('fileInput').click()">
          <i class="bi bi-file-earmark-arrow-up" style="font-size:2.5rem; color:#6c757d"></i>
          <p class="mt-2 mb-0 text-muted">Click to upload PDF or TXT</p>
          <small class="text-muted">Max 16MB</small>
        </div>
        <input type="file" id="fileInput" accept=".pdf,.txt" class="d-none" onchange="uploadFile(this)">
        <div id="uploadProgress" class="d-none">
          <div class="progress mb-2"><div class="progress-bar progress-bar-striped progress-bar-animated w-100"></div></div>
          <small class="text-muted">Processing document...</small>
        </div>
        <div id="uploadResult" class="d-none alert alert-success p-2 small"></div>
      </div>
    </div>

    <div class="card">
      <div class="card-header bg-info text-white"><h6 class="mb-0"><i class="bi bi-pencil-square me-2"></i>Paste Notes</h6></div>
      <div class="card-body">
        <input type="text" id="notesTitle" class="form-control mb-2" placeholder="Notes title (optional)">
        <textarea id="notesText" class="form-control mb-2" rows="6" placeholder="Paste your study notes here..."></textarea>
        <button class="btn btn-info w-100" onclick="uploadNotes()"><i class="bi bi-plus-circle me-2"></i>Add Notes</button>
        <div id="notesResult" class="mt-2 d-none alert alert-success p-2 small"></div>
      </div>
    </div>
  </div>

  <!-- Documents List & Actions -->
  <div class="col-lg-7">
    <div class="card mb-4">
      <div class="card-header bg-white d-flex justify-content-between align-items-center">
        <h6 class="mb-0"><i class="bi bi-collection text-primary me-2"></i>Your Documents ({{ docs|length }})</h6>
      </div>
      <div class="card-body p-0">
        {% if docs %}
        <div class="list-group list-group-flush">
          {% for doc in docs %}
          <div class="list-group-item d-flex align-items-center gap-3">
            <i class="bi bi-file-earmark-{{ 'pdf text-danger' if doc.doc_type == 'PDF' else 'text text-info' }}" style="font-size:1.4rem"></i>
            <div class="flex-grow-1">
              <div class="fw-semibold">{{ doc.original_name }}</div>
              <small class="text-muted">{{ doc.doc_type }} · {{ doc.text_length }} chars · {{ doc.created_at[:10] }}</small>
            </div>
            <div class="d-flex gap-1">
              {% if doc.processed %}
              <button class="btn btn-sm btn-outline-primary" onclick="summarizeDoc('{{ doc.id }}')"><i class="bi bi-file-text"></i></button>
              {% endif %}
            </div>
          </div>
          {% endfor %}
        </div>
        {% else %}
        <div class="text-center text-muted py-5">
          <i class="bi bi-folder-x" style="font-size:3rem"></i>
          <p class="mt-2">No documents uploaded yet.</p>
        </div>
        {% endif %}
      </div>
    </div>

    <div class="card" id="summaryCard" style="display:none!important">
      <div class="card-header bg-white"><h6 class="mb-0"><i class="bi bi-file-richtext text-success me-2"></i>Document Summary</h6></div>
      <div class="card-body">
        <div id="summaryLoading" class="text-center py-3 d-none">
          <div class="spinner-border text-primary"></div><p class="mt-2">Generating summary...</p>
        </div>
        <div id="summaryContent" style="white-space:pre-wrap; font-size:0.9rem; line-height:1.6"></div>
      </div>
    </div>
  </div>
</div>
{% endblock %}
{% block scripts %}
<script>
function uploadFile(input) {
  const file = input.files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append('file', file);
  document.getElementById('uploadProgress').classList.remove('d-none');
  document.getElementById('uploadResult').classList.add('d-none');
  fetch('/api/upload-document', { method: 'POST', body: fd })
    .then(r => r.json())
    .then(d => {
      document.getElementById('uploadProgress').classList.add('d-none');
      const res = document.getElementById('uploadResult');
      res.classList.remove('d-none');
      if (d.success) {
        res.className = 'alert alert-success p-2 small';
        res.textContent = '✓ ' + d.filename + ' uploaded and processed successfully!';
        setTimeout(() => location.reload(), 1500);
      } else {
        res.className = 'alert alert-danger p-2 small';
        res.textContent = '✗ ' + (d.error || 'Upload failed');
      }
    })
    .catch(e => {
      document.getElementById('uploadProgress').classList.add('d-none');
      const res = document.getElementById('uploadResult');
      res.className = 'alert alert-danger p-2 small';
      res.textContent = 'Error: ' + e.message;
      res.classList.remove('d-none');
    });
}
function uploadNotes() {
  const text = document.getElementById('notesText').value.trim();
  const title = document.getElementById('notesTitle').value.trim() || 'My Notes';
  if (text.length < 20) { alert('Please enter at least 20 characters.'); return; }
  fetch('/api/upload-notes', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ text, title })
  })
  .then(r => r.json())
  .then(d => {
    const el = document.getElementById('notesResult');
    el.classList.remove('d-none');
    if (d.success) {
      el.className = 'mt-2 alert alert-success p-2 small';
      el.textContent = '✓ ' + d.message;
      document.getElementById('notesText').value = '';
      setTimeout(() => location.reload(), 1500);
    } else {
      el.className = 'mt-2 alert alert-danger p-2 small';
      el.textContent = '✗ ' + (d.error || 'Failed to add notes');
    }
  });
}
function summarizeDoc(docId) {
  const card = document.getElementById('summaryCard');
  card.style.display = 'block';
  document.getElementById('summaryLoading').classList.remove('d-none');
  document.getElementById('summaryContent').textContent = '';
  fetch('/api/summarize-document', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({doc_id: docId})
  })
  .then(r => r.json())
  .then(d => {
    document.getElementById('summaryLoading').classList.add('d-none');
    document.getElementById('summaryContent').textContent = d.summary || d.error;
    card.scrollIntoView({behavior:'smooth'});
  });
}
</script>
{% endblock %}
""")

STUDY_PLANNER_TEMPLATE = BASE_TEMPLATE.replace(
    "{% block title %}Smart Study Generator{% endblock %}", "{% block title %}Study Planner{% endblock %}"
).replace(
    "{% block page_title %}StudyMate AI{% endblock %}", "{% block page_title %}Study Planner{% endblock %}"
).replace("{% block content %}{% endblock %}", """{% block content %}
<div class="row g-4">
  <!-- Profile Form -->
  <div class="col-lg-5">
    <div class="card">
      <div class="card-header bg-primary text-white"><h6 class="mb-0"><i class="bi bi-person-gear me-2"></i>Student Profile</h6></div>
      <div class="card-body">
        <div class="row g-2">
          <div class="col-12"><input type="text" id="pName" class="form-control" placeholder="Your name (optional)" value="{{ profile.name or '' }}"></div>
          <div class="col-6"><input type="text" id="pCourse" class="form-control" placeholder="Course (e.g. B.Tech)" value="{{ profile.course or '' }}"></div>
          <div class="col-6"><input type="text" id="pBranch" class="form-control" placeholder="Branch (e.g. CSE)" value="{{ profile.branch or '' }}"></div>
          <div class="col-12"><input type="text" id="pUniv" class="form-control" placeholder="University name" value="{{ profile.university or '' }}"></div>
          <div class="col-6">
            <select id="pSemester" class="form-select">
              {% for i in range(1,9) %}<option value="{{ i }}" {{ 'selected' if profile.semester == i else '' }}>Semester {{ i }}</option>{% endfor %}
            </select>
          </div>
          <div class="col-6"><input type="number" id="pHours" class="form-control" placeholder="Study hrs/day" min="1" max="16" value="{{ profile.study_hours or 4 }}"></div>
          <div class="col-12"><textarea id="pSubjects" class="form-control" rows="2" placeholder="Subjects (comma separated)">{{ profile.subjects or '' }}</textarea></div>
          <div class="col-12"><input type="date" id="pExamDate" class="form-control" value="{{ profile.exam_date or '' }}"></div>
          <div class="col-12">
            <select id="pPrepLevel" class="form-select">
              <option value="beginner" {{ 'selected' if profile.prep_level == 'beginner' else '' }}>Beginner (just starting)</option>
              <option value="intermediate" {{ 'selected' if profile.prep_level == 'intermediate' else '' }}>Intermediate (some topics done)</option>
              <option value="advanced" {{ 'selected' if profile.prep_level == 'advanced' else '' }}>Advanced (most topics done)</option>
            </select>
          </div>
          <div class="col-12"><textarea id="pDifficult" class="form-control" rows="2" placeholder="Difficult subjects/topics">{{ profile.difficult_subjects or '' }}</textarea></div>
          <div class="col-12"><input type="number" id="pTargetSGPA" class="form-control" placeholder="Target SGPA (e.g. 8.5)" min="0" max="10" step="0.1" value="{{ profile.target_sgpa or 8.0 }}"></div>
          <div class="col-12">
            <button class="btn btn-primary w-100" id="generatePlanBtn" onclick="generatePlan()">
              <span class="spinner-border spinner-border-sm me-2 d-none" id="planSpinner"></span>
              <i class="bi bi-calendar-plus me-2"></i>Generate My Study Plan
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Plan Output & Tasks -->
  <div class="col-lg-7">
    <div class="card mb-4" id="planCard" style="display:none">
      <div class="card-header bg-success text-white"><h6 class="mb-0"><i class="bi bi-calendar3 me-2"></i>Your Personalized Study Plan</h6></div>
      <div class="card-body"><pre id="planContent" style="white-space:pre-wrap;font-size:0.87rem;line-height:1.7;max-height:400px;overflow-y:auto"></pre></div>
    </div>

    <div class="card">
      <div class="card-header bg-white d-flex justify-content-between align-items-center">
        <h6 class="mb-0"><i class="bi bi-list-check text-primary me-2"></i>Study Tasks ({{ tasks|length }})</h6>
      </div>
      <div class="card-body p-0">
        {% if tasks %}
        <div style="max-height:400px; overflow-y:auto">
          <table class="table table-hover mb-0">
            <thead class="table-light"><tr><th>Subject</th><th>Topic</th><th>Date</th><th>Status</th><th></th></tr></thead>
            <tbody>
              {% for t in tasks[:30] %}
              <tr id="task_{{ t.id }}">
                <td><span class="badge badge-subject">{{ t.subject }}</span></td>
                <td><small>{{ t.topic[:40] }}</small></td>
                <td><small>{{ t.scheduled_date }}</small></td>
                <td>
                  {% if t.status == 'completed' %}
                  <span class="badge bg-success">Done</span>
                  {% else %}
                  <span class="badge bg-warning text-dark">Pending</span>
                  {% endif %}
                </td>
                <td>
                  {% if t.status != 'completed' %}
                  <button class="btn btn-xs btn-outline-success btn-sm" onclick="completeTask({{ t.id }})"><i class="bi bi-check"></i></button>
                  {% endif %}
                </td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
        {% else %}
        <div class="text-center text-muted py-5">
          <i class="bi bi-calendar-x" style="font-size:3rem"></i>
          <p class="mt-2">No tasks yet. Generate a study plan to get started!</p>
        </div>
        {% endif %}
      </div>
    </div>
  </div>
</div>
{% endblock %}
{% block scripts %}
<script>
function generatePlan() {
  const btn = document.getElementById('generatePlanBtn');
  const spinner = document.getElementById('planSpinner');
  btn.disabled = true; spinner.classList.remove('d-none');
  const data = {
    name: document.getElementById('pName').value,
    course: document.getElementById('pCourse').value,
    branch: document.getElementById('pBranch').value,
    university: document.getElementById('pUniv').value,
    semester: parseInt(document.getElementById('pSemester').value),
    study_hours: parseFloat(document.getElementById('pHours').value) || 4,
    subjects: document.getElementById('pSubjects').value,
    exam_date: document.getElementById('pExamDate').value,
    prep_level: document.getElementById('pPrepLevel').value,
    difficult_subjects: document.getElementById('pDifficult').value,
    target_sgpa: parseFloat(document.getElementById('pTargetSGPA').value) || 8.0
  };
  fetch('/api/generate-plan', {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)
  })
  .then(r => r.json())
  .then(d => {
    btn.disabled = false; spinner.classList.add('d-none');
    const card = document.getElementById('planCard');
    card.style.display = 'block';
    document.getElementById('planContent').textContent = d.plan || d.error;
    card.scrollIntoView({behavior:'smooth'});
    setTimeout(() => location.reload(), 3000);
  })
  .catch(e => { btn.disabled = false; spinner.classList.add('d-none'); alert('Error: ' + e.message); });
}
function completeTask(id) {
  fetch('/api/complete-task', {
    method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({task_id: id})
  })
  .then(r => r.json())
  .then(d => {
    if (d.success) {
      const row = document.getElementById('task_' + id);
      row.querySelector('td:nth-child(4)').innerHTML = '<span class="badge bg-success">Done</span>';
      row.querySelector('td:nth-child(5)').innerHTML = '';
    }
  });
}
</script>
{% endblock %}
""")

QUESTION_ANALYZER_TEMPLATE = BASE_TEMPLATE.replace(
    "{% block title %}Smart Study Generator{% endblock %}", "{% block title %}Paper Analyzer{% endblock %}"
).replace(
    "{% block page_title %}StudyMate AI{% endblock %}", "{% block page_title %}Question Paper Analyzer{% endblock %}"
).replace("{% block content %}{% endblock %}", """{% block content %}
<div class="alert alert-warning small">
  <i class="bi bi-exclamation-triangle me-2"></i>
  <strong>Disclaimer:</strong> Analysis is based on historical patterns only and does not guarantee future exam questions. Use this as a study aid, not a prediction tool.
</div>
<div class="row g-4">
  <div class="col-lg-5">
    <div class="card">
      <div class="card-header bg-warning text-dark"><h6 class="mb-0"><i class="bi bi-upload me-2"></i>Upload Question Paper</h6></div>
      <div class="card-body">
        <div class="mb-3">
          <label class="form-label">Subject Name</label>
          <input type="text" id="paperSubject" class="form-control" placeholder="e.g. Data Structures">
        </div>
        <div class="mb-3">
          <label class="form-label">Upload PDF/TXT</label>
          <input type="file" id="paperFile" class="form-control" accept=".pdf,.txt">
        </div>
        <hr><p class="text-center text-muted small">— or paste text —</p>
        <textarea id="paperText" class="form-control mb-3" rows="6" placeholder="Paste question paper text here..."></textarea>
        <button class="btn btn-warning w-100 text-dark" id="analyzeBtn" onclick="analyzePaper()">
          <span class="spinner-border spinner-border-sm me-2 d-none" id="analyzeSpinner"></span>
          <i class="bi bi-search me-2"></i>Analyze Paper
        </button>
      </div>
    </div>
  </div>

  <div class="col-lg-7">
    <div id="analysisResults" style="display:none">
      <div class="card mb-4">
        <div class="card-header bg-white"><h6 class="mb-0"><i class="bi bi-bar-chart text-warning me-2"></i>Topic Frequency Analysis</h6></div>
        <div class="card-body"><canvas id="topicChart" height="200"></canvas></div>
      </div>
      <div class="row g-3 mb-4" id="analysisStats"></div>
      <div class="card mb-4">
        <div class="card-header bg-white"><h6 class="mb-0"><i class="bi bi-star text-warning me-2"></i>Important Topics</h6></div>
        <div class="card-body" id="importantTopics"></div>
      </div>
      <div class="card mb-4">
        <div class="card-header bg-white"><h6 class="mb-0"><i class="bi bi-arrow-repeat text-primary me-2"></i>Repeated Questions</h6></div>
        <div class="card-body" id="repeatedQuestions"></div>
      </div>
      <div class="card mb-4">
        <div class="card-header bg-white"><h6 class="mb-0"><i class="bi bi-lightbulb text-success me-2"></i>Recommendations</h6></div>
        <div class="card-body" id="recommendations"></div>
      </div>
      <button class="btn btn-success w-100" onclick="generatePracticeQs()">
        <i class="bi bi-question-circle me-2"></i>Generate Practice Questions
      </button>
      <div class="card mt-3" id="practiceCard" style="display:none">
        <div class="card-header bg-success text-white"><h6 class="mb-0">Practice Questions</h6></div>
        <div class="card-body"><pre id="practiceContent" style="white-space:pre-wrap;font-size:0.87rem;line-height:1.7"></pre></div>
      </div>
    </div>
    <div id="analyzerEmpty" class="text-center text-muted py-5">
      <i class="bi bi-file-earmark-bar-graph" style="font-size:4rem; opacity:0.3"></i>
      <p class="mt-3">Upload a question paper to see analysis</p>
    </div>
  </div>
</div>
{% endblock %}
{% block scripts %}
<script>
let analysisData = null;
let currentSubject = '';
let topicChart = null;

function analyzePaper() {
  const btn = document.getElementById('analyzeBtn');
  const spinner = document.getElementById('analyzeSpinner');
  btn.disabled = true; spinner.classList.remove('d-none');

  const file = document.getElementById('paperFile').files[0];
  const text = document.getElementById('paperText').value.trim();
  currentSubject = document.getElementById('paperSubject').value;

  let fetchPromise;
  if (file) {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('subject', currentSubject);
    fetchPromise = fetch('/api/analyze-paper', { method: 'POST', body: fd });
  } else if (text) {
    fetchPromise = fetch('/api/analyze-paper', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ text, subject: currentSubject })
    });
  } else {
    alert('Please upload a file or paste question paper text.');
    btn.disabled = false; spinner.classList.add('d-none');
    return;
  }

  fetchPromise.then(r => r.json()).then(d => {
    btn.disabled = false; spinner.classList.add('d-none');
    if (!d.success) { alert(d.error || 'Analysis failed.'); return; }
    analysisData = d.analysis;
    renderAnalysis(d.analysis);
  }).catch(e => { btn.disabled = false; spinner.classList.add('d-none'); alert('Error: ' + e.message); });
}

function renderAnalysis(a) {
  document.getElementById('analyzerEmpty').style.display = 'none';
  document.getElementById('analysisResults').style.display = 'block';

  // Chart
  const topics = a.topics || [];
  const labels = topics.map(t => t.name).slice(0,8);
  const data = topics.map(t => t.frequency).slice(0,8);
  if (topicChart) topicChart.destroy();
  const ctx = document.getElementById('topicChart').getContext('2d');
  topicChart = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ label: 'Frequency', data, backgroundColor: '#4361ee' }] },
    options: { responsive: true, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true } } }
  });

  // Important topics
  const it = document.getElementById('importantTopics');
  it.innerHTML = (a.important_topics || []).map(t =>
    '<span class="badge bg-warning text-dark me-2 mb-2 p-2">' + t + '</span>'
  ).join('') || '<span class="text-muted">None identified</span>';

  // Repeated questions
  const rq = document.getElementById('repeatedQuestions');
  rq.innerHTML = (a.repeated_questions || []).map((q,i) =>
    '<p class="mb-1"><span class="badge bg-danger me-2">' + (i+1) + '</span>' + q + '</p>'
  ).join('') || '<p class="text-muted">None detected</p>';

  // Recommendations
  const rec = document.getElementById('recommendations');
  rec.innerHTML = '<ul>' + (a.recommendations || ['Review all topics thoroughly.']).map(r => '<li>' + r + '</li>').join('') + '</ul>';

  document.getElementById('analysisResults').scrollIntoView({behavior:'smooth'});
}

function generatePracticeQs() {
  if (!analysisData) { alert('Please analyze a paper first.'); return; }
  fetch('/api/generate-practice-questions', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ subject: currentSubject, topics: analysisData.important_topics || [] })
  })
  .then(r => r.json())
  .then(d => {
    document.getElementById('practiceCard').style.display = 'block';
    document.getElementById('practiceContent').textContent = d.questions || 'Could not generate questions.';
  });
}
</script>
{% endblock %}
""")

QUIZ_TEMPLATE = BASE_TEMPLATE.replace(
    "{% block title %}Smart Study Generator{% endblock %}", "{% block title %}AI Quiz{% endblock %}"
).replace(
    "{% block page_title %}StudyMate AI{% endblock %}", "{% block page_title %}AI Quiz Generator{% endblock %}"
).replace("{% block content %}{% endblock %}", """{% block content %}
<div class="row g-4">
  <div class="col-lg-4">
    <div class="card" id="quizSetup">
      <div class="card-header bg-secondary text-white"><h6 class="mb-0"><i class="bi bi-gear me-2"></i>Quiz Settings</h6></div>
      <div class="card-body">
        <div class="mb-3"><label class="form-label">Subject</label><input type="text" id="qSubject" class="form-control" placeholder="e.g. Data Structures"></div>
        <div class="mb-3"><label class="form-label">Topic</label><input type="text" id="qTopic" class="form-control" placeholder="e.g. Sorting Algorithms"></div>
        <div class="mb-3">
          <label class="form-label">Difficulty</label>
          <select id="qDifficulty" class="form-select">
            <option value="easy">Easy</option>
            <option value="medium" selected>Medium</option>
            <option value="hard">Hard</option>
          </select>
        </div>
        <div class="mb-3">
          <label class="form-label">Number of Questions</label>
          <select id="qCount" class="form-select">
            <option value="3">3 Questions</option>
            <option value="5" selected>5 Questions</option>
            <option value="7">7 Questions</option>
            <option value="10">10 Questions</option>
          </select>
        </div>
        <button class="btn btn-secondary w-100" id="genQuizBtn" onclick="generateQuiz()">
          <span class="spinner-border spinner-border-sm me-2 d-none" id="quizSpinner"></span>
          <i class="bi bi-play-fill me-2"></i>Generate Quiz
        </button>
      </div>
    </div>
  </div>

  <div class="col-lg-8">
    <div id="quizEmpty" class="text-center text-muted py-5">
      <i class="bi bi-patch-question" style="font-size:4rem; opacity:0.3"></i>
      <p class="mt-3">Configure your quiz and click Generate</p>
    </div>

    <div id="quizContainer" style="display:none">
      <div class="d-flex justify-content-between align-items-center mb-3">
        <h6 class="mb-0" id="quizTitle">Quiz</h6>
        <span class="badge bg-secondary" id="quizProgress">Q 0/0</span>
      </div>
      <div id="questionsDiv"></div>
      <button class="btn btn-primary w-100 mt-3" id="submitQuizBtn" onclick="submitQuiz()">
        <i class="bi bi-check2-square me-2"></i>Submit Quiz
      </button>
    </div>

    <div id="quizResults" style="display:none" class="card">
      <div class="card-header bg-white"><h6 class="mb-0"><i class="bi bi-trophy text-warning me-2"></i>Quiz Results</h6></div>
      <div class="card-body">
        <div class="text-center mb-4">
          <div class="display-4 fw-bold text-primary" id="scoreDisplay"></div>
          <p class="text-muted" id="feedbackText"></p>
          <div class="progress mb-3" style="height:12px"><div class="progress-bar" id="scoreBar"></div></div>
        </div>
        <div id="resultsDetail"></div>
        <div id="weakTopicsDiv" class="mt-3"></div>
        <div class="d-flex gap-2 mt-3">
          <button class="btn btn-primary flex-fill" onclick="retakeQuiz()"><i class="bi bi-arrow-clockwise me-2"></i>New Quiz</button>
        </div>
      </div>
    </div>
  </div>
</div>
{% endblock %}
{% block scripts %}
<script>
let currentQuestions = [];
let userAnswers = {};

function generateQuiz() {
  const btn = document.getElementById('genQuizBtn');
  const spinner = document.getElementById('quizSpinner');
  btn.disabled = true; spinner.classList.remove('d-none');
  document.getElementById('quizEmpty').style.display = 'none';
  document.getElementById('quizContainer').style.display = 'none';
  document.getElementById('quizResults').style.display = 'none';

  fetch('/api/generate-quiz', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({
      subject: document.getElementById('qSubject').value || 'General Knowledge',
      topic: document.getElementById('qTopic').value,
      difficulty: document.getElementById('qDifficulty').value,
      num_questions: parseInt(document.getElementById('qCount').value)
    })
  })
  .then(r => r.json())
  .then(d => {
    btn.disabled = false; spinner.classList.add('d-none');
    if (!d.success) { alert(d.error || 'Failed to generate quiz. Try again.'); document.getElementById('quizEmpty').style.display = 'block'; return; }
    currentQuestions = d.questions;
    userAnswers = {};
    renderQuiz();
  })
  .catch(e => { btn.disabled = false; spinner.classList.add('d-none'); alert('Error: ' + e.message); });
}

function renderQuiz() {
  const subj = document.getElementById('qSubject').value || 'Quiz';
  const diff = document.getElementById('qDifficulty').value;
  document.getElementById('quizTitle').textContent = subj + ' – ' + diff.charAt(0).toUpperCase() + diff.slice(1);
  document.getElementById('quizProgress').textContent = 'Q ' + currentQuestions.length;
  const div = document.getElementById('questionsDiv');
  div.innerHTML = '';
  currentQuestions.forEach((q, i) => {
    const qDiv = document.createElement('div');
    qDiv.className = 'card mb-3';
    qDiv.innerHTML = '<div class="card-body"><p class="fw-semibold mb-3">' + (i+1) + '. ' + escHtml(q.question) + '</p>' +
      q.options.map(opt => {
        const letter = opt.charAt(0);
        return '<div class="quiz-option" id="opt_' + i + '_' + letter + '" onclick="selectAnswer(' + i + ',\'' + letter + '\')">' + escHtml(opt) + '</div>';
      }).join('') + '</div>';
    div.appendChild(qDiv);
  });
  document.getElementById('quizContainer').style.display = 'block';
}

function selectAnswer(qIdx, letter) {
  const prev = userAnswers[qIdx];
  if (prev) document.getElementById('opt_' + qIdx + '_' + prev)?.classList.remove('selected');
  userAnswers[qIdx] = letter;
  document.getElementById('opt_' + qIdx + '_' + letter)?.classList.add('selected');
}

function submitQuiz() {
  if (Object.keys(userAnswers).length < currentQuestions.length) {
    if (!confirm('You have not answered all questions. Submit anyway?')) return;
  }
  fetch('/api/submit-quiz', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({
      questions: currentQuestions,
      answers: userAnswers,
      subject: document.getElementById('qSubject').value,
      topic: document.getElementById('qTopic').value,
      difficulty: document.getElementById('qDifficulty').value
    })
  })
  .then(r => r.json())
  .then(d => {
    document.getElementById('quizContainer').style.display = 'none';
    const resDiv = document.getElementById('quizResults');
    resDiv.style.display = 'block';
    document.getElementById('scoreDisplay').textContent = d.score + '/' + d.total + ' (' + d.percentage + '%)';
    document.getElementById('feedbackText').textContent = d.feedback;
    const bar = document.getElementById('scoreBar');
    bar.style.width = d.percentage + '%';
    bar.className = 'progress-bar ' + (d.percentage >= 80 ? 'bg-success' : d.percentage >= 60 ? 'bg-warning' : 'bg-danger');

    const detail = document.getElementById('resultsDetail');
    detail.innerHTML = d.results.map((r, i) => '<div class="mb-3 p-3 rounded ' + (r.is_correct ? 'bg-success bg-opacity-10' : 'bg-danger bg-opacity-10') + '">' +
      '<p class="fw-semibold mb-1">' + (i+1) + '. ' + escHtml(r.question) + '</p>' +
      '<p class="mb-1 small">Your answer: <strong>' + (r.user_answer || 'Not answered') + '</strong> ' + (r.is_correct ? '✅' : '❌') + '</p>' +
      (r.is_correct ? '' : '<p class="mb-1 small text-success">Correct: <strong>' + r.correct_answer + '</strong></p>') +
      '<p class="mb-0 small text-muted">' + escHtml(r.explanation) + '</p></div>'
    ).join('');

    if (d.weak_topics && d.weak_topics.length > 0) {
      document.getElementById('weakTopicsDiv').innerHTML = '<div class="alert alert-warning"><strong>Weak Topics:</strong> ' + d.weak_topics.join(', ') + '<br><small>Review these topics in the Study Assistant.</small></div>';
    }
    resDiv.scrollIntoView({behavior:'smooth'});
  });
}

function retakeQuiz() {
  document.getElementById('quizResults').style.display = 'none';
  document.getElementById('quizEmpty').style.display = 'block';
  currentQuestions = []; userAnswers = {};
}

function escHtml(t) {
  if (!t) return '';
  return String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
</script>
{% endblock %}
""")

PROGRESS_TEMPLATE = BASE_TEMPLATE.replace(
    "{% block title %}Smart Study Generator{% endblock %}", "{% block title %}Progress{% endblock %}"
).replace(
    "{% block page_title %}StudyMate AI{% endblock %}", "{% block page_title %}Progress Dashboard{% endblock %}"
).replace("{% block content %}{% endblock %}", """{% block content %}
<div class="row g-4">
  <!-- Stats -->
  <div class="col-6 col-md-3">
    <div class="stat-card text-center">
      <div class="stat-icon bg-primary bg-opacity-10 text-primary mx-auto mb-2"><i class="bi bi-percent"></i></div>
      <h3 class="text-primary">{{ progress.overall_pct }}%</h3>
      <small class="text-muted">Overall Progress</small>
    </div>
  </div>
  <div class="col-6 col-md-3">
    <div class="stat-card text-center">
      <div class="stat-icon bg-success bg-opacity-10 text-success mx-auto mb-2"><i class="bi bi-check-circle"></i></div>
      <h3 class="text-success">{{ progress.completed_tasks }}</h3>
      <small class="text-muted">Completed Tasks</small>
    </div>
  </div>
  <div class="col-6 col-md-3">
    <div class="stat-card text-center">
      <div class="stat-icon bg-warning bg-opacity-10 text-warning mx-auto mb-2"><i class="bi bi-clock"></i></div>
      <h3 class="text-warning">{{ progress.pending_tasks }}</h3>
      <small class="text-muted">Pending Tasks</small>
    </div>
  </div>
  <div class="col-6 col-md-3">
    <div class="stat-card text-center">
      <div class="stat-icon bg-info bg-opacity-10 text-info mx-auto mb-2"><i class="bi bi-journal-richtext"></i></div>
      <h3 class="text-info">{{ quizzes|length }}</h3>
      <small class="text-muted">Quizzes Taken</small>
    </div>
  </div>

  <!-- Charts -->
  <div class="col-lg-6">
    <div class="card">
      <div class="card-header bg-white"><h6 class="mb-0"><i class="bi bi-bar-chart text-primary me-2"></i>Subject Progress</h6></div>
      <div class="card-body">
        {% if progress.subject_data %}
        <canvas id="subjectChart" height="200"></canvas>
        {% else %}
        <div class="text-center text-muted py-4">
          <i class="bi bi-bar-chart" style="font-size:3rem; opacity:0.3"></i>
          <p class="mt-2">Create a study plan to track subject progress</p>
        </div>
        {% endif %}
      </div>
    </div>
  </div>
  <div class="col-lg-6">
    <div class="card">
      <div class="card-header bg-white"><h6 class="mb-0"><i class="bi bi-graph-up text-success me-2"></i>Quiz Performance</h6></div>
      <div class="card-body">
        {% if quizzes %}
        <canvas id="quizChart" height="200"></canvas>
        {% else %}
        <div class="text-center text-muted py-4">
          <i class="bi bi-patch-question" style="font-size:3rem; opacity:0.3"></i>
          <p class="mt-2">Take some quizzes to see your performance</p>
        </div>
        {% endif %}
      </div>
    </div>
  </div>

  <!-- Quiz History -->
  {% if quizzes %}
  <div class="col-12">
    <div class="card">
      <div class="card-header bg-white"><h6 class="mb-0"><i class="bi bi-clock-history text-primary me-2"></i>Quiz History</h6></div>
      <div class="card-body p-0">
        <div class="table-responsive">
          <table class="table table-hover mb-0">
            <thead class="table-light"><tr><th>Subject</th><th>Topic</th><th>Difficulty</th><th>Score</th><th>%</th><th>Date</th></tr></thead>
            <tbody>
              {% for q in quizzes %}
              <tr>
                <td>{{ q.subject or '—' }}</td>
                <td>{{ q.topic or '—' }}</td>
                <td><span class="badge bg-{{ 'success' if q.difficulty == 'easy' else 'warning text-dark' if q.difficulty == 'medium' else 'danger' }}">{{ q.difficulty }}</span></td>
                <td>{{ q.score }}/{{ q.total }}</td>
                <td>
                  <div class="d-flex align-items-center gap-2">
                    <div class="progress flex-grow-1" style="height:6px; width:60px">
                      <div class="progress-bar {{ 'bg-success' if q.percentage >= 80 else 'bg-warning' if q.percentage >= 60 else 'bg-danger' }}" style="width:{{ q.percentage }}%"></div>
                    </div>
                    <small>{{ q.percentage }}%</small>
                  </div>
                </td>
                <td><small class="text-muted">{{ q.created_at[:10] }}</small></td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
  {% endif %}

  <!-- Avg Quiz Scores -->
  {% if progress.avg_quiz_scores %}
  <div class="col-12">
    <div class="card">
      <div class="card-header bg-white"><h6 class="mb-0"><i class="bi bi-star text-warning me-2"></i>Average Quiz Scores by Subject</h6></div>
      <div class="card-body">
        {% for subj, avg in progress.avg_quiz_scores.items() %}
        <div class="mb-2">
          <div class="d-flex justify-content-between mb-1">
            <span>{{ subj }}</span><strong>{{ avg }}%</strong>
          </div>
          <div class="progress"><div class="progress-bar {{ 'bg-success' if avg >= 80 else 'bg-warning' if avg >= 60 else 'bg-danger' }}" style="width:{{ avg }}%"></div></div>
        </div>
        {% endfor %}
      </div>
    </div>
  </div>
  {% endif %}
</div>
{% endblock %}
{% block scripts %}
<script>
{% if progress.subject_data %}
const subjLabels = {{ progress.subject_data.keys()|list|tojson }};
const subjDone = {{ progress.subject_data.values()|map(attribute='completed')|list|tojson }};
const subjTotal = {{ progress.subject_data.values()|map(attribute='total')|list|tojson }};
new Chart(document.getElementById('subjectChart'), {
  type: 'bar',
  data: { labels: subjLabels, datasets: [
    { label: 'Completed', data: subjDone, backgroundColor: '#40c057' },
    { label: 'Pending', data: subjTotal.map((t,i) => t - subjDone[i]), backgroundColor: '#dee2e6' }
  ]},
  options: { responsive: true, scales: { x: { stacked: true }, y: { stacked: true, beginAtZero: true } } }
});
{% endif %}

{% if quizzes %}
const quizDates = {{ quizzes|map(attribute='created_at')|list|tojson }}.map(d => d.slice(0,10));
const quizPcts = {{ quizzes|map(attribute='percentage')|list|tojson }};
new Chart(document.getElementById('quizChart'), {
  type: 'line',
  data: { labels: quizDates, datasets: [{
    label: 'Score %', data: quizPcts, borderColor: '#4361ee', backgroundColor: 'rgba(67,97,238,0.1)', fill: true, tension: 0.4
  }]},
  options: { responsive: true, scales: { y: { beginAtZero: true, max: 100 } } }
});
{% endif %}
</script>
{% endblock %}
""")

ABOUT_TEMPLATE = BASE_TEMPLATE.replace(
    "{% block title %}Smart Study Generator{% endblock %}", "{% block title %}About{% endblock %}"
).replace(
    "{% block page_title %}StudyMate AI{% endblock %}", "{% block page_title %}About StudyMate AI{% endblock %}"
).replace("{% block content %}{% endblock %}", """{% block content %}
<div class="row g-4">
  <div class="col-12">
    <div class="card" style="background: linear-gradient(135deg,#4361ee,#7c5cd8); color:#fff;">
      <div class="card-body p-5 text-center">
        <i class="bi bi-mortarboard-fill" style="font-size:4rem"></i>
        <h2 class="mt-3">StudyMate AI</h2>
        <p class="lead mb-2">Personalized Smart Study Generator</p>
        <p class="opacity-75">An intelligent multi-agent AI study companion for college students</p>
        <span class="badge bg-warning text-dark me-2">Hackathon Project</span>
        <span class="badge bg-light text-dark">Educational Use Only</span>
      </div>
    </div>
  </div>

  <!-- Problem & Solution -->
  <div class="col-md-6">
    <div class="card h-100">
      <div class="card-header bg-danger text-white"><h6 class="mb-0"><i class="bi bi-exclamation-triangle me-2"></i>Problem Statement</h6></div>
      <div class="card-body">
        <ul>
          <li>Students struggle with large syllabi and lack personalized guidance</li>
          <li>No AI tool that integrates study notes, quizzes, and exam planning together</li>
          <li>Difficult to identify weak topics and create effective revision strategies</li>
          <li>Time management during exam preparation is a major challenge</li>
          <li>Students lack insight into which topics are most exam-relevant</li>
        </ul>
      </div>
    </div>
  </div>
  <div class="col-md-6">
    <div class="card h-100">
      <div class="card-header bg-success text-white"><h6 class="mb-0"><i class="bi bi-lightbulb me-2"></i>Our Solution</h6></div>
      <div class="card-body">
        <ul>
          <li>4 specialized AI agents that work together intelligently</li>
          <li>RAG-based Q&A using students' own study materials</li>
          <li>Personalized study plans based on exam dates and prep level</li>
          <li>Adaptive quizzes that identify and address weak topics</li>
          <li>Question paper analysis for informed exam preparation</li>
        </ul>
      </div>
    </div>
  </div>

  <!-- Agents -->
  <div class="col-12">
    <div class="card">
      <div class="card-header bg-primary text-white"><h6 class="mb-0"><i class="bi bi-robot me-2"></i>AI Agents</h6></div>
      <div class="card-body">
        <div class="row g-3">
          <div class="col-md-3">
            <div class="p-3 rounded-3 bg-primary bg-opacity-10">
              <h6><i class="bi bi-book text-primary me-2"></i>Study Knowledge Agent</h6>
              <p class="small text-muted mb-0">Answers academic questions using Grok AI and your uploaded study materials (RAG). Generates exam-ready answers for 2, 5, 8, and 10-mark questions.</p>
            </div>
          </div>
          <div class="col-md-3">
            <div class="p-3 rounded-3 bg-success bg-opacity-10">
              <h6><i class="bi bi-calendar3 text-success me-2"></i>Study Planner Agent</h6>
              <p class="small text-muted mb-0">Creates personalized daily and weekly study timetables, subject priorities, and revision schedules based on your profile and exam date.</p>
            </div>
          </div>
          <div class="col-md-3">
            <div class="p-3 rounded-3 bg-warning bg-opacity-10">
              <h6><i class="bi bi-patch-question text-warning me-2"></i>Quiz & Assessment Agent</h6>
              <p class="small text-muted mb-0">Generates MCQ quizzes from uploaded materials, evaluates answers, identifies weak topics, and recommends targeted revision.</p>
            </div>
          </div>
          <div class="col-md-3">
            <div class="p-3 rounded-3 bg-danger bg-opacity-10">
              <h6><i class="bi bi-file-earmark-bar-graph text-danger me-2"></i>Paper Analyzer Agent</h6>
              <p class="small text-muted mb-0">Analyzes previous-year question papers to identify topic frequencies, repeated questions, and generates practice questions based on patterns.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Tech Stack -->
  <div class="col-md-6">
    <div class="card">
      <div class="card-header bg-dark text-white"><h6 class="mb-0"><i class="bi bi-stack me-2"></i>Technology Stack</h6></div>
      <div class="card-body">
        <table class="table table-sm">
          <tbody>
            <tr><td><strong>Backend</strong></td><td>Python, Flask</td></tr>
            <tr><td><strong>AI LLM</strong></td><td>Grok API (xAI)</td></tr>
            <tr><td><strong>RAG</strong></td><td>FAISS + Sentence Transformers</td></tr>
            <tr><td><strong>PDF Processing</strong></td><td>PyMuPDF (fitz)</td></tr>
            <tr><td><strong>Database</strong></td><td>SQLite</td></tr>
            <tr><td><strong>Frontend</strong></td><td>Bootstrap 5, Chart.js, JS</td></tr>
            <tr><td><strong>Icons</strong></td><td>Bootstrap Icons</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- How it works -->
  <div class="col-md-6">
    <div class="card">
      <div class="card-header bg-info text-white"><h6 class="mb-0"><i class="bi bi-diagram-3 me-2"></i>How It Works</h6></div>
      <div class="card-body">
        <ol>
          <li><strong>Upload</strong> your PDF notes or paste text</li>
          <li>Documents are <strong>chunked and indexed</strong> in FAISS vector store</li>
          <li>Ask a question → <strong>relevant chunks retrieved</strong> (RAG)</li>
          <li>Grok AI generates a <strong>context-aware answer</strong></li>
          <li>Study plan generated from <strong>your profile + exam date</strong></li>
          <li>Quizzes adapt to your <strong>performance and weak topics</strong></li>
          <li>PYQ analysis reveals <strong>high-frequency topics</strong></li>
          <li>Progress tracked in <strong>SQLite across sessions</strong></li>
        </ol>
      </div>
    </div>
  </div>

  <!-- Disclaimer -->
  <div class="col-12">
    <div class="alert alert-warning">
      <h6><i class="bi bi-shield-exclamation me-2"></i>Disclaimer</h6>
      <p class="mb-0">StudyMate AI is an educational tool designed to assist with study preparation. AI-generated content should be verified against authoritative textbooks and course materials. Question paper analysis identifies historical patterns and does not predict or guarantee future examination questions. This application is intended for educational purposes only.</p>
    </div>
  </div>
</div>
{% endblock %}
""")

# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    print("=" * 60)
    print("  StudyMate AI – Personalized Smart Study Generator")
    print("=" * 60)
    print(f"  PyMuPDF available:           {PYMUPDF_AVAILABLE}")
    print(f"  Sentence Transformers:       {SENTENCE_TRANSFORMERS_AVAILABLE}")
    print(f"  FAISS available:             {FAISS_AVAILABLE}")
    print(f"  OpenAI SDK available:        {OPENAI_SDK_AVAILABLE}")
    print(f"  Grok API Key configured:     {'Yes' if GROK_API_KEY else 'NO – set GROK_API_KEY in .env'}")
    print(f"  Grok Model:                  {GROK_MODEL}")
    print("=" * 60)
    print("  Starting server at http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)
