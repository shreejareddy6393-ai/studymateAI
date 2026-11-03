# StudyMate AI – Personalized Smart Study Generator

> An intelligent multi-agent AI web application that helps college students understand difficult subjects, prepare for examinations, generate personalized study plans, analyze previous-year question papers, and track academic progress.

---

## Features

| Feature | Description |
|---|---|
| **Study Knowledge Agent** | Answers academic questions using Grok AI + RAG on your own notes |
| **Study Planner Agent** | Generates personalized daily/weekly study timetables |
| **AI Quiz Agent** | Creates adaptive MCQ quizzes with scoring and weak-topic analysis |
| **Paper Analyzer Agent** | Analyzes previous-year papers for topic frequency and patterns |
| **RAG System** | FAISS + Sentence Transformers for document-aware Q&A |
| **Progress Tracking** | SQLite-backed subject and quiz performance tracking |

---

## Technology Stack

- **Backend:** Python, Flask
- **AI LLM:** Grok API (xAI)
- **RAG:** FAISS + Sentence Transformers (`all-MiniLM-L6-v2`)
- **PDF Processing:** PyMuPDF (fitz)
- **Database:** SQLite
- **Frontend:** Bootstrap 5, Chart.js, Bootstrap Icons

---

## Prerequisites

- Python 3.9 or higher
- A Grok API key from [https://console.x.ai/](https://console.x.ai/)
- ~2 GB disk space (for Sentence Transformers model download on first run)
- Internet connection for first-time model download

---

## Installation

### Step 1 – Clone or download the project

```bash
# If using git
git clone <your-repo-url>
cd studymateAI

# Or just place app.py, requirements.txt, and .env.example in a folder
```

### Step 2 – Create a Python virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 – Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first install downloads PyTorch and Sentence Transformers (~1-2 GB). This is a one-time download.

> **Tip:** If FAISS installation fails on Windows, try:
> ```bash
> pip install faiss-cpu --no-deps
> pip install numpy
> ```

### Step 4 – Configure environment variables

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Edit `.env` and fill in your values:

```env
GROK_API_KEY=xai-your-actual-key-here
GROK_MODEL=grok-3-mini
GROK_BASE_URL=https://api.x.ai/v1
SECRET_KEY=any-long-random-string-here
```

### Step 5 – Run the application

```bash
python app.py
```

The application will:
1. Initialize the SQLite database automatically
2. Print a startup summary showing which libraries are available
3. Start the server at **http://127.0.0.1:5000**

---

## How to Get a Grok API Key

1. Visit [https://console.x.ai/](https://console.x.ai/)
2. Create an account or sign in with X (Twitter)
3. Go to **API Keys** section
4. Click **Create API Key**
5. Copy the key and paste it as `GROK_API_KEY` in your `.env` file

Available models:
- `grok-3-mini` – Fast, cost-effective (recommended)
- `grok-3` – More capable, higher cost
- `grok-2` – Previous generation

---

## Application Pages

| Page | URL | Description |
|---|---|---|
| Dashboard | `/` | Welcome, stats, activity |
| Study Assistant | `/study-assistant` | AI Q&A with document context |
| Study Material | `/study-material` | Upload PDFs/notes |
| Study Planner | `/study-planner` | Generate study timetables |
| Paper Analyzer | `/question-analyzer` | Analyze PYQs |
| AI Quiz | `/quiz` | Take adaptive quizzes |
| Progress | `/progress` | Charts and performance |
| About | `/about` | Project info |

---

## Using the Four AI Agents

### 1. Study Knowledge Agent (Study Assistant page)

1. Go to **Study Assistant**
2. Optionally set Subject and Topic
3. Select answer type (2-mark, 5-mark, etc.)
4. Type your question and press **Ctrl+Enter** or click Send
5. Upload study materials first for document-aware answers

### 2. Study Planner Agent (Study Planner page)

1. Go to **Study Planner**
2. Fill in your profile (course, semester, subjects, exam date)
3. Set study hours per day and preparation level
4. List your difficult subjects/topics
5. Click **Generate My Study Plan**
6. Mark tasks as complete as you finish them

### 3. AI Quiz Agent (AI Quiz page)

1. Go to **AI Quiz**
2. Enter subject and topic
3. Select difficulty and number of questions
4. Click **Generate Quiz**
5. Select answers for each question
6. Click **Submit Quiz** to see score and explanations

### 4. Question Paper Analyzer (Paper Analyzer page)

1. Go to **Paper Analyzer**
2. Enter the subject name
3. Upload a PDF question paper OR paste the text
4. Click **Analyze Paper**
5. View topic frequency chart and recommendations
6. Click **Generate Practice Questions**

---

## Uploading Study Materials

1. Go to **Study Material**
2. **PDF upload:** Click the upload zone and select a PDF (max 16 MB)
3. **Text notes:** Paste notes in the text area and click Add Notes
4. Processed documents are automatically used as context in the Study Assistant
5. Click the document summary button (📄) to get an AI summary

---

## Troubleshooting

### "GROK_API_KEY is not configured"
- Make sure `.env` exists (not `.env.example`)
- Check the key is correctly copied without extra spaces
- Verify the key is active at [https://console.x.ai/](https://console.x.ai/)

### "PyMuPDF not available" / PDF extraction fails
```bash
pip install PyMuPDF
```

### "Sentence Transformers not available" / RAG not working
```bash
pip install sentence-transformers
```
The model downloads ~90 MB on first use. Ensure internet connection.

### "FAISS not available" — keyword search fallback is used
```bash
pip install faiss-cpu
```

### Port 5000 already in use
```bash
# Windows – find and kill the process
netstat -ano | findstr :5000
taskkill /PID <pid> /F
```

### Scanned PDF — no text extracted
Scanned PDFs require OCR. Convert to text-based PDF using Adobe Acrobat, Microsoft Lens, or Google Drive (open → Download as .docx → copy text).

### Quiz returns "Could not parse questions"
- Ensure the subject/topic is specific enough
- Try a simpler topic name
- Check your API key has sufficient credits

---

## Project Structure

```
studymateAI/
├── app.py              # Complete Flask application (single file)
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .env                # Your actual environment variables (DO NOT COMMIT)
├── README.md           # This file
├── studymate.db        # SQLite database (auto-created on first run)
└── uploads/            # Uploaded document storage (auto-created)
```

---

## Disclaimer

StudyMate AI is an educational tool for study assistance only. AI-generated content should be verified against authoritative textbooks and course materials. Historical question paper analysis identifies patterns only and **does not predict or guarantee future examination questions**. Use responsibly.

---

*Built for AI/ML Hackathon – Educational Use Only*
