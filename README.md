# genai-youtube-tutor
# 🎓 AI-Powered Tutor (LangChain + Hugging Face + Streamlit)

This project is an AI-powered tutor that extracts a transcript from a YouTube video and allows users to ask questions based on its content. It uses `LangChain`, `Hugging Face Transformers`, and `Streamlit` to create an interactive and intelligent learning experience.

---

## 🚀 Features

- 🎥 Fetches transcripts from English YouTube videos
- 🤖 Embeds the content using Hugging Face sentence transformers
- 🔍 Uses LangChain and FAISS for semantic search
- 🧠 Answers user questions using a local transformer model (e.g., `flan-t5-small`)
- 🌐 Simple Streamlit-based web UI

---

## 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/ai-powered-tutor.git
cd ai-powered-tutor

## (Optional) Create and activate a virtual environment

Edit
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate


## Install dependencies

Copy
Edit
pip install -r requirements.txt

## 🛠️ Running the App
Edit
streamlit run app.py
Then open your browser at http://localhost:8501.

## 📁 .env Configuration (Optional)
If using Hugging Face API (instead of local models), create a .env file:


HUGGING_FACE_API_KEY=hf_your_actual_key_here
For local usage with transformers, this file is not required.

##📚 Example Usage
Enter a YouTube video URL (with English transcript).

Click "Process Video" to extract and embed the transcript.

Ask any question like:

"What is fine-tuning?"

"Explain the main topic."

##🧰 Tech Stack
Frontend/UI: Streamlit

NLP Backend: LangChain, Hugging Face Transformers

Vector DB: FAISS

Transcript Extraction: pytube, youtube-transcript-api


