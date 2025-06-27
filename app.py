import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from pytube import YouTube
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    CouldNotRetrieveTranscript
)
from dotenv import load_dotenv

# ✅ Load environment variables from .env
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGING_FACE_API_KEY")

# ✅ Function to extract YouTube transcript
def get_youtube_transcript(url):
    try:
        video_id = YouTube(url).video_id
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript(["en"])
        transcript_data = transcript.fetch()
        text = " ".join([item.text for item in transcript_data])
        st.success("✅ Transcript extracted successfully!")
        return text
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        st.error("No transcript found for this video.")
    except VideoUnavailable:
        st.error("This video is unavailable.")
    except CouldNotRetrieveTranscript:
        st.error("Transcript may not be available in your region.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    return ""

# ✅ Function to save transcript to file
def save_transcript_to_file(text, filename="transcript.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

# ✅ Streamlit UI
st.title("🎓 AI-Powered Tutor (Hugging Face Version)")
st.write("Paste a YouTube video URL and ask questions from its transcript.")

video_url = st.text_input("📺 Enter YouTube Video URL")

if st.button("🔍 Process Video"):
    if video_url:
        with st.spinner("Extracting transcript..."):
            transcript_text = get_youtube_transcript(video_url)

        if transcript_text:
            save_transcript_to_file(transcript_text)

            # Load and split the document
            loader = TextLoader("transcript.txt", encoding="utf-8")
            documents = loader.load()

            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.split_documents(documents)

            # Use Hugging Face embeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(docs, embeddings)
            retriever = vectorstore.as_retriever()

            # ✅ Use a local Hugging Face pipeline (no API required)
            pipe = pipeline("text2text-generation", model="google/flan-t5-small")
            llm = HuggingFacePipeline(pipeline=pipe)

            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
            st.session_state.qa_chain = qa_chain
            st.success("✅ Transcript processed! Ask your questions below.")
    else:
        st.warning("Please enter a valid YouTube URL.")

# ✅ Question/Answer section
if "qa_chain" in st.session_state:
    user_question = st.text_input("💬 Ask a question from the transcript")
    if user_question:
        with st.spinner("Generating answer..."):
            prompt = f"Answer this question based on the transcript:\n{user_question}"
            answer = st.session_state.qa_chain.run(prompt)
        st.success("✅ Answer generated!")
        st.write("**Answer:**", answer)
