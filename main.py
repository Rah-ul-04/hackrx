from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os
import tempfile
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.text_splitter import RecursiveCharacterTextSplitter
import mimetypes
from tika import parser

app = FastAPI()

class HackrxRequest(BaseModel):
    documents: str
    questions: list

@app.post("/hackrx/run")
async def hackrx_run(req: HackrxRequest):
    # Step 1: Download file
    try:
        response = requests.get(req.documents)
        response.raise_for_status()
    except Exception as e:
        return {"error": f"Failed to download document: {e}"}

    # Step 2: Save to temp file
    suffix = get_file_suffix_from_url(req.documents)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    # Step 3: Extract text
    try:
        text = extract_text(tmp_path)
    except Exception as e:
        os.unlink(tmp_path)
        return {"error": f"Text extraction failed: {e}"}

    os.unlink(tmp_path)  # cleanup

    # Step 4: Chunk & embed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Step 5: Save FAISS index to disk
    index_dir = "./faiss_index"
    os.makedirs(index_dir, exist_ok=True)
    vectorstore.save_local(index_dir)

    return {"message": "Document embedded and FAISS index saved successfully."}

def get_file_suffix_from_url(url: str):
    guess = mimetypes.guess_extension(mimetypes.guess_type(url)[0] or '')
    return guess or ".pdf"  # default to pdf

def extract_text(path: str):
    parsed = parser.from_file(path)
    text = parsed.get("content")
    return text.strip() if text else ""
