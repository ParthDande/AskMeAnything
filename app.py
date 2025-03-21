from flask import Flask, request, jsonify, render_template
import os
import pdfplumber
import faiss
import pickle
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import requests

nltk.download('punkt')
load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

chunk_size = 3

together_api_key = os.getenv("TOGETHER_API_KEY")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text(x_tolerance=1, y_tolerance=1) + "\n"

    with open("extracted_text.txt", "w", encoding="utf-8") as debug_file:
        debug_file.write(text)
    return text

def clean_text(text):
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line or len(line) < 25:
            continue
        if '@' in line or 'google.com' in line or 'gmail' in line:
            continue
        if any(char.isdigit() for char in line) and len(line.split()) < 4:
            continue
        cleaned.append(line)
    return " ".join(cleaned)

def chunk_text(text, size=chunk_size):
    sentences = sent_tokenize(text)
    chunks = [". ".join(sentences[i:i+size]).strip() for i in range(0, len(sentences), size)]
    return [chunk for chunk in chunks if len(chunk.split()) > 10]

def create_faiss_index(chunks):
    embeddings = embedder.encode(chunks).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    file = request.files['pdf']
    if not file:
        return "No file uploaded.", 400

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    raw_text = extract_text_from_pdf(path)
    cleaned_text = clean_text(raw_text)
    chunks = chunk_text(cleaned_text)

    index, chunk_data = create_faiss_index(chunks)

    faiss.write_index(index, "my_vector.index")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunk_data, f)

    return "PDF uploaded and vector DB created. You can now ask questions."

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    query = data.get("query", "")
    query_embedding = embedder.encode([query]).astype("float32")

    index = faiss.read_index("my_vector.index")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    D, I = index.search(query_embedding, k=1)
    context = " ".join([chunks[i] for i in I[0]])
    prompt = f"""
You are a helpful AI assistant answering user questions based ONLY on the provided context.

Guidelines:
- ONLY use information found in the context.
- If the answer is NOT present in the context, reply: "I’m not sure based on the given information."
- Do NOT make assumptions or add extra knowledge.
- Do NOT list facts unless the question specifically asks for them.
- Your goal is to directly answer what is asked, no more, no less.
- You have to give the response as if you're talking like a human to another human.
- Keep your response short and simple.
- If the answer is just one word, reply with that one word only.

Context:
{context}

Question:
{query}

Answer:
"""




    headers = {
        "Authorization": f"Bearer {together_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",
        "prompt": prompt,
        "max_tokens": 300,
        "temperature": 0.7,
        "top_p": 0.9,
        "stop": ["User:", "Assistant:"]
    }

    response = requests.post("https://api.together.xyz/v1/completions", headers=headers, json=payload)
    result = response.json()

    answer = result.get("choices", [{}])[0].get("text", "Sorry, I couldn't find an answer.")
    cleaned = answer.strip()

# Remove "Question:" and "Answer:" if the model includes them
    if "Answer:" in cleaned:
        cleaned = cleaned.split("Answer:")[-1].strip()

    # Split into lines and make bullets
    lines = cleaned.split('\n')
    bulleted = "\n".join(f"- {line.strip()}" for line in lines if line.strip())

    return jsonify({'answer': bulleted})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000,debug=False)
