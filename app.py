from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import traceback
import pdfplumber
import re
import openai
import anthropic
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from summary import retrieve_relevant_chunks
from e_learning import generate_detailed_notes
from chatbot import ask_claude

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Set up directories
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
anthropic_client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Pinecone initialization
index_name = "nervous"
if index_name not in [idx.name for idx in pinecone_client.list_indexes()]:
    pinecone_client.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
index = pinecone_client.Index(index_name)

# Utility functions
def chunk_text(text, chunk_size=500):
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def process_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    if not text.strip():
        raise ValueError("The uploaded PDF is empty or contains no extractable text.")
    return chunk_text(text)

def index_pdf_chunks(chunks):
    for i, chunk in enumerate(chunks):
        embedding = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=[chunk]
        )['data'][0]['embedding']
        metadata = {"text": chunk}
        index.upsert([{"id": f"chunk-{i}", "values": embedding, "metadata": metadata}])

def beautify_text(text):
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    return '\n'.join(f"- {line}" for line in cleaned_lines)

# Routes
@app.route('/')
def index():
    dashboard_data = {"images": [], "tables": []}
    return render_template('index.html', dashboard_data=dashboard_data)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message')
        if not user_message:
            return jsonify({'response': 'No message provided.'}), 400
        bot_response = ask_claude(user_message)
        return jsonify({'response': bot_response})
    except Exception as e:
        return jsonify({'response': f'Error: {str(e)}'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error="No file part provided.", dashboard_data={"images": [], "tables": []})
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No file selected.", dashboard_data={"images": [], "tables": []})
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    return render_template('index.html', message="File uploaded successfully!", uploaded_file=file.filename, dashboard_data={"images": [], "tables": []})

@app.route('/generate_summary', methods=['POST'])
def generate_summary_route():
    uploaded_file = request.form.get('uploaded_file')
    if not uploaded_file:
        return render_template('index.html', error="No uploaded file found for summary generation.", dashboard_data={"images": [], "tables": []})
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file)
    try:
        chunks = process_pdf(file_path)
        index_pdf_chunks(chunks)
        retrieved_chunks = retrieve_relevant_chunks("Summarize the document", top_k=5)
        summary = beautify_text("\n".join(retrieved_chunks))
        return render_template('index.html', summary=summary, uploaded_file=uploaded_file, dashboard_data={"images": [], "tables": []})
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error generating summary: {e}")
        print(error_details)
        return render_template('index.html', error=f"Error generating summary: {str(e)}", dashboard_data={"images": [], "tables": []})

@app.route('/e_learning', methods=['POST'])
def e_learning():
    uploaded_file = request.form.get('uploaded_file')
    if not uploaded_file:
        return render_template('index.html', error="No uploaded file found for e-learning.", dashboard_data={"images": [], "tables": []})
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file)
    try:
        chapter_notes = {}
        for chapter in range(1, 3):
            query = f"Detailed notes for Chapter {chapter}"
            retrieved_chunks = retrieve_relevant_chunks(query, top_k=5)
            notes = generate_detailed_notes(retrieved_chunks, chapter)
            chapter_notes[chapter] = notes
        return render_template('index.html', chapter_notes=chapter_notes, uploaded_file=uploaded_file, dashboard_data={"images": [], "tables": []})
    except Exception as e:
        return render_template('index.html', error=f"Error generating e-learning content: {str(e)}", dashboard_data={"images": [], "tables": []})

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
