from flask import Flask, render_template, request, jsonify
import os
from chatbot import ask_claude
import pdfplumber
import openai
import anthropic
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import re
from summary import summarize_pdf
from e_learning import generate_detailed_notes

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

def chunk_text(text, chunk_size=500):
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def process_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages])
    return chunk_text(text)

def index_pdf_chunks(chunks):
    for i, chunk in enumerate(chunks):
        embedding = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=[chunk]
        )['data'][0]['embedding']
        metadata = {"text": chunk}
        index.upsert([{"id": f"chunk-{i}", "values": embedding, "metadata": metadata}])

def retrieve_relevant_chunks(query, top_k=5):
    query_embedding = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=[query]
    )['data'][0]['embedding']
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return [match['metadata']['text'] for match in results['matches']]

def generate_summary(retrieved_chunks):
    context = "\n".join(retrieved_chunks)
    prompt = f"""
    {anthropic.HUMAN_PROMPT}
    Using the following context, create a structured summary of the document:
    Context:
    {context}
    {anthropic.AI_PROMPT}
    """
    response = anthropic_client.completions.create(
        model="claude-2",
        max_tokens_to_sample=2000,
        temperature=0,
        prompt=prompt
    )
    return response.completion

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message')
        if not user_message:
            return jsonify({'response': 'No message provided.'}), 400

        # Call the `ask_claude` function from chatbot.py
        bot_response = ask_claude(user_message)
        return jsonify({'response': bot_response})
    except Exception as e:
        return jsonify({'response': f'Error: {str(e)}'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error="No file part provided.")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No file selected.")

    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    return render_template('index.html', message="File uploaded successfully!", uploaded_file=file.filename)

@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    uploaded_file = request.form.get('uploaded_file')
    if not uploaded_file:
        return render_template('index.html', error="No uploaded file found for summary generation.")
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file)
    try:
        summary = summarize_pdf(file_path)
        return render_template('index.html', summary=summary, uploaded_file=uploaded_file)
    except Exception as e:
        return render_template('index.html', error=f"Error generating summary: {str(e)}")

@app.route('/e_learning', methods=['POST'])
def e_learning():
    uploaded_file = request.form.get('uploaded_file')
    if not uploaded_file:
        return render_template('index.html', error="No uploaded file found for e-learning.")
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file)
    try:
        # Example: Generate notes for chapters 1-3
        chapter_notes = {}
        for chapter in range(1, 3):
            query = f"Detailed notes for Chapter {chapter}"
            retrieved_chunks = retrieve_relevant_chunks(query, top_k=5)
            notes = generate_detailed_notes(retrieved_chunks, chapter)
            chapter_notes[chapter] = notes

        return render_template('index.html', chapter_notes=chapter_notes, uploaded_file=uploaded_file)
    except Exception as e:
        return render_template('index.html', error=f"Error generating e-learning content: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
