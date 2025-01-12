from flask import Flask, render_template, request, jsonify, send_from_directory
from e_learning import generate_e_learning_notes  # Import the necessary function
import os
import traceback
from dotenv import load_dotenv
from summary import process_pdf, retrieve_relevant_chunks, generate_detailed_notes
import openai
import anthropic
from chatbot import ask_claude
from e_learning import extract_pdf_text , generate_detailed_notes_with_anthropic 

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Set up directories
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
anthropic_client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))

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
        # Call the `ask_claude` function from chatbot.py
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
        chunks = process_pdf(file_path)  # Process PDF into chunks
        retrieved_chunks = retrieve_relevant_chunks("Summarize the document", top_k=5)  # Retrieve relevant chunks from Pinecone
        summary = generate_detailed_notes(retrieved_chunks)  # Generate detailed notes
        return render_template('index.html', message="Summary generated successfully!", summary=summary, uploaded_file=uploaded_file, dashboard_data={"images": [], "tables": []})
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
        # Generate detailed e-learning notes for each chapter from e_learning.py
        chapter_notes = generate_e_learning_notes(file_path)
        
        if chapter_notes:
            return render_template('index.html', message="E-learning content generated successfully!", chapter_notes=chapter_notes, uploaded_file=uploaded_file, dashboard_data={"images": [], "tables": []})
        else:
            return render_template('index.html', error="Failed to generate e-learning content.", dashboard_data={"images": [], "tables": []})
    except Exception as e:
        return render_template('index.html', error=f"Error generating e-learning content: {str(e)}", dashboard_data={"images": [], "tables": []})

def generate_e_learning_notes(pdf_path):
    """
    Extracts text from the PDF and generates e-learning notes for multiple chapters.
    Args:
        pdf_path (str): Path to the PDF file.
    Returns:
        dict: Dictionary containing notes for each chapter.
    """
    try:
        # Extract text from PDF
        text = extract_pdf_text(pdf_path)
        if not text:
            raise ValueError("Failed to extract text from PDF.")

        chapter_notes = {}
        total_chapters = 3  # Modify as needed for your PDF structure
        
        # Split the text into chunks for each chapter
        chunk_size = len(text) // total_chapters  # Divide text roughly into 3 chapters
        for chapter in range(1, total_chapters + 1):
            chapter_text = text[(chapter - 1) * chunk_size : chapter * chunk_size]
            notes = generate_detailed_notes_with_anthropic(chapter_text, chapter)
            chapter_notes[chapter] = notes
        
        return chapter_notes
    except Exception as e:
        print(f"Error generating e-learning notes: {e}")
        return None



@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
