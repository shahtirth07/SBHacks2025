import os
import openai
import anthropic
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import pdfplumber
import re

# Load environment variables
load_dotenv()

# Set API keys
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone using Pinecone class
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Specify the Pinecone index name
index_name = "demo3"

# Check if the index exists; if not, create it
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

# Initialize Anthropic
anthropic_client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Function to clean and chunk PDF text
def chunk_text(text, chunk_size=500):
    """
    Split the text into chunks of approximately 'chunk_size' tokens.
    """
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Function to process the PDF
def process_pdf(pdf_path):
    """
    Extract text from the PDF file and chunk it.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages])
    return chunk_text(text)

# Function to generate embeddings and store them in Pinecone
def index_pdf_chunks(chunks):
    """
    Generate embeddings for chunks and store them in Pinecone.
    """
    for i, chunk in enumerate(chunks):
        embedding = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=[chunk]
        )['data'][0]['embedding']
        metadata = {"text": chunk}
        index.upsert([{"id": f"chunk-{i}", "values": embedding, "metadata": metadata}])

# Function to query Pinecone and retrieve relevant chunks
def retrieve_relevant_chunks(query, top_k=5):
    """
    Query Pinecone to retrieve the most relevant chunks for the input query.
    """
    query_embedding = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=[query]
    )['data'][0]['embedding']
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return [match['metadata']['text'] for match in results['matches']]

# Function to generate detailed notes for a chapter
def generate_detailed_notes(retrieved_chunks, chapter_number):
    """
    Use Anthropic Claude to generate detailed notes for a chapter.
    """
    context = "\n".join(retrieved_chunks)
    prompt = f"""
    {anthropic.HUMAN_PROMPT}
    Using the following context, create detailed study notes for Chapter {chapter_number}. The notes should include:
    
    1. A detailed explanation of key concepts, facts, and figures in the chapter.
    2. Relevant examples or applications of the concepts.
    3. Key definitions, terminologies, and their explanations.
    4. A suggestion for a YouTube video reference for further learning (if available).
    
    Ensure the notes are comprehensive and easy to understand.
    
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

# Main function
if __name__ == "__main__":
    # Dynamically fetch the PDF file path
    uploaded_file_name = input("Enter the uploaded file name (from ./uploads/ directory): ").strip()
    pdf_path = f"./uploads/{uploaded_file_name}"  # Construct the dynamic path
    
    try:
        print("Processing PDF and indexing chunks...")
        
        # Process PDF and index chunks
        chunks = process_pdf(pdf_path)
        index_pdf_chunks(chunks)
        
        print("PDF content indexed successfully.")
        
        # Generate notes for each chapter
        chapter_numbers = [1, 2, 3, 4, 5]  # Example chapter numbers
        for chapter in chapter_numbers:
            print(f"Generating notes for Chapter {chapter}...")
            
            # Customize query for each chapter
            query = f"Detailed notes for Chapter {chapter}"
            retrieved_chunks = retrieve_relevant_chunks(query, top_k=5)
            notes = generate_detailed_notes(retrieved_chunks, chapter)
            
            # Save notes to a chapter-specific file
            output_file = f"./uploads/Chapter_{chapter}_notes.txt"
            with open(output_file, "w", encoding="utf-8") as f:  # Ensure UTF-8 encoding
                f.write(notes)
            
            print(f"Notes for Chapter {chapter} saved to {output_file}.")
    
    except FileNotFoundError as e:
        print(str(e))
    except Exception as e:
        print(f"An error occurred: {str(e)}")
