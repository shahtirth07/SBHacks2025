import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import anthropic
import pdfplumber

# Load environment variables
load_dotenv()

Client = OpenAI()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")

pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "nervous"
model_name = "text-embedding-3-small"
max_tokens = 8191
dimensions = 1536

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

anthropic_client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))

def chunk_text(text, chunk_size=500):
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def process_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages])
    return chunk_text(text)

def index_pdf_chunks(chunks):
    for i, chunk in enumerate(chunks):
        response = Client.embeddings.create(
            model="text-embedding-ada-002",
            input=chunk
        )
        embedding = response.data[0].embedding
        metadata = {"text": chunk}
        index.upsert([{"id": f"chunk-{i}", "values": embedding, "metadata": metadata}])

def retrieve_relevant_chunks(query, top_k=5):
    response = Client.embeddings.create(
        model="text-embedding-ada-002",
        input=query
    )
    query_embedding = response.data[0].embedding
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return [match['metadata']['text'] for match in results['matches']]

def generate_detailed_notes(retrieved_chunks):
    notes = []
    for chunk in retrieved_chunks:
        # Customize this logic based on your requirement to generate more detailed notes
        note = f"Details: \n{chunk}\n"
        notes.append(note)
    return "\n".join(notes)
