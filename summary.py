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

# Initialize Anthropic
anthropic_client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))

def chunk_text(text, chunk_size=500):
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def process_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")
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
    Using the following context, create a structured summary of the document. Include:
    1. Title
    2. Introduction
    3. Chapter summaries
    4. Key topics
    5. Conclusion
    Context: {context}
    {anthropic.AI_PROMPT}
    """
    response = anthropic_client.completions.create(
        model="claude-2",
        max_tokens_to_sample=2000,
        temperature=0,
        prompt=prompt
    )
    return response.completion

def summarize_pdf(pdf_path):
    """
    Full workflow to process a PDF and generate a summary.
    """
    chunks = process_pdf(pdf_path)
    index_pdf_chunks(chunks)
    retrieved_chunks = retrieve_relevant_chunks("Generate a detailed summary of the PDF.")
    summary = generate_summary(retrieved_chunks)
    return summary
