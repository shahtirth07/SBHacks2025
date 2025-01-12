import os
import openai
import anthropic
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import pdfplumber
import re
from openai import OpenAI

# Load environment variables
load_dotenv()

Client = OpenAI()
openai.api_key = os.getenv("OPENAI_API_KEY")

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

def generate_summary(retrieved_chunks):
    context = "\n".join(retrieved_chunks)
    prompt = f"""
    {anthropic.HUMAN_PROMPT}
    Using the following context, create a structured summary of the document. The summary should include:
    1. Title of the document.
    2. A brief introduction (2-3 sentences).
    3. Detailed chapter summaries with bullet points for key concepts covered in each chapter.
    4. A section on "Key Topics to Study" that highlights important points, terminologies, or concepts from the document.
    5. A conclusion summarizing the document's purpose and main takeaways.
    Ensure the summary is clear, concise, and easy to read with bullet points wherever applicable.
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

if __name__ == "__main__":
    pdf_path = r"/Users/baddalagovardhanreddy/Desktop/Test/Data/NERVOUS.pdf"
    
    try:
        print("Processing PDF and indexing chunks...")
        chunks = process_pdf(pdf_path)
        index_pdf_chunks(chunks)
        
        print("PDF content indexed successfully.")
        
        user_query = "Generate a detailed summary of the PDF."
        retrieved_chunks = retrieve_relevant_chunks(user_query, top_k=5)
        summary = generate_summary(retrieved_chunks)
        
        output_file = "NERVOUS_System_Summary.txt"
        with open(output_file, "w") as f:
            f.write(summary)
        
        print(f"Summary saved to {output_file}")
        print("\nGenerated Summary:\n")
        print(summary)
    
    except FileNotFoundError as e:
        print(str(e))
    except Exception as e:
        print(f"An error occurred: {str(e)}")
