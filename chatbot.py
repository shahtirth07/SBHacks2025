import os
import openai
import anthropic
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["ANTHROPIC_API_KEY"] = os.getenv("Anthropic_API_KEY")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Specify index name
index_name = "nervous"

# Check if the index exists; if not, create it
if index_name not in pinecone_client.list_indexes().names():
    pinecone_client.create_index(
        name=index_name,
        dimension=1536,  # Dimension of OpenAI embeddings
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",  # Specify cloud provider
            region="us-east-1"  # Specify region
        )
    )

# Access the index
index = pinecone_client.Index(index_name)

# Initialize Anthropic
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Function to generate query embedding
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(
        model=model,
        input=text
    )
    return response['data'][0]['embedding']

# Function to query Pinecone and send data to Claude
def ask_claude(user_input):
    try:
        # Generate embedding for the question
        query_embedding = get_embedding(user_input)

        # Query Pinecone for relevant context
        query_result = index.query(
            vector=query_embedding,
            top_k=5,  # Retrieve top 5 relevant matches
            include_metadata=True
        )

        # Extract context from Pinecone results
        context = "\n".join([item['metadata']['text_representation'] for item in query_result['matches']])

        # Combine the context with user input
        prompt = anthropic.HUMAN_PROMPT + f"Context: {context}\n\nUser Question: {user_input}" + anthropic.AI_PROMPT

        # Send the prompt to Claude
        message = anthropic_client.completions.create(
            model="claude-2",
            max_tokens_to_sample=1000,
            temperature=0,
            prompt=prompt
        )

        # Return Claude's response
        return message.completion

    except Exception as e:
        return f"Error: {str(e)}"

# Example Usage
if __name__ == "__main__":
    user_question = input("Enter your question: ")
    response = ask_claude(user_question)
    print("\nClaude's Response:")
    print(response)
