import openai
import anthropic
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
index_name = "nervous"
model_name = "text-embedding-3-small"
max_tokens = 8191
dimensions = 1536
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pinecone_client.Index(index_name)

Client = OpenAI()

# Initialize Anthropic
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def get_embedding(text, model="text-embedding-3-small"):
    """
    Generate embeddings for the input text using OpenAI's updated API.
    """
    try:
        # Updated API call
        response = Client.embeddings.create(
            model=model,
            input=text
        )

        # Extract the embedding from the response object
        embedding = response.data[0].embedding  # Access the first embedding

        return embedding
    except openai.BadRequestError as e:
        raise ValueError(f"Error generating embedding: {str(e)}")


def ask_claude(user_input):
    """
    Query Pinecone for relevant context and send the prompt to Anthropic Claude.
    """
    try:
        # Generate embedding for the user query
        query_embedding = get_embedding(user_input)

        # Query Pinecone
        query_result = index.query(
            vector=query_embedding,
            top_k=5,  # Retrieve top 5 relevant matches
            include_metadata=True
        )

        # Extract context from Pinecone results
        context = "\n".join([item['metadata']['text_representation'] for item in query_result['matches']])

        # Prepare the prompt for Claude
        prompt = anthropic.HUMAN_PROMPT + f"Context: {context}\n\nUser Question: {user_input}" + anthropic.AI_PROMPT

        # Send prompt to Claude
        response = anthropic_client.completions.create(
            model="claude-2",
            max_tokens_to_sample=1000,
            temperature=0,
            prompt=prompt
        )

        # Access the completion text from the response object
        return response.completion

    except Exception as e:
        # Handle generic errors
        return f"Error: {str(e)}"

if __name__ == "__main__":
    print("Chatbot Ready!")
    user_question = input("Enter your question: ")
    response = ask_claude(user_question)
    print("\nClaude's Response:")
    print(response)
