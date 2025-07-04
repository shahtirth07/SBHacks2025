# -*- coding: utf-8 -*-
"""plots.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1r3ZZzppxNJg1SIt7BdL2Jc1GVThoOfyl
"""

pip install anthropic

pip install pinecone

pip uninstall tabula -y

pip install tabula-py

pip install pdfplumber

import os
import pandas as pd
import matplotlib.pyplot as plt
from tabula import read_pdf
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Set environment variables

pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Pinecone setup
index_name = "demo3"
index = pinecone_client.Index(index_name)

# Load embedding model
embedding_model = SentenceTransformer("all-mpnet-base-v2")

# Function to generate 1536-dimensional embeddings
def get_embedding(text):
    try:
        embedding = embedding_model.encode(text).tolist()
        if len(embedding) == 768:
            return embedding + embedding
        else:
            raise ValueError("Embedding dimension mismatch.")
    except Exception as e:
        print(f"Failed to generate embedding: {str(e)}")
        return None

# Extract tables from PDF

def process_pdf_tables(pdf_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    try:
        dfs = read_pdf(pdf_path, pages="all", multiple_tables=True, pandas_options={"header": None})
        if dfs:
            for i, df in enumerate(dfs):
                df.to_csv(os.path.join(output_dir, f"table_{i + 1}.csv"), index=False)
            print(f"Extracted {len(dfs)} tables from PDF.")
        else:
            print("No tables found in the PDF.")
    except Exception as e:
        print(f"Failed to extract data from PDF: {e}")

# Extract tables from JSON
def process_json_tables(json_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    try:
        tables = [item for item in json_data if item["type"] == "table"]
        for i, table in enumerate(tables):
            df = pd.DataFrame(table["data"])
            df.to_csv(os.path.join(output_dir, f"table_{i + 1}.csv"), index=False)
        print(f"Extracted {len(tables)} tables from JSON.")
    except Exception as e:
        print(f"Failed to extract data from JSON: {e}")

# Generate plots from tables
def generate_plots_from_tables(output_dir):
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    for file in os.listdir(output_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(output_dir, file)
            try:
                df = pd.read_csv(file_path)
                if len(df.columns) >= 2:
                    plt.figure(figsize=(10, 6))
                    df.iloc[:, 0].value_counts().plot(kind="bar", alpha=0.7)
                    plt.title(f"Plot for {file}")
                    plt.savefig(os.path.join(plot_dir, f"{file}_plot.png"))
                    plt.close()
            except Exception as e:
                print(f"Error generating plot for {file}: {e}")

# Upload plots to Pinecone
def upload_plots_to_pinecone(output_dir):
    id_counter = 1
    for file in os.listdir(output_dir):
        if file.endswith(".png"):
            file_path = os.path.join(output_dir, file)
            metadata = {"type": "plot", "text_representation": f"Plot: {file}"}
            embedding = get_embedding(metadata["text_representation"])
            if embedding:
                index.upsert([{"id": f"item_{id_counter}", "values": embedding, "metadata": metadata}])
                id_counter += 1
            else:
                print(f"Skipping {file} due to embedding failure.")

# Main function
if __name__ == "__main__":
    file_path = "/content/NERVOUS.pdf"
    output_dir = "output"

    if file_path.endswith(".json"):
        with open(file_path, "r") as f:
            json_data = json.load(f)
        process_json_tables(json_data, output_dir)
    elif file_path.endswith(".pdf"):
        process_pdf_tables(file_path, output_dir)
    else:
        print("Unsupported file format.")

    # Generate plots
    generate_plots_from_tables(output_dir)

    # Upload plots to Pinecone
    upload_plots_to_pinecone(os.path.join(output_dir, "plots"))
