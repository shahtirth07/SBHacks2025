import sycamore
from dotenv import load_dotenv
from sycamore.functions.tokenizer import OpenAITokenizer
from sycamore.transforms.merge_elements import GreedySectionMerger
from sycamore.transforms.partition import ArynPartitioner
from sycamore.transforms.embed import OpenAIEmbedder
from pinecone import Pinecone, ServerlessSpec
import os
from sycamore.context import ExecMode


def process_and_encode_file(file_path):
    load_dotenv()

    # Ensure API keys are available
    if not all([os.getenv("ARYN_API_KEY"), os.getenv("PINECONE_API_KEY"), os.getenv("OPENAI_API_KEY")]):
        raise ValueError("Missing required API keys. Add them to the .env file.")

    ctx = sycamore.init(ExecMode.LOCAL)

    # Configure tokenizer and embedding model
    model_name = "text-embedding-3-small"
    max_tokens = 8191
    dimensions = 1536
    tokenizer = OpenAITokenizer(model_name)

    try:
        # Process and embed file
        ds = (
            ctx.read.binary(file_path, binary_format="pdf")
            .partition(partitioner=ArynPartitioner(
                threshold="auto",
                use_ocr=True,
                extract_table_structure=True,
                extract_images=True,
                source="docprep"
            ))
            .merge(merger=GreedySectionMerger(tokenizer=tokenizer, max_tokens=max_tokens, merge_across_pages=False))
            .split_elements(tokenizer=tokenizer, max_tokens=max_tokens)
            .embed(embedder=OpenAIEmbedder(model_name=model_name))
        )

        ds.execute()

        # Write data to Pinecone
        pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = "nervous"
        spec = ServerlessSpec(cloud="aws", region="us-east-1")
        
        if index_name not in pinecone_client.list_indexes().names():
            pinecone_client.create_index(
                name=index_name,
                dimension=dimensions,
                metric="cosine",
                spec=spec
            )
        
        ds.write.pinecone(
            index_name=index_name,
            dimensions=dimensions,
            distance_metric="cosine",
            index_spec=spec
        )

        return "file encoded successfully and stored in Pinecone."

    except Exception as e:
        raise RuntimeError(f"Error during file processing and encoding: {e}")

