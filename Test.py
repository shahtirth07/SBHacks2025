import pyarrow.fs
import sycamore
import json
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sycamore.functions.tokenizer import OpenAITokenizer
from sycamore.llms import OpenAIModels, OpenAI
from sycamore.transforms import COALESCE_WHITESPACE
from sycamore.transforms.merge_elements import GreedySectionMerger
from sycamore.transforms.partition import ArynPartitioner
from sycamore.transforms.embed import OpenAIEmbedder
from sycamore.materialize_config import MaterializeSourceMode
from sycamore.utils.pdf_utils import show_pages
from sycamore.transforms.summarize_images import SummarizeImages
from sycamore.context import ExecMode
from pinecone import ServerlessSpec

load_dotenv()

# Access the keys
os.environ["ARYN_API_KEY"] = os.getenv("ARYN_API_KEY")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Verify the keys
if not all([os.getenv("ARYN_API_KEY"), os.getenv("PINECONE_API_KEY"), os.getenv("OPENAI_API_KEY")]):
    print("YOU ARE MISSING REQUIRED API KEYS FOR THIS PIPELINE. Add them to the .env file.")

paths = ["C:/Studies/Hackathon/SB hacks/Data/Test1.pdf"]
ctx = sycamore.init(ExecMode.LOCAL)
# Set the embedding model and its parameters
model_name = "text-embedding-3-small"
max_tokens = 8191
dimensions = 1536
# Initialize the tokenizer
tokenizer = OpenAITokenizer(model_name)

ds = (
    ctx.read.binary(paths, binary_format="pdf")
    # Partition and extract tables and images
    .partition(partitioner=ArynPartitioner(
        threshold="auto",
        use_ocr=True,
        extract_table_structure=True,
        extract_images=True,
        source="docprep"
    ))
    # Use materialize to cache output. If changing upstream code or input files, change setting from USE_STORED to RECOMPUTE to create a new cache.
    .materialize(path="./materialize/partitioned", source_mode=MaterializeSourceMode.USE_STORED)
    # Merge elements into larger chunks
    .merge(merger=GreedySectionMerger(
      tokenizer=tokenizer,  max_tokens=max_tokens, merge_across_pages=False
    ))
    # Split elements that are too big to embed
    .split_elements(tokenizer=tokenizer, max_tokens=max_tokens)
)

ds.execute()

# Display the first 3 pages after chunking
show_pages(ds, limit=3)