# General-purpose libraries
import os
import json
import base64

# Image processing
from PIL import Image, ImageDraw, ImageFont

# Data manipulation
import pandas as pd

# OCR
import pytesseract

# Embedding model
from sentence_transformers import SentenceTransformer

# Pinecone for vector storage
from pinecone import Pinecone

# Anthropic for Claude API integration
import anthropic

# Pinecone and Anthropic setup
def initialize_clients():
    pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return pinecone_client, anthropic_client

 # Initialize Pinecone and Anthropic
pinecone_client, anthropic_client = initialize_clients()

# Define constants
JSON_PATH = "Digestive_System_508-bbox.json"
OUTPUT_DIR = "data"
IMAGE_TEXT_DIR = os.path.join(OUTPUT_DIR, "extracted_text")

# Function to extract images with captions
def extract_images_with_captions(json_data, output_dir):
    images = [item for item in json_data if item["type"] == "Image"]
    captions = [item for item in json_data if item["type"] == "Caption"]

    for idx, image in enumerate(images):
        binary_data = image.get("binary_representation")
        if binary_data:
            try:
                # Decode Base64 and save the image
                img_data = base64.b64decode(binary_data)
                image_size = image["properties"]["image_size"]
                image_mode = image["properties"]["image_mode"]
                img = Image.frombytes(image_mode, tuple(image_size), img_data)

                # Find associated caption
                caption_text = ""
                image_bbox = image["bbox"]
                for caption in captions:
                    caption_bbox = caption["bbox"]
                    if (
                        caption_bbox[0] >= image_bbox[0]
                        and caption_bbox[2] <= image_bbox[2]
                        and caption_bbox[1] >= image_bbox[3]
                    ):
                        caption_text = caption["text_representation"]
                        break

                # Add caption to the image
                if caption_text:
                    new_height = img.height + 40
                    new_img = Image.new("RGB", (img.width, new_height), "white")
                    new_img.paste(img, (0, 0))
                    draw = ImageDraw.Draw(new_img)
                    font = ImageFont.load_default()
                    draw.text((10, img.height + 10), caption_text, fill="black", font=font)
                    img = new_img

                # Save the image
                img.save(os.path.join(output_dir, f"image_with_caption_{idx + 1}.png"))
                print(f"Image {idx + 1} with caption saved successfully!")
            except Exception as e:
                print(f"Failed to decode image {idx + 1}: {e}")
        else:
            print(f"No binary data for image {idx + 1}.")

# Function to extract tables with captions
def extract_tables_with_captions(json_data, output_dir):
    tables = [item for item in json_data if item["type"] == "Table"]
    captions = [item for item in json_data if item["type"] == "Caption"]

    for idx, table in enumerate(tables):
        try:
            cells = table["table"]["cells"]
            rows = {}
            for cell in cells:
                row_idx = cell["rows"][0]
                col_idx = cell["cols"][0]
                content = cell.get("content", "")
                if row_idx not in rows:
                    rows[row_idx] = {}
                rows[row_idx][col_idx] = content

            # Create a pandas DataFrame
            df = pd.DataFrame.from_dict(rows, orient="index").sort_index(axis=1)

            # Find associated caption
            caption_text = ""
            table_bbox = table["bbox"]
            for caption in captions:
                caption_bbox = caption["bbox"]
                if (
                    caption_bbox[0] >= table_bbox[0]
                    and caption_bbox[2] <= table_bbox[2]
                    and caption_bbox[1] >= table_bbox[3]
                ):
                    caption_text = caption["text_representation"]
                    break

            # Save the table and caption
            table_path = os.path.join(output_dir, f"table_with_caption_{idx + 1}.csv")
            df.to_csv(table_path, index=False)
            if caption_text:
                with open(table_path.replace(".csv", ".txt"), "w") as caption_file:
                    caption_file.write(caption_text)
            print(f"Table {idx + 1} with caption saved successfully!")
        except Exception as e:
            print(f"Failed to process table {idx + 1}: {e}")

# Function to extract text from images
def extract_text_from_images(image_dir, output_text_dir):
    os.makedirs(output_text_dir, exist_ok=True)
    for image_file in os.listdir(image_dir):
        if image_file.endswith(".png"):  # Process only PNG files
            image_path = os.path.join(image_dir, image_file)
            try:
                # Open image and perform OCR
                img = Image.open(image_path)
                text = pytesseract.image_to_string(img)

                # Save extracted text to a file
                text_file = os.path.join(output_text_dir, f"{os.path.splitext(image_file)[0]}.txt")
                with open(text_file, "w") as f:
                    f.write(text)

                print(f"Extracted text from {image_file} and saved to {text_file}")
            except Exception as e:
                print(f"Failed to process {image_file}: {e}")

# Main function
def main():
    # Load JSON data
    with open(JSON_PATH, "r") as f:
        json_data = json.load(f)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Extract images and tables
    extract_images_with_captions(json_data, OUTPUT_DIR)
    extract_tables_with_captions(json_data, OUTPUT_DIR)

    # Extract text from images
    extract_text_from_images(OUTPUT_DIR, IMAGE_TEXT_DIR)

    # Further logic for Pinecone and Anthropic integration goes here...

if __name__ == "__main__":
    main()
