import os
import pdfplumber
from anthropic import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
anthropic_client = Client(api_key=os.getenv("ANTHROPIC_API_KEY"))

def extract_pdf_text(pdf_path):
    """
    Extracts text from a PDF file.
    Args:
        pdf_path (str): Path to the PDF file.
    Returns:
        str: Extracted text from the PDF.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Extract text from each page of the PDF
            return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return None

def generate_detailed_notes_with_anthropic(text, chapter_number):
    """
    Generates detailed notes for a chapter using Anthropic Claude.
    Args:
        text (str): Text to analyze.
        chapter_number (int): Chapter number for contextualization.
    Returns:
        str: Generated notes.
    """
    try:
        prompt = f"""
        \n\nHuman: You are an expert e-learning content creator. Your task is to analyze the given text and create comprehensive e-learning materials for Chapter {chapter_number}. Include the following:
        
        1. **Introduction**:
           - Provide an engaging and informative introduction to the chapter's topic.
        
        2. **Definitions and Concepts**:
           - Highlight and define all important terms and concepts.

        3. **Methods and Processes**:
           - Explain any methods, processes, or techniques described in the text in detail.

        4. **Key Points and Insights**:
           - Summarize the main ideas with bullet points.

        5. **Examples and Applications**:
           - Provide examples, case studies, or applications of the concepts discussed.

        6. **Concluding Remarks**:
           - Conclude with a summary of what the reader has learned in this chapter.

        7. **Suggested YouTube Video**:
           - If possible, suggest a relevant YouTube video link for further learning.
        
        The e-learning material should be clear, concise, and structured, with appropriate bullet points and sections.

        Analyze and generate the material for Chapter {chapter_number} based on the following text:
        
        {text}

        Begin generating the detailed notes for Chapter {chapter_number}:
        \n\nAssistant:
        """
        response = anthropic_client.completions.create(
            model="claude-2",
            max_tokens_to_sample=3000,
            temperature=0.5,
            prompt=prompt,
        )
        return response.completion.strip()
    except Exception as e:
        print(f"Error generating notes for Chapter {chapter_number}: {e}")
        return None

def generate_e_learning_notes(pdf_path):
    """
    Extracts text from the PDF and generates e-learning notes for multiple chapters.
    Args:
        pdf_path (str): Path to the PDF file.
    Returns:
        dict: Dictionary containing notes for each chapter.
    """
    try:
        # Extract text from the PDF
        text = extract_pdf_text(pdf_path)
        if not text:
            raise ValueError("Failed to extract text from PDF.")

        chapter_notes = {}
        total_chapters = 3  # Modify as needed for your PDF structure
        
        # Split the text into chunks for each chapter
        chunk_size = len(text) // total_chapters  # Divide text roughly into 3 chapters
        for chapter in range(1, total_chapters + 1):
            chapter_text = text[(chapter - 1) * chunk_size : chapter * chunk_size]
            notes = generate_detailed_notes_with_anthropic(chapter_text, chapter)  # Use the correct function here
            chapter_notes[chapter] = notes
        
        return chapter_notes
    except Exception as e:
        print(f"Error generating e-learning notes: {e}")
        return None

# Main function to handle the PDF processing and e-learning content generation
def main():
    pdf_path = input("Enter the path to the PDF file: ").strip().strip('"')
    
    if not os.path.exists(pdf_path):
        print("The specified PDF file does not exist.")
        return
    
    print("Extracting text from PDF...")
    text = extract_pdf_text(pdf_path)
    
    if not text:
        print("Failed to extract text from the PDF.")
        return

    print("Generating e-learning content...")
    for chapter in range(1, 4):  # Generate for 3 chapters
        print(f"Processing Chapter {chapter}...")
        chapter_text = text[:min(len(text), len(text) // 3)]  # Divide roughly into three sections
        text = text[len(text) // 3:]  # Shift to the next section
        notes = generate_detailed_notes_with_anthropic(chapter_text, chapter)
        
        if not notes:
            print(f"Failed to generate notes for Chapter {chapter}.")
            continue

        save_notes_to_file(notes, chapter)

def save_notes_to_file(notes, chapter_number):
    """
    Saves notes for a chapter to a text file.
    Args:
        notes (str): Notes text.
        chapter_number (int): Chapter number.
    """
    file_name = f"Chapter_{chapter_number}_eLearning.txt"
    try:
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(notes)
        print(f"Chapter {chapter_number} notes saved to {file_name}")
    except Exception as e:
        print(f"Error saving Chapter {chapter_number} notes: {e}")

# If the script is run directly, the main function is invoked
if __name__ == "__main__":
    main()
