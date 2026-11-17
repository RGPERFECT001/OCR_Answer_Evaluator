import pytesseract
from PIL import Image
import sys

def extract_text_from_image(image_path):
    try:
        # Open the image using PIL
        img = Image.open(image_path)
        
        # Extract text from the image
        text = pytesseract.image_to_string(img)
        
        return text.strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ocr_reader.py <image_path>")
        sys.exit(1)
    
    image_path = r"C:\Users\rhyth\OneDrive\Desktop\Rhythm\AgilityAI\OCR_and_API_test\Invoice.png"
    extracted_text = extract_text_from_image(image_path)
    print("\nExtracted text from the image:")
    print("-" * 40)
    print(extracted_text)
    print("-" * 40)
