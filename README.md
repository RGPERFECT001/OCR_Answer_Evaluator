# Auto Grader Prototype - README

## Overview
This project provides an automated grading tool for student answer sheets using OCR and AI-based assessment. The main interface is built with Gradio, allowing users to upload answer files (images or PDFs) and specify assessment criteria. The tool extracts text from the uploaded documents, parses questions and answers, and evaluates them based on user-defined criteria using an AI model.

## Features
- Upload answer sheets as images or PDFs
- Extracts and parses questions/answers using OCR
- Supports custom assessment criteria
- Multiple checking modes: lenient, normal, strict
- Supports subject selection and custom subjects
- Displays detailed feedback and scores for each question

## Usage
1. **Install dependencies**:
   - Install required Python packages (see below).
2. **Set API Keys**:
   - Set your `OPENAI_API_KEY` and `MISTRAL_API_KEY` in a `.env` file in the project root.
3. **Run the app**:
   ```bash
   python main.py
   ```
4. **Interact with the UI**:
   - Upload an answer file (image or PDF)
   - Enter assessment criteria
   - Select checking mode and subject
   - Click "Assess" to get results

## File Structure
- `main.py` - Main Gradio app entry point
- `prototype2.py` - Core grading logic and OCR functions
- `ocr_reader.py` - (if present) Additional OCR utilities
- `frontend/` - Static frontend files (HTML, JS, CSS)
- `assessment_data/` - Example assessment data
- `data/` - Example answer files

## Requirements
- Python 3.8+
- Packages: gradio, openai, mistralai, pypdfium2, pillow, python-dotenv, gradio_pdf

Install with:
```bash
pip install gradio openai mistralai pypdfium2 pillow python-dotenv gradio_pdf
```

## Environment Variables
Create a `.env` file with:
```
OPENAI_API_KEY=your_openai_key
MISTRAL_API_KEY=your_mistral_key
```

## Notes
- The tool uses AI models for assessment; results may vary based on OCR quality and criteria.
- For best results, provide clear scanned answer sheets and detailed assessment criteria.

## License
MIT License
