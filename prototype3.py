import gradio as gr
from PIL import Image
from mistralai import ImageURLChunk
import base64
import json
from pathlib import Path
import os
from dotenv import load_dotenv
import re
import io
import gradio_pdf as gr_pdf
import pypdfium2 as pdfium
import csv
import pandas as pd
from datetime import datetime

load_dotenv()

import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
checker = openai.OpenAI()
from mistralai import Mistral
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

# PDF processing functionality
import fitz  # PyMuPDF
import google.generativeai as genai
from typing import List, Dict, Any

class PDFToCSVExtractor:
    def __init__(self):
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
    def pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5)) 
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
        
        doc.close()
        return images
    
    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        
        unwanted_patterns = [
            r'(?i)page\s+\d+\s*of\s*\d+',
            r'(?i)page\s+\d+',
            r'(?i)watermark',
            r'(?i)confidential',
            r'(?i)draft',
            r'(?i)copyright\s+©?\s*\d*',
            r'(?i)all\s+rights\s+reserved',
            r'(?i)do\s+not\s+distribute',
            r'(?i)sample\s+paper',
            r'(?i)test\s+paper',
            r'^\s*\d+\s*$',  
            r'^\s*-\s*\d+\s*-\s*$',  
        ]
        
        cleaned_text = text
        for pattern in unwanted_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.MULTILINE)
        
        cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        return cleaned_text.strip()
    
    def extract_questions_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        images = self.pdf_to_images(pdf_path)
        all_questions = []
        
        for page_num, image in enumerate(images):
            
            prompt = """
            You are an expert at extracting questions from educational documents. Analyze this page carefully and extract ONLY genuine questions, ignoring headers, footers, watermarks, page numbers, and other non-question content.

            IMPORTANT INSTRUCTIONS:
            1. ONLY extract actual questions - ignore page numbers, watermarks, headers, footers, copyright notices
            2. For tables: Convert tabular data into clear, readable text within the question
            3. For mathematical equations: Preserve mathematical notation and formulas accurately
            4. Extract the complete question including all parts, sub-questions, and context
            5. If marks are shown (like "5 marks", "[10]", "(15 points)"), extract the number only
            6. Question numbers can be in formats like: Q1, 1., (1), Question 1, etc.

            Return ONLY valid JSON in this exact format:
            {
                "questions": [
                    {
                        "question_number": "1",
                        "question_text": "Complete question text with all parts, tables converted to text, equations preserved",
                        "maximum_marks": "10"
                    }
                ]
            }

            RULES FOR TABLES:
            - Convert tables into readable text format
            - Example: "Based on the following data: Item A: 10 units at $5 each, Item B: 15 units at $3 each, Item C: 8 units at $7 each"
            - Preserve all data relationships clearly

            RULES FOR EQUATIONS:
            - Keep mathematical notation: x² + 2x - 3 = 0
            - Preserve fractions: 3/4 + 1/2
            - Keep symbols: ∑, ∫, √, π, etc.

            FILTERING RULES:
            - Skip text that looks like: "Page 1 of 5", "Confidential", "Sample Paper", standalone numbers
            - Only include substantial question content
            - Minimum question length should be at least 10 words

            If no valid questions are found on this page, return: {"questions": []}
            """
            
            try:
                response = self.model.generate_content([prompt, image])
                response_text = response.text.strip()
                
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                
                page_data = json.loads(response_text)
                
                if 'questions' in page_data and page_data['questions']:
                    for question in page_data['questions']:
                        question_text = self.clean_text(question.get('question_text', ''))
                        
                        if len(question_text) >= 10 and not self._is_unwanted_text(question_text):
                            question['question_text'] = question_text
                            all_questions.append(question)
                    
            except Exception as e:
                print(f"Error processing questions page {page_num + 1}: {str(e)}")
                continue
        
        return all_questions
    
    def extract_answers_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        images = self.pdf_to_images(pdf_path)
        all_answers = []
        
        for page_num, image in enumerate(images):
            
            prompt = """
            You are an expert at extracting answer keys and marking criteria from educational documents. Analyze this page and extract ONLY genuine answers and marking schemes, ignoring headers, footers, watermarks, and page numbers.

            IMPORTANT INSTRUCTIONS:
            1. ONLY extract actual answers and marking criteria - ignore page numbers, watermarks, headers, footers.
            2. For tabular answers: Convert tables into clear, readable text.
            3. For mathematical solutions: Preserve all mathematical notation, steps, and formulas.
            4. Extract complete answers including all solution steps.
            5. Extract detailed marking criteria and point allocation. ALWAYS provide a value for the 'criteria' field.
            6. Match answer format with question numbering (Q1, 1., (1), etc.).

            Return ONLY valid JSON in this exact format:
            {
                "answers": [
                    {
                        "question_number": "1",
                        "answer": "Complete answer with all steps, mathematical notation preserved, tables converted to readable text",
                        "criteria": "Detailed marking criteria: 2 marks for correct formula, 3 marks for calculation, 1 mark for final answer"
                    }
                ]
            }

            RULES FOR TABULAR ANSWERS:
            - Convert answer tables into readable text.
            - Example: "Correct answers: A=25, B=30, C=45. Total=100"
            - Preserve all relationships and data clearly.

            RULES FOR MATHEMATICAL SOLUTIONS:
            - Preserve step-by-step solutions.
            - Keep all mathematical notation: derivatives, integrals, equations.
            - Show complete working: "Step 1: x² + 2x - 3 = 0, Step 2: (x+3)(x-1) = 0, Step 3: x = -3 or x = 1"

            RULES FOR MARKING CRITERIA:
            - ALWAYS populate the 'criteria' field for every answer.
            - For Multiple Choice Questions (MCQs) where the answer is just a letter or option (e.g., "(b)", "Option C"), the criteria should be exactly "Correct option".
            - If no explicit marking scheme is given for a long-form answer, briefly describe the components required for marks (e.g., "Correct formula and final calculation" or "Correct final answer").
            - Extract specific point allocations if available (e.g., "Method: 2 marks, Calculation: 3 marks, Final answer: 1 mark").
            - Include partial marking schemes if mentioned.

            FILTERING RULES:
            - Skip headers, footers, page numbers, watermarks.
            - Only include substantial answer content.
            - Minimum answer length should be at least 5 words.

            If no valid answers are found on this page, return: {"answers": []}
            """
            
            try:
                response = self.model.generate_content([prompt, image])
                response_text = response.text.strip()
                
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                
                page_data = json.loads(response_text)
                
                if 'answers' in page_data and page_data['answers']:
                    for answer in page_data['answers']:
                        answer_text = self.clean_text(answer.get('answer', ''))
                        criteria_text = self.clean_text(answer.get('criteria', ''))
                        
                        if len(answer_text) >= 5 and not self._is_unwanted_text(answer_text):
                            answer['answer'] = answer_text
                            answer['criteria'] = criteria_text
                            all_answers.append(answer)
                    
            except Exception as e:
                print(f"Error processing answers page {page_num + 1}: {str(e)}")
                continue
        
        return all_answers
    
    def _is_unwanted_text(self, text: str) -> bool:
        if not text or len(text.strip()) < 5:
            return True
            
        unwanted_keywords = [
            'page', 'watermark', 'confidential', 'draft', 'copyright',
            'all rights reserved', 'sample paper', 'test paper',
            'do not distribute', 'internal use'
        ]
        
        text_lower = text.lower()
        for keyword in unwanted_keywords:
            if keyword in text_lower and len(text.strip()) < 50: 
                return True
        
        if re.match(r'^\s*\d+\s*$', text) or re.match(r'^\s*-\s*\d+\s*-\s*$', text):
            return True
            
        return False
    
    def normalize_question_number(self, q_num: str) -> str:
        if not q_num:
            return ""
        
        match = re.search(r'(\d+)', str(q_num))
        if match:
            return match.group(1)
        return str(q_num).strip()
    
    def merge_questions_and_answers(self, questions: List[Dict], answers: List[Dict]) -> List[Dict]:
        merged_data = []
        
        answer_lookup = {}
        for answer in answers:
            q_num = self.normalize_question_number(answer.get('question_number', ''))
            if q_num:
                answer_lookup[q_num] = answer
        
        unmatched_answers = set(answer_lookup.keys())
        
        for question in questions:
            q_num = self.normalize_question_number(question.get('question_number', ''))
            
            matching_answer = answer_lookup.get(q_num)
            
            if matching_answer:
                unmatched_answers.discard(q_num)
            
            merged_entry = {
                'question_number': question.get('question_number', ''),
                'question': question.get('question_text', ''),
                'answer': matching_answer.get('answer', '') if matching_answer else '',
                'criteria': matching_answer.get('criteria', '') if matching_answer else '',
                'maximum_marks': question.get('maximum_marks', '') or ''
            }
            
            merged_data.append(merged_entry)
        
        if unmatched_answers:
            print(f"Unmatched answer keys: {sorted(unmatched_answers)}")
        
        return merged_data
    
    def export_to_csv(self, data: List[Dict], output_path: str):
        if not data:
            print("No data to export")
            return
        
        fieldnames = ['question_number', 'question', 'answer', 'criteria', 'maximum_marks']
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
    
    def process_documents(self, questions_pdf: str, answers_pdf: str, output_csv: str):
        if not os.path.exists(questions_pdf):
            raise FileNotFoundError(f"Questions PDF not found: {questions_pdf}")
        if not os.path.exists(answers_pdf):
            raise FileNotFoundError(f"Answers PDF not found: {answers_pdf}")
    
        questions = self.extract_questions_from_pdf(questions_pdf)
        
        answers = self.extract_answers_from_pdf(answers_pdf)
        merged_data = self.merge_questions_and_answers(questions, answers)
        
        self.export_to_csv(merged_data, output_csv)
        
        return merged_data

# Storage management functions
def ensure_storage_directories():
    """Create storage directories for assessment data"""
    base_dir = Path("assessment_data")
    base_dir.mkdir(exist_ok=True)
    
    subjects = ["English", "Mathematics", "Physics", "Chemistry", "Biology", "History", "Geography", "Political Science"]
    for subject in subjects:
        subject_dir = base_dir / subject
        subject_dir.mkdir(exist_ok=True)
    
    return base_dir

def get_existing_csvs(subject: str) -> List[str]:
    """Get list of existing CSV files for a subject"""
    if not subject:
        return []
    base_dir = ensure_storage_directories()
    subject_dir = base_dir / subject
    
    if not subject_dir.exists():
        return []
    
    csv_files = []
    for csv_file in subject_dir.glob("*.csv"):
        csv_files.append(csv_file.stem)  
    
    return sorted(csv_files)

def load_csv_data(subject: str, csv_name: str) -> List[Dict]:
    """Load CSV data for assessment"""
    base_dir = ensure_storage_directories()
    csv_path = base_dir / subject / f"{csv_name}.csv"
    
    if not csv_path.exists():
        return []
    
    data = []
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    
    return data

def get_markdown_from_image(image_path: str) -> str:
    image_file = Path(image_path)
    assert image_file.is_file()

    encoded = base64.b64encode(image_file.read_bytes()).decode()
    base64_data_url = f"data:image/jpeg;base64,{encoded}"
    
    image_response = client.ocr.process(document=ImageURLChunk(image_url=base64_data_url), model="mistral-ocr-latest")
    
    response_dict = json.loads(image_response.model_dump_json())
    return response_dict["pages"][0]["markdown"]

def get_markdown_from_pdf(pdf_path: str) -> str:
    pdf = pdfium.PdfDocument(pdf_path)
    full_markdown = ""
    
    for i in range(len(pdf)):
        page = pdf.get_page(i)
        image = page.render(scale=2).to_pil()
        
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()

        encoded = base64.b64encode(img_bytes).decode()
        base64_data_url = f"data:image/jpeg;base64,{encoded}"
        
        image_response = client.ocr.process(document=ImageURLChunk(image_url=base64_data_url), model="mistral-ocr-latest")
        
        response_dict = json.loads(image_response.model_dump_json())
        page_markdown = response_dict["pages"][0]["markdown"]
        full_markdown += f"\n\n--- Page {i+1} ---\n\n" + page_markdown
        
    print(full_markdown.strip())
    return full_markdown.strip()

def assess_document_with_csv(file_path: str, mode: str, subject: str, custom_subject: str, selected_csv: str, 
                           question_pdf: str, answer_pdf: str) -> str:
    if file_path is None:
        return "Please provide a student answer file (Image/PDF)."

    final_subject = custom_subject.strip() if subject == "Other" and custom_subject else subject
    
    assessment_data = []
    if selected_csv and selected_csv != "None":
        assessment_data = load_csv_data(final_subject, selected_csv)
        if not assessment_data:
            return f"Error: Could not load selected CSV data for {selected_csv}"
    elif question_pdf and answer_pdf:
        # This state should be handled by the UI, but as a fallback:
        return "Error: You have uploaded new PDFs. Please click 'Prepare Assessment Data' first to process them and create the required CSV file, then select it from the dropdown."
    else:
        return "Please either select an existing assessment CSV or upload question and answer PDFs to create one."

    if not assessment_data:
        return "Critical Error: No assessment data is available to perform the evaluation. Please select a valid CSV."

    p = Path(file_path)
    if p.suffix.lower() == '.pdf':
        full_markdown_text = get_markdown_from_pdf(file_path)
    elif p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
        full_markdown_text = get_markdown_from_image(file_path)
    else:
        return "Unsupported file type. Please upload an image or a PDF."
    # Create a detailed, structured reference from the assessment data CSV
    csv_data_text = "OFFICIAL ASSESSMENT DATA (Question Number | Question | Expected Answer | Marking Criteria | Maximum Marks):\n"
    total_possible_marks = 0
    individual_max_marks_map = {}
    for i, row in enumerate(assessment_data):
        q_num_norm = str(i + 1) # Use a simple 1-based index for reliability
        q_num_orig = row.get('question_number', q_num_norm)
        marks = int(row.get('maximum_marks', 0))
        total_possible_marks += marks
        individual_max_marks_map[f"Q{q_num_norm}"] = marks
        
        csv_data_text += f"Q{q_num_norm}: {row['question']} | Expected: {row['answer']} | Criteria: {row['criteria']} | Max Marks: {marks}\n"

    # This is the new, much more robust prompt
    prompt = (
        f"You are a meticulous and strict {final_subject} examiner. Your task is to assess a student's answer sheet using a precise, step-by-step process. The student's answer sheet has been converted to text via OCR and may contain significant errors in question numbering.\n\n"
        f"**YOUR PROCESS MUST FOLLOW THESE STEPS EXACTLY:**\n\n"
        f"**STEP 1: PARSE AND RENUMBER THE STUDENT'S ANSWERS**\n"
        f"First, analyze the raw OCR text from the student's sheet. Your goal is to identify every answer the student attempted and renumber them sequentially based on their order of appearance. IGNORE the question numbers written by the student, as they are unreliable due to OCR errors.\n"
        f"  - The very first question/answer block you find in the text is **Student's Q1**.\n"
        f"  - The second block you find is **Student's Q2**.\n"
        f"  - Continue this for all answers found (Q3, Q4, ...).\n"
        f"  - OCR Correction Hints: Be aware of common OCR mistakes. For example:\n"
        f"    - 'Q10.' at the start of a paper is almost certainly 'Q1.'\n"
        f"    - A '9.' followed by a '11.' could be 'Q11'.\n"
        f"    - 'l.' is often '1.'\n"
        f"    - 'O' can be '0'.\n"
        f"  - Your goal is to create an ordered list of the student's responses, irrespective of the numbers they wrote.\n\n"
        f"**STEP 2: ALIGN AND EVALUATE AGAINST OFFICIAL DATA**\n"
        f"Now, iterate through the OFFICIAL ASSESSMENT DATA provided below. For each official question (Q1, Q2, Q3, etc.), you will perform an evaluation.\n"
        f"  - For **Official Q1**, you will evaluate the **Student's Q1** (the first answer you parsed).\n"
        f"  - For **Official Q2**, you will evaluate the **Student's Q2** (the second answer you parsed).\n"
        f"  - And so on.\n\n"
        f"**EVALUATION CRITERIA FOR EACH QUESTION:**\n"
        f"1.  **ATTEMPT CHECK:** Did the student provide an answer for this sequential position? If the official assessment has 19 questions but you only parsed 14 student answers, then for Official Q15 through Q19, the feedback must be 'No response provided' and the score is 0.\n"
        f"2.  **RELEVANCE CHECK (CRITICAL):** Is the student's answer relevant to the official question's topic? \n"
        f"    - **EXAMPLE:** If the official question is about 'Accounting Ratios' and the student's answer is about 'Newton's Laws of Motion', the answer is COMPLETELY IRRELEVANT. \n"
        f"    - If irrelevant, the score is **0**. The feedback must state: 'The response is completely irrelevant to the question, which is about [Official Question Topic]. The student appears to have answered a question from a different subject.'\n"
        f"3.  **SCORING:** If the answer is attempted and relevant, evaluate it strictly against the 'Expected Answer' and 'Marking Criteria' from the official data. Assign a score based on the quality and correctness of the response, up to the 'Max Marks' for that question.\n\n"
        f"**--- OFFICIAL ASSESSMENT DATA ---**\n"
        f"{csv_data_text}\n"
        f"**--- END OF OFFICIAL ASSESSMENT DATA --**\n\n"
        f"**--- START OF STUDENT'S OCR TEXT ---**\n"
        f"{full_markdown_text}\n"
        f"**--- END OF STUDENT'S OCR TEXT --**\n\n"
        f"**FINAL OUTPUT: Respond with ONLY a single, valid JSON object.** Do not add any text before or after the JSON. The JSON object must have the following structure:\n"
        f"{{\n"
        f"  \"total_max_marks\": {total_possible_marks},\n"
        f"  \"total_score\": <calculated_total_score>,\n"
        f"  \"feedback\": {{ \"Q1\": \"<feedback_for_official_q1>\", \"Q2\": \"<feedback_for_official_q2>\", ... }},\n"
        f"  \"individual_scores\": {{ \"Q1\": <score_for_q1>, \"Q2\": <score_for_q2>, ... }},\n"
        f"  \"individual_max_marks\": {json.dumps(individual_max_marks_map)},\n"
        f"  \"question_analysis\": {{\n"
        f"    \"Q1\": {{ \"attempted\": <true_or_false>, \"response\": \"<actual_text_parsed_as_student_q1>\" }},\n"
        f"    \"Q2\": {{ \"attempted\": <true_or_false>, \"response\": \"<actual_text_parsed_as_student_q2_or_empty>\" }},\n"
        f"    ...\n"
        f"  }}\n"
        f"}}"
    )
    
    message = [{'role': 'user', 'content': prompt}]
    # Using a more powerful model is recommended for this level of complex reasoning.
    # gpt-4-turbo is a good choice. gpt-4.1-nano might struggle.
    response = checker.chat.completions.create(model="gpt-4-turbo", messages=message, temperature=0.1, response_format={"type": "json_object"})
    
    js_response = response.choices[0].message.content
    
    if js_response is None:
        return "Error: No response from model."

    try:
        feedback_data = json.loads(js_response)
        
        individual_scores = feedback_data.get('individual_scores', {})
        individual_max_marks = feedback_data.get('individual_max_marks', {})
        question_analysis = feedback_data.get('question_analysis', {})
        feedback_dict = feedback_data.get('feedback', {})

        # Verify and recalculate totals as a safeguard
        calculated_total_score = sum(v for v in individual_scores.values() if isinstance(v, (int, float)))
        calculated_total_max_marks = sum(v for v in individual_max_marks.values() if isinstance(v, (int, float)))

        # Handle case where model might fail to calculate total max marks correctly
        if calculated_total_max_marks == 0 and total_possible_marks > 0:
            calculated_total_max_marks = total_possible_marks

        result = f"Assessment Results\n"
        result += f"Subject: {final_subject}\n"
        result += f"Assessment Data: {selected_csv if selected_csv != 'None' else 'Custom PDFs'}\n"
        result += f"Total Score: {calculated_total_score}/{calculated_total_max_marks}\n"
        if calculated_total_max_marks > 0:
            result += f"Percentage: {(calculated_total_score/calculated_total_max_marks*100):.1f}%\n\n"
        
        result += "=== QUESTION-BY-QUESTION ANALYSIS ===\n\n"
        
        # Sort questions by their number (Q1, Q2, Q10...) for ordered output
        question_keys = sorted(feedback_dict.keys(), key=lambda q: int(re.findall(r'\d+', q)[0]))

        for q_key in question_keys:
            q_num = re.findall(r'\d+', q_key)[0]
            
            # Find the corresponding original question text from assessment_data
            original_question_text = ""
            if int(q_num) <= len(assessment_data):
                original_question_text = assessment_data[int(q_num)-1].get('question', 'N/A')

            q_feedback = feedback_dict.get(q_key, 'No feedback provided.')
            q_score = individual_scores.get(q_key, 'N/A')
            q_max_marks = individual_max_marks.get(q_key, 'N/A')
            q_analysis = question_analysis.get(q_key, {})
            attempted = q_analysis.get('attempted', False)
            student_response = q_analysis.get('response', 'N/A')

            result += f"--- {q_key} ---\n"
            result += f"Official Question: {original_question_text}\n"
            result += f"Score: {q_score}/{q_max_marks}\n"
            result += f"Attempted by Student: {'Yes' if attempted else 'No'}\n"
            if attempted:
                result += f"Student's Response (as parsed): \"{student_response}\"\n"
            result += f"Evaluation Feedback: {q_feedback}\n\n"
        
        return result.strip()

    except json.JSONDecodeError as e:
        return f"Error: Could not decode the model's JSON response. This is often due to an incomplete or malformed reply from the AI. \n\nDetails: {e}\n\nRaw response:\n{js_response}"
    except Exception as e:
        return f"An unexpected error occurred while processing the results: {str(e)}\n\nRaw Response:\n{js_response}"

def prepare_assessment_data(question_pdf: str, answer_pdf: str, subject: str, custom_subject: str) -> str:
    """Process PDFs and create assessment CSV"""
    if not question_pdf or not answer_pdf:
        return "Please upload both question paper and answer sheet PDFs."
    
    final_subject = custom_subject.strip() if subject == "Other" and custom_subject else subject
    if not final_subject:
        return "Error: Subject name cannot be empty."

    try:
        extractor = PDFToCSVExtractor()
        
        question_pdf_name = Path(question_pdf).stem
        csv_filename = f"{question_pdf_name}_assessment_data"
        
        base_dir = ensure_storage_directories()
        subject_dir = base_dir / final_subject
        subject_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = subject_dir / f"{csv_filename}.csv"
        
        merged_data = extractor.process_documents(question_pdf, answer_pdf, str(csv_path))
        
        if merged_data:
            return f"Assessment data prepared successfully!\nSaved as: {csv_filename}\nSubject: {final_subject}\nQuestions processed: {len(merged_data)}"
        else:
            return "Error: No data was extracted from the PDFs."
            
    except Exception as e:
        return f"Error processing PDFs: {str(e)}"

# UI control functions
def update_subject_visibility(subject):
    return gr.update(visible=True) if subject == "Other" else gr.update(visible=False)

def display_uploaded_file(filepath):
    if filepath is None:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    p = Path(filepath)
    if p.suffix.lower() == '.pdf':
        return gr.update(visible=False), gr.update(visible=True, value=filepath), gr.update(visible=True)
    elif p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
        return gr.update(visible=True, value=filepath), gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

def update_csv_dropdown(subject, custom_subject):
    final_subject = custom_subject.strip() if subject == "Other" and custom_subject else subject
    existing_csvs = get_existing_csvs(final_subject)
    choices = ["None"] + existing_csvs
    return gr.update(choices=choices, value="None")

def toggle_pdf_inputs(selected_csv):
    show_pdfs = selected_csv == "None"
    return (
        gr.update(visible=show_pdfs), 
        gr.update(visible=show_pdfs), 
        gr.update(visible=show_pdfs) 
    )

ensure_storage_directories()

css = """
.gradio-pdf-page-controls {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    gap: 1rem !important;
}
"""

with gr.Blocks(css=css, title="Enhanced Auto Grader") as demo:
    gr.Markdown("# Auto Grader\nUpload student answer sheets and assess them using structured question-answer data.")
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(type="filepath", label="Upload Student Answer Sheet (Image or PDF)")
            image_viewer = gr.Image(label="Image Preview", visible=False)
            pdf_viewer = gr_pdf.PDF(label="PDF Preview", visible=False)
        
        with gr.Column(visible=False) as assessment_config:
            gr.Markdown("### Assessment Configuration")
            
            with gr.Row():
                subject_input = gr.Dropdown(
                    choices=["English", "Mathematics", "Physics", "Chemistry", "Biology", "History", "Geography", "Political Science", "Other"], 
                    value="Mathematics", 
                    label="Subject"
                )
                custom_subject_input = gr.Textbox(label="Custom Subject", visible=False, placeholder="Enter subject name")
            
            csv_selector = gr.Dropdown(
                choices=["None"], 
                value="None", 
                label="Select Existing Assessment Data",
                info="Choose existing CSV or select 'None' to create new from PDFs"
            )
            
            question_pdf_input = gr.File(type="filepath", label="Question Paper PDF", visible=True)
            answer_pdf_input = gr.File(type="filepath", label="Answer Sheet PDF", visible=True)
            prepare_btn = gr.Button("Prepare Assessment Data", visible=True)
            preparation_status = gr.Textbox(label="Preparation Status", visible=False)
            
            mode_input = gr.Dropdown(choices=["lenient", "normal", "strict"], value="normal", label="Checking Mode")
    
    file_input.change(
        display_uploaded_file, 
        inputs=[file_input], 
        outputs=[image_viewer, pdf_viewer, assessment_config]
    )
    
    subject_input.change(update_subject_visibility, inputs=[subject_input], outputs=[custom_subject_input])
    subject_input.change(update_csv_dropdown, inputs=[subject_input, custom_subject_input], outputs=[csv_selector])
    custom_subject_input.change(update_csv_dropdown, inputs=[subject_input, custom_subject_input], outputs=[csv_selector])
    
    csv_selector.change(
        toggle_pdf_inputs,
        inputs=[csv_selector],
        outputs=[question_pdf_input, answer_pdf_input, prepare_btn]
    )
    
    prepare_btn.click(
        lambda: gr.update(value="Processing PDFs..."),
        outputs=[preparation_status]
    ).then(
        lambda: gr.update(visible=True),
        outputs=[preparation_status]
    ).then(
        prepare_assessment_data,
        inputs=[question_pdf_input, answer_pdf_input, subject_input, custom_subject_input],
        outputs=[preparation_status]
    ).then(
        update_csv_dropdown,
        inputs=[subject_input, custom_subject_input],
        outputs=[csv_selector]
    )
    
    output = gr.Textbox(label="Assessment Results", lines=15)
    submit_btn = gr.Button("Assess Student's Answer sheet", size="lg")
    
    submit_btn.click(
        lambda: gr.update(value="Processing assessment..."),
        outputs=[output]
    ).then(
        assess_document_with_csv,
        inputs=[
            file_input, 
            mode_input, 
            subject_input, 
            custom_subject_input, 
            csv_selector,
            question_pdf_input,
            answer_pdf_input
        ],
        outputs=[output]
    )

if __name__ == "__main__":
    demo.launch(debug=True, share=False)