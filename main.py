import gradio as gr
from PIL import Image
from mistralai import ImageURLChunk, Mistral
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
from typing import List, Dict, Any
import openai
import fitz

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")
checker = openai.OpenAI(api_key=openai_api_key)

mistral_api_key = os.getenv("MISTRAL_API_KEY")
if not mistral_api_key:
    raise ValueError("MISTRAL_API_KEY not found in environment variables.")
client = Mistral(api_key=mistral_api_key)


def normalize_question_number(q_num_str: str) -> str:
    if not q_num_str:
        return ""
    normalized = str(q_num_str).lower().strip()
    match = re.search(r'(\d+[a-z]?)', normalized)
    return match.group(1) if match else normalized

def reconstruct_table_from_text(text: str) -> str:
    if '|' in text and '---' in text:
        return text 

    if len(re.findall(r'\n', text)) < 2 or len(re.findall(r'\s{2,}', text)) < 2:
        return text

    prompt = f"""
    The following OCR text is supposed to be a table, but the formatting is broken. Reconstruct it into a clean, valid Markdown table.
    Preserve all text and numerical data. If the text is not a table, return it unchanged.

    **Broken OCR Text:**
    ---
    {text}
    ---

    Respond with ONLY the reconstructed Markdown table.
    """
    try:
        response = checker.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Table reconstruction failed: {e}")
        return text

def format_markdown_table_for_display(markdown_table: str) -> str:
    lines = markdown_table.strip().split('\n')
    if len(lines) < 2 or not re.search(r'---\s*\|', lines[1]):
        return markdown_table

    table_data = [re.split(r'\s*\|\s*', line.strip().strip('|')) for line in lines if '|' in line]
    if not table_data or len(table_data) < 2:
        return markdown_table
    
    table_data.pop(1) 
    
    num_columns = max(len(row) for row in table_data) if table_data else 0
    if num_columns == 0:
        return markdown_table

    padded_data = [row + [''] * (num_columns - len(row)) for row in table_data]

    column_widths = [max(len(cell) for cell in col) for col in zip(*padded_data)]

    formatted_lines = []
    header_line = "| " + " | ".join([cell.ljust(column_widths[i]) for i, cell in enumerate(padded_data[0])]) + " |"
    formatted_lines.append(header_line)
    
    separator_line = "|-" + "-|-".join(["-" * width for width in column_widths]) + "-|"
    formatted_lines.append(separator_line)

    for row in padded_data[1:]:
        formatted_lines.append("| " + " | ".join([cell.ljust(column_widths[i]) for i, cell in enumerate(row)]) + " |")

    return "\n".join(formatted_lines)


def get_markdown_from_image_enhanced(image_path: str) -> dict:
    try:
        image_file = Path(image_path)
        encoded = base64.b64encode(image_file.read_bytes()).decode()
        base64_data_url = f"data:image/jpeg;base64,{encoded}"
        
        image_response = client.ocr.process(document=ImageURLChunk(image_url=base64_data_url), model="mistral-ocr-latest")
        response_dict = json.loads(image_response.model_dump_json())
        raw_markdown = response_dict["pages"][0]["markdown"]
        return {"processed_markdown": raw_markdown, "tables_detected": raw_markdown.count("|---")}
    except Exception as e:
        return {"processed_markdown": "", "tables_detected": 0, "error": str(e)}


def get_markdown_from_pdf_enhanced(pdf_path: str) -> dict:
    try:
        pdf = pdfium.PdfDocument(pdf_path)
        full_processed_content = ""
        total_tables = 0
        
        for i in range(len(pdf)):
            page = pdf.get_page(i)
            image = page.render(scale=1.0).to_pil()
            
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=100)
            encoded = base64.b64encode(buffered.getvalue()).decode()
            base64_data_url = f"data:image/jpeg;base64,{encoded}"
            
            image_response = client.ocr.process(document=ImageURLChunk(image_url=base64_data_url), model="mistral-ocr-latest")
            response_dict = json.loads(image_response.model_dump_json())
            raw_page_markdown = response_dict["pages"][0]["markdown"]
            total_tables += raw_page_markdown.count("|---")
            full_processed_content += f"\n\n--- Page {i+1} ---\n\n" + raw_page_markdown
        
        return {"full_processed_content": full_processed_content.strip(), "total_pages": len(pdf), "total_tables_detected": total_tables}
    except Exception as e:
        return {"error": str(e), "full_processed_content": "", "total_pages": 0, "total_tables_detected": 0}


def parse_student_answers_enhanced(content: str) -> List[Dict[str, str]]:
    if not content or not content.strip():
        return []

    prompt = f"""
    You are a text-parsing expert processing OCR output from a student's answer sheet. Your task is to segment the text into a structured list of all attempted answers.

    **CRITICAL INSTRUCTIONS:**
    1.  **Identify Answer Blocks:** An answer begins with a clear identifier (e.g., 'Q.3', '4)', 'Ans 5') OR is a distinct block of text/calculation that is visually separate from others.
    2.  **Be Forgiving but Accurate:** Capture every distinct answer. If an answer block is missing a number, still treat it as a valid answer and do your best to infer its number from the context or assign a placeholder.
    
    **OUTPUT FORMAT:**
    Return a single JSON object: `{{"answers": [...]}}`. Each item in the list must be an object with two keys:
    1.  `"question_number"`: The question's identifier (e.g., "1", "2", "3b").
    2.  `"answer_text"`: The complete and unmodified text for that answer.

    **Example:**
    `{{"answers": [{{"question_number": "1", "answer_text": "Q.1 The capital of France is Paris."}}, {{"question_number": "2", "answer_text": "The formula for water is H2O.\\nIt is a liquid at room temperature."}}]}}`

    Now, process the following text:
    ---
    {content}
    ---
    """
    try:
        response = checker.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        result = json.loads(response.choices[0].message.content)
        if isinstance(result.get("answers"), list):
            for answer in result["answers"]:
                answer["answer_text"] = reconstruct_table_from_text(answer["answer_text"])
            return result["answers"]
        return []
    except Exception as e:
        print(f"Error parsing answers with AI: {e}")
        return [{"question_number": "1", "answer_text": reconstruct_table_from_text(content)}]


def assess_document_with_csv_enhanced(file_path: str, mode: str, subject: str, custom_subject: str, 
                                    selected_csv: str, question_pdf: str, answer_pdf: str) -> str:
    if file_path is None:
        return "Please provide a student answer file (Image/PDF)."

    final_subject = custom_subject.strip() if subject == "Other" and custom_subject else subject
    
    assessment_data = load_csv_data(final_subject, selected_csv)
    if not assessment_data:
        return "Error: Could not load selected CSV data. Please select or prepare a valid assessment CSV."

    p = Path(file_path)
    if p.suffix.lower() == '.pdf':
        ocr_result = get_markdown_from_pdf_enhanced(file_path)
    elif p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
        ocr_result = get_markdown_from_image_enhanced(file_path)
    else:
        return "Unsupported file type. Please upload an image or a PDF."

    if "error" in ocr_result:
        return f"OCR Failed: {ocr_result['error']}"

    full_markdown_text = ocr_result.get("full_processed_content") or ocr_result.get("processed_markdown", "")
    processing_info = f"File processed: {ocr_result.get('total_pages', 1)} page(s), {ocr_result.get('total_tables_detected', 0)} tables detected."

    parsed_answers = parse_student_answers_enhanced(full_markdown_text)
    
    student_answers_map = {
        normalize_question_number(ans.get("question_number", "")): ans.get("answer_text", "")
        for ans in parsed_answers if ans.get("question_number")
    }

    all_results = []
    total_score = 0
    total_max_marks = 0

    for official_question in assessment_data:
        q_max_marks = int(official_question.get('maximum_marks', 0))
        total_max_marks += q_max_marks
        
        official_q_num_str = official_question.get('question_number', '')
        normalized_q_num = normalize_question_number(official_q_num_str)
        
        student_answer_text = student_answers_map.get(normalized_q_num)
        attempted = student_answer_text is not None

        current_result = {
            "q_key": f"Q{official_q_num_str}",
            "official_question": official_question.get('question', 'N/A'),
            "score": 0,
            "max_marks": q_max_marks,
            "attempted": attempted,
            "student_response": student_answer_text or "Not Attempted",
            "feedback": "Not attempted by student."
        }
        
        if attempted:
            prompt = f"""
            You are a highly intelligent and meticulous {final_subject} examiner. Your task is to evaluate a student's answer based on the provided information, with special attention to OCR errors and mathematical notation.

            **EVALUATION RULES:**
            1.  **MCQ (Multiple Choice):** An answer is CORRECT if the student provides the correct option letter (e.g., 'c') OR the correct option text (e.g., '94%'). Award full marks if either matches the official answer. Penalize only if both are wrong.
            2.  **MATHEMATICAL NOTATION:** Student work is from OCR. A fraction like a/b might appear as `a` on one line and `b` on the next. Interpret mathematical notation contextually. Evaluate the logical correctness of the steps and formulas used, not just the visual layout.
            3.  **TABLES:** The student's response may have been reconstructed into a Markdown table. Compare the data within the student's table to the official answer table cell by cell.
            4.  **PARTIAL CREDIT:** Award partial marks for correct methodology, formulas, or partially correct data, as per the marking criteria.

            **ASSESSMENT DETAILS:**
            - **Official Question:** {official_question.get('question', '')}
            - **Official Answer/Key:** {official_question.get('answer', '')}
            - **Marking Criteria:** {official_question.get('criteria', 'Standard marking scheme')}
            - **Maximum Marks:** {q_max_marks}

            **STUDENT'S RESPONSE (AI Pre-processed):**
            ---
            {student_answer_text}
            ---

            **YOUR TASK:**
            Provide your evaluation as a JSON object with two keys:
            1.  `score`: A number for the marks awarded (e.g., 2.5, can be a float).
            2.  `feedback`: A concise, professional explanation for the score.

            Respond with ONLY the valid JSON object.
            """
            try:
                response = checker.chat.completions.create(
                    model="gpt-4o",
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=0.1, 
                    response_format={"type": "json_object"}
                )
                single_eval = json.loads(response.choices[0].message.content)
                score = single_eval.get("score", 0)
                current_result["score"] = max(0, min(q_max_marks, float(score))) if isinstance(score, (int, float)) else 0
                current_result["feedback"] = single_eval.get("feedback", "Error: Feedback not generated.")
            except Exception as e:
                current_result["feedback"] = f"Error during AI evaluation: {str(e)}"

        total_score += current_result["score"]
        all_results.append(current_result)

    result = f"Assessment Results\n"
    result += f"Subject: {final_subject}\n"
    result += f"Assessment Data: {selected_csv}\n"
    result += f"Processing Info: {processing_info}\n"
    result += f"Total Score: {total_score}/{total_max_marks}\n"
    if total_max_marks > 0:
        result += f"Percentage: {(total_score/total_max_marks*100):.1f}%\n\n"
    
    result += "=== QUESTION-BY-QUESTION ANALYSIS ===\n"
    result += f"({len(parsed_answers)} answers were parsed from the student's sheet)\n\n"
    
    for res in all_results:
        result += f"--- {res['q_key']} ---\n"
        formatted_question = format_markdown_table_for_display(res['official_question'])
        result += f"Official Question:\n```\n{formatted_question}\n```\n"
        result += f"Score: {res['score']}/{res['max_marks']}\n"
        result += f"Attempted by Student: {'Yes' if res['attempted'] else 'No'}\n"
        formatted_response = format_markdown_table_for_display(res['student_response'])
        result += f"Student's Response:\n```\n{formatted_response}\n```\n"
        result += f"Evaluation Feedback: {res['feedback']}\n\n"
            
    return result.strip()


class PDFToCSVExtractor:
    def pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        doc = fitz.open(pdf_path)
        images = []
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            images.append(img)
        doc.close()
        return images

    def _get_markdown_from_image(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=100)
        encoded = base64.b64encode(buffered.getvalue()).decode()
        base64_data_url = f"data:image/jpeg;base64,{encoded}"
        try:
            response = client.ocr.process(document=ImageURLChunk(image_url=base64_data_url), model="mistral-ocr-latest")
            return json.loads(response.model_dump_json())["pages"][0]["markdown"]
        except Exception as e:
            print(f"Mistral OCR error during CSV prep: {e}")
            return ""

    def _parse_text_with_openai(self, text: str, prompt: str) -> Dict:
        full_prompt = f"{prompt}\n\n--- OCR TEXT ---\n{text}"
        try:
            response = checker.chat.completions.create(
                model="gpt-4o",
                messages=[{'role': 'user', 'content': full_prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"OpenAI parsing error during CSV prep: {e}")
            return {}

    def _extract_data(self, pdf_path: str, prompt_template: str, data_key: str) -> List[Dict]:
        images = self.pdf_to_images(pdf_path)
        all_items = []
        for image in images:
            markdown = self._get_markdown_from_image(image)
            if markdown:
                page_data = self._parse_text_with_openai(markdown, prompt_template)
                if data_key in page_data and page_data[data_key]:
                    all_items.extend(page_data[data_key])
        return all_items

    def extract_questions_from_pdf(self, pdf_path: str) -> List[Dict]:
        prompt = """
        From the OCR text, extract all questions. Reconstruct tables into clean Markdown.
        Return JSON: {"questions": [{"question_number": "...", "question_text": "...", "maximum_marks": "..."}]}.
        """
        return self._extract_data(pdf_path, prompt, "questions")

    def extract_answers_from_pdf(self, pdf_path: str) -> List[Dict]:
        prompt = """
        From the OCR text, extract all answers. Reconstruct tables into clean Markdown.
        Return JSON: {"answers": [{"question_number": "...", "answer": "...", "criteria": "..."}]}.
        """
        return self._extract_data(pdf_path, prompt, "answers")

    def merge_questions_and_answers(self, questions: List[Dict], answers: List[Dict]) -> List[Dict]:
        answer_lookup = {normalize_question_number(ans.get('question_number', '')): ans for ans in answers}
        merged_data = []
        for q in questions:
            q_num_norm = normalize_question_number(q.get('question_number', ''))
            matching_answer = answer_lookup.get(q_num_norm, {})
            merged_data.append({
                'question_number': q.get('question_number', ''),
                'question': q.get('question_text', ''),
                'answer': matching_answer.get('answer', ''),
                'criteria': matching_answer.get('criteria', 'Standard marking scheme.'),
                'maximum_marks': q.get('maximum_marks', '')
            })
        return merged_data

    def export_to_csv(self, data: List[Dict], output_path: str):
        if not data: return
        fieldnames = ['question_number', 'question', 'answer', 'criteria', 'maximum_marks']
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    def process_documents(self, questions_pdf: str, answers_pdf: str, output_csv: str):
        questions = self.extract_questions_from_pdf(questions_pdf)
        answers = self.extract_answers_from_pdf(answers_pdf)
        merged_data = self.merge_questions_and_answers(questions, answers)
        self.export_to_csv(merged_data, output_csv)
        return merged_data


def ensure_storage_directories():
    base_dir = Path("assessment_data")
    base_dir.mkdir(exist_ok=True)
    subjects = ["English", "Mathematics", "Accounts", "Physics", "Chemistry", "Other"]
    for subject in subjects:
        (base_dir / subject).mkdir(exist_ok=True)

def get_existing_csvs(subject: str) -> List[str]:
    if not subject: return []
    subject_dir = Path("assessment_data") / subject
    return sorted([f.stem for f in subject_dir.glob("*.csv")]) if subject_dir.exists() else []

def load_csv_data(subject: str, csv_name: str) -> List[Dict]:
    csv_path = Path("assessment_data") / subject / f"{csv_name}.csv"
    if not csv_path.exists(): return []
    with open(csv_path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def prepare_assessment_data(question_pdf: str, answer_pdf: str, subject: str, custom_subject: str) -> str:
    if not question_pdf or not answer_pdf:
        return "Please upload both the Question Paper and Answer Key PDFs."
    final_subject = custom_subject.strip() if subject == "Other" and custom_subject else subject
    if not final_subject: return "Error: Subject name cannot be empty."
    try:
        extractor = PDFToCSVExtractor()
        csv_filename = f"{Path(question_pdf).stem}_assessment_data"
        subject_dir = Path("assessment_data") / final_subject
        subject_dir.mkdir(parents=True, exist_ok=True)
        csv_path = subject_dir / f"{csv_filename}.csv"
        merged_data = extractor.process_documents(question_pdf, answer_pdf, str(csv_path))
        if merged_data:
            return f"Success! Assessment data prepared.\nSaved as: {csv_filename}\nSubject: {final_subject}\nQuestions processed: {len(merged_data)}"
        else:
            return "Error: No data was extracted. Check the documents and try again."
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def update_subject_visibility(subject):
    return gr.update(visible=(subject == "Other"))

def display_uploaded_file(filepath):
    if not filepath:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    p = Path(filepath)
    is_pdf = p.suffix.lower() == '.pdf'
    is_image = p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']
    return gr.update(visible=is_image, value=filepath if is_image else None), \
           gr.update(visible=is_pdf, value=filepath if is_pdf else None), \
           gr.update(visible=(is_image or is_pdf))

def update_csv_dropdown(subject, custom_subject):
    final_subject = custom_subject.strip() if subject == "Other" and custom_subject else subject
    return gr.update(choices=["None"] + get_existing_csvs(final_subject), value="None")

def toggle_pdf_inputs(selected_csv):
    show_pdfs = selected_csv == "None"
    return gr.update(visible=show_pdfs), gr.update(visible=show_pdfs), gr.update(visible=show_pdfs)


ensure_storage_directories()

css = """
.gradio-pdf-page-controls { display: flex !important; justify-content: center !important; align-items: center !important; gap: 1rem !important; }
.output-text { font-family: 'Courier New', monospace; white-space: pre-wrap; overflow-x: auto; }
"""

with gr.Blocks(css=css, title="AI Auto Grader") as demo:
    gr.Markdown("# AI Auto Grader with Mistral OCR & OpenAI Reasoning\nUpload a student's answer sheet to automatically grade it against a prepared answer key.")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(type="filepath", label="Upload Student Answer Sheet (Image or PDF)")
            image_viewer = gr.Image(label="Image Preview", visible=False)
            pdf_viewer = gr_pdf.PDF(label="PDF Preview", visible=False)
        
        with gr.Column(scale=1, visible=False) as assessment_config:
            gr.Markdown("### Assessment Configuration")
            with gr.Row():
                subject_input = gr.Dropdown(choices=["English", "Mathematics", "Accounts", "Physics", "Chemistry", "Other"], value="Accounts", label="Subject")
                custom_subject_input = gr.Textbox(label="Custom Subject", visible=False, placeholder="Enter subject name")
            
            csv_selector = gr.Dropdown(choices=["None"], value="None", label="Select Existing Assessment Data", info="Choose a CSV or select 'None' to create a new one.")
            
            with gr.Group(visible=True) as prep_group:
                question_pdf_input = gr.File(type="filepath", label="Question Paper PDF")
                answer_pdf_input = gr.File(type="filepath", label="Answer Key PDF")
                prepare_btn = gr.Button("Prepare Assessment Data")
                preparation_status = gr.Textbox(label="Preparation Status", interactive=False, visible=False)
            
            mode_input = gr.Dropdown(choices=["normal"], value="normal", label="Checking Mode")
    
    output = gr.Textbox(label="Assessment Results", lines=30, elem_classes=["output-text"], interactive=False)
    submit_btn = gr.Button("Assess Student's Answer Sheet", variant="primary", size="lg")
    
    file_input.change(display_uploaded_file, inputs=[file_input], outputs=[image_viewer, pdf_viewer, assessment_config])
    
    subject_input.change(update_subject_visibility, inputs=[subject_input], outputs=[custom_subject_input])
    subject_input.change(update_csv_dropdown, inputs=[subject_input, custom_subject_input], outputs=[csv_selector])
    custom_subject_input.change(update_csv_dropdown, inputs=[subject_input, custom_subject_input], outputs=[csv_selector])
    
    csv_selector.change(toggle_pdf_inputs, inputs=[csv_selector], outputs=[question_pdf_input, answer_pdf_input, prepare_btn])
    
    prepare_btn.click(
        lambda: (gr.update(visible=True, value="Processing PDFs with Mistral OCR & OpenAI... This may take a few minutes."), gr.update(visible=True)),
        outputs=[preparation_status, preparation_status]
    ).then(
        prepare_assessment_data,
        inputs=[question_pdf_input, answer_pdf_input, subject_input, custom_subject_input],
        outputs=[preparation_status]
    ).then(
        update_csv_dropdown,
        inputs=[subject_input, custom_subject_input],
        outputs=[csv_selector]
    )
    
    submit_btn.click(
        lambda: "Processing assessment with Mistral OCR & OpenAI Reasoning...",
        outputs=[output]
    ).then(
        assess_document_with_csv_enhanced,
        inputs=[file_input, mode_input, subject_input, custom_subject_input, csv_selector, question_pdf_input, answer_pdf_input],
        outputs=[output]
    )

if __name__ == "__main__":
    demo.launch(debug=True, share=False)