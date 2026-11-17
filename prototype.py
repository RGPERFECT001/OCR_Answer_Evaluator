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

load_dotenv()

import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
checker = openai.OpenAI()
from mistralai import Mistral
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

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
    
    return full_markdown.strip()

def assess_document(file_path: str, criteria: str, mode: str, max_marks_input: str, subject: str, custom_subject: str) -> str:
    if file_path is None or not criteria.strip():
        return "Please provide both a file (Image/PDF) and assessment criteria."

    p = Path(file_path)
    if p.suffix.lower() == '.pdf':
        full_markdown_text = get_markdown_from_pdf(file_path)
    elif p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
        full_markdown_text = get_markdown_from_image(file_path)
    else:
        return "Unsupported file type. Please upload an image or a PDF."

    final_subject = custom_subject.strip() if subject == "Other" and custom_subject.strip() else subject

    prompt = (
        f"You are a {final_subject} teacher/professor. Your task is to assess a student's answer sheet based on the provided criteria. "
        f"The input is raw text extracted from the student's handwritten document via OCR, so it may contain errors, scrambled question numbers, or other artifacts. "
        f"For example, '9.11' might actually be 'Q11', and questions might be written out of order.\n\n"
        f"Your first step is to intelligently parse the entire text to identify each distinct question and its corresponding answer. "
        f"Please consolidate and re-number the questions sequentially (Q1, Q2, Q3, ...) based on their logical order in the document.\n\n"
        f"Once parsed, evaluate each answer. Checking mode: {mode}. Be lenient or strict with scoring accordingly, but do not mention the mode in your feedback.\n\n"
        f"If mode is 'lenient', be more forgiving in your assessment. If 'strict', be very precise and critical don't give full marks easily, do not mention the mode in your feedback.\n\n"
        f"The maximum marks are specified as: '{max_marks_input}'.\n"
        f"- If this is a single number, it is the maximum mark for every question.\n"
        f"- If this is a comma-separated list (e.g., '2,3,5'), assign the marks to the questions in sequential order (Q1 gets 2, Q2 gets 3, etc.).\n\n"
        f"Provide your assessment in a single, valid JSON object. The JSON must contain these keys:\n"
        f"1. 'total_max_marks': (number) The sum of maximum marks for all identified questions.\n"
        f"2. 'total_score': (number) The total score the student achieved.\n"
        f"3. 'feedback': (object) Detailed feedback for each question, using corrected sequential question numbers (e.g., 'Q1', 'Q2') as keys.\n"
        f"4. 'individual_scores': (object) The score for each question, using corrected question numbers as keys.\n"
        f"5. 'individual_max_marks': (object) The maximum possible marks for each question, using corrected question numbers as keys.\n\n"
        f"Give only the JSON format and nothing else. Ensure your response is a valid JSON.\n\n"
        f"--- START OF OCR TEXT ---\n"
        f"{full_markdown_text}\n"
        f"--- END OF OCR TEXT ---\n\n"
        f"--- ASSESSMENT CRITERIA ---\n"
        f"{criteria}\n"
        f"--- END OF CRITERIA ---"
    )

    message = [{'role': 'user', 'content': prompt}]
    response = checker.chat.completions.create(model="gpt-4.1-nano-2025-04-14", messages=message, temperature=0,)
    #print(f"Response received from model.\n{response}")
    Js = response.choices[0].message.content
    #print(Js)
    if Js is None:
        return "Error: No response from model."

    try:
        feedback = json.loads(Js)
        
        individual_scores = feedback.get('individual_scores', {})
        individual_max_marks = feedback.get('individual_max_marks', {})

        calculated_total_score = sum(score for score in individual_scores.values() if isinstance(score, (int, float)))
        calculated_total_max_marks = sum(score for score in individual_max_marks.values() if isinstance(score, (int, float)))

        result = f"Total Marks: {calculated_total_score}/{calculated_total_max_marks}\n\n"
        
        feedback_dict = feedback.get('feedback', {})
        
        question_keys = sorted(feedback_dict.keys(), key=lambda q: int(re.findall(r'\d+', q)[0]))

        for q_key in question_keys:
            q_feedback = feedback_dict.get(q_key, 'No feedback provided.')
            q_score = individual_scores.get(q_key, 'N/A')
            q_max_marks = individual_max_marks.get(q_key, 'N/A')
            result += f"{q_key} - Score: {q_score}/{q_max_marks}\nFeedback: {q_feedback}\n\n"
        
        return result.strip()

    except json.JSONDecodeError as e:
        return f"Error: Could not decode the model's JSON response. Details: {e}\nRaw response:\n{Js}"
    except (AttributeError, TypeError, IndexError):
        return f"Error: The model's response was not in the expected format.\nResponse:\n{feedback}"


def update_subject_visibility(subject):
    return gr.update(visible=True) if subject == "Other" else gr.update(visible=False)

def display_uploaded_file(filepath):
    if filepath is None:
        return gr.update(visible=False), gr.update(visible=False)
    
    p = Path(filepath)
    if p.suffix.lower() == '.pdf':
        return gr.update(visible=False), gr.update(visible=True, value=filepath)
    elif p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
        return gr.update(visible=True, value=filepath), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=False)

css = """
.gradio-pdf-page-controls {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    gap: 1rem !important;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# Auto Grader Prototype\nUpload an answer image or PDF and specify assessment criteria.")
    with gr.Row():
        with gr.Column():
            file_input = gr.File(type="filepath", label="Upload Answer File (Image or PDF)")
            image_viewer = gr.Image(label="Image Preview", visible=False)
            pdf_viewer = gr_pdf.PDF(label="PDF Preview", visible=False)
        criteria_input = gr.Textbox(label="Assessment Criteria", placeholder="e.g. Correctness, Steps, Neatness")

    file_input.change(display_uploaded_file, inputs=[file_input], outputs=[image_viewer, pdf_viewer])
    
    with gr.Row():
        mode_input = gr.Dropdown(choices=["lenient", "normal", "strict"], value="normal", label="Checking Mode")
        max_marks_input = gr.Textbox(label="Maximum Marks per Question", value="4", placeholder="e.g. 4 or 2,3,5 for Q1,Q2,Q3")
    with gr.Row():
        subject_input = gr.Dropdown(
            choices=["English", "Mathematics", "Physics", "Chemistry", "Biology", "History", "Geography", "Political Science", "Other"], 
            value="Mathematics", 
            label="Subject"
        )
        custom_subject_input = gr.Textbox(label="Custom Subject", visible=False, placeholder="Enter subject name")
    
    subject_input.change(update_subject_visibility, inputs=[subject_input], outputs=[custom_subject_input])
    
    output = gr.Textbox(label="Assessment Result")
    submit_btn = gr.Button("Assess")
    submit_btn.click(assess_document, inputs=[file_input, criteria_input, mode_input, max_marks_input, subject_input, custom_subject_input], outputs=output)

if __name__ == "__main__":
    demo.launch()