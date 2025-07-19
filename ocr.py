import os
import webbrowser
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
import sys
import torch
import json
import openai  

#configurations
image_path = r'C:\Users\draco\OneDrive\Desktop\New Project/Screenshot 2025-07-19 150307.png'
model_path = "nanonets/Nanonets-OCR-s"
openai.api_key = "YOUR_OPENAI_API_KEY" 
max_new_tokens = 4096
html_page_path = os.path.abspath("index.html")  
json_file_name = "mcqs.json"
model = AutoModelForImageTextToText.from_pretrained(
    model_path, 
    torch_dtype="auto", 
    device_map="auto", 
    attn_implementation="flash_attention_2"
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path)
def ocr_page_with_nanonets_s(image_path, model, processor, max_new_tokens=4096):
    prompt = """Extract the text from the above document as if you were reading it naturally..."""
    image = Image.open(image_path)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": prompt},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]
def generate_mcqs(text, n=3):
    prompt = f"""Read the following text and create {n} multiple choice questions..."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message['content']

extracted_text = ocr_page_with_nanonets_s(image_path, model, processor, max_new_tokens)
mcq_json_text = generate_mcqs(extracted_text)

try:
    mcqs = json.loads(mcq_json_text)
except json.JSONDecodeError:
    mcqs = [{"error": "Failed to parse JSON from GPT output."}]

with open(json_file_name, "w", encoding="utf-8") as f:
    json.dump(mcqs, f, indent=2, ensure_ascii=False)

print("MCQs saved to mcqs.json")

webbrowser.open(f"file:///{html_page_path}")
