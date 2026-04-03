import json
import os
from PIL import Image

ocr_path = r'd:\Internship\Fine-tuning\ocr_without_ner_tags\invoice_CPC000692108_page0.json'
gt_path = r'd:\Internship\Fine-tuning\annotations_docile\invoice_CPC000692108.json'
img_path = r'd:\Internship\Fine-tuning\images\invoice_CPC000692108_page0.png'

with open(ocr_path, 'r') as f:
    ocr_data = json.load(f)

with open(gt_path, 'r') as f:
    gt_data = json.load(f)

img = Image.open(img_path)
width, height = img.size
print(f"Image dimensions: {width}x{height}")

# Find index of "CPC000692108" in OCR
try:
    idx = ocr_data['tokens'].index("CPC000692108")
    ocr_bbox = ocr_data['bboxes'][idx]
    print(f"OCR token 'CPC000692108' at index {idx}, bbox: {ocr_bbox}")
except ValueError:
    print("Token 'CPC000692108' not found in OCR tokens")

# Find "CPC000692108" in GT
for item in gt_data['field_extractions']:
    if item.get('text') == "CPC000692108":
        print(f"GT field 'CPC000692108', bbox: {item['bbox']}, page: {item['page']}")
