import sys
import torch
import json
import os
import torch.nn.functional as nnf
from PIL import Image
from transformers import LayoutLMv3Processor

# Add src to path
sys.path.append('src')
from trainer import ModelModule
from utils import dataSetFormat

SRC_DIR = 'src'
PROJECT_ROOT = '.'
model_weights = os.path.join(SRC_DIR, 'model.bin')
model_path = os.path.join('inputs', 'layoutlmv3Microsoft')
if not os.path.exists(model_path): model_path = 'microsoft/layoutlmv3-base'
config_path = os.path.join('inputs', 'label_config.json')
# User's specific image from active document
IMAGE_PATH = os.path.join('images', 'invoice_CPC000692108_page0.png')

with open(config_path, 'r') as f: config = json.load(f)
id2label = {int(k): v for k, v in config['id2label'].items()}

processor = LayoutLMv3Processor.from_pretrained(model_path, apply_ocr=False)
model = ModelModule(len(id2label))
model.load_state_dict(torch.load(model_weights, map_location='cpu'))
model.eval()

image = Image.open(IMAGE_PATH).convert('RGB')
test_dict, w, h = dataSetFormat(image)
encoding = processor(image, test_dict['tokens'], boxes=test_dict['bboxes'], max_length=512, stride=128, padding='max_length', truncation=True, return_overflowing_tokens=True, return_offsets_mapping=True, return_tensors='pt')

final_predictions = {}
num_chunks = len(encoding.input_ids)

for i in range(num_chunks):
    with torch.no_grad():
        inputs = {'input_ids': encoding['input_ids'][i].unsqueeze(0), 'attention_mask': encoding['attention_mask'][i].unsqueeze(0), 'bbox': encoding['bbox'][i].unsqueeze(0), 'pixel_values': encoding['pixel_values'][i].unsqueeze(0)}
        logits, _ = model(**inputs)
        chunk_probs = nnf.softmax(logits.squeeze(0), dim=-1).numpy()
        chunk_labels = logits.argmax(dim=-1).squeeze().tolist()
        word_ids = encoding.word_ids(i)
        for idx in range(len(word_ids)):
            word_id = word_ids[idx]
            if word_id is None or encoding['offset_mapping'][i][idx][0] != 0: continue
            label_id = chunk_labels[idx]
            prob = chunk_probs[idx][label_id]
            if word_id not in final_predictions or prob > final_predictions[word_id]['prob']:
                final_predictions[word_id] = {'label_id': label_id, 'prob': float(prob), 'text': test_dict['tokens'][word_id], 'box': encoding['bbox'][i][idx].tolist()}

print('--- Final Predictions Details for CPC000692108 ---')
# Check specifically for tokens that should be amount_due
for idx in range(len(test_dict['tokens'])):
    p = final_predictions.get(idx)
    if p:
        text = p['text'].replace('\n', ' ')
        label = id2label.get(p['label_id'], 'O')
        if idx > 110:
           print(f'{idx}: [{text}] -> {label} (conf: {p["prob"]:.4f}) at box {p["box"]}')
