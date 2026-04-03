import json
import os
from PIL import Image, ImageDraw, ImageFont
import random

def visualize_new_format():
    input_file = r'd:\Internship\Fine-tuning\invoice_CPC000692108_page0_new.json'
    img_file = r'd:\Internship\Fine-tuning\images\invoice_CPC000692108_page0.png'
    output_dir = r'd:\Internship\Fine-tuning\ners_visualize'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    img = Image.open(img_file).convert('RGB')
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    # Try to load a font, otherwise use default
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except:
        font = ImageFont.load_default()
        
    # Generate colors for each label category
    labels = list(set([item['label'] for item in data if item['label'] != 'O']))
    label_colors = {}
    random.seed(42) # Consistent colors
    for label in labels:
        base_label = label[2:] if label.startswith(('B-', 'I-')) else label
        if base_label not in label_colors:
            label_colors[base_label] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
    for item in data:
        label = item['label']
        if label == 'O':
            continue
            
        base_label = label[2:] if label.startswith(('B-', 'I-')) else label
        color = label_colors[base_label]
        
        # Bbox in JSON is [x1, y1, x2, y2] normalized to 0-1000
        x1, y1, x2, y2 = item['box']
        
        # Scale back to pixels
        px1 = x1 * width / 1000
        py1 = y1 * height / 1000
        px2 = x2 * width / 1000
        py2 = y2 * height / 1000
        
        # Draw rectangle
        draw.rectangle([px1, py1, px2, py2], outline=color, width=2)
        
        # Draw label text if it's a "B-" tag (to avoid clutter)
        if label.startswith('B-'):
            draw.text((px1, py1 - 15), base_label, fill=color, font=font)
            
    output_path = os.path.join(output_dir, "invoice_CPC000692108_page0_new_viz.png")
    img.save(output_path)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    visualize_new_format()
