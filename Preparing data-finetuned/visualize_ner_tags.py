import json
import os
import glob
from PIL import Image, ImageDraw, ImageFont
import random

def visualize_ner_batch():
    ocr_dir = r'd:\Internship\Fine-tuning\ocr_with_ner_tags'
    label_map_file = r'd:\Internship\Fine-tuning\ocr_with_ner_tags\label_map.json'
    img_dir = r'd:\Internship\Fine-tuning\images'
    output_dir = r'd:\Internship\Fine-tuning\ners_visualize'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(label_map_file, 'r') as f:
        label_map = json.load(f)
        id2label = label_map['id2label']
        
    ocr_files = glob.glob(os.path.join(ocr_dir, "*.json"))
    # Exclude the label_map itself
    ocr_files = [f for f in ocr_files if "label_map.json" not in f]
    
    print(f"Found {len(ocr_files)} files to visualize.")
    
    # Try to load a font, otherwise use default
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except:
        font = ImageFont.load_default()
        
    for ocr_file in ocr_files:
        filename = os.path.basename(ocr_file)
        base_name = filename.replace("_page0.json", "")
        img_file = os.path.join(img_dir, base_name + "_page0.png")
        
        if not os.path.exists(img_file):
            print(f"Skipping {filename}: Missing image.")
            continue
            
        print(f"Visualizing {filename}...")
        
        with open(ocr_file, 'r') as f:
            ocr_data = json.load(f)
            
        img = Image.open(img_file).convert('RGB')
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        tokens = ocr_data['tokens']
        bboxes = ocr_data['bboxes']
        ner_tags = ocr_data['ner_tags']
        
        # Generate colors for each label category
        # Using a fixed seed for consistent colors across multiple images
        random.seed(42)
        unique_labels = list(set([id2label[str(tag)] for tag in ner_tags if tag != 0]))
        label_colors = {}
        for label in unique_labels:
            base_label = label[2:] if label.startswith(('B-', 'I-')) else label
            if base_label not in label_colors:
                label_colors[base_label] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                
        for i, tag in enumerate(ner_tags):
            if tag == 0:
                continue
                
            label = id2label[str(tag)]
            base_label = label[2:] if label.startswith(('B-', 'I-')) else label
            color = label_colors[base_label]
            
            x1, y1, x2, y2 = bboxes[i]
            px1 = x1 * width / 1000
            py1 = y1 * height / 1000
            px2 = x2 * width / 1000
            py2 = y2 * height / 1000
            
            draw.rectangle([px1, py1, px2, py2], outline=color, width=2)
            
            if label.startswith('B-'):
                draw.text((px1, py1 - 15), base_label, fill=color, font=font)
                
        output_path = os.path.join(output_dir, base_name + "_ner.png")
        img.save(output_path)
        
    print(f"All visualizations saved to {output_dir}")

if __name__ == "__main__":
    visualize_ner_batch()
