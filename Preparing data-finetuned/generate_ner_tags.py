import json
import os
import glob
from PIL import Image

def normalize_bbox(bbox, width, height):
    """Normalize bbox [x1, y1, x2, y2] from pixels to 0-1000."""
    return [
        int(1000 * bbox[0] / width),
        int(1000 * bbox[1] / height),
        int(1000 * bbox[2] / width),
        int(1000 * bbox[3] / height)
    ]

def is_inside(box_inner, box_outer):
    """Check if the center of box_inner is inside box_outer."""
    cx = (box_inner[0] + box_inner[2]) / 2
    cy = (box_inner[1] + box_inner[3]) / 2
    return (box_outer[0] <= cx <= box_outer[2]) and (box_outer[1] <= cy <= box_outer[3])

def get_label_map(labels_file):
    with open(labels_file, 'r') as f:
        lines = f.readlines()
    
    unique_labels = []
    for line in lines:
        line = line.strip()
        if line and not line.endswith(':') and not line.startswith('General') and not line.startswith('Line Item') and not line.startswith('Based on'):
            if line not in unique_labels:
                unique_labels.append(line)
    
    label2id = {"O": 0}
    id2label = {0: "O"}
    idx = 1
    for label in unique_labels:
        label2id[f"B-{label}"] = idx
        id2label[idx] = f"B-{label}"
        idx += 1
        label2id[f"I-{label}"] = idx
        id2label[idx] = f"I-{label}"
        idx += 1
    return label2id, id2label

def generate_tags_batch():
    ocr_dir = r'd:\Internship\Fine-tuning\ocr_without_ner_tags'
    ann_dir = r'd:\Internship\Fine-tuning\annotations_docile'
    img_dir = r'd:\Internship\Fine-tuning\images'
    labels_file = r'd:\Internship\Fine-tuning\Labels.txt'
    output_dir = r'd:\Internship\Fine-tuning\ocr_with_ner_tags'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    label2id, id2label = get_label_map(labels_file)
    
    # Save the mapping for reference
    mapping_path = os.path.join(output_dir, 'label_map.json')
    with open(mapping_path, 'w') as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=4)
        
    ocr_files = glob.glob(os.path.join(ocr_dir, "*.json"))
    print(f"Found {len(ocr_files)} files to process.")
    
    for ocr_file in ocr_files:
        filename = os.path.basename(ocr_file)
        # Assuming filename is like: invoice_CPC000692108_page0.json
        base_name = filename.replace("_page0.json", "")
        
        ann_file = os.path.join(ann_dir, base_name + ".json")
        img_file = os.path.join(img_dir, base_name + "_page0.png")
        
        if not os.path.exists(ann_file) or not os.path.exists(img_file):
            print(f"Skipping {filename}: Missing annotation or image.")
            continue
            
        print(f"Processing {filename}...")
        
        with open(ocr_file, 'r') as f:
            ocr_data = json.load(f)
        
        with open(ann_file, 'r') as f:
            ann_data = json.load(f)
            
        try:
            img = Image.open(img_file)
            width, height = img.size
        except Exception as e:
            print(f"Error opening image {img_file}: {e}")
            continue
            
        fields = []
        for item in ann_data.get('field_extractions', []):
            if item.get('bbox') and item.get('page') == 1: # Assuming page 1 is page0.png
                fields.append(item)
        for item in ann_data.get('line_item_extractions', []):
            if item.get('bbox') and item.get('page') == 1:
                fields.append(item)
                
        tokens = ocr_data['tokens']
        ocr_bboxes = ocr_data['bboxes']
        token_to_field_map = {}
        
        for i, ocr_bbox in enumerate(ocr_bboxes):
            for f_idx, field in enumerate(fields):
                gt_bbox_pixels = field['bbox']
                gt_bbox_norm = normalize_bbox(gt_bbox_pixels, width, height)
                
                if is_inside(ocr_bbox, gt_bbox_norm):
                    token_to_field_map[i] = f_idx
                    break
                    
        ner_tags_str = ["O"] * len(tokens)
        field_started = {}
        
        for i in range(len(tokens)):
            if i in token_to_field_map:
                f_idx = token_to_field_map[i]
                field_type = fields[f_idx]['fieldtype']
                if f_idx not in field_started:
                    ner_tags_str[i] = f"B-{field_type}"
                    field_started[f_idx] = True
                else:
                    ner_tags_str[i] = f"I-{field_type}"
            else:
                ner_tags_str[i] = "O"
                
        # Map to IDs
        ner_tags_id = [label2id.get(tag, 0) for tag in ner_tags_str]
        ocr_data['ner_tags'] = ner_tags_id
        
        # Save the output
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(ocr_data, f, indent=4)
            
    print(f"Exported all tagged JSON files to {output_dir}")

if __name__ == "__main__":
    generate_tags_batch()
