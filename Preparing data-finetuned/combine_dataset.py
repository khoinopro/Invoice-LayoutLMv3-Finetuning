import json
import os
import glob
from tqdm import tqdm

def combine_dataset():
    # Paths
    ocr_dir = r'd:\Internship\Fine-tuning\ocr_with_ner_tags'
    img_dir = r'd:\Internship\Fine-tuning\images'
    output_path = r'd:\Internship\Fine-tuning\LayoutLMv3\Training_layoutLMV3.json'
    label_map_path = os.path.join(ocr_dir, 'label_map.json')
    
    # Check dependencies
    if not os.path.exists(label_map_path):
        print(f"Error: label_map.json not found in {ocr_dir}")
        return
        
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
        id2label = label_map['id2label']
        
    json_files = glob.glob(os.path.join(ocr_dir, "*.json"))
    # Exclude label_map itself
    json_files = [f for f in json_files if 'label_map.json' not in f]
    
    print(f"Found {len(json_files)} files to combine.")
    
    combined_data = []
    
    for json_file in tqdm(json_files, desc="Combining JSONs"):
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        filename = data['file_name']
        tokens = data['tokens']
        bboxes = data['bboxes']
        ner_tags = data['ner_tags']
        
        # Absolute image path
        img_path = os.path.join(img_dir, filename)
        if not os.path.exists(img_path):
            print(f"Warning: Image not found at {img_path}")
            # We use absolute path as requested for the trainer
            img_path = os.path.abspath(img_path)
        else:
            img_path = os.path.abspath(img_path)
            
        annotations = []
        for i in range(len(tokens)):
            label_id = str(ner_tags[i])
            label_name = id2label.get(label_id, "O")
            
            annotations.append({
                "text": tokens[i],
                "box": bboxes[i],
                "label": label_name
            })
            
        doc_entry = {
            "file_name": img_path,
            "annotations": annotations
        }
        
        combined_data.append(doc_entry)
        
    # Save output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=4, ensure_ascii=False)
        
    print(f"Successfully combined {len(combined_data)} documents into {output_path}")

if __name__ == "__main__":
    combine_dataset()
