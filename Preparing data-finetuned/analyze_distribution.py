import json
import os
import glob
from collections import Counter

def analyze_distribution():
    ocr_dir = r'd:\Internship\Fine-tuning\ocr_with_ner_tags'
    label_map_path = os.path.join(ocr_dir, 'label_map.json')
    
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
        id2label = label_map['id2label']
    
    all_tags = []
    json_files = glob.glob(os.path.join(ocr_dir, "*.json"))
    
    # Exclude label_map.json itself if it matches the glob
    json_files = [f for f in json_files if 'label_map.json' not in f]
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            if 'ner_tags' in data:
                all_tags.extend(data['ner_tags'])
    
    counts = Counter(all_tags)
    
    print("Class Distribution:")
    print(f"{'ID':<5} {'Label':<30} {'Count':<10}")
    print("-" * 50)
    for label_id, count in sorted(counts.items(), key=lambda x: int(x[0])):
        label_name = id2label.get(str(label_id), "Unknown")
        print(f"{label_id:<5} {label_name:<30} {count:<10}")

if __name__ == "__main__":
    analyze_distribution()
