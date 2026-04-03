# Inference script using the BASE LayoutLMv3 model (No fine-tuning)

import torch
import torch.nn.functional as nnf
import os
import json
import numpy as np
from PIL import Image
from transformers import (
    LayoutLMv3ImageProcessor,
    LayoutLMv3TokenizerFast,
    LayoutLMv3Processor,
)
from trainer import ModelModule
from utils import dataSetFormat, plot_img

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# Note: We are specifically NOT loading model.bin here
model_path    = "microsoft/layoutlmv3-base"
config_path   = os.path.join(PROJECT_ROOT, "inputs", "label_config.json")

# ---------------------------------------------------------------------------
# Load label mapping (required to define the model head size)
# ---------------------------------------------------------------------------
with open(config_path, "r") as f:
    config = json.load(f)
id2label = {int(v): k for k, v in config["label2id"].items()}
num_labels = len(id2label)

# ---------------------------------------------------------------------------
# Build processor
# ---------------------------------------------------------------------------
feature_extractor = LayoutLMv3ImageProcessor(apply_ocr=False)
tokenizer         = LayoutLMv3TokenizerFast.from_pretrained(model_path, ignore_mismatched_sizes=True)
processor         = LayoutLMv3Processor(tokenizer=tokenizer, image_processor=feature_extractor)

# ---------------------------------------------------------------------------
# Load model (Base weights only, NO model.bin)
# ---------------------------------------------------------------------------
print(f"[INFO] Initializing BASE model: {model_path}")
model = ModelModule(num_labels) # This creates the model with a random head
model.eval()
print("[OK] Base model initialized (untrained classification head).")

# ---------------------------------------------------------------------------
# Core Inference Function
# ---------------------------------------------------------------------------
def run_inference(image_path, output_root):
    if not os.path.exists(image_path):
        print(f"[Error] Image not found: {image_path}")
        return

    # Create subdirectories as requested: annotations_base and visualize_base
    ann_dir = os.path.join(output_root, "annotations_base")
    viz_dir = os.path.join(output_root, "visualize_base")
    os.makedirs(ann_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    # 1. Load image and run OCR
    image = Image.open(image_path).convert("RGB")
    print(f"\n[OK] Processing: {os.path.basename(image_path)}")
    
    test_dict, width_scale, height_scale = dataSetFormat(image)

    # 2. Encode with Sliding Window
    encoding = processor(
        image,
        test_dict["tokens"],
        boxes=test_dict["bboxes"],
        max_length=512,
        stride=128,
        padding="max_length",
        truncation=True,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    # 3. Inference Loop
    num_chunks = len(encoding.input_ids)
    final_predictions = {}

    for i in range(num_chunks):
        with torch.no_grad():
            inputs = {
                "input_ids":      encoding["input_ids"][i].unsqueeze(0),
                "attention_mask": encoding["attention_mask"][i].unsqueeze(0),
                "bbox":           encoding["bbox"][i].unsqueeze(0),
                "pixel_values":   encoding["pixel_values"][i].unsqueeze(0),
            }
            logits, _ = model(**inputs)
            chunk_probs = nnf.softmax(logits.squeeze(0), dim=-1).numpy()
            chunk_labels = logits.argmax(dim=-1).squeeze().tolist()
            word_ids = encoding.word_ids(i)
            
            for idx, word_id in enumerate(word_ids):
                if word_id is None or encoding["offset_mapping"][i][idx][0] != 0:
                    continue
                label_id = chunk_labels[idx]
                prob = chunk_probs[idx][label_id]
                if word_id not in final_predictions or prob > final_predictions[word_id]["prob"]:
                    final_predictions[word_id] = {
                        "label_id": label_id,
                        "prob": float(prob),
                        "box": encoding["bbox"][i][idx].tolist(),
                        "text": test_dict["tokens"][word_id]
                    }

    # 4. Filter and Results
    image_name = os.path.basename(image_path)
    image_basename = os.path.splitext(image_name)[0]
    results = []
    for word_id in sorted(final_predictions.keys()):
        pred = final_predictions[word_id]
        label = id2label.get(pred["label_id"], "O")
        # Base model predictions will likely be random, but we still filter 'O'
        if label != "O":
            results.append({
                "label": label,
                "confidence": pred["prob"],
                "text": pred["text"],
                "box_2d": pred["box"]
            })

    # 5. Save JSON in structured folder
    json_path = os.path.join(ann_dir, f"{image_basename}.json")
    with open(json_path, "w") as f:
        json.dump({"image": image_name, "predictions": results}, f, indent=4)

    # 6. Visualise in structured folder
    if len(results) > 0:
        plot_img(
            image,
            [r["box_2d"] for r in results],
            [r["label"] for r in results],
            [r["confidence"] for r in results],
            width_scale,
            height_scale,
            output_name=os.path.join(viz_dir, f"{image_basename}.jpg")
        )
    else:
        print(f"[INFO] No non-'O' detections for {image_name} with Base Model.")

if __name__ == "__main__":
    # Configure input and output paths
    IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")
    OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "inference_base_model")
    
    if not os.path.exists(IMAGES_DIR):
        print(f"[Error] Images folder not found at: {IMAGES_DIR}")
        exit(1)

    print(f"--- Starting Batch Inference (BASE MODEL) from {IMAGES_DIR} ---")
    
    exts = (".png", ".jpg", ".jpeg")
    images = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(exts)]
    
    print(f"[INFO] Found {len(images)} images to process.")
    
    for i, img_name in enumerate(images):
        img_path = os.path.join(IMAGES_DIR, img_name)
        print(f"[{i+1}/{len(images)}] ", end="")
        try:
            run_inference(img_path, OUTPUT_ROOT)
        except Exception as e:
            print(f"[Error] Failed to process {img_name}: {e}")
    
    print(f"\n--- Base Model Inference Complete! Results stored in {OUTPUT_ROOT} ---")
