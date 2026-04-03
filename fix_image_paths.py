"""
Converts absolute `file_name` paths in Training_layoutLMV3.json to relative paths.
Result: "images/<filename>.png"
Run once from the project root before transferring to another machine.
"""

import json
import os

JSON_PATH = os.path.join(os.path.dirname(__file__), "inputs", "Training_layoutLMV3.json")
OUTPUT_PATH = JSON_PATH  # overwrite in-place (backup first)

# --- backup ---
backup_path = JSON_PATH + ".bak"
if not os.path.exists(backup_path):
    with open(JSON_PATH, "r") as f:
        raw = f.read()
    with open(backup_path, "w") as f:
        f.write(raw)
    print(f"Backup saved to: {backup_path}")
else:
    print(f"Backup already exists: {backup_path} (skipping overwrite)")

# --- load ---
with open(JSON_PATH, "r") as f:
    data = json.load(f)

# --- convert ---
changed = 0
for item in data:
    old_path = item.get("file_name", "")
    # Extract just the filename and build a relative path
    filename = os.path.basename(old_path)
    new_path = f"images/{filename}"
    if old_path != new_path:
        item["file_name"] = new_path
        changed += 1

print(f"Updated {changed} / {len(data)} records.")

# --- save ---
with open(OUTPUT_PATH, "w") as f:
    json.dump(data, f, indent=4)

print(f"Done. Saved to: {OUTPUT_PATH}")
