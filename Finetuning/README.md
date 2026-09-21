# LayoutLMv3 Batch Inference Tool

This folder is a standalone tool for extracting information from invoice PDFs using a fine-tuned LayoutLMv3 model and PaddleOCR. It is designed to automatically process all PDF documents found in the `inputs/` directory in a single run.

---

## 🛠 Prerequisites

Before running this tool on a new machine, ensure you have the following installed:

1.  **Python 3.10**: Download from [python.org](https://www.python.org/). 
    *   *Note: Ensure "Add Python to PATH" is checked during installation.*
2.  **Pipenv**: Open your terminal (PowerShell or CMD) and run:
    ```powershell
    pip install pipenv
    ```

---

## 🚀 Setup & Execution (3 Steps)

Follow these steps to run the inference on any computer:

### 1. Copy the Folder
Copy the entire `Finetuning` folder to the target computer. 

> [!IMPORTANT]
> Do not move files out of this folder! The script depends on the `src` and `inputs` folders being exactly where they are.

### 2. Initialize the Environment
Open a terminal **inside** the `Finetuning` folder and run:
```powershell
pipenv install
```
*This will automatically create a virtual environment and install all heavy dependencies (Torch, Transformers, PaddleOCR, etc.). This step takes 5-10 minutes.*

### 3. Run the Inference
Ensure you have at least one invoice PDF inside the **`inputs/`** folder, then **Double-Click** the batch file:
*   **`run_inference_PORTABLE.bat`**

---

## 📁 Folder Structure Explained

*   **`inference_script.py`**: The main entry point.
*   **`src/`**: The "Engine" (architecture and OCR logic).
*   **`inputs/`**: Place your **invoice PDFs** here. Also contains label configurations.
*   **`run_inference_PORTABLE.bat`**: A convenient script to run the tool.

---

## 💡 Troubleshooting

*   **"No module named 'torch'"**: Ensure you have run `pipenv install` as described in Step 2.
*   **"No PDF files found"**: Place a PDF file inside the **`inputs/`** folder.
*   **Windows Path Issues**: If you see errors about "Path too long", try moving the folder to a shorter path like `C:\Inference\`.
