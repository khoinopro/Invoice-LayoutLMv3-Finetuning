import os
import sys

# Adding CUDA paths manually
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin"
if cuda_path not in os.environ["PATH"]:
    os.environ["PATH"] = cuda_path + os.pathsep + os.environ["PATH"]

import paddle
try:
    from paddleocr import PaddleOCR
    print("Paddle version:", paddle.version.full_version)
    print("GPU available:", paddle.is_compiled_with_cuda())
    print("CUDA version:", paddle.version.cuda())
    print("PaddleOCR imported successfully!")
    
    # Simple initialization test
    ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=paddle.is_compiled_with_cuda())
    print("PaddleOCR engine loaded successfully!")
except Exception as e:
    print(f"Verification failed: {e}")
