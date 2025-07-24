This file provides detailed instructions for installing and setting up Ob-Pro 5.0.File Name: docs/installation.md

# Installation Guide for Ob-Pro 5.0

This guide outlines the steps to install and set up Ob-Pro 5.0, an advanced obfuscation and encryption tool.

## Prerequisites
- **Python**: Version 3.8 or higher. Download from [python.org](https://www.python.org/downloads/).
- **Git**: Required to clone the repository. Install from [git-scm.com](https://git-scm.com).
- **Optional Tools**:
  - **Tesseract OCR**: For image text extraction. Install from [github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract).
  - **FontForge**: For custom font generation. Install from [fontforge.org](https://fontforge.org).
- A system running Windows, Linux, or macOS.

## Installation Steps
1. **Clone the Repository**:
   ```bash
   git clone [https://github.com/BGGremlin-Group/Ob-Pro.git](https://github.com/BGGremlin-Group/Ob-Pro/main.git
   cd Ob-Pro



Install Dependencies:
Install required Python packages listed in requirements.txt:bash



pip install -r requirements.txt



The following packages will be installed:cryptography>=41.0.7: For encryption (AES, ChaCha20, RSA).

pillow>=10.4.0: For image processing.

pygame>=2.6.0: For GUI music playback.

qrcode>=7.4.2: For QR code generation.

colorama>=0.4.6: For CLI color output.

numpy>=1.26.4: For numerical operations.

rich>=13.7.1: For enhanced CLI formatting.

prompt_toolkit>=3.0.47: For CLI interaction.

python-barcode>=0.15.1: For barcode generation.

fontforge-python>=20230101: For font creation.

reportlab>=4.2.2: For PDF output.

pytesseract>=0.3.10: For OCR.

opencv-python>=4.10.0: For image preprocessing.

psutil>=5.9.8: For system resource monitoring.


Install Optional Tools:Tesseract OCR:Windows: Download and install from UB-Mannheim/tesseract.

Linux: sudo apt-get install tesseract-ocr

macOS: brew install tesseract

Update config/image_processing.json with the correct tesseract_path.


FontForge:Windows/Linux/macOS: Follow instructions at fontforge.org.

Ensure fontforge-python is installed via pip.


Verify Setup:
Run the script to ensure it works:bash



python main.py



For CLI mode:bash



python main.py --cli


TroubleshootingMissing Dependencies: Ensure all packages in requirements.txt are installed.

Tesseract Not Found: Verify the tesseract_path in config/image_processing.json.

FontForge Errors: Install FontForge and fontforge-python.

Permission Issues: Run commands with sudo on Linux/macOS if needed.

Contact: Open a GitHub issue.



**Legal** By downloading or using Ob-Pro 5.0, you agree to the BGGG License v1.0 (../LICENSE.md) and NDA (../NDA.md). The software is provided "AS IS" with no guarantees
