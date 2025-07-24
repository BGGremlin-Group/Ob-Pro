# Usage Guide for Ob-Pro 5.0

Ob-Pro 5.0 is an advanced obfuscation and encryption tool with GUI and CLI interfaces. This guide covers basic usage.

## Running the Software
Ensure you’ve followed the [Installation Guide](installation.md).

- **GUI Mode**:
  ```bash
  python main.py



  Launches the Tkinter-based GUI with options for:Text obfuscation (layered ciphers, wordlists).

Image processing (steganography, OCR).

Custom glyph creation (TTF/OTF fonts).

Exporting outputs (PNG, PDF, QR code, barcode).

CLI Mode:bash



python main.py --cli



Displays a menu-driven interface with options:Obfuscate text

Decrypt text

Process image

Create custom characters

Exit


Key FeaturesText Obfuscation:Select ciphers (AES, ChaCha20, RSA, Polybius) via config/cipher_rules.json.

Use wordlists (obf_list1.json, obf_list2.json, braille_morse.json).

Enable quadrant mode for advanced obfuscation.


Image Processing:Extract text from images using OCR (Tesseract).

Hide data in images via steganography (config/image_processing.json).

Export as PNG, PDF, QR code, or barcode.


Custom Glyphs:Create TTF/OTF fonts with custom symbols (config/font_generator.json).

Output saved to data/fonts/.


Cloud Storage:Save/load encrypted outputs via config/cloud_storage.json.

Share outputs with obpro://cloud/ links.


Accessibility:Customize GUI/CLI appearance (config/accessibility.json).

Supports colorblind and vision-impaired modes.


ExamplesGUI: Launch python main.py, enter text in the input field, select “High” obfuscation, and click “Obfuscate”. Export as QR code to data/output/.

CLI: Run python main.py --cli, select option 1, enter text, choose ciphers, and save output to data/output/encrypted.txt.



ConfigurationEdit files in config/ to customize:config.json: Theme, grid size, rules.

cipher_rules.json: Cipher settings.

image_processing.json: Steganography and OCR settings.

accessibility.json: GUI/CLI appearance.

cloud_storage.json: Cloud storage options.

font_generator.json: Font creation settings.

playlist.json: GUI music playback.



TroubleshootingGUI Not Loading: Ensure Tkinter is installed (pip install tk).

OCR Errors: Verify Tesseract path in config/image_processing.json.

Font Issues: Install FontForge for glyph creation.

Contact: bggg-contact@neon-grid.io (mailto:bggg-contact@neon-grid.io) or open a GitHub issue.



**Legal** By using Ob-Pro 5.0, you agree to the BGGG License v1.0 (../LICENSE.md) and NDA (../NDA.md). The software is provided "AS IS" with no guarantees.
