### Ob-Pro 5.0 README
---
⚖️ **Legal Notice**  
Ob-Pro 5.0 is proprietary software licensed under the **BG Gremlin Group License v1.0 (BGGG License v1.0)** by BG Gremlin Group (“BGGG”), a Swiss GmbH operating exclusively online, contactable via [https://github.com/BGGremlin-Group/Ob-Pro]([https://github.com/BGGremlin-Group/Ob-Pro](https://github.com/BGGremlin-Group/Ob-Pro/main/LICENSE.md)](https://github.com/BGGremlin-Group/Ob-Pro). By downloading the Software, you irrevocably agree to be bound by the BGGG License v1.0 and the associated Non-Disclosure Agreement (NDA) in full, as these agreements are universally applicable to all who obtain the Software. Use is strictly limited to Authorized Users as defined in the License. Reverse engineering, redistribution, publication, or modification is prohibited unless expressly permitted in writing by BGGG. The Software is provided “AS IS” with no warranties, guarantees, or promises of successful obfuscation or encryption. BGGG does not endorse any use or misuse of the Software, and Licensee is solely responsible for all outcomes and legal compliance. BGGG disclaims all liability for damages, including indirect, incidental, or consequential damages, to the maximum extent permitted by law.  

**Contract Changes**: The License, NDA, and all terms governing Ob-Pro 5.0 may be changed, updated, or replaced at any time, without prior notice, at the sole discretion of BGGG, to the maximum extent permitted by applicable law. By downloading, possessing, or using the Software after any change, you agree to the then-current terms in full.  

**Full License and NDA**: See [LICENSE](LICENSE), [NDA](NDA.md), or [https://[your-domain]/licenses/bggg-v1](https://[your-domain]/licenses/bggg-v1).
---

**Ob-Pro 5.0** is a powerful Python application for advanced text and image obfuscation, encryption, and custom glyph creation. Built for security enthusiasts, developers, and cyberpunk coders, it offers layered cryptographic ciphers, steganography, a Tkinter-based GUI with accessibility themes, a CLI fallback for terminal environments, and a character creation wizard for crafting custom fonts. Whether you’re hiding secrets in images, generating cryptic QR codes, or designing unique runes, Ob-Pro 5.0 is your ultimate tool in the digital underground—use it at your own risk, as BGGG makes no promises of success.

---

## Table of Contents

- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [GUI Mode](#gui-mode)
  - [CLI Mode](#cli-mode)
  - [Walkthrough: Creating a Custom Font](#walkthrough-creating-a-custom-font)
  - [Walkthrough: Obfuscating Text](#walkthrough-obfuscating-text)
  - [Walkthrough: Encrypting an Image](#walkthrough-encrypting-an-image)
- [Configuration](#configuration)
- [Supported Ciphers](#supported-ciphers)
- [Output Formats](#output-formats)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Features

Ob-Pro 5.0 delivers a robust suite of tools for secure and creative data manipulation:

| Feature | Description |
|---------|-------------|
| **Text Obfuscation** | Apply layered ciphers (AES, ChaCha20, RSA, etc.) with custom wordlists and junk characters. Supports quadrant mode for splitting text. |
| **Image Encryption** | Encrypt images with steganography and extract/obfuscate text via OCR. |
| **Custom Glyph Creation** | Design glyphs in a Tkinter-based wizard and export as TTF/OTF fonts using FontForge. |
| **GUI Interface** | Tkinter-based GUI with dark, light, vision-impaired, and colorblind themes. Features animated glitch effects and music playback. |
| **CLI Fallback** | Menu-driven interface for terminal environments (e.g., Termux). |
| **Output Options** | Export to text, JSON, PNG, PDF, QR codes, and barcodes. Cloud storage with encrypted links. |
| **Accessibility** | Low-resource mode for limited CPU/RAM, Termux compatibility, and adjustable GUI themes. |
| **Dependency Management** | Auto-installs missing Python packages on first run. |

> **Info Block: Why Ob-Pro 5.0?**  
> Ob-Pro 5.0 combines military-grade encryption with creative tools like glyph design, making it ideal for developers, security researchers, and artists. Its cross-platform support (Windows, Linux, Termux) ensures accessibility, but downloading binds you to the BGGG License v1.0 and NDA. BGGG neither endorses use nor misuse, and offers no guarantee of obfuscation or encryption success—you’re on your own in the digital sprawl.

---

## System Requirements

To run Ob-Pro 5.0, ensure your system meets the following:

| Component | Requirement |
|-----------|-------------|
| **Operating System** | Windows 10/11, Linux (Ubuntu, Debian, etc.), or Android (Termux) |
| **Python** | Python 3.8 or higher |
| **RAM** | Minimum 2 GB (4 GB recommended for image processing) |
| **Storage** | 500 MB for dependencies and outputs |
| **Dependencies** | See [Installation](#installation) for required Python packages and system tools |

> **Note**: The GUI requires Tkinter, which may need manual installation on some Linux distributions (`sudo apt install python3-tk`) or Termux (`pkg install python-tkinter`). By downloading, you agree to the BGGG License v1.0 and NDA.

---

## Installation

Follow these steps to set up Ob-Pro 5.0:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/ob-pro-5.0.git
   cd ob-pro-5.0
   ```

2. **Install Python Dependencies**:
   Ob-Pro 5.0 auto-installs most Python packages on first run, but you can pre-install them:
   ```bash
   pip install cryptography pillow pygame qrcode colorama numpy rich prompt_toolkit python-barcode reportlab opencv-python psutil
   ```

3. **Install System Dependencies**:
   - **Tesseract OCR** (for image text extraction):
     - Linux: `sudo apt install tesseract-ocr`
     - Windows: Download and install from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
     - Termux: `pkg install tesseract`
   - **FontForge** (for font creation):
     - Linux: `sudo apt install fontforge`
     - Windows: Download from [FontForge website](https://fontforge.org/) and install `fontforge-python` via `pip`.
     - Termux: Limited support; manual setup may be required.
   - **Tkinter** (for GUI):
     - Linux: `sudo apt install python3-tk`
     - Termux: `pkg install python-tkinter`
     - Windows: Included with Python.

4. **Verify Setup**:
   ```bash
   python -c "import tkinter, cryptography, PIL, pygame, qrcode, fontforge, pytesseract"
   ```
   If no errors appear, you’re ready to go!

> **Info Block: Termux Users**  
> Termux requires storage permissions (`termux-setup-storage`) and may struggle with `fontforge` or `opencv-python`. Use CLI mode (`--cli`) for best compatibility. Downloading implies consent to the BGGG License v1.0 and NDA, including restrictions on redistribution and reverse engineering, with no endorsement of use or misuse.

---

## Usage

Ob-Pro 5.0 supports both GUI and CLI modes, with detailed walkthroughs for key features. By downloading the Software, you agree to comply with the BGGG License v1.0 and NDA, including restrictions on unauthorized access, modification, or redistribution, and acknowledge that BGGG does not guarantee successful obfuscation or encryption.

### GUI Mode
1. Run the script:
   ```bash
   python ob_pro_5_0.py
   ```
2. The Tkinter GUI opens with options for:
   - **Text Obfuscation**: Enter text, select ciphers, and choose output formats.
   - **Image Encryption**: Upload an image, extract text (OCR), and obfuscate/encrypt.
   - **Character Creator**: Design custom glyphs and export fonts.
   - **Settings**: Adjust themes, music, and low-resource mode.
3. Use buttons like "Obfuscate Text," "Encrypt Image," or "Create Character" to access features.

### CLI Mode
1. Run with the `--cli` flag:
   ```bash
   python ob_pro_5_0.py --cli
   ```
2. A menu appears with options:
   ```
   [1] Obfuscate Text
   [2] Encrypt/Decrypt Image
   [3] Export to File/Cloud
   [4] Create Custom Font (GUI-only)
   [5] Exit
   ```
3. Enter a number to select an option and follow prompts.

### Walkthrough: Creating a Custom Font
Design and export a custom font with the character creation wizard, compliant with the BGGG License v1.0 and NDA.

1. **Launch the Wizard**:
   - In GUI mode, click "Create Character."
   - In CLI mode, select option 4 (warns that GUI is required).
2. **Draw a Glyph**:
   - A Tkinter window opens with an 8x8 or 16x16 grid (based on resource mode).
   - Select a drawing mode: Freehand, Line, or Curve.
   - Adjust brush size (1–5 pixels) via the Spinbox.
   - Draw your glyph (e.g., a diagonal line for 'A').
   - Use the AR Preview to rotate/scale the glyph.
3. **Assign a Character**:
   - Enter a source character (e.g., 'A') in the Entry field.
   - Click "Save Character" to map it to a Unicode PUA codepoint (U+E000–U+F8FF).
4. **Export the Font**:
   - Click "Export Font" to generate `fonts/ObProFont_{timestamp}.ttf` and `.otf`.
   - A JSON wordlist (`obf_custom_{timestamp}.json`) is created with mappings (e.g., `{'A': '\ue000'}`).
   - Optionally upload to the cloud via "Save to Cloud" (encrypted per the License and NDA).
5. **Use the Font**:
   - Use in `obfuscate_text` or `obfuscate_image_text` with the `substitution` cipher.
   - Install the TTF file in your system Fonts directory for external use, subject to License and NDA restrictions.

> **Tip**: Save multiple glyphs before exporting to create a complete font. Check `ob_pro_5_0.log` for errors if FontForge fails. BGGG does not guarantee font creation success, per the anonymity clause.

### Walkthrough: Obfuscating Text
Obfuscate text with layered ciphers and custom glyphs.

1. **Select Mode**:
   - GUI: Enter text in the input field and select ciphers (e.g., ChaCha20, Substitution).
   - CLI: Choose option 1 and enter text when prompted.
2. **Configure Options**:
   - Choose an obfuscation level: `light` (1 layer, 0.1 junk frequency), `standard` (2 layers, 0.3), or `heavy` (3 layers, 0.5).
   - Select a wordlist (e.g., `obf_list1.json`) or use `CHARACTER_POOLS['custom']` for your glyphs.
   - Specify keys for each cipher (e.g., `mysecretkey` for ChaCha20).
3. **Run Obfuscation**:
   - Example input: `"Hello, World!"`
   - Ciphers: `['chacha20', 'substitution']`
   - Output: A string like `☠⚮AE3...⚯==` (base64 with junk glyphs).
4. **Export**:
   - Save as text, JSON, PNG, PDF, QR code, or barcode.
   - Upload to the cloud with encryption for sharing, per Section 5 (Confidentiality) of the License and NDA.

> **Note**: BGGG does not guarantee successful obfuscation, per Section 8.2 of the License and Section 6.2 of the NDA.

### Walkthrough: Encrypting an Image
Hide data in an image or obfuscate extracted text.

1. **Upload an Image**:
   - GUI: Click "Upload Image" and select a PNG/JPG.
   - CLI: Choose option 2 and enter the file path.
2. **Choose Mode**:
   - **Steganography**: Hide text in the image’s least significant bits.
   - **OCR + Obfuscation**: Extract text (via Tesseract), obfuscate it, and re-encode the image.
3. **Configure**:
   - For steganography, enter text to hide and a key (e.g., `mysecretkey`).
   - For OCR, select ciphers and a wordlist.
4. **Run Encryption**:
   - Example: Extract text "Secret" from `image.png`, obfuscate to `⚮SE...⚯`, and save as `output/encrypted_image.png`.
   - A `.key` file is generated for decryption.
5. **Export**:
   - Save the encrypted image or upload to the cloud, ensuring compliance with License and NDA restrictions.

> **Info Block: Steganography**  
> Ob-Pro 5.0 hides up to 1 bit per pixel in RGB channels, ensuring minimal visual impact. Use high-resolution images for larger payloads. BGGG does not guarantee encryption success, and all outputs remain subject to BGGG’s intellectual property rights per Section 4.2 of the License and Section 5 of the NDA.

---

## Configuration

Ob-Pro 5.0 uses JSON files and in-memory settings for customization:

| File | Purpose | Location |
|------|---------|----------|
| `obf_list1.json`, `obf_list2.json` | Wordlists for substitution ciphers (e.g., `{"a": "☠"}`) | Script directory |
| `playlist.json` | Music tracks for GUI playback | Script directory |
| `ob_pro_5_0.log` | Logs errors and actions | Script directory |

**Settings**:
- **Themes**: `dark`, `light`, `vision_impaired`, `colorblind` (GUI only).
- **Low-Resource Mode**: Auto-detected via `psutil`; enables smaller GUI and 8x8 glyph grids.
- **Cloud Storage**: Saves to `cloud/` or `~/bggg_cloud` (Termux) with ChaCha20 encryption.

> **Tip**: Create custom wordlists in JSON format to map characters to glyphs or symbols for unique obfuscation. Protect all configurations as Confidential Information per Section 5 of the License and Section 2 of the NDA.

---

## Supported Ciphers

Ob-Pro 5.0 supports a variety of cryptographic and obfuscation techniques:

| Cipher | Description | Key Requirement |
|--------|-------------|-----------------|
| AES | 256-bit symmetric encryption | 16/24/32-byte key |
| ChaCha20 | Stream cipher with high performance | 32-byte key + nonce |
| RSA | Asymmetric encryption | Public/private key pair |
| Serpent | Secure block cipher | 16/24/32-byte key |
| Twofish | Fast symmetric cipher | 16/24/32-byte key |
| Blowfish | Legacy block cipher | 4–56-byte key |
| Polybius | 5x5 grid substitution | None |
| Homophonic | Multiple substitutions per character | Wordlist |
| Stream Cipher | Custom XOR-based cipher (not secure) | Any key |
| Transposition | Rearranges characters | Numeric key |

> **Warning**: The custom `stream_cipher` is not cryptographically secure. Use AES or ChaCha20 for sensitive data. BGGG does not guarantee cipher effectiveness, per Section 8.2 of the License and Section 6.2 of the NDA.

---

## Output Formats

Ob-Pro 5.0 supports multiple output formats for obfuscated/encrypted data:

| Format | Description | Use Case |
|--------|-------------|----------|
| Text | Plaintext or obfuscated string | Quick sharing |
| JSON | Structured data with mappings | Archiving |
| PNG | Image with steganography or rendered text | Visual output |
| PDF | Formatted document with text/image | Professional reports |
| QR Code | Scannable code for text | Mobile sharing |
| Barcode | EAN-13 or Code 128 | Physical media |

> **Note**: Licensee owns Output but not embedded BGGG IP (e.g., fonts, glyphs, algorithms) per Section 4.2 of the License and Section 5 of the NDA. No guarantee of output usability is provided.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Dependency Errors** | Run `pip install -r requirements.txt` or check `ob_pro_5_0.log` for missing packages. |
| **GUI Fails** | Ensure `tkinter` is installed. Use `--cli` for terminal-only mode. |
| **FontForge Fails** | Install FontForge system package. On Windows, add to PATH. |
| **Tesseract Fails** | Install Tesseract OCR and verify `pytesseract` path. |
| **Permission Denied** | Grant write access to `output/`, `fonts/`, and `cloud/` directories. |
| **Unauthorized Use** | Ensure only Authorized Users access the Software per Section 1.2 of the License and NDA. |

> **Info Block: Logging**  
> Check `ob_pro_5_0.log` for detailed error messages. Enable verbose logging in settings for debugging. Report any unauthorized use to BGGG immediately via [bggg-contact@neon-grid.io](mailto:bggg-contact@neon-grid.io), per Section 6.3 of the License and Section 2.4 of the NDA.

---

## Contributing

Contributions to Ob-Pro 5.0 are welcome but must comply with the BGGG License v1.0 and NDA, particularly Sections 3 (Restrictions) and 4 (Ownership) of the License and Section 2 of the NDA. To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request via [https://github.com/yourusername/ob-pro-5.0](https://github.com/yourusername/ob-pro-5.0).

Please follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/). All Feedback is subject to Section 4.3 of the License and Section 5 of the NDA. BGGG does not endorse contributions that misuse the Software.

---

## License

Ob-Pro 5.0 is licensed under the **BG Gremlin Group License v1.0 (BGGG License v1.0)**, a proprietary license, and is subject to the associated Non-Disclosure Agreement (NDA). By downloading the Software, you irrevocably agree to be bound by the License and NDA in their entirety, as these agreements are universally applicable to all who obtain the Software. Below is a summary; see the full [LICENSE](LICENSE) file, [NDA](NDA.md), for complete terms.

### Key Terms
- **License Grant**: BGGG grants a non-exclusive, non-transferable, non-sublicensable, revocable license to Authorized Users for internal use, subject to compliance with the License and any Order Form/SOW (Section 2).
- **Consent by Download**: Downloading the Software constitutes irrevocable acceptance of the License and NDA in their entirety (Section 2.2; NDA Section 2.5).
- **Restrictions**: Prohibits copying, modifying, reverse engineering, redistributing, or circumventing controls without BGGG’s written consent (Section 3).
- **Ownership**: BGGG retains all rights to the Software, including fonts, glyphs, and algorithms. Licensee owns Output but not embedded BGGG IP (Section 4).
- **Confidentiality**: Licensee must protect BGGG’s Confidential Information (e.g., source code, keys) with reasonable care and limit access to Authorized Users (Section 5; NDA Section 2).
- **Security & Audit**: Licensee must implement security controls and allow BGGG audits for compliance (Section 6; NDA Schedule A).
- **Compliance & Export**: Use must comply with all applicable laws, including Swiss and US export controls. The Software may not be used in embargoed jurisdictions (Section 7).
- **Anonymity and Non-Endorsement**: BGGG does not endorse any use or misuse of the Software, and offers no guarantee or promise of successful obfuscation, encryption, or any other outcome. Licensee is solely responsible for all outcomes and legal compliance (Section 8; NDA Section 6).
- **Warranties & Liability**: Provided “AS IS” with no warranties. BGGG’s liability is capped at fees paid in the prior 12 months or USD/EUR/CHF 100 if no fees were paid (Sections 9–10).
- **Changes to License**: BGGG may revise the License or NDA at any time without notice. Continued possession or use of the Software after changes constitutes acceptance (Section 14; NDA Section 10).

### Unilateral Change Clause
> **Section 14: Changes to this License**  
> 14.1 **Unilateral Changes**. BGGG may revise, update, or replace this License (including any referenced schedules, exhibits, or policies) at any time, in its sole discretion, to the maximum extent permitted by applicable law. Unless expressly stated otherwise, changes are effective immediately upon publication, delivery, or being made reasonably available (including via the Software UI, GitHub repository, or Discord server.
> 14.2 **Continued Possession or Use = Acceptance**. Licensee’s continued possession, access, or use of the Software after any change constitutes acceptance of the then-current License and NDA in full. If Licensee does not agree, Licensee must immediately cease all use, uninstall the Software, and comply with Section 11.3 (Effect of Termination).  
> 14.3 **Order Forms**. In case of a direct conflict between a currently active, fully signed Order Form/SOW and a subsequent unilateral update to this License, the relevant Order Form/SOW controls for its stated term, unless the parties agree otherwise in writing.

### Governing Law and Venue
This License and NDA are governed by the substantive laws of Switzerland, excluding conflict-of-law rules, and, where applicable, 
