# v4.x → v5.0 Migration Note & Change Log

# Changelog

All notable changes to **Ob-Pro** will be documented in this file.

## [5.0.0] – 2025-XX-XX

### Added
- **New ciphers**: Serpent, Twofish, Blowfish (CBC mode with Argon2-derived keys).
- **HMAC-SHA256** integrity tagging for cloud sync and data at rest.
- **OCR pipeline (pytesseract + OpenCV)** with preprocessing (thresholding, contrast, denoise).
- **Quadrant / grid OCR & obfuscation** (2x2, 3x3, custom).
- **FontForge-based glyph editor** with PUA mapping, TTF export, and automatic JSON wordlist generation.
- **Secure cloud sync** using ChaCha20 + HMAC + dummy `bggg://` share links.
- **Dynamic wordlist mutation** (frequency-aware) to increase anti-frequency-analysis resilience.
- **Low-resource auto-detection** (≤2 cores or ≤2GB RAM): reduced GUI footprint, smaller grids, disabled animations.
- **Colorblind & vision-impaired themes** with high-contrast defaults.
- **Click-through license acceptance** (GUI & CLI) with hash logging.

### Changed
- **Dependency bootstrapper** on first run: auto-installs missing packages.
- **Config & output directories** standardized across Windows/Linux/Termux, with Termux-friendly cloud/output mappings.
- **Enhanced logging** w/ redactable paths & error details.

### Fixed
- Various crashes in Termux/Android due to GUI attempts — now falls back to CLI cleanly.
- Cryptography exceptions now correctly logged and surfaced to the user.
- Wordlist rotation & adaptive cipher edge-cases.

### Security
- Non-cryptographic “stream cipher” now labeled clearly as **NOT SECURE**.
- HMAC added to detect tampering in cloud saves.
- Keys erased from memory & files securely wiped (best-effort, platform-dependent).

### Deprecated / Removed
- Any non-reproducible random key generation without KDF/nonce handling.
- Legacy poorly-documented config keys in `config.json`.

---

## [4.5.x] – 2024-XX-XX

 Security Hardening Checklist

Create SECURITY-HARDENING.md:

Cryptography & Key Management

[ ] Use Argon2 / PBKDF2 for all password→key derivations (in code already).

[ ] Use unique IVs/nonces per encryption (already done for AES/ChaCha20).

[ ] Store keys in KMS / Vault / OS keychain, never plaintext in configs.

[ ] HMAC-SHA256 all blobs stored in cloud sync (already present).

[ ] Consider AEAD modes (ChaCha20-Poly1305, AES-GCM) for combined enc+auth (future roadmap).


File & Memory Hygiene

[ ] Secure-delete temp files (secure_delete) — already in code, but verify FS actually overwrites.

[ ] Zeroize secrets in memory (Python makes this tricky; use best effort and minimize exposure).

[ ] Redact sensitive values from logs (keys, plaintexts, personally identifiable data).


Operational Controls

[ ] Enforce license acceptance logging and version/hash pinning.

[ ] Lock cloud sync to known directories and check write permissions.

[ ] Provide export control / sanctions screening for enterprise distribution.

[ ] Use pip-compile or hash-pinned requirements for supply-chain integrity.

[ ] Run pip-audit or safety in CI to detect vulnerable dependencies.


Platform / Environment

[ ] Termux / Android: confirm no restricted crypto exports.

[ ] Linux servers: disable GUI/autoplay features by default.

[ ] Windows: ensure FontForge path is sanitized and verified.





> Note: Versions & licenses change. Auto-generate this file on every release with pip-licenses (or pipdeptree + licensecheck). Below is a starter table; replace with the generated one at build time.

Auto-generate command (recommended)

pip install pip-licenses
pip-licenses \
  --format=markdown \
  --with-license-file \
  --order=license \
  --output-file LICENSES-3RD-PARTY.md

Typical Core Dependencies (Verify w/ the command above before publishing)

Package	Typical License*	Why We Use It

cryptography	Apache-2.0 / BSD	Modern crypto primitives
pillow (PIL)	HPND (PIL)	Image processing
pygame	LGPL / GPL mix	Music playback
qrcode	MIT	QR code generation
colorama	BSD-3-Clause	CLI color support
numpy	BSD-3-Clause	Array ops / image handling
rich	MIT	Beautiful CLI output
prompt_toolkit	BSD-3-Clause	CLI UX (completions)
python-barcode	MIT	Barcode generation
fontforge-python	GPLv3 (verify!)	Font creation / PUA export
reportlab	BSD-like (RLPL)	PDF generation
pytesseract	Apache-2.0	OCR bridge to Tesseract
opencv-python	Apache-2.0	Image preprocessing for OCR
argon2-cffi	MIT	KDF (secure password → key)
psutil	BSD-3-Clause	Resource detection (low-res mode)
tkinter	Tcl/Tk (bundled)	GUI framework


* Always verify: Run pip-licenses at build time to ensure compliance and accuracy.
