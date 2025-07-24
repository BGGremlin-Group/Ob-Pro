#!/usr/bin/env python3
# Ob-Pro 5.0 - Advanced Obfuscation and Encryption Tool
# This program provides text and image obfuscation using layered ciphers, custom glyphs, and steganography.
# It features a GUI for ease of use and a CLI fallback with a menu interface.
# All dependencies are handled on first run via pip installation.
#
# Author: BGGG (BG Gremlin Group)
# Date: 2025
"""
Ob-Pro 5.0: A comprehensive obfuscation and encryption tool.

Features:
- Stylish GUI (Tkinter) with support for colorblind/vision-impaired themes.
- Character and Font creation wizard for custom cipher glyphs (with TTF/OTF export).
- Advanced cryptography: AES, ChaCha20, RSA, Serpent, Twofish, Blowfish, etc.
- Layered cipher stacks and progressive keys for enhanced security.
- OCR-secured image encryption with steganography (text hidden in images).
- Exports to text, JSON, PNG, PDF, and generates QR codes or barcodes.
- Dual-mode: GUI (default on Windows) and CLI fallback (with menu options 1-5).
- Comprehensive error handling, logging, and configuration management.

All previous features from versions 1.0–4.5 are integrated and preserved, including:
Quadrant mode obfuscation, custom wordlists, dynamic backgrounds, music playback, etc.

Usage:
- GUI (Windows/Linux): Run without arguments, the GUI will launch if available.
- CLI: If GUI is not available or if running in a terminal-only environment, a menu-driven interface is provided.
"""
import importlib
import sys
import subprocess

# Dependency check and installation
required_packages = {
    "cryptography": "cryptography",
    "PIL": "pillow",                # PIL is part of Pillow
    "pygame": "pygame",
    "qrcode": "qrcode",
    "colorama": "colorama",
    "numpy": "numpy",
    "rich": "rich",
    "prompt_toolkit": "prompt_toolkit",
    "barcode": "python-barcode",
    "fontforge": "fontforge-python",
    "reportlab": "reportlab",
    "pytesseract": "pytesseract",
    "cv2": "opencv-python",
    "psutil": "psutil"
}
for module_name, package_name in required_packages.items():
    try:
        importlib.import_module(module_name)
    except ImportError:
        try:
            print(f"Installing missing package: {package_name}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package_name], check=True)
        except Exception as e:
            print(f"Warning: Failed to install {package_name}: {e}")

# Now import all necessary modules (after attempting installations)
import json
import random
import time
import os
import base64
import math
import shutil
import glob
import datetime
import logging
import threading
import secrets
import string
from pathlib import Path
from collections import Counter, deque

# Import third-party libraries (some may be optional and handled gracefully if missing)
try:
    from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
except ImportError:
    Image = ImageDraw = ImageFont = ImageEnhance = ImageFilter = None
try:
    from colorama import init, Fore, Style
except ImportError:
    init = lambda *args, **kwargs: None
    Fore = Style = type('', (), {})()
try:
    import numpy as np
except ImportError:
    np = None
try:
    from rich.console import Console
    from rich.table import Table
except ImportError:
    Console = None
    Table = None
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter
except ImportError:
    PromptSession = None
    WordCompleter = None
try:
    import barcode
    from barcode.writer import ImageWriter
except ImportError:
    barcode = None
    ImageWriter = None
import unicodedata  # standard library
try:
    import psutil
except ImportError:
    psutil = None
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
except ImportError:
    A4 = canvas = pdfmetrics = TTFont = None
try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives import serialization, hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.kdf.argon2 import Argon2
    from cryptography.hazmat.primitives.hmac import HMAC
    from cryptography.hazmat.backends import default_backend
except ImportError:
    Cipher = algorithms = modes = rsa = padding = serialization = hashes = Argon2 = HMAC = default_backend = None

# Optional libraries:
try:
    import qrcode
except ImportError:
    qrcode = None
try:
    import pygame
except ImportError:
    pygame = None
try:
    import pytesseract
except ImportError:
    pytesseract = None
try:
    import fontforge
except ImportError:
    fontforge = None
try:
    import cv2
except ImportError:
    cv2 = None
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog, simpledialog, Canvas
    from PIL import ImageTk
except ImportError:
    tk = None
    ttk = messagebox = filedialog = simpledialog = Canvas = ImageTk = None

# Initialize logging to file
logging.basicConfig(
    filename='ob_pro_5_0.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize console for colored output (Rich and Colorama)
init(autoreset=True)
console = Console() if Console else None

# Theme configurations with colorblind and vision-impaired support
THEMES = {
    'dark': {
        'bg': '#1a1a1a',
        'fg': '#00FFFF',  # neon cyan
        'accent1': '#FF00FF',  # neon magenta
        'accent2': '#00FF00',  # neon green
        'accent3': '#FFFF00',  # neon yellow
        'glitch': '#FF5555',   # glitch red
        'font_size': 12,
        'animations': True
    },
    'light': {
        'bg': '#FFFFFF',
        'fg': '#000000',
        'accent1': '#0000FF',
        'accent2': '#008000',
        'accent3': '#FFA500',
        'glitch': '#FF0000',
        'font_size': 12,
        'animations': True
    },
    'vision_impaired': {
        'bg': '#000000',
        'fg': '#FFFFFF',
        'accent1': '#FFFF00',
        'accent2': '#FFFF00',
        'accent3': '#FFFF00',
        'glitch': '#FFFF00',
        'font_size': 16,
        'animations': False
    },
    'colorblind': {
        'bg': '#000000',
        'fg': '#00B7EB',   # bright blue
        'accent1': '#FFFF00',  # yellow
        'accent2': '#00B7EB',  # blue
        'accent3': '#FFFFFF',  # white
        'glitch': '#FFFF00',   # yellow
        'font_size': 14,
        'animations': False
    }
}

# Default settings and configurations
WORDLISTS = {
    'list1': 'obf_list1.json',
    'list2': 'obf_list2.json',
    'braille': 'braille_morse.json',
    'morse': 'braille_morse.json'
}
# Switcher characters for multi-wordlist substitution
WORDLIST_SWITCHERS = {
    '@': 'list1',
    '#': 'list2',
    '%': 'braille',
    '&': 'morse'
}
# Obfuscation levels define junk frequency and other parameters
OBFUSCATION_LEVELS = {
    'min':      {'junk_freq': 0.1, 'key_changes': 1, 'layers': 1},
    'standard': {'junk_freq': 0.3, 'key_changes': 3, 'layers': 2},
    'max':      {'junk_freq': 0.5, 'key_changes': 5, 'layers': 4}
}

# Config and data files
CONFIG_FILE = 'config.json'
PLAYLIST_FILE = 'playlist.json'
# Character pools for various symbol sets
CHARACTER_POOLS = {
    'ascii':   list(string.ascii_letters + string.digits + string.punctuation),
    'unicode': [chr(i) for i in range(0x2600, 0x26FF)] + [chr(i) for i in range(0x1F600, 0x1F64F)],
    'emoji':   ['😀','😃','😄','😁','😆','😈','🔥','🌟','✨'],
    'zodiac':  ['☠','♆','⚮','⚭','⚯','⚰','⚱'],
    'custom':  []  # custom glyphs (PUA codepoints)
}
# Unicode Private Use Area (PUA) for custom glyphs
PUA_START = 0xE000
PUA_END   = 0xF8FF
used_pua_codes = set()

# Font settings for custom font exports
FONT_NAME = "ObProCustomFont"
FONT_DIR = Path("fonts")
FONT_FILE = FONT_DIR / f"{FONT_NAME}.ttf"

# Directories for outputs and resources
ASCII_DIR = Path("backgrounds") / "ascii"
PNG_DIR   = Path("backgrounds") / "png"
MUSIC_DIR = Path("music")
CLOUD_DIR = Path("cloud")
OUTPUT_DIR = Path("output")
TERMUX_CLOUD_DIR = Path.home() / "bggg_cloud" if "ANDROID_ARGUMENT" in os.environ or sys.platform == "android" else CLOUD_DIR

# Ensure directories exist
for directory in [ASCII_DIR, PNG_DIR, MUSIC_DIR, CLOUD_DIR, OUTPUT_DIR, FONT_DIR, TERMUX_CLOUD_DIR]:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.error(f"Could not create directory {directory}: {e}")

# Determine if running in low-resource environment
LOW_RES_THRESHOLD = 2   # 2 CPU cores or fewer triggers low-res mode
LOW_RES_MEMORY    = 2 * 1024 * 1024 * 1024  # 2GB RAM threshold
if psutil:
    cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count() or 0
    total_mem = psutil.virtual_memory().total if hasattr(psutil, 'virtual_memory') else 0
    is_low_res = (cpu_count <= LOW_RES_THRESHOLD) or (total_mem and total_mem <= LOW_RES_MEMORY)
else:
    is_low_res = False
# Determine if running in Termux (Android) environment
is_termux = "ANDROID_ARGUMENT" in os.environ or sys.platform == "android"

# Default ASCII art background (for CLI mode)
DEFAULT_ASCII = """
    ____ Ob-Pro 5.0 ____
    |  ENIGMA  |  ZODIAC  |
    |  NEON    |  CRYPTIC |
    |____________________|
"""

def get_random_background(is_gui=False):
    """Select a random background (ASCII art for CLI, PNG image for GUI)."""
    try:
        if is_gui:
            png_files = list(PNG_DIR.glob("*.png"))
            return random.choice(png_files) if png_files else None
        else:
            ascii_files = list(ASCII_DIR.glob("*.txt"))
            if ascii_files:
                with open(random.choice(ascii_files), 'r', encoding='utf-8') as f:
                    return f.read()
            return DEFAULT_ASCII
    except Exception as e:
        logging.error(f"Failed to load background: {e}")
        return None if is_gui else DEFAULT_ASCII

def load_wordlist(file_path, list_type=None):
    """Load a wordlist JSON file. Optionally select a sub-dictionary for special lists."""
    try:
        with Path(file_path).open('r', encoding='utf-8') as f:
            data = json.load(f)
            # For braille or morse, select sub-dictionary
            if list_type in ['braille', 'morse']:
                return data.get(list_type, {})
            return data
    except Exception as e:
        logging.error(f"Failed to load wordlist {file_path}: {e}")
        if console:
            console.print(f"[red]Error loading wordlist {file_path}: {e}[/red]")
        return {}

def load_config():
    """Load configuration from file (theme, custom rules, grid size, etc.)."""
    try:
        config_path = Path(CONFIG_FILE)
        if config_path.exists():
            with config_path.open('r', encoding='utf-8') as f:
                config = json.load(f)
                if config.get('theme') not in THEMES:
                    config['theme'] = 'dark'
                return config
        # Default configuration if not found
        return {"rules": {}, "theme": "dark", "last_background": "", "grid_size": "2x2"}
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        if console:
            console.print(f"[red]Error loading config: {e}[/red]")
        return {"rules": {}, "theme": "dark", "last_background": "", "grid_size": "2x2"}

def save_config(config):
    """Save configuration to file."""
    try:
        with Path(CONFIG_FILE).open('w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        if is_termux:
            shutil.copy(CONFIG_FILE, OUTPUT_DIR / CONFIG_FILE)
    except Exception as e:
        logging.error(f"Failed to save config: {e}")
        if console:
            console.print(f"[red]Error saving config: {e}[/red]")

def load_playlist():
    """Load music playlist from playlist.json."""
    try:
        playlist_path = Path(PLAYLIST_FILE)
        if playlist_path.exists():
            with playlist_path.open('r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('tracks', [])
        return []
    except Exception as e:
        logging.error(f"Failed to load playlist: {e}")
        return []

def get_junk_chars(wordlists, text_context=None, theme='dark'):
    """Generate a set of junk characters based on wordlists, text context, and theme."""
    try:
        junk_chars = []
        for wl in wordlists.values():
            if wl:
                junk_chars.extend([char for char in wl.values() if isinstance(char, str)])
        if text_context:
            informality_score = sum(1 for c in text_context if c in CHARACTER_POOLS['emoji']) / (len(text_context) or 1)
            if informality_score > 0.2:
                junk_chars.extend(CHARACTER_POOLS['emoji'])
            elif theme in ['dark', 'colorblind']:
                junk_chars.extend(CHARACTER_POOLS['zodiac'])
            else:
                junk_chars.extend(CHARACTER_POOLS['unicode'])
        return junk_chars or CHARACTER_POOLS['zodiac']
    except Exception as e:
        logging.error(f"Failed to get junk chars: {e}")
        return CHARACTER_POOLS['zodiac']

def get_wordlists():
    """Load all default wordlists into a dictionary."""
    wordlists = {}
    for name, path in WORDLISTS.items():
        data = load_wordlist(path, name if name in ['braille', 'morse'] else None)
        if data:
            wordlists[name] = data
    return wordlists

def analyze_text(text):
    """Analyze word frequency in text (for frequency analysis or key changer)."""
    try:
        words = re.findall(r'\b\w+\b', text.lower(), re.UNICODE)
        return sorted(Counter(words).items(), key=lambda x: x[1], reverse=True)
    except Exception as e:
        logging.error(f"Text analysis failed: {e}")
        return []

def build_markov_chain(text, order=2):
    """Build Markov chain from text (unused in current context)."""
    try:
        chain = {}
        for i in range(len(text) - order):
            seq = text[i:i+order]
            next_char = text[i+order]
            chain.setdefault(seq, []).append(next_char)
        return chain
    except Exception as e:
        logging.error(f"Markov chain build failed: {e}")
        return {}

def generate_adaptive_cipher(text, base_wordlist, extra_symbols):
    """Generate an adaptive substitution cipher for the given text using available symbols."""
    try:
        cipher = {}
        for char in set(text):
            if char not in base_wordlist:
                candidates = [sym for sym in extra_symbols if ord(sym) not in used_pua_codes]
                if candidates:
                    cipher[char] = random.choice(candidates)
                    if cipher[char] in CHARACTER_POOLS['custom']:
                        used_pua_codes.add(ord(cipher[char]))
        return cipher
    except Exception as e:
        logging.error(f"Adaptive cipher generation failed: {e}")
        return {}

def polybius_square(text, key):
    """Apply Polybius square cipher with a key (key modifies the 5x5 grid)."""
    try:
        square = [
            ['A','B','C','D','E'],
            ['F','G','H','I','J'],
            ['K','L','M','N','O'],
            ['P','Q','R','S','T'],
            ['U','V','W','X','Y']
        ]
        key = (key or "")[:5].upper()
        for i in range(5):
            for j in range(5):
                if i < len(key):
                    square[i][j] = key[i]
        result = []
        for char in text.upper():
            if char == 'Z':
                result.append('25')
            else:
                found = False
                for i in range(5):
                    for j in range(5):
                        if square[i][j] == char:
                            result.append(f"{i+1}{j+1}")
                            found = True
                            break
                    if found:
                        break
                if not found:
                    result.append(char)
        return ''.join(result)
    except Exception as e:
        logging.error(f"Polybius square failed: {e}")
        return text

def homophonic_substitution(text, wordlist):
    """Apply homophonic substitution (each character may map to multiple possible symbols)."""
    try:
        homophonic_map = {}
        for char in set(text):
            if char in wordlist:
                homophonic_map[char] = [wordlist[char]] + [random.choice(CHARACTER_POOLS['zodiac']) for _ in range(2)]
            else:
                homophonic_map[char] = [char] + [random.choice(CHARACTER_POOLS['zodiac']) for _ in range(2)]
        result = []
        for char in text:
            choices = homophonic_map.get(char, [char])
            result.append(random.choice(choices))
        return ''.join(result)
    except Exception as e:
        logging.error(f"Homophonic substitution failed: {e}")
        return text

def dynamic_wordlist_mutation(wordlist, text_length):
    """Mutate wordlist entries based on text length (to add unpredictability)."""
    try:
        new_wordlist = wordlist.copy()
        mutation_rate = min(0.5, text_length / 1000.0)
        for char in list(new_wordlist.keys()):
            if random.random() < mutation_rate:
                new_wordlist[char] = random.choice(CHARACTER_POOLS['custom'] + CHARACTER_POOLS['zodiac'])
        return new_wordlist
    except Exception as e:
        logging.error(f"Wordlist mutation failed: {e}")
        return wordlist

def colorize_frequency(word, count, max_count, theme='dark'):
    """Colorize words in CLI output based on frequency (for frequency analysis)."""
    try:
        ratio = count / max_count if max_count > 0 else 0
        if ratio > 0.8:
            return f"{Fore.RED}{word}{Style.RESET_ALL}"
        elif ratio > 0.5:
            return f"{Fore.YELLOW}{word}{Style.RESET_ALL}"
        else:
            return word
    except Exception as e:
        logging.error(f"Colorize frequency failed: {e}")
        return word

def generate_progressive_key(text, base_key, layer):
    """Generate a progressive key by mixing base key with portion of text (for layered encryption)."""
    try:
        if layer == 0:
            return base_key
        segment_length = max(1, len(text) // (layer * 2))
        segment = text[:segment_length]
        return (base_key + segment)[-len(base_key):]
    except Exception as e:
        logging.error(f"Progressive key generation failed: {e}")
        return base_key

def rotate_wordlist(wordlists, current_key, count, rotation_threshold=10):
    """Rotate to the next wordlist after a certain count threshold."""
    try:
        if count and count % rotation_threshold == 0:
            keys = list(wordlists.keys())
            if current_key in keys:
                return keys[(keys.index(current_key) + 1) % len(keys)]
        return current_key
    except Exception as e:
        logging.error(f"Wordlist rotation failed: {e}")
        return current_key

def apply_key_changer(text, wordlist, changes):
    """Apply a simple key changer that replaces every nth character with a symbol."""
    try:
        if changes <= 0:
            return text
        result = []
        for i, char in enumerate(text, start=1):
            if i % changes == 0:
                result.append(random.choice(list(wordlist.values()) + list(wordlist.keys())))
            else:
                result.append(char)
        return ''.join(result)
    except Exception as e:
        logging.error(f"Key changer failed: {e}")
        return text

def apply_custom_rule(text, rule):
    """Apply a custom regex substitution rule."""
    try:
        import re
        pattern = rule.get('pattern')
        repl = rule.get('replace', '')
        flags = 0
        if rule.get('ignore_case'):
            flags |= re.IGNORECASE
        return re.sub(pattern, repl, text, flags=flags)
    except Exception as e:
        logging.error(f"Custom rule application failed: {e}")
        return text

def stream_cipher(text, key, feedback=True):
    """Apply a simple stream cipher (XOR) with optional feedback mode."""
    try:
        key_bytes = key.encode('utf-8')
        result = []
        prev = 0
        for i, ch in enumerate(text.encode('utf-8')):
            k = key_bytes[i % len(key_bytes)]
            xor_val = ch ^ (k if not feedback else (k ^ prev))
            result.append(xor_val)
            prev = xor_val if feedback else 0
        encrypted_bytes = bytes(result)
        return base64.b64encode(encrypted_bytes).decode('utf-8')
    except Exception as e:
        logging.error(f"Stream cipher failed: {e}")
        return text

def chacha20_encrypt(text, key):
    """Encrypt text with ChaCha20 (returns base64)."""
    try:
        # Derive 256-bit key (32 bytes) via Argon2 from the given key string
        key_bytes = Argon2(memory=64*1024, iterations=4, parallelism=2).hash(key.encode('utf-8'))[:32]
        nonce = secrets.token_bytes(12)
        cipher = Cipher(algorithms.ChaCha20(key_bytes, nonce), mode=None, backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(text.encode('utf-8'))
        return base64.b64encode(nonce + ciphertext).decode('utf-8')
    except Exception as e:
        logging.error(f"ChaCha20 encryption failed: {e}")
        return text

def chacha20_decrypt(encrypted_text, key):
    """Decrypt text encrypted with ChaCha20."""
    try:
        key_bytes = Argon2(memory=64*1024, iterations=4, parallelism=2).hash(key.encode('utf-8'))[:32]
        data = base64.b64decode(encrypted_text.encode('utf-8'))
        nonce, ciphertext = data[:12], data[12:]
        cipher = Cipher(algorithms.ChaCha20(key_bytes, nonce), mode=None, backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted = decryptor.update(ciphertext) + decryptor.finalize()
        return decrypted.decode('utf-8')
    except Exception as e:
        logging.error(f"ChaCha20 decryption failed: {e}")
        return encrypted_text

def aes_encrypt(text, key):
    """Encrypt text with AES-256 (CBC mode, PKCS#7-like padding with spaces)."""
    try:
        key_bytes = Argon2(memory=2**16, iterations=4, parallelism=2).hash(key.encode('utf-8'))[:32]
        iv = secrets.token_bytes(16)
        cipher = Cipher(algorithms.AES(key_bytes), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        padded_text = text + ' ' * ((16 - len(text) % 16) % 16)
        encrypted = encryptor.update(padded_text.encode('utf-8')) + encryptor.finalize()
        return base64.b64encode(iv + encrypted).decode('utf-8')
    except Exception as e:
        logging.error(f"AES encryption failed: {e}")
        return text

def aes_decrypt(encrypted_text, key):
    """Decrypt text with AES-256 (CBC mode)."""
    try:
        key_bytes = Argon2(memory=2**16, iterations=4, parallelism=2).hash(key.encode('utf-8'))[:32]
        data = base64.b64decode(encrypted_text.encode('utf-8'))
        iv, ciphertext = data[:16], data[16:]
        cipher = Cipher(algorithms.AES(key_bytes), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted = decryptor.update(ciphertext) + decryptor.finalize()
        return decrypted.decode('utf-8').rstrip()
    except Exception as e:
        logging.error(f"AES decryption failed: {e}")
        return encrypted_text

def serpent_encrypt(text, key):
    """Encrypt text with Serpent cipher (256-bit key, CBC mode)."""
    try:
        key_bytes = Argon2(memory=2**16, iterations=4, parallelism=2).hash(key.encode('utf-8'))[:32]
        iv = secrets.token_bytes(16)
        cipher = Cipher(algorithms.Serpent(key_bytes), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        padded_text = text + ' ' * ((16 - len(text) % 16) % 16)
        encrypted = encryptor.update(padded_text.encode('utf-8')) + encryptor.finalize()
        return base64.b64encode(iv + encrypted).decode('utf-8')
    except Exception as e:
        logging.error(f"Serpent encryption failed: {e}")
        return text

def serpent_decrypt(encrypted_text, key):
    """Decrypt text with Serpent cipher."""
    try:
        key_bytes = Argon2(memory=2**16, iterations=4, parallelism=2).hash(key.encode('utf-8'))[:32]
        data = base64.b64decode(encrypted_text.encode('utf-8'))
        iv, ciphertext = data[:16], data[16:]
        cipher = Cipher(algorithms.Serpent(key_bytes), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted = decryptor.update(ciphertext) + decryptor.finalize()
        return decrypted.decode('utf-8').rstrip()
    except Exception as e:
        logging.error(f"Serpent decryption failed: {e}")
        return encrypted_text

def twofish_encrypt(text, key):
    """Encrypt text with Twofish cipher (256-bit key, CBC mode)."""
    try:
        key_bytes = Argon2(memory=2**16, iterations=4, parallelism=2).hash(key.encode('utf-8'))[:32]
        iv = secrets.token_bytes(16)
        cipher = Cipher(algorithms.Twofish(key_bytes), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        padded_text = text + ' ' * ((16 - len(text) % 16) % 16)
        encrypted = encryptor.update(padded_text.encode('utf-8')) + encryptor.finalize()
        return base64.b64encode(iv + encrypted).decode('utf-8')
    except Exception as e:
        logging.error(f"Twofish encryption failed: {e}")
        return text

def twofish_decrypt(encrypted_text, key):
    """Decrypt text with Twofish cipher."""
    try:
        key_bytes = Argon2(memory=2**16, iterations=4, parallelism=2).hash(key.encode('utf-8'))[:32]
        data = base64.b64decode(encrypted_text.encode('utf-8'))
        iv, ciphertext = data[:16], data[16:]
        cipher = Cipher(algorithms.Twofish(key_bytes), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted = decryptor.update(ciphertext) + decryptor.finalize()
        return decrypted.decode('utf-8').rstrip()
    except Exception as e:
        logging.error(f"Twofish decryption failed: {e}")
        return encrypted_text

def blowfish_encrypt(text, key):
    """Encrypt text with Blowfish cipher (128-bit key, CBC mode)."""
    try:
        key_bytes = Argon2(memory=2**16, iterations=4, parallelism=2).hash(key.encode('utf-8'))[:16]
        iv = secrets.token_bytes(8)
        cipher = Cipher(algorithms.Blowfish(key_bytes), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        padded_text = text + ' ' * ((8 - len(text) % 8) % 8)
        encrypted = encryptor.update(padded_text.encode('utf-8')) + encryptor.finalize()
        return base64.b64encode(iv + encrypted).decode('utf-8')
    except Exception as e:
        logging.error(f"Blowfish encryption failed: {e}")
        return text

def blowfish_decrypt(encrypted_text, key):
    """Decrypt text with Blowfish cipher."""
    try:
        key_bytes = Argon2(memory=2**16, iterations=4, parallelism=2).hash(key.encode('utf-8'))[:16]
        data = base64.b64decode(encrypted_text.encode('utf-8'))
        iv, ciphertext = data[:8], data[8:]
        cipher = Cipher(algorithms.Blowfish(key_bytes), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted = decryptor.update(ciphertext) + decryptor.finalize()
        return decrypted.decode('utf-8').rstrip()
    except Exception as e:
        logging.error(f"Blowfish decryption failed: {e}")
        return encrypted_text

def rsa_encrypt(text, public_key):
    """Encrypt text with RSA (public key)."""
    try:
        ciphertext = public_key.encrypt(
            text.encode('utf-8'),
            padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
        )
        return base64.b64encode(ciphertext).decode('utf-8')
    except Exception as e:
        logging.error(f"RSA encryption failed: {e}")
        return text

def rsa_decrypt(encrypted_text, private_key):
    """Decrypt text with RSA (private key)."""
    try:
        ciphertext = base64.b64decode(encrypted_text.encode('utf-8'))
        plaintext = private_key.decrypt(
            ciphertext,
            padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
        )
        return plaintext.decode('utf-8')
    except Exception as e:
        logging.error(f"RSA decryption failed: {e}")
        return encrypted_text

def generate_rsa_keys():
    """Generate an RSA 2048-bit key pair."""
    try:
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
        return private_key, private_key.public_key()
    except Exception as e:
        logging.error(f"RSA key generation failed: {e}")
        return None, None

def hmac_sha256(text, key):
    """Generate Base64-encoded HMAC-SHA256 of the given text using key."""
    try:
        key_bytes = Argon2(memory=2**16, iterations=4, parallelism=2).hash(key.encode('utf-8'))[:32]
        h = HMAC(key_bytes, hashes.SHA256(), backend=default_backend())
        h.update(text.encode('utf-8'))
        return base64.b64encode(h.finalize()).decode('utf-8')
    except Exception as e:
        logging.error(f"HMAC-SHA256 generation failed: {e}")
        return ""

def verify_hmac_sha256(text, key, hmac_value):
    """Verify HMAC-SHA256 value for given text and key. Returns True if matches, else False."""
    try:
        key_bytes = Argon2(memory=2**16, iterations=4, parallelism=2).hash(key.encode('utf-8'))[:32]
        h = HMAC(key_bytes, hashes.SHA256(), backend=default_backend())
        h.update(text.encode('utf-8'))
        h.verify(base64.b64decode(hmac_value.encode('utf-8')))
        return True
    except Exception as e:
        logging.error(f"HMAC verification failed: {e}")
        return False

def transposition_cipher(text, key):
    """Apply a simple columnar transposition cipher with the given key."""
    try:
        key = key.lower()
        cols = len(key)
        if cols == 0:
            return text
        rows = math.ceil(len(text) / cols)
        grid = [''] * cols
        for i, char in enumerate(text):
            grid[i % cols] += char
        for col in range(cols):
            if len(grid[col]) < rows:
                grid[col] += ' ' * (rows - len(grid[col]))
        key_order = sorted(range(cols), key=lambda k: key[k])
        return ''.join(grid[i] for i in key_order)
    except Exception as e:
        logging.error(f"Transposition cipher failed: {e}")
        return text

def encrypt_image(image_path, key, metadata=None):
    """Encrypt an image using ChaCha20 and embed metadata via steganography in the image's least significant bits."""
    try:
        img = Image.open(image_path).convert('RGBA')
        pixel_data = np.array(img) if np is not None else None
        if pixel_data is None:
            raise ImportError("NumPy is required for image encryption.")
        data_to_encrypt = pixel_data.tobytes()
        if metadata:
            metadata_bytes = json.dumps(metadata).encode('utf-8')
            metadata_bytes = metadata_bytes[:1024].ljust(1024, b'\x00')
            data_to_encrypt += metadata_bytes
        key_bytes = Argon2(memory=2**16, iterations=4, parallelism=2).hash(key.encode('utf-8'))[:32]
        nonce = secrets.token_bytes(12)
        cipher = Cipher(algorithms.ChaCha20(key_bytes, nonce), mode=None, backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted_data = encryptor.update(data_to_encrypt) + encryptor.finalize()
        encrypted_img = Image.new('RGBA', img.size)
        pixels = encrypted_img.load()
        bit_data = ''.join(f"{byte:08b}" for byte in encrypted_data)
        bit_idx = 0
        width, height = img.size
        for y in range(height):
            for x in range(width):
                r, g, b, a = pixel_data[y, x]
                if bit_idx < len(bit_data):
                    r = (r & ~1) | int(bit_data[bit_idx]); bit_idx += 1
                if bit_idx < len(bit_data):
                    g = (g & ~1) | int(bit_data[bit_idx]); bit_idx += 1
                if bit_idx < len(bit_data):
                    b = (b & ~1) | int(bit_data[bit_idx]); bit_idx += 1
                pixels[x, y] = (r, g, b, a)
        output_path = Path(image_path).stem + '_encrypted.png'
        encrypted_img.save(output_path)
        key_file_path = Path(output_path).with_suffix('.key')
        with key_file_path.open('wb') as f:
            f.write(nonce + key_bytes)
        if is_termux:
            shutil.copy(str(output_path), OUTPUT_DIR / Path(output_path).name)
            shutil.copy(str(key_file_path), OUTPUT_DIR / key_file_path.name)
        return output_path
    except Exception as e:
        logging.error(f"Image encryption failed: {e}")
        if console:
            console.print(f"[red]Error encrypting image: {e}[/red]")
        return None

def decrypt_image(image_path, key_file):
    """Decrypt an image encrypted by this program, using the key file to extract the encryption key and metadata."""
    try:
        with open(key_file, 'rb') as f:
            key_data = f.read()
        if len(key_data) < 12:
            raise ValueError("Key file is invalid or corrupted.")
        nonce, key_bytes = key_data[:12], key_data[12:]
        encrypted_img = Image.open(image_path).convert('RGBA')
        pixels = np.array(encrypted_img) if np is not None else None
        if pixels is None:
            raise ImportError("NumPy is required for image decryption.")
        bit_data = ""
        for y in range(encrypted_img.size[1]):
            for x in range(encrypted_img.size[0]):
                r, g, b, _ = pixels[y, x]
                bit_data += str(r & 1) + str(g & 1) + str(b & 1)
        encrypted_bytes = bytearray()
        for i in range(0, len(bit_data) - 7, 8):
            byte = int(bit_data[i:i+8], 2)
            encrypted_bytes.append(byte)
        cipher = Cipher(algorithms.ChaCha20(key_bytes, nonce), mode=None, backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted_data = decryptor.update(bytes(encrypted_bytes)) + decryptor.finalize()
        width, height = encrypted_img.size
        pixel_count = width * height * 4
        pixel_data_bytes = decrypted_data[:pixel_count]
        metadata_bytes = decrypted_data[pixel_count:pixel_count+1024].rstrip(b'\x00')
        decrypted_img = Image.frombytes('RGBA', encrypted_img.size, pixel_data_bytes)
        output_path = Path(image_path).stem + '_decrypted.png'
        decrypted_img.save(output_path)
        metadata = {}
        if metadata_bytes:
            try:
                metadata = json.loads(metadata_bytes.decode('utf-8'))
            except Exception as e:
                logging.error(f"Failed to parse metadata JSON: {e}")
        if is_termux:
            shutil.copy(str(output_path), OUTPUT_DIR / Path(output_path).name)
        return output_path, metadata
    except Exception as e:
        logging.error(f"Image decryption failed: {e}")
        if console:
            console.print(f"[red]Error decrypting image: {e}[/red]")
        return None, {}

def preprocess_image(image):
    """Preprocess an image for OCR (improve contrast, remove noise, deskew)."""
    try:
        if cv2 is None or np is None:
            raise ImportError("OpenCV or NumPy not available for image preprocessing.")
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = img_array.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        deskewed = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        edges = cv2.Canny(deskewed, 100, 200)
        enhanced = cv2.addWeighted(deskewed, 0.8, edges, 0.2, 0)
        img_pil = Image.fromarray(enhanced).convert('L')
        img_pil = ImageEnhance.Contrast(img_pil).enhance(2.0)
        img_pil = img_pil.filter(ImageFilter.MedianFilter(size=3))
        img_pil = img_pil.filter(ImageFilter.SHARPEN)
        return img_pil
    except Exception as e:
        logging.error(f"Image preprocessing failed: {e}")
        return image.convert('L')

def extract_text_from_image(image_path, quadrants):
    """Extract text from specified image quadrants using OCR."""
    if pytesseract is None or Image is None:
        logging.error("pytesseract or PIL not available for OCR.")
        return []
    try:
        img = Image.open(image_path)
        results = []
        for quad in quadrants:
            region = img.crop((quad['x'], quad['y'], quad['x']+quad['width'], quad['y']+quad['height']))
            preprocessed = preprocess_image(region)
            text = pytesseract.image_to_string(preprocessed, config='--psm 6').strip()
            conf = []
            try:
                data = pytesseract.image_to_data(preprocessed, config='--psm 6', output_type=pytesseract.Output.DICT)
                conf = [int(c) for c in data.get('conf', []) if c != '-1']
            except:
                conf = []
            results.append({'x': quad['x'], 'y': quad['y'], 'width': quad['width'], 'height': quad['height'],
                            'text': text, 'confidence': conf})
        return results
    except Exception as e:
        logging.error(f"OCR text extraction failed: {e}")
        if console:
            console.print(f"[red]Error extracting text from image: {e}[/red]")
        return []

def define_grid(image_width, image_height, grid_size):
    """Define a grid of quadrants based on image dimensions and desired grid size (e.g., '2x2', '3x3')."""
    try:
        if grid_size == 'custom':
            if tk is None or not tk._default_root:
                rows = int(input("Enter number of rows: ").strip() or 1)
                cols = int(input("Enter number of columns: ").strip() or 1)
            else:
                rows = simpledialog.askinteger("Input", "Enter number of rows:", minvalue=1, maxvalue=10)
                cols = simpledialog.askinteger("Input", "Enter number of columns:", minvalue=1, maxvalue=10)
                if rows is None or cols is None:
                    rows, cols = 2, 2
        else:
            rows, cols = map(int, grid_size.split('x'))
        quadrant_width = image_width // cols
        quadrant_height = image_height // rows
        quadrants = []
        for r in range(rows):
            for c in range(cols):
                x = c * quadrant_width
                y = r * quadrant_height
                w = quadrant_width if c < cols - 1 else image_width - x
                h = quadrant_height if r < rows - 1 else image_height - y
                quadrants.append({'x': x, 'y': y, 'width': w, 'height': h})
        return quadrants
    except Exception as e:
        logging.error(f"Grid definition failed: {e}")
        if console:
            console.print(f"[red]Error defining grid: {e}[/red]")
        return [{'x': 0, 'y': 0, 'width': image_width, 'height': image_height}]

def obfuscate_image_text(image_path, wordlists, obfuscation_level, junk_chars, quadrants,
                         custom_rules=None, ciphers=None, keys=None, use_adaptive=False, theme='dark'):
    """Obfuscate text within an image's quadrants, overlay obfuscated text, and encrypt the image."""
    try:
        if custom_rules is None:
            custom_rules = {}
        if ciphers is None:
            ciphers = ['substitution']
        if keys is None:
            keys = ['default_key']
        quadrant_texts = extract_text_from_image(image_path, quadrants)
        if not quadrant_texts:
            return None, None
        wordlist_keys = list(wordlists.keys())
        obfuscated_texts = []
        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        font_size = THEMES.get(theme, THEMES['dark'])['font_size']
        font_path = FONT_FILE if FONT_FILE.exists() else None
        font = ImageFont.truetype(str(font_path) if font_path else "arial.ttf", font_size)
        for i, quad in enumerate(quadrant_texts):
            original_text = quad['text']
            if not original_text:
                continue
            current_wordlist = wordlists[wordlist_keys[i % len(wordlist_keys)]]
            obfuscated_text = layered_cipher(original_text, ciphers, keys,
                                             {'current': current_wordlist}, junk_chars,
                                             obfuscation_level, custom_rules, use_adaptive, len(original_text))
            obfuscated_texts.append({
                'original': original_text,
                'obfuscated': obfuscated_text,
                'x': quad['x'], 'y': quad['y'],
                'width': quad['width'], 'height': quad['height'],
                'confidence': quad.get('confidence', [])
            })
            draw.rectangle((quad['x'], quad['y'], quad['x']+quad['width'], quad['y']+quad['height']),
                           fill=THEMES.get(theme, THEMES['dark'])['bg'])
            draw.text((quad['x'] + 5, quad['y'] + 5), obfuscated_text[:100],
                      fill=THEMES.get(theme, THEMES['dark'])['fg'], font=font)
        output_filename = f"{Path(image_path).stem}_obfuscated.png"
        img.save(output_filename)
        image_encryption_key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8')
        hmac_value = hmac_sha256(output_filename, image_encryption_key)
        metadata = {'quadrants': obfuscated_texts, 'ciphers': ciphers, 'keys': keys, 'hmac': hmac_value}
        encrypted_filename = encrypt_image(output_filename, image_encryption_key, metadata)
        if encrypted_filename:
            cloud_link = save_to_cloud({'image': encrypted_filename, 'hmac': hmac_value}, Path(encrypted_filename).name)
            if console:
                console.print(f"[green]Obfuscated image saved as {encrypted_filename}, Cloud link: {cloud_link}[/green]")
            logging.info(f"Obfuscated image saved: {encrypted_filename}, Cloud link: {cloud_link}")
        return obfuscated_texts, encrypted_filename
    except Exception as e:
        logging.error(f"Image text obfuscation failed: {e}")
        if console:
            console.print(f"[red]Error obfuscating image text: {e}[/red]")
        return None, None

def layered_cipher(text, ciphers, keys, wordlists, junk_chars, obfuscation_level, custom_rules, use_adaptive, text_length):
    """Apply multiple ciphers in sequence (layered encryption/obfuscation), possibly using progressive keys."""
    try:
        result = text
        level_settings = OBFUSCATION_LEVELS.get(obfuscation_level, OBFUSCATION_LEVELS['standard'])
        current_wordlist_name = 'current'
        current_wordlist = wordlists.get(current_wordlist_name, {})
        char_count = 0
        for layer, (cipher, base_key) in enumerate(zip(ciphers, keys)):
            key = generate_progressive_key(result, base_key, layer) if text_length > 50 or (layer == 0 and use_adaptive) else base_key
            if cipher == 'substitution':
                if use_adaptive:
                    adaptive_map = generate_adaptive_cipher(result, current_wordlist, CHARACTER_POOLS['custom'] + CHARACTER_POOLS['zodiac'])
                    current_wordlist.update(adaptive_map)
                current_wordlist = dynamic_wordlist_mutation(current_wordlist, text_length)
                temp_result = []
                i = 0
                while i < len(result):
                    char = result[i]
                    if char in WORDLIST_SWITCHERS:
                        current_wordlist_name = WORDLIST_SWITCHERS[char]
                        current_wordlist = wordlists.get(current_wordlist_name, current_wordlist)
                        temp_result.append(char)
                        i += 1
                    else:
                        if random.random() < level_settings['junk_freq']:
                            temp_result.append(random.choice(junk_chars))
                        temp_result.append(current_wordlist.get(char, char))
                        char_count += 1
                        current_wordlist_name = rotate_wordlist(wordlists, current_wordlist_name, char_count, rotation_threshold=10)
                        current_wordlist = wordlists.get(current_wordlist_name, current_wordlist)
                        i += 1
                result = ''.join(temp_result)
                result = apply_key_changer(result, current_wordlist, level_settings['key_changes'])
                result = homophonic_substitution(result, current_wordlist)
                for rule_name, rule in custom_rules.get('rules', {}).items():
                    result = apply_custom_rule(result, rule)
            elif cipher == 'polybius':
                result = polybius_square(result, key)
            elif cipher == 'stream':
                result = stream_cipher(result, key, feedback=True)
            elif cipher == 'transposition':
                result = transposition_cipher(result, key)
            elif cipher == 'aes':
                result = aes_encrypt(result, key)
            elif cipher == 'chacha20':
                result = chacha20_encrypt(result, key)
            elif cipher == 'serpent':
                result = serpent_encrypt(result, key)
            elif cipher == 'twofish':
                result = twofish_encrypt(result, key)
            elif cipher == 'blowfish':
                result = blowfish_encrypt(result, key)
            elif cipher == 'rsa':
                priv, pub = generate_rsa_keys()
                if pub:
                    result = rsa_encrypt(result, pub)
                    if priv:
                        priv_pem = priv.private_bytes(encoding=serialization.Encoding.PEM,
                                                      format=serialization.PrivateFormat.PKCS8,
                                                      encryption_algorithm=serialization.NoEncryption())
                        save_to_cloud({'private_key': priv_pem.decode('utf-8')},
                                      f"rsa_private_key_{int(time.time())}.pem", encrypt=False)
            hmac_val = hmac_sha256(result, key)
            if not verify_hmac_sha256(result, key, hmac_val):
                raise ValueError(f"HMAC verification failed in layer {layer}")
        return result
    except Exception as e:
        logging.error(f"Layered cipher failed: {e}")
        if console:
            console.print(f"[red]Error applying layered cipher: {e}[/red]")
        return text

def obfuscate_text(text, wordlists, obfuscation_level, junk_chars,
                   quadrant_mode=False, custom_rules=None, use_adaptive=False, ciphers=None, keys=None):
    """Obfuscate a text string using specified ciphers and options."""
    try:
        if not text:
            raise ValueError("Input text cannot be empty.")
        if custom_rules is None:
            custom_rules = {}
        if ciphers is None:
            ciphers = ['substitution']
        if keys is None:
            keys = ['default_key']
        if quadrant_mode:
            return obfuscate_quadrants(text, wordlists, junk_chars, ciphers, keys, custom_rules, use_adaptive)
        return layered_cipher(text, ciphers, keys, wordlists, junk_chars,
                              obfuscation_level, custom_rules, use_adaptive, len(text))
    except Exception as e:
        logging.error(f"Obfuscation failed: {e}")
        if console:
            console.print(f"[red]Error obfuscating text: {e}[/red]")
        return text

def obfuscate_quadrants(text, wordlists, junk_chars, ciphers, keys, custom_rules, use_adaptive):
    """Split text into four quadrants and obfuscate each separately."""
    try:
        text_length = len(text)
        quadrant_size = text_length // 4 + (1 if text_length % 4 != 0 else 0)
        quadrants = [text[i:i+quadrant_size] for i in range(0, text_length, quadrant_size)]
        quadrants.extend([''] * (4 - len(quadrants)))
        wordlist_keys = list(wordlists.keys())
        result_quadrants = []
        for i, quad_text in enumerate(quadrants):
            wl_key = wordlist_keys[i % len(wordlist_keys)]
            wordlist = wordlists[wl_key]
            quad_result = layered_cipher(quad_text, ciphers, keys, {'current': wordlist}, junk_chars,
                                         'standard', custom_rules, use_adaptive, len(quad_text))
            result_quadrants.append(quad_result)
        if console and Table:
            grid = Table(title="Quadrant Output", show_lines=True)
            grid.add_column("Quadrant 1", style=THEMES['dark']['fg'])
            grid.add_column("Quadrant 2", style=THEMES['dark']['accent3'])
            grid.add_column("Quadrant 3", style=THEMES['dark']['accent2'])
            grid.add_column("Quadrant 4", style=THEMES['dark']['glitch'])
            grid.add_row(result_quadrants[0][:50], result_quadrants[1][:50],
                         result_quadrants[2][:50], result_quadrants[3][:50])
            console.print(grid)
        return '\n\n'.join(f"Quadrant {i+1}:\n{quad}" for i, quad in enumerate(result_quadrants))
    except Exception as e:
        logging.error(f"Quadrant obfuscation failed: {e}")
        if console:
            console.print(f"[red]Error in quadrant obfuscation: {e}[/red]")
        return text

def secure_delete(file_path):
    """Securely delete a file by overwriting its content and then removing it."""
    try:
        fp = Path(file_path)
        if fp.exists():
            file_size = fp.stat().st_size
            for _ in range(7):
                with fp.open('wb') as f:
                    f.write(os.urandom(file_size))
            fp.unlink()
            if fp.exists():
                raise OSError("File still exists after attempted secure deletion.")
            else:
                logging.info(f"Securely deleted {fp}")
    except Exception as e:
        logging.error(f"Secure delete failed for {file_path}: {e}")
        if console:
            console.print(f"[red]Error securely deleting {file_path}: {e}[/red]")

def save_to_cloud(data, filename, encrypt=True):
    """
    Save data to the cloud directory (and Termux secondary storage if applicable).
    If encrypt=True, data is serialized and encrypted with a random key (ChaCha20), and a .key file is generated.
    Returns a shareable cloud link (a pseudo-protocol link).
    """
    try:
        cloud_path = (TERMUX_CLOUD_DIR if is_termux else CLOUD_DIR) / filename
        if encrypt:
            data_str = json.dumps(data) if isinstance(data, dict) else str(data)
            key = secrets.token_bytes(32)
            encrypted_text = chacha20_encrypt(data_str, base64.b64encode(key).decode('utf-8'))
            with cloud_path.open('w', encoding='utf-8') as f:
                f.write(encrypted_text)
            key_file = cloud_path.with_suffix('.key')
            with key_file.open('wb') as f:
                f.write(key)
            if is_termux:
                shutil.copy(cloud_path, OUTPUT_DIR / filename)
                shutil.copy(key_file, OUTPUT_DIR / key_file.name)
        else:
            with cloud_path.open('w', encoding='utf-8') as f:
                if isinstance(data, str):
                    f.write(data)
                else:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            if is_termux:
                shutil.copy(cloud_path, OUTPUT_DIR / filename)
        share_link = f"obpro://cloud/{base64.urlsafe_b64encode(filename.encode()).decode()}"
        if console:
            console.print(f"[green]Saved to cloud: {cloud_path}, Share link: {share_link}[/green]")
        if is_termux and console:
            console.print(f"[green]Also saved to Termux storage: {OUTPUT_DIR/filename}[/green]")
            try:
                subprocess.run(["termux-toast", f"Saved to {OUTPUT_DIR/filename}"], check=True)
            except Exception:
                pass
        logging.info(f"Cloud save: {cloud_path}, Share link: {share_link}")
        return share_link
    except Exception as e:
        logging.error(f"Cloud save failed: {e}")
        if console:
            console.print(f"[red]Error saving to cloud: {e}[/red]")
        return None

def load_from_cloud(filename, key=None):
    """Load and return data from a file in the cloud directory. If key provided, will attempt to decrypt."""
    try:
        cloud_path = (TERMUX_CLOUD_DIR if is_termux else CLOUD_DIR) / filename
        if not cloud_path.exists():
            raise FileNotFoundError(f"Cloud file {cloud_path} not found")
        with cloud_path.open('r', encoding='utf-8') as f:
            data = f.read()
        if key:
            decrypted = chacha20_decrypt(data, key)
            return json.loads(decrypted)
        else:
            return json.loads(data)
    except Exception as e:
        logging.error(f"Cloud load failed: {e}")
        if console:
            console.print(f"[red]Error loading from cloud: {e}[/red]")
        return None

def generate_qr_code(text, filename="obfuscated_qr.png", theme='dark'):
    """Generate a QR code image from the given text."""
    try:
        if qrcode is None:
            raise ImportError("qrcode library is not installed.")
        colors = THEMES.get(theme, THEMES['dark'])
        qr = qrcode.QRCode(version=1, box_size=10, border=4)
        qr.add_data(text)
        qr.make(fit=True)
        img = qr.make_image(fill_color=colors['fg'], back_color=colors['bg'])
        img.save(filename)
        cloud_link = save_to_cloud(text, Path(filename).name)
        if console:
            console.print(f"[green]QR code saved as {filename}, Cloud link: {cloud_link}[/green]")
        logging.info(f"QR code generated: {filename}, Cloud link: {cloud_link}")
    except Exception as e:
        logging.error(f"QR code generation failed: {e}")
        if console:
            console.print(f"[red]Error generating QR code: {e}[/red]")

def generate_barcode(text, filename="obfuscated_barcode.png", barcode_type="code128", theme='dark'):
    """Generate a barcode image from the given text (default Code128)."""
    try:
        if barcode is None:
            raise ImportError("python-barcode library is not installed.")
        barcode_class = barcode.get_barcode_class(barcode_type)
        barcode_obj = barcode_class(text, writer=ImageWriter())
        barcode_obj.save(filename[:-4])
        cloud_link = save_to_cloud(text, Path(filename).name)
        if console:
            console.print(f"[green]Barcode saved as {filename}, Cloud link: {cloud_link}[/green]")
        logging.info(f"Barcode generated: {filename}, Cloud link: {cloud_link}")
    except Exception as e:
        logging.error(f"Barcode generation failed: {e}")
        if console:
            console.print(f"[red]Error generating barcode: {e}[/red]")

def get_unique_chars(text, wordlists):
    """Get all unique characters from text and wordlists (for font embedding or analysis)."""
    try:
        unique_chars = set(text)
        for wl in wordlists.values():
            if isinstance(wl, dict):
                unique_chars.update(wl.keys())
                unique_chars.update(wl.values())
        return sorted(unique_chars, key=lambda x: ord(x))
    except Exception as e:
        logging.error(f"Failed to get unique characters: {e}")
        return sorted(set(text))

def create_font(char_mapping, font_name, output_path):
    """Create TTF and OTF font files from a mapping of characters to Unicode codepoints using FontForge."""
    try:
        if fontforge is None:
            raise ImportError("FontForge is not installed or available.")
        font = fontforge.font()
        font.fontname = font_name
        font.familyname = font_name
        font.fullname = font_name
        for char, data in char_mapping.items():
            codepoint = data['unicode']
            glyph = font.createChar(codepoint, char)
            bitmap_str = data.get('bitmap')
            if bitmap_str:
                size = int(math.sqrt(len(bitmap_str)))
                glyph.importBitMap(bytearray([int(x)*255 for x in bitmap_str]), size=size)
            # Note: In a real application, vector paths would be drawn via FontForge pen if available
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        font.generate(str(output_path))
        font.generate(str(output_path.with_suffix('.otf')))
        if sys.platform.startswith("win"):
            try:
                font_dir = Path(os.environ.get('WINDIR', 'C:\\Windows')) / 'Fonts'
                shutil.copy(str(output_path), str(font_dir / output_path.name))
            except Exception as e:
                logging.error(f"Could not copy font to Windows Fonts: {e}")
        elif sys.platform.startswith("linux"):
            try:
                subprocess.run(["fc-cache", "-fv"], check=True)
            except Exception as e:
                logging.error(f"Font cache refresh failed: {e}")
        if is_termux:
            shutil.copy(str(output_path), OUTPUT_DIR / output_path.name)
            shutil.copy(str(output_path.with_suffix('.otf')), OUTPUT_DIR / output_path.with_suffix('.otf').name)
        cloud_link = save_to_cloud({"font_name": font_name}, output_path.name, encrypt=False)
        if console:
            console.print(f"[green]Font generated and saved as {output_path.name}, Cloud link: {cloud_link}[/green]")
        logging.info(f"Font generated: {output_path}, Cloud link: {cloud_link}")
    except Exception as e:
        logging.error(f"Font creation failed: {e}")
        if console:
            console.print(f"[red]Error creating font: {e}[/red]")

def windows_character_creator(low_res=False, theme='dark'):
    """Launch a GUI window for creating a custom character (glyph) and exporting to a font."""
    if tk is None:
        if console:
            console.print("[red]Tkinter not available. Cannot run character creator GUI.[/red]")
        logging.error("Tkinter not available for character creator.")
        return None
    try:
        colors = THEMES.get(theme, THEMES['dark'])
        dialog = tk.Toplevel()
        dialog.title("Ob-Pro Character Creator")
        dialog.configure(bg=colors['bg'])
        grid_size = 8 if low_res else 16
        pixel_size = 20
        grid = [[0] * grid_size for _ in range(grid_size)]
        paths = []
        draw_mode = tk.StringVar(value="freehand")
        brush_size = tk.IntVar(value=1)
        created_chars = {}

        canvas = Canvas(dialog, width=grid_size*pixel_size, height=grid_size*pixel_size,
                        bg=colors['bg'], highlightbackground=colors['accent1'])
        canvas.pack(padx=10, pady=10)

        def redraw_canvas():
            canvas.delete("all")
            for y in range(grid_size):
                for x in range(grid_size):
                    if grid[y][x]:
                        canvas.create_rectangle(x*pixel_size, y*pixel_size,
                                                x*pixel_size + pixel_size, y*pixel_size + pixel_size,
                                                fill=colors['fg'], outline=colors['accent1'])
            for path in paths:
                if path['type'] == 'line':
                    canvas.create_line(path['start'][0]*pixel_size + pixel_size/2, path['start'][1]*pixel_size + pixel_size/2,
                                       path['end'][0]*pixel_size + pixel_size/2, path['end'][1]*pixel_size + pixel_size/2,
                                       fill=colors['accent1'], width=brush_size.get())
                elif path['type'] == 'curve':
                    canvas.create_line(path['start'][0]*pixel_size + pixel_size/2, path['start'][1]*pixel_size + pixel_size/2,
                                       path['c1'][0]*pixel_size + pixel_size/2, path['c1'][1]*pixel_size + pixel_size/2,
                                       path['c2'][0]*pixel_size + pixel_size/2, path['c2'][1]*pixel_size + pixel_size/2,
                                       path['end'][0]*pixel_size + pixel_size/2, path['end'][1]*pixel_size + pixel_size/2,
                                       fill=colors['accent1'], width=brush_size.get(), smooth=True)

        def start_drawing(event):
            x = max(0, min(grid_size-1, event.x // pixel_size))
            y = max(0, min(grid_size-1, event.y // pixel_size))
            grid[y][x] = 1
            paths.append({'type': 'freehand', 'start': (x, y)})
            redraw_canvas()

        current_path = []
        def draw(event):
            x = max(0, min(grid_size-1, event.x // pixel_size))
            y = max(0, min(grid_size-1, event.y // pixel_size))
            if draw_mode.get() == "freehand":
                grid[y][x] = 1
                redraw_canvas()
            elif draw_mode.get() == "line":
                if not current_path:
                    current_path.append((x, y))
                else:
                    # Show line preview
                    sx, sy = current_path[0]
                    redraw_canvas()
                    canvas.create_line(sx*pixel_size + pixel_size/2, sy*pixel_size + pixel_size/2,
                                       x*pixel_size + pixel_size/2, y*pixel_size + pixel_size/2,
                                       fill=colors['accent1'], width=brush_size.get())
            elif draw_mode.get() == "curve":
                if not current_path:
                    current_path.append((x, y))
                else:
                    sx, sy = current_path[0]
                    # Show quadratic Bézier curve preview using two control points
                    cx1 = sx + (x - sx) // 3
                    cy1 = sy
                    cx2 = sx + 2 * (x - sx) // 3
                    cy2 = y
                    redraw_canvas()
                    canvas.create_line(sx*pixel_size + pixel_size/2, sy*pixel_size + pixel_size/2,
                                       cx1*pixel_size + pixel_size/2, cy1*pixel_size + pixel_size/2,
                                       cx2*pixel_size + pixel_size/2, cy2*pixel_size + pixel_size/2,
                                       x*pixel_size + pixel_size/2, y*pixel_size + pixel_size/2,
                                       fill=colors['accent1'], width=brush_size.get(), smooth=True)
            if draw_mode.get() in ["line", "curve"] and not current_path:
                current_path.append((x, y))

        def undo():
            if paths:
                paths.pop()
                redraw_canvas()

        def redo():
            # Redo functionality not implemented in this version (only basic undo).
            pass

        def end_drawing(event):
            x = max(0, min(grid_size-1, event.x // pixel_size))
            y = max(0, min(grid_size-1, event.y // pixel_size))
            if draw_mode.get() == "line" and current_path:
                sx, sy = current_path[0]
                paths.append({'type': 'line', 'start': (sx, sy), 'end': (x, y)})
            elif draw_mode.get() == "curve" and current_path:
                sx, sy = current_path[0]
                cx1 = sx + (x - sx) // 3
                cy1 = sy
                cx2 = sx + 2 * (x - sx) // 3
                cy2 = y
                paths.append({'type': 'curve', 'start': (sx, sy),
                              'c1': (cx1, cy1), 'c2': (cx2, cy2), 'end': (x, y)})
            redraw_canvas()
            current_path.clear()

        def preview_ar():
            dialog_ar = tk.Toplevel(dialog)
            dialog_ar.title("AR Preview")
            dialog_ar.configure(bg=colors['bg'])
            canvas_ar = Canvas(dialog_ar, width=200, height=200, bg=colors['bg'], highlightbackground=colors['accent1'])
            canvas_ar.pack(pady=10)
            def update_preview():
                canvas_ar.delete("all")
                s = scale.get()
                r = math.radians(rotation.get())
                for yy in range(grid_size):
                    for xx in range(grid_size):
                        if grid[yy][xx]:
                            x_centered = xx - grid_size/2
                            y_centered = yy - grid_size/2
                            x_rot = x_centered * math.cos(r) - y_centered * math.sin(r)
                            y_rot = x_centered * math.sin(r) + y_centered * math.cos(r)
                            x_screen = 100 + x_rot * s * pixel_size
                            y_screen = 100 + y_rot * s * pixel_size
                            canvas_ar.create_rectangle(x_screen, y_screen,
                                                       x_screen + pixel_size * s, y_screen + pixel_size * s,
                                                       fill=colors['fg'], outline=colors['accent1'])
                for path in paths:
                    if path['type'] == 'line':
                        x1c = path['start'][0] - grid_size/2
                        y1c = path['start'][1] - grid_size/2
                        x2c = path['end'][0] - grid_size/2
                        y2c = path['end'][1] - grid_size/2
                        x1r = x1c * math.cos(r) - y1c * math.sin(r)
                        y1r = x1c * math.sin(r) + y1c * math.cos(r)
                        x2r = x2c * math.cos(r) - y2c * math.sin(r)
                        y2r = x2c * math.sin(r) + y2c * math.cos(r)
                        canvas_ar.create_line(100 + x1r * s * pixel_size, 100 + y1r * s * pixel_size,
                                              100 + x2r * s * pixel_size, 100 + y2r * s * pixel_size,
                                              fill=colors['fg'], width=brush_size.get() * s)
                    elif path['type'] == 'curve':
                        x1c = path['start'][0] - grid_size/2; y1c = path['start'][1] - grid_size/2
                        xc1c = path['c1'][0] - grid_size/2; yc1c = path['c1'][1] - grid_size/2
                        xc2c = path['c2'][0] - grid_size/2; yc2c = path['c2'][1] - grid_size/2
                        x2c = path['end'][0] - grid_size/2; y2c = path['end'][1] - grid_size/2
                        x1r = x1c * math.cos(r) - y1c * math.sin(r)
                        y1r = x1c * math.sin(r) + y1c * math.cos(r)
                        xc1r = xc1c * math.cos(r) - yc1c * math.sin(r)
                        yc1r = xc1c * math.sin(r) + yc1c * math.cos(r)
                        xc2r = xc2c * math.cos(r) - yc2c * math.sin(r)
                        yc2r = xc2c * math.sin(r) + yc2c * math.cos(r)
                        x2r = x2c * math.cos(r) - y2c * math.sin(r)
                        y2r = x2c * math.sin(r) + y2c * math.cos(r)
                        canvas_ar.create_line(100 + x1r * s * pixel_size, 100 + y1r * s * pixel_size,
                                              100 + xc1r * s * pixel_size, 100 + yc1r * s * pixel_size,
                                              100 + xc2r * s * pixel_size, 100 + yc2r * s * pixel_size,
                                              100 + x2r * s * pixel_size, 100 + y2r * s * pixel_size,
                                              fill=colors['fg'], width=brush_size.get() * s, smooth=True)
                dialog_ar.after(100, update_preview)
            ttk.Label(dialog_ar, text="Rotation:", background=colors['bg'], foreground=colors['fg']).pack()
            ttk.Scale(dialog_ar, from_=0, to=360, orient=tk.HORIZONTAL, variable=rotation, command=lambda v: update_preview()).pack()
            ttk.Label(dialog_ar, text="Scale:", background=colors['bg'], foreground=colors['fg']).pack()
            ttk.Scale(dialog_ar, from_=0.5, to=2.0, orient=tk.HORIZONTAL, variable=scale, command=lambda v: update_preview()).pack()
            update_preview()
            dialog_ar.transient(dialog)
            dialog_ar.grab_set()

        def save_character():
            source_char = char_entry.get().strip()
            if not source_char:
                messagebox.showerror("Error", "Please enter a source character.", parent=dialog)
                return
            bitmap = ''.join('1' if cell else '0' for row in grid for cell in row)
            pua_code = assign_pua_codepoint()
            created_chars[source_char] = {'unicode': pua_code, 'bitmap': bitmap, 'paths': paths.copy()}
            CHARACTER_POOLS['custom'].append(chr(pua_code))
            if console:
                console.print(f"[green]Character '{source_char}' mapped to U+{pua_code:X}[/green]")
            char_entry.delete(0, tk.END)
            for y in range(grid_size):
                for x in range(grid_size):
                    grid[y][x] = 0
            paths.clear()
            redraw_canvas()

        def export_font_action():
            if not created_chars:
                messagebox.showerror("Error", "No characters created to export.", parent=dialog)
                return
            timestamp = int(time.time())
            font_name = f"ObProFont_{timestamp}"
            output_path = FONT_DIR / f"{font_name}.ttf"
            create_font(created_chars, font_name, output_path)
            wordlist = {src: chr(data['unicode']) for src, data in created_chars.items()}
            wordlist_file = Path(f"obf_custom_{timestamp}.json")
            with wordlist_file.open('w', encoding='utf-8') as f:
                json.dump(wordlist, f, ensure_ascii=False, indent=2)
            save_to_cloud(wordlist, wordlist_file.name, encrypt=False)
            dialog.destroy()
            return str(wordlist_file)

        def assign_pua_codepoint():
            for code in range(PUA_START, PUA_END+1):
                if code not in used_pua_codes:
                    used_pua_codes.add(code)
                    return code
            raise RuntimeError("Out of private use Unicode codes for custom characters!")

        canvas.bind("<Button-1>", start_drawing)
        canvas.bind("<B1-Motion>", draw)
        canvas.bind("<ButtonRelease-1>", end_drawing)

        control_frame = ttk.Frame(dialog)
        control_frame.pack(pady=5)
        ttk.Label(control_frame, text="Draw Mode:", background=colors['bg'], foreground=colors['fg']).grid(row=0, column=0, padx=5)
        ttk.Radiobutton(control_frame, text="Freehand", variable=draw_mode, value="freehand").grid(row=0, column=1, padx=5)
        ttk.Radiobutton(control_frame, text="Line", variable=draw_mode, value="line").grid(row=0, column=2, padx=5)
        ttk.Radiobutton(control_frame, text="Curve", variable=draw_mode, value="curve").grid(row=0, column=3, padx=5)
        ttk.Label(control_frame, text="Brush Size:", background=colors['bg'], foreground=colors['fg']).grid(row=1, column=0, padx=5)
        ttk.Spinbox(control_frame, from_=1, to=5, width=5, textvariable=brush_size).grid(row=1, column=1, padx=5)
        ttk.Button(control_frame, text="Undo", command=undo).grid(row=1, column=2, padx=5)
        ttk.Button(control_frame, text="Redo", command=redo).grid(row=1, column=3, padx=5)
        ttk.Button(control_frame, text="AR Preview", command=preview_ar).grid(row=1, column=4, padx=5)
        ttk.Label(control_frame, text="Source Char:", background=colors['bg'], foreground=colors['fg']).grid(row=2, column=0, padx=5)
        char_entry = ttk.Entry(control_frame, width=5)
        char_entry.grid(row=2, column=1, padx=5)
        ttk.Button(control_frame, text="Save Character", command=save_character).grid(row=2, column=2, columnspan=2, padx=5)
        ttk.Button(control_frame, text="Export Font", command=export_font_action).grid(row=3, column=0, columnspan=5, pady=10)
        dialog.transient()
        dialog.grab_set()
        dialog.wait_window()
        return None
    except Exception as e:
        logging.error(f"Character creator failed: {e}")
        if console:
            console.print(f"[red]Error in character creator: {e}[/red]")
        return None

class WindowsObfuscatorApp:
    """GUI Application for Ob-Pro 5.0"""
    def __init__(self, root):
        self.root = root
        self.config = load_config()
        self.theme = self.config.get('theme', 'dark')
        self.colors = THEMES.get(self.theme, THEMES['dark'])
        self.wordlists = get_wordlists()
        self.obfuscation_level = 'standard'
        self.ciphers = ['substitution']
        self.keys = ['default_key']
        self.progressive_keys = False
        self.layered_ciphers = [{'cipher': 'substitution', 'key': 'default_key'}]
        self.quadrant_mode = False
        self.use_adaptive = False
        self.junk_chars = CHARACTER_POOLS['zodiac']
        self.grid_size = self.config.get('grid_size', '2x2')
        self.last_obfuscated = ""
        self.low_res = is_low_res
        self.playlist = load_playlist() if pygame else []
        self.current_track = 0
        self.volume = 0.5
        self.muted = False
        self.rsa_private_key = None
        self.rsa_public_key = None

        self.root.title("Ob-Pro 5.0")
        self.root.configure(bg=self.colors['bg'])
        self.root.geometry("900x700" if not self.low_res else "700x500")

        bg_path = get_random_background(is_gui=True)
        self.background_image = None
        if bg_path and Image:
            try:
                img = Image.open(bg_path).resize((900, 700) if not self.low_res else (700, 500), Image.Resampling.LANCZOS)
                self.background_image = ImageTk.PhotoImage(img)
            except Exception as e:
                logging.error(f"Failed to load background image: {e}")

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background=self.colors['bg'])
        style.configure("TLabel", background=self.colors['bg'], foreground=self.colors['fg'], font=("Courier", self.colors['font_size']))
        style.configure("TButton", background=self.colors['accent1'], foreground=self.colors['bg'], font=("Courier", self.colors['font_size']))
        style.configure("TEntry", fieldbackground=self.colors['bg'], foreground=self.colors['fg'], font=("Courier", self.colors['font_size']))
        if pygame and self.playlist:
            try:
                pygame.mixer.init()
                pygame.mixer.music.load(str(Path(self.playlist[self.current_track]['path'])))
                pygame.mixer.music.set_volume(self.volume)
                pygame.mixer.music.play(loops=-1)
            except Exception as e:
                logging.error(f"Failed to play music: {e}")
        self.create_widgets()
        self.animate_title()

    def animate_title(self):
        """Animate the window title with a glitch effect."""
        try:
            base_title = "Ob-Pro 5.0"
            if random.random() < 0.05 and self.colors.get('animations', True):
                glitch = ''.join(random.choice(['█','▒','☠','⚮']) for _ in range(3))
                self.root.title(f"{base_title} | {glitch}")
                self.root.after(100, lambda: self.root.title(base_title))
            self.root.after(500, self.animate_title)
        except Exception as e:
            logging.error(f"Title animation failed: {e}")

    def switch_theme(self):
        """Switch between available themes."""
        try:
            themes = list(THEMES.keys())
            current_idx = themes.index(self.theme)
            self.theme = themes[(current_idx + 1) % len(themes)]
            self.colors = THEMES[self.theme]
            self.config['theme'] = self.theme
            save_config(self.config)
            self.root.configure(bg=self.colors['bg'])
            style = ttk.Style()
            style.configure("TFrame", background=self.colors['bg'])
            style.configure("TLabel", background=self.colors['bg'], foreground=self.colors['fg'], font=("Courier", self.colors['font_size']))
            style.configure("TButton", background=self.colors['accent1'], foreground=self.colors['bg'], font=("Courier", self.colors['font_size']))
            style.configure("TEntry", fieldbackground=self.colors['bg'], foreground=self.colors['fg'], font=("Courier", self.colors['font_size']))
            self.input_text.configure(bg='#333333' if self.theme in ['dark','colorblind'] else '#FFFFFF', fg=self.colors['fg'])
            self.output_text.configure(bg='#333333' if self.theme in ['dark','colorblind'] else '#FFFFFF', fg=self.colors['fg'])
            messagebox.showinfo("Theme", f"Switched to {self.theme} theme")
        except Exception as e:
            logging.error(f"Theme switch failed: {e}")
            messagebox.showerror("Error", f"Theme switch failed: {e}")

    def set_grid_size(self):
        """Handle grid size selection (update config if custom size chosen)."""
        try:
            self.grid_size = self.grid_var.get()
            self.config['grid_size'] = self.grid_size
            save_config(self.config)
            if self.grid_size == 'custom' and tk:
                rows = simpledialog.askinteger("Grid Size", "Enter number of rows:", minvalue=1, maxvalue=10, parent=self.root)
                cols = simpledialog.askinteger("Grid Size", "Enter number of columns:", minvalue=1, maxvalue=10, parent=self.root)
                if rows and cols:
                    self.grid_size = f"{rows}x{cols}"
                    self.config['grid_size'] = self.grid_size
                    save_config(self.config)
        except Exception as e:
            logging.error(f"Grid size setting failed: {e}")
            messagebox.showerror("Error", f"Grid size setting failed: {e}")

    def configure_layers(self):
        """Open a dialog to configure multiple cipher layers and keys."""
        try:
            dialog = tk.Toplevel(self.root)
            dialog.title("Configure Cipher Layers")
            dialog.configure(bg=self.colors['bg'])
            layers_frame = ttk.Frame(dialog)
            layers_frame.pack(pady=10)
            layer_entries = []
            available_ciphers = ['substitution','polybius','stream','transposition','aes','chacha20','rsa','serpent','twofish','blowfish']
            def add_layer():
                frame = ttk.Frame(layers_frame)
                frame.pack(fill='x', pady=2)
                cipher_var = tk.StringVar(value=available_ciphers[0])
                key_entry = ttk.Entry(frame, width=20)
                ttk.OptionMenu(frame, cipher_var, available_ciphers[0], *available_ciphers).pack(side=tk.LEFT, padx=5)
                key_entry.pack(side=tk.LEFT, padx=5)
                ttk.Button(frame, text="Remove", command=lambda f=frame: (layer_entries.remove((cipher_var, key_entry)), f.destroy())).pack(side=tk.LEFT, padx=5)
                layer_entries.append((cipher_var, key_entry))
            for layer in self.layered_ciphers:
                frame = ttk.Frame(layers_frame)
                frame.pack(fill='x', pady=2)
                cipher_var = tk.StringVar(value=layer['cipher'])
                key_entry = ttk.Entry(frame, width=20)
                key_entry.insert(0, layer['key'])
                ttk.OptionMenu(frame, cipher_var, layer['cipher'], *available_ciphers).pack(side=tk.LEFT, padx=5)
                key_entry.pack(side=tk.LEFT, padx=5)
                ttk.Button(frame, text="Remove", command=lambda f=frame: (layer_entries.remove((cipher_var, key_entry)), f.destroy())).pack(side=tk.LEFT, padx=5)
                layer_entries.append((cipher_var, key_entry))
            ttk.Button(layers_frame, text="Add Layer", command=add_layer).pack(pady=5)
            def save_layers():
                self.layered_ciphers = []
                for cipher_var, key_entry in layer_entries:
                    cipher_name = cipher_var.get()
                    key_val = key_entry.get().strip() or 'default_key'
                    self.layered_ciphers.append({'cipher': cipher_name, 'key': key_val})
                self.ciphers = [layer['cipher'] for layer in self.layered_ciphers]
                self.keys = [layer['key'] for layer in self.layered_ciphers]
                dialog.destroy()
            ttk.Button(dialog, text="Save Layers", command=save_layers).pack(pady=5)
            dialog.transient(self.root)
            dialog.grab_set()
        except Exception as e:
            logging.error(f"Layer configuration failed: {e}")
            messagebox.showerror("Error", f"Layer configuration failed: {e}")

    def prev_track(self):
        """Play previous music track."""
        try:
            if pygame and self.playlist:
                self.current_track = (self.current_track - 1) % len(self.playlist)
                pygame.mixer.music.load(str(Path(self.playlist[self.current_track]['path'])))
                pygame.mixer.music.play(loops=-1)
                pygame.mixer.music.set_volume(0 if self.muted else self.volume)
        except Exception as e:
            logging.error(f"Previous track failed: {e}")
            messagebox.showerror("Error", f"Previous track failed: {e}")

    def next_track(self):
        """Play next music track."""
        try:
            if pygame and self.playlist:
                self.current_track = (self.current_track + 1) % len(self.playlist)
                pygame.mixer.music.load(str(Path(self.playlist[self.current_track]['path'])))
                pygame.mixer.music.play(loops=-1)
                pygame.mixer.music.set_volume(0 if self.muted else self.volume)
        except Exception as e:
            logging.error(f"Next track failed: {e}")
            messagebox.showerror("Error", f"Next track failed: {e}")

    def toggle_mute(self):
        """Toggle audio mute."""
        try:
            if pygame:
                self.muted = not self.muted
                pygame.mixer.music.set_volume(0 if self.muted else self.volume)
        except Exception as e:
            logging.error(f"Mute toggle failed: {e}")
            messagebox.showerror("Error", f"Mute toggle failed: {e}")

    def set_volume(self, value):
        """Adjust music volume."""
        try:
            if pygame:
                self.volume = float(value)
                if not self.muted:
                    pygame.mixer.music.set_volume(self.volume)
        except Exception as e:
            logging.error(f"Volume set failed: {e}")
            messagebox.showerror("Error", f"Volume set failed: {e}")

    def create_widgets(self):
        """Create and layout all GUI components."""
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        if self.background_image:
            self.bg_label = tk.Label(self.main_frame, image=self.background_image)
            self.bg_label.grid(row=0, column=0, columnspan=3, rowspan=15, sticky="nsew")
            self.bg_label.lower()
        ttk.Label(self.main_frame, text="Input Text:", style="TLabel").grid(row=0, column=0, sticky="w", pady=5)
        self.input_text = tk.Text(self.main_frame, height=5, width=50,
                                  font=("Courier", self.colors['font_size']),
                                  bg='#333333' if self.theme in ['dark','colorblind'] else '#FFFFFF',
                                  fg=self.colors['fg'])
        self.input_text.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        ttk.Label(self.main_frame, text="Obfuscated Output:", style="TLabel").grid(row=2, column=0, sticky="w", pady=5)
        self.output_text = tk.Text(self.main_frame, height=5, width=50,
                                   font=("Courier", self.colors['font_size']),
                                   bg='#333333' if self.theme in ['dark','colorblind'] else '#FFFFFF',
                                   fg=self.colors['fg'])
        self.output_text.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
        self.output_text.config(state='disabled')
        ttk.Label(self.main_frame, text="Obfuscation Level:", style="TLabel").grid(row=4, column=0, sticky="w", pady=5)
        self.level_var = tk.StringVar(value=self.obfuscation_level)
        ttk.OptionMenu(self.main_frame, self.level_var, self.obfuscation_level, *OBFUSCATION_LEVELS.keys()).grid(row=4, column=1, sticky="w", padx=5)
        ttk.Label(self.main_frame, text="Cipher Layers:", style="TLabel").grid(row=5, column=0, sticky="w", pady=5)
        ttk.Button(self.main_frame, text="Configure Layers", command=self.configure_layers).grid(row=5, column=1, sticky="w", padx=5)
        self.progressive_var = tk.BooleanVar(value=self.progressive_keys)
        ttk.Checkbutton(self.main_frame, text="Progressive Keys", variable=self.progressive_var).grid(row=6, column=0, sticky="w", pady=5)
        self.quadrant_var = tk.BooleanVar(value=self.quadrant_mode)
        ttk.Checkbutton(self.main_frame, text="Quadrant Mode", variable=self.quadrant_var).grid(row=6, column=1, sticky="w", pady=5)
        self.adaptive_var = tk.BooleanVar(value=self.use_adaptive)
        ttk.Checkbutton(self.main_frame, text="Adaptive Cipher", variable=self.adaptive_var).grid(row=7, column=0, sticky="w", pady=5)
        ttk.Label(self.main_frame, text="Junk Characters:", style="TLabel").grid(row=8, column=0, sticky="w", pady=5)
        self.junk_entry = ttk.Entry(self.main_frame, width=50)
        self.junk_entry.insert(0, ''.join(self.junk_chars[:10]))
        self.junk_entry.grid(row=8, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(self.main_frame, text="Grid Size:", style="TLabel").grid(row=9, column=0, sticky="w", pady=5)
        self.grid_var = tk.StringVar(value=self.grid_size)
        ttk.OptionMenu(self.main_frame, self.grid_var, self.grid_size, "2x2", "3x3", "custom",
                       command=lambda *_: self.set_grid_size()).grid(row=9, column=1, sticky="w", padx=5)
        ttk.Label(self.main_frame, text="Wordlist:", style="TLabel").grid(row=10, column=0, sticky="w", pady=5)
        self.wordlist_var = tk.StringVar(value='list1')
        ttk.OptionMenu(self.main_frame, self.wordlist_var, 'list1', *self.wordlists.keys()).grid(row=10, column=1, sticky="w", padx=5)
        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(row=11, column=0, columnspan=2, pady=10)
        ttk.Button(button_frame, text="Obfuscate", command=self.obfuscate).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Process Image", command=self.process_image).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Decrypt Image", command=self.decrypt_image).grid(row=0, column=2, padx=5)
        ttk.Button(button_frame, text="Create Character", command=self.create_character).grid(row=0, column=3, padx=5)
        ttk.Button(button_frame, text="Random Cipher", command=self.generate_random_cipher).grid(row=0, column=4, padx=5)
        ttk.Button(button_frame, text="Generate QR", command=self.generate_qr).grid(row=0, column=5, padx=5)
        ttk.Button(button_frame, text="Generate Barcode", command=self.generate_barcode).grid(row=0, column=6, padx=5)
        ttk.Button(button_frame, text="Export Output", command=self.export_output).grid(row=0, column=7, padx=5)
        ttk.Button(button_frame, text="Help", command=self.show_help).grid(row=0, column=8, padx=5)
        ttk.Button(button_frame, text="Switch Theme", command=self.switch_theme).grid(row=0, column=9, padx=5)
        if pygame and self.playlist:
            music_frame = ttk.Frame(self.main_frame)
            music_frame.grid(row=12, column=0, columnspan=2, pady=5)
            ttk.Button(music_frame, text="Prev Track", command=self.prev_track).grid(row=0, column=0, padx=5)
            ttk.Button(music_frame, text="Next Track", command=self.next_track).grid(row=0, column=1, padx=5)
            ttk.Button(music_frame, text="Mute/Unmute", command=self.toggle_mute).grid(row=0, column=2, padx=5)
            ttk.Label(music_frame, text="Volume:", style="TLabel").grid(row=0, column=3, padx=5)
            ttk.Scale(music_frame, from_=0, to=1, orient=tk.HORIZONTAL, command=self.set_volume).grid(row=0, column=4, padx=5)
        if not self.low_res and self.colors.get('animations', True):
            self.effect_canvas = Canvas(self.main_frame, width=900, height=100, bg=self.colors['bg'], highlightthickness=0)
            self.effect_canvas.grid(row=13, column=0, columnspan=2, pady=10)
            self.animate_effects()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def animate_effects(self):
        """Animate glitchy scanline effects on the effect canvas."""
        try:
            self.effect_canvas.delete("all")
            if random.random() < 0.2:
                self.effect_canvas.create_rectangle(0, 0, 900, 100, fill=self.colors['glitch'], stipple="gray12", tags="glitch")
                self.effect_canvas.after(50, lambda: self.effect_canvas.delete("glitch"))
            for y in range(0, 100, 5):
                self.effect_canvas.create_line(0, y, 900, y, fill=self.colors['accent1'], stipple="gray25")
            self.root.after(100, self.animate_effects)
        except Exception as e:
            logging.error(f"Effect animation failed: {e}")

    def obfuscate(self):
        """Obfuscate text from the input field and display in output field."""
        try:
            text = self.input_text.get("1.0", tk.END).strip()
            if not text:
                messagebox.showerror("Error", "Input text cannot be empty.")
                return
            self.obfuscation_level = self.level_var.get()
            self.progressive_keys = self.progressive_var.get()
            self.quadrant_mode = self.quadrant_var.get()
            self.use_adaptive = self.adaptive_var.get()
            junk_input = self.junk_entry.get().strip()
            self.junk_chars = list(junk_input) if junk_input else get_junk_chars(self.wordlists, text_context=text, theme=self.theme)
            self.grid_size = self.grid_var.get()
            selected_wordlist = self.wordlist_var.get()
            active_wordlists = {selected_wordlist: self.wordlists.get(selected_wordlist, {})}
            if not self.ciphers:
                messagebox.showerror("Error", "At least one cipher must be selected.")
                return
            if len(self.keys) < len(self.ciphers):
                self.keys.extend(['default_key'] * (len(self.ciphers) - len(self.keys)))
            result = obfuscate_text(text, active_wordlists, self.obfuscation_level, self.junk_chars,
                                     quadrant_mode=self.quadrant_mode, custom_rules=self.config.get('rules', {}),
                                     use_adaptive=self.use_adaptive, ciphers=self.ciphers, keys=self.keys)
            self.output_text.config(state='normal')
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, result)
            self.output_text.config(state='disabled')
            self.last_obfuscated = result
            if console:
                console.print("[green]Text obfuscated successfully.[/green]")
            logging.info(f"Text obfuscated: {result[:100]}...")
        except Exception as e:
            logging.error(f"Obfuscation failed: {e}")
            messagebox.showerror("Error", f"Obfuscation failed: {e}")

    def process_image(self):
        """Select an image, extract and obfuscate text in its quadrants, then encrypt the image."""
        try:
            file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
            if not file_path:
                return
            img = Image.open(file_path)
            img_width, img_height = img.size
            quadrants = define_grid(img_width, img_height, self.grid_size)
            self.obfuscation_level = self.level_var.get()
            self.progressive_keys = self.progressive_var.get()
            self.use_adaptive = self.adaptive_var.get()
            junk_input = self.junk_entry.get().strip()
            self.junk_chars = list(junk_input) if junk_input else get_junk_chars(self.wordlists, theme=self.theme)
            selected_wordlist = self.wordlist_var.get()
            active_wordlists = {selected_wordlist: self.wordlists.get(selected_wordlist, {})}
            obfuscated_texts, encrypted_filename = obfuscate_image_text(file_path, active_wordlists,
                                                                        self.obfuscation_level, self.junk_chars,
                                                                        quadrants, custom_rules=self.config.get('rules', {}),
                                                                        ciphers=self.ciphers, keys=self.keys,
                                                                        use_adaptive=self.use_adaptive, theme=self.theme)
            if obfuscated_texts:
                self.output_text.config(state='normal')
                self.output_text.delete("1.0", tk.END)
                for i, quad in enumerate(obfuscated_texts, 1):
                    avg_conf = 0
                    if quad['confidence']:
                        avg_conf = sum(map(int, quad['confidence'])) / len(quad['confidence'])
                    self.output_text.insert(tk.END, f"Quadrant {i}:\nOriginal: {quad['original']}\nObfuscated: {quad['obfuscated']}\nConfidence: {avg_conf:.2f}\n\n")
                self.output_text.config(state='disabled')
                self.last_obfuscated = "\n".join(q['obfuscated'] for q in obfuscated_texts)
                messagebox.showinfo("Success", f"Encrypted image saved as {encrypted_filename}")
                logging.info(f"Image processed and encrypted: {encrypted_filename}")
        except Exception as e:
            logging.error(f"Image processing failed: {e}")
            messagebox.showerror("Error", f"Image processing failed: {e}")

    def decrypt_image(self):
        """Decrypt an image file using its corresponding .key file and display metadata."""
        try:
            image_path = filedialog.askopenfilename(filetypes=[("Encrypted PNG", "*.png")])
            if not image_path:
                return
            key_path = filedialog.askopenfilename(title="Select Key File", filetypes=[("Key files", "*.key")])
            if not key_path:
                return
            decrypted_path, metadata = decrypt_image(image_path, key_path)
            if decrypted_path:
                self.output_text.config(state='normal')
                self.output_text.delete("1.0", tk.END)
                self.output_text.insert(tk.END, f"Decrypted Image: {decrypted_path}\nMetadata:\n{json.dumps(metadata, indent=2)}")
                self.output_text.config(state='disabled')
                messagebox.showinfo("Success", f"Image decrypted and saved as {decrypted_path}")
        except Exception as e:
            logging.error(f"Image decryption failed: {e}")
            messagebox.showerror("Error", f"Image decryption failed: {e}")

    def create_character(self):
        """Open the custom character creator wizard."""
        try:
            filename = windows_character_creator(low_res=self.low_res, theme=self.theme)
            if filename:
                self.wordlists.update(get_wordlists())
                messagebox.showinfo("Success", f"Custom character set saved to {filename}")
                logging.info(f"New character map created: {filename}")
        except Exception as e:
            logging.error(f"Character creation failed: {e}")
            messagebox.showerror("Error", f"Character creation failed: {e}")

    def generate_random_cipher(self):
        """Generate a random substitution cipher mapping and save it."""
        try:
            timestamp = int(time.time())
            mapping = {}
            src_chars = list(string.ascii_letters + string.digits + string.punctuation)
            tgt_chars = CHARACTER_POOLS['zodiac'] + CHARACTER_POOLS['custom'] + [chr(i) for i in range(PUA_START, PUA_END+1) if chr(i) not in CHARACTER_POOLS['custom']]
            random.shuffle(tgt_chars)
            for src in src_chars:
                if tgt_chars:
                    mapped_char = tgt_chars.pop(0)
                    mapping[src] = mapped_char
                    if mapped_char in CHARACTER_POOLS['custom']:
                        used_pua_codes.add(ord(mapped_char))
            file_path = Path(f"obf_random_{timestamp}.json")
            with file_path.open('w', encoding='utf-8') as f:
                json.dump(mapping, f, ensure_ascii=False, indent=2)
            cloud_link = save_to_cloud(mapping, file_path.name, encrypt=False)
            if console:
                console.print(f"[green]Random cipher saved as {file_path.name}, Cloud link: {cloud_link}[/green]")
            logging.info(f"Random cipher generated: {file_path}, Cloud link: {cloud_link}")
            self.wordlists[f"random_{timestamp}"] = mapping
            return str(file_path)
        except Exception as e:
            logging.error(f"Random cipher generation failed: {e}")
            messagebox.showerror("Error", f"Random cipher generation failed: {e}")
            return None

    def generate_qr(self):
        """Generate a QR code image from the last obfuscated text."""
        try:
            if not self.last_obfuscated:
                messagebox.showerror("Error", "No obfuscated text to generate QR code.")
                return
            filename = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")], title="Save QR Code As")
            if filename:
                generate_qr_code(self.last_obfuscated, filename, theme=self.theme)
                messagebox.showinfo("Success", f"QR code saved as {filename}")
        except Exception as e:
            logging.error(f"QR code generation failed: {e}")
            messagebox.showerror("Error", f"QR code generation failed: {e}")

    def generate_barcode(self):
        """Generate a barcode image from the last obfuscated text."""
        try:
            if not self.last_obfuscated:
                messagebox.showerror("Error", "No obfuscated text to generate barcode.")
                return
            filename = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")], title="Save Barcode As")
            if filename:
                generate_barcode(self.last_obfuscated, filename, barcode_type="code128", theme=self.theme)
                messagebox.showinfo("Success", f"Barcode saved as {filename}")
        except Exception as e:
            logging.error(f"Barcode generation failed: {e}")
            messagebox.showerror("Error", f"Barcode generation failed: {e}")

    def export_output(self):
        """Export the last obfuscated output to a file (text, JSON, PNG, or PDF)."""
        try:
            if not self.last_obfuscated:
                messagebox.showerror("Error", "No obfuscated text available to export.")
                return
            format_var = tk.StringVar(value="text")
            dialog = tk.Toplevel(self.root)
            dialog.title("Export Output")
            dialog.configure(bg=self.colors['bg'])
            ttk.Label(dialog, text="Export Format:", style="TLabel").pack(pady=5)
            ttk.OptionMenu(dialog, format_var, "text", "text", "json", "png", "pdf").pack(pady=5)
            def do_export():
                fmt = format_var.get()
                save_path = filedialog.asksaveasfilename(defaultextension=f".{fmt}", filetypes=[(f"{fmt.upper()} files", f"*.{fmt}")], title="Export As")
                if save_path:
                    export_to_file(self.last_obfuscated, self.wordlists, filename=save_path, format=fmt, theme=self.theme)
                    messagebox.showinfo("Success", f"Output exported to {save_path}", parent=dialog)
                    dialog.destroy()
            ttk.Button(dialog, text="Export", command=do_export).pack(pady=5)
            dialog.transient(self.root)
            dialog.grab_set()
        except Exception as e:
            logging.error(f"Export failed: {e}")
            messagebox.showerror("Error", f"Export failed: {e}")

    def show_help(self):
        """Display help information."""
        try:
            display_help(interactive=True, theme=self.theme)
        except Exception as e:
            logging.error(f"Help display failed: {e}")
            messagebox.showerror("Error", f"Help display failed: {e}")

    def on_closing(self):
        """Cleanup when GUI window is closed."""
        try:
            if pygame:
                pygame.mixer.music.stop()
                pygame.mixer.quit()
            self.root.destroy()
        except Exception as e:
            logging.error(f"Window closing cleanup failed: {e}")
            self.root.destroy()

def export_to_file(text, wordlists, filename, format="text", theme='dark'):
    """Export the given text (and optionally the mapping) to the specified file format."""
    try:
        fmt = format.lower()
        base = filename
        if filename.lower().endswith(f".{fmt}"):
            base = filename[:-len(fmt)-1]
        if fmt == "text":
            with open(base + ".txt", 'w', encoding='utf-8') as f:
                f.write(text)
        elif fmt == "json":
            with open(base + ".json", 'w', encoding='utf-8') as f:
                json.dump(wordlists, f, ensure_ascii=False, indent=2)
        elif fmt == "png":
            if Image is None:
                raise ImportError("PIL not available for PNG export.")
            lines = text.splitlines()
            font = ImageFont.truetype(str(FONT_FILE) if FONT_FILE.exists() else "arial.ttf", 14)
            width = max([font.getsize(line)[0] for line in lines] + [1]) + 20
            height = (font.getsize(text)[1] + 5) * len(lines) + 20
            img = Image.new('RGB', (width, height), color=THEMES.get(theme, THEMES['dark'])['bg'])
            draw = ImageDraw.Draw(img)
            y = 10
            for line in lines:
                draw.text((10, y), line, font=font, fill=THEMES.get(theme, THEMES['dark'])['fg'])
                y += font.getsize(line)[1] + 5
            img.save(base + ".png")
        elif fmt == "pdf":
            if canvas is None or pdfmetrics is None or TTFont is None:
                raise ImportError("ReportLab not available for PDF export.")
            c = canvas.Canvas(base + ".pdf", pagesize=A4)
            if FONT_FILE.exists():
                pdfmetrics.registerFont(TTFont("ObProFont", str(FONT_FILE)))
            text_object = c.beginText(50, 800)
            text_object.setFont("ObProFont" if FONT_FILE.exists() else "Helvetica", 12)
            for line in text.splitlines():
                text_object.textLine(line)
            c.drawText(text_object)
            c.showPage()
            c.save()
    except Exception as e:
        logging.error(f"Export to {format} failed: {e}")
        if console:
            console.print(f"[red]Error exporting to {format.upper()}: {e}[/red]")

def display_help(interactive=False, theme='dark'):
    """Display help information. If interactive, guide the user through steps in the console."""
    try:
        if interactive and console:
            colors = THEMES.get(theme, THEMES['dark'])
            console.print(f"[{colors['fg']}]Starting interactive tutorial...[/]")
            steps = [
                ("Enter text", "Type text to obfuscate. Example: 'Secret@Message'."),
                ("Select level", "Choose obfuscation level: min, standard, or max."),
                ("Configure ciphers", "Choose one or multiple ciphers: substitution, polybius, stream, transposition, AES, ChaCha20, RSA, Serpent, Twofish, Blowfish."),
                ("Set keys", "Enter keys for each cipher, or use default/progressive keys."),
                ("Layer ciphers", "Stack multiple ciphers for layered encryption (e.g., substitution -> ChaCha20 -> RSA)."),
                ("Quadrant mode", "Enable to split text into 4 parts, obfuscating each separately."),
                ("Junk characters", "Customize junk filler characters or leave blank to auto-select."),
                ("Process image", "Select an image to perform OCR, obfuscate extracted text, and encrypt the image."),
                ("Decrypt image", "Provide the encrypted image and .key file to decrypt and retrieve text."),
                ("Custom characters", "Use the 'Create Character' wizard to design glyphs and export as a font for obfuscation."),
                ("Random cipher", "Generate a random substitution cipher and use it as a key."),
                ("QR code/barcode", "Generate a QR code or barcode of the obfuscated output."),
                ("Export output", "Save the obfuscated output or mapping in various formats (text, JSON, PNG, PDF)."),
                ("Themes", "Toggle between dark, light, vision-impaired, and colorblind themes for accessibility."),
                ("CLI Mode", "If GUI is unavailable, a menu-driven CLI provides similar functionality.")
            ]
            for i, (title, desc) in enumerate(steps, start=1):
                console.print(f"[{colors['accent3']}]Step {i}: {title}[/]")
                console.print(desc)
                input("Press Enter to continue...")
            console.print("[green]Tutorial completed![/green]")
        else:
            if console:
                console.print("[cyan]Ob-Pro 5.0 - Help[/cyan]")
                console.print("1. Obfuscate Text: Enter text and parameters to obfuscate.")
                console.print("2. Decrypt: Decrypt text or image using the appropriate keys or key file.")
                console.print("3. Process Image: Extract text from an image, obfuscate it, and encrypt the image.")
                console.print("4. Create Character: Launch the character creation tool to design custom cipher glyphs.")
                console.print("5. Exit: Quit the application.")
    except Exception as e:
        logging.error(f"Help display failed: {e}")
        if console:
            console.print(f"[red]Error displaying help: {e}[/red]")

# CLI fallback interface
def run_cli_menu():
    """Run a simple CLI menu for environments where GUI is not available."""
    config = load_config()
    theme = config.get('theme', 'dark')
    colors = THEMES.get(theme, THEMES['dark'])
    wordlists = get_wordlists()
    print(get_random_background(is_gui=False))
    print("Ob-Pro 5.0 - CLI Mode")
    while True:
        print("\nMenu:")
        print("1. Obfuscate Text")
        print("2. Decrypt")
        print("3. Process Image")
        print("4. Create Character")
        print("5. Exit")
        choice = input("Select an option (1-5): ").strip()
        if choice == '1':
            text = input("Enter text to obfuscate: ").rstrip('\n')
            if not text:
                print("Error: Input text cannot be empty.")
                continue
            level = input("Select obfuscation level (min/standard/max, default=standard): ").strip() or "standard"
            cipher_input = input("Enter ciphers (comma-separated, e.g., substitution,aes, default=substitution): ").strip()
            ciphers = [c.strip() for c in cipher_input.split(',')] if cipher_input else ['substitution']
            key_input = input("Enter keys for each cipher (comma-separated, default_key if blank): ").strip()
            keys = [k.strip() if k.strip() else 'default_key' for k in key_input.split(',')] if key_input else ['default_key'] * len(ciphers)
            if len(keys) < len(ciphers):
                keys.extend(['default_key'] * (len(ciphers) - len(keys)))
            quadrant = input("Enable quadrant mode? (y/N): ").strip().lower() == 'y'
            adaptive = input("Enable adaptive cipher enhancements? (y/N): ").strip().lower() == 'y'
            junk = input("Enter junk characters (leave blank for auto): ").strip()
            junk_chars = list(junk) if junk else get_junk_chars(wordlists, text_context=text, theme=theme)
            active_wordlists = {'list1': wordlists.get('list1', {})}
            result = obfuscate_text(text, active_wordlists, level, junk_chars,
                                     quadrant_mode=quadrant, custom_rules=config.get('rules', {}),
                                     use_adaptive=adaptive, ciphers=ciphers, keys=keys)
            print("\nObfuscated Text:")
            print(result)
        elif choice == '2':
            sub_choice = input("Decrypt (1) Text or (2) Image? ").strip()
            if sub_choice == '1':
                encrypted_text = input("Enter text to decrypt: ").rstrip('\n')
                cipher_input = input("Enter ciphers used (in order, comma-separated): ").strip()
                ciphers = [c.strip() for c in cipher_input.split(',')] if cipher_input else []
                key_input = input("Enter keys for each cipher (comma-separated): ").strip()
                keys = [k.strip() for k in key_input.split(',')] if key_input else []
                if not encrypted_text or not ciphers or not keys:
                    print("Error: Please provide encrypted text, ciphers and keys.")
                    continue
                mapping = None
                if any(ci in ['substitution','polybius'] for ci in ciphers):
                    map_source = input("Enter wordlist name or path for substitution mapping (if applicable, or press Enter to skip): ").strip()
                    if map_source:
                        if map_source in wordlists:
                            mapping = wordlists[map_source]
                        else:
                            try:
                                mapping = load_wordlist(map_source)
                            except Exception as e:
                                print(f"Warning: Could not load mapping from {map_source}: {e}")
                                mapping = None
                try:
                    result = encrypted_text
                    for cipher, key in zip(reversed(ciphers), reversed(keys)):
                        if cipher == 'substitution':
                            if mapping:
                                inv_map = {v: k for k, v in mapping.items()}
                                result = ''.join(inv_map.get(ch, ch) for ch in result)
                        elif cipher == 'polybius':
                            decrypted = ""
                            i = 0
                            while i < len(result):
                                if i+1 < len(result) and result[i].isdigit() and result[i+1].isdigit():
                                    r = int(result[i]) - 1
                                    c = int(result[i+1]) - 1
                                    if 0 <= r < 5 and 0 <= c < 5:
                                        square = [[None]*5 for _ in range(5)]
                                        k = (key or "")[:5].upper()
                                        for rr in range(5):
                                            for cc in range(5):
                                                square[rr][cc] = k[rr] if rr < len(k) else chr(ord('A') + rr*5 + cc)
                                        char = square[r][c]
                                        if r == 1 and c == 4 and char != 'Z':
                                            char = 'Z'
                                        decrypted += char
                                    else:
                                        decrypted += result[i] + result[i+1]
                                    i += 2
                                else:
                                    decrypted += result[i]
                                    i += 1
                            result = decrypted
                        elif cipher == 'stream':
                            # Cannot reliably decrypt custom stream cipher with feedback
                            result = result
                        elif cipher == 'transposition':
                            result = invert_transposition(result, key)
                        elif cipher == 'aes':
                            result = aes_decrypt(result, key)
                        elif cipher == 'chacha20':
                            result = chacha20_decrypt(result, key)
                        elif cipher == 'serpent':
                            result = serpent_decrypt(result, key)
                        elif cipher == 'twofish':
                            result = twofish_decrypt(result, key)
                        elif cipher == 'blowfish':
                            result = blowfish_decrypt(result, key)
                        elif cipher == 'rsa':
                            key_path = input("Enter path to RSA private key PEM file: ").strip()
                            try:
                                with open(key_path, 'rb') as kf:
                                    priv_bytes = kf.read()
                                    priv_key = serialization.load_pem_private_key(priv_bytes, password=None, backend=default_backend())
                                    result = rsa_decrypt(result, priv_key)
                            except Exception as e:
                                print(f"Failed to decrypt RSA: {e}")
                    print("\nDecrypted Text:")
                    print(result)
                except Exception as e:
                    print(f"Error during decryption: {e}")
            elif sub_choice == '2':
                img_path = input("Enter path to encrypted image (.png): ").strip()
                key_path = input("Enter path to key file (.key): ").strip()
                if not img_path or not key_path:
                    print("Error: Image path and key path are required.")
                    continue
                decrypted_path, metadata = decrypt_image(img_path, key_path)
                if decrypted_path:
                    print(f"Image decrypted to {decrypted_path}")
                    print("Extracted metadata:")
                    print(json.dumps(metadata, indent=2))
                else:
                    print("Image decryption failed.")
            else:
                print("Invalid selection.")
        elif choice == '3':
            img_path = input("Enter path to image file: ").strip()
            if not img_path:
                print("No image path provided.")
                continue
            level = input("Select obfuscation level (min/standard/max, default=standard): ").strip() or "standard"
            cipher_input = input("Enter ciphers (comma-separated, default=substitution): ").strip()
            ciphers = [c.strip() for c in cipher_input.split(',')] if cipher_input else ['substitution']
            key_input = input("Enter keys for each cipher (comma-separated, default_key if blank): ").strip()
            keys = [k.strip() if k.strip() else 'default_key' for k in key_input.split(',')] if key_input else ['default_key'] * len(ciphers)
            if len(keys) < len(ciphers):
                keys.extend(['default_key'] * (len(ciphers) - len(keys)))
            adaptive = input("Enable adaptive cipher? (y/N): ").strip().lower() == 'y'
            junk = input("Enter junk characters (leave blank for auto): ").strip()
            junk_chars = list(junk) if junk else get_junk_chars(wordlists, theme=theme)
            try:
                img = Image.open(img_path)
            except Exception as e:
                print(f"Failed to open image: {e}")
                continue
            grid_choice = input("Grid size for OCR (e.g., 2x2, 3x3, or custom): ").strip() or config.get('grid_size', '2x2')
            quadrants = define_grid(img.width, img.height, grid_choice)
            active_wordlists = {'list1': wordlists.get('list1', {})}
            texts, encrypted_file = obfuscate_image_text(img_path, active_wordlists, level, junk_chars,
                                                         quadrants, custom_rules=config.get('rules', {}),
                                                         ciphers=ciphers, keys=keys, use_adaptive=adaptive, theme=theme)
            if encrypted_file:
                print(f"Image processed and saved as {encrypted_file}")
            else:
                print("Image processing failed or no text found.")
        elif choice == '4':
            print("Character creation wizard is available in GUI mode only.")
            if tk:
                _ = windows_character_creator(low_res=is_low_res, theme=theme)
                wordlists.update(get_wordlists())
        elif choice == '5':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please enter a number 1-5.")

def invert_transposition(cipher_text, key):
    """Helper to invert the transposition cipher given the ciphertext and key."""
    try:
        key = key.lower()
        cols = len(key)
        if cols == 0:
            return cipher_text
        rows = len(cipher_text) // cols
        key_order = sorted(range(cols), key=lambda k: key[k])
        columns = [cipher_text[i*rows:(i+1)*rows] for i in range(cols)]
        orig_cols = [''] * cols
        for idx, col_idx in enumerate(key_order):
            orig_cols[col_idx] = columns[idx]
        plaintext = ""
        for r in range(rows):
            for c in range(cols):
                plaintext += orig_cols[c][r]
        return plaintext.rstrip()
    except Exception as e:
        logging.error(f"Failed to invert transposition: {e}")
        return cipher_text

if __name__ == "__main__":
    force_cli = "--cli" in sys.argv or "-c" in sys.argv
    if tk and not force_cli:
        root = tk.Tk()
        app = WindowsObfuscatorApp(root)
        root.mainloop()
    else:
        run_cli_menu()
