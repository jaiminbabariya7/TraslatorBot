# Multilingual Translator Bot

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?logo=huggingface)
![Languages](https://img.shields.io/badge/Languages-50%2B%20pairs-brightgreen)
![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey?logo=flask)
![License](https://img.shields.io/badge/License-MIT-green)

> Multilingual translator chatbot powered by Hugging Face transformer seq2seq models (Helsinki-NLP/opus-mt). Supports 50+ language pairs via a clean web interface and Python API.

## Architecture
```
User Input (source text + target language)
        ↓
Language Detection (langdetect)
        ↓
Hugging Face Pipeline
  └── Helsinki-NLP/opus-mt-{src}-{tgt}
      (MarianMT seq2seq transformer)
        ↓
Translated Output
        ↓
Flask Web Interface / REST API
```

## Supported Languages (sample)
English ↔ French, German, Spanish, Italian, Portuguese, Dutch, Russian, Chinese, Japanese, Hindi, Arabic, and 40+ more via opus-mt models.

## Project Structure
```
├── app.py                 # Flask web application
├── cbot.py                # Core translation logic
├── app.ipynb              # Interactive Jupyter demo
├── cbot.ipynb             # Model exploration notebook
└── templates/
    └── index.html         # Web interface
```

## Usage
```python
from cbot import translate

result = translate("Hello, how are you?", src="en", tgt="fr")
print(result)  # "Bonjour, comment allez-vous?"
```

## Web App
```bash
pip install transformers flask langdetect sentencepiece
python app.py   # → http://localhost:5000
```

## Skills Demonstrated
`Python` · `Transformers` · `Hugging Face` · `seq2seq` · `NLP` · `Flask` · `MarianMT` · `Multilingual AI`
