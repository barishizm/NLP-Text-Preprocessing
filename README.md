# NLP-Text-Preprocessing

Comprehensive demonstration of text preprocessing techniques using [NLTK](https://www.nltk.org/) and [spaCy](https://spacy.io/).

## Features
- Automatic download of required NLTK corpora and the `en_core_web_sm` spaCy model.
- Sentence and word tokenization with both NLTK and spaCy for comparison.
- Case normalisation and punctuation stripping with regular expressions.
- Stop‑word filtering using NLTK's stopword list and spaCy's built‑in flags.
- Stemming and lemmatisation (Porter, Snowball, WordNet, spaCy).
- Word cloud generation before and after stop‑word removal.
- Frequency analysis of tokens and n‑grams plus word/sentence length histograms.

## Installation
```bash
pip install nltk spacy wordcloud matplotlib pandas
```

## Usage
Run the main demonstration script:
```bash
python main.py
```
The script loads resources, tokenises the provided sample text, generates visualisations and prints common words and n‑grams. In headless environments you can disable interactive windows with `MPLBACKEND=Agg`.

To analyse your own corpus, replace the `text` variable in `main.py`.

## License
This project is released under the [MIT License](LICENSE).
