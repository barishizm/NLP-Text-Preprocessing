import re
import string
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk import word_tokenize, sent_tokenize

import spacy

# Safe downloader for NLTK resources
resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']
for r in resources:
    try:
        nltk.data.find(r)
    except LookupError:
        print(f"Downloading NLTK resource: {r}")
        nltk.download(r)

print("NLTK resources loaded successfully.")

# Safe loader for spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading 'en_core_web_sm' model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

print("spaCy model loaded successfully.")

# Sample text for processing
text = """
Universal Declaration of Human Rights

Preamble 
Whereas recognition of the inherent dignity and of the equal and inalienable rights of all members of the human family is the foundation of freedom, justice and peace in the world, 

Whereas disregard and contempt for human rights have resulted in barbarous acts which have outraged the conscience of mankind, and the advent of a world in which human beings shall enjoy freedom of speech and belief and freedom from fear and want has been proclaimed as the highest aspiration of the common people, 

Whereas it is essential, if man is not to be compelled to have recourse, as a last resort, to rebellion against tyranny and oppression, that human rights should be protected by the rule of law, 

Whereas it is essential to promote the development of friendly relations between nations, 

Whereas the peoples of the United Nations have in the Charter reaffirmed their faith in fundamental human rights, in the dignity and worth of the human person and in the equal rights of men and women and have determined to promote social progress and better standards of life in larger freedom, 

Whereas Member States have pledged themselves to achieve, in cooperation with the United Nations, the promotion of universal respect for and observance of human rights and fundamental freedoms, 

Whereas a common understanding of these rights and freedoms is of the greatest importance for the full realization of this pledge, Now, therefore, The General Assembly, Proclaims this Universal Declaration of Human Rights as a common standard of achievement for all peoples and all nations, to the end that every individual and every organ of society, keeping this Declaration constantly in mind, shall strive by teaching and education to promote respect for these rights and freedoms and by progressive measures, national and international, to secure their universal and effective recognition and observance, both among the peoples of Member States themselves and among the peoples of territories under their jurisdiction. 

Article I 
All human beings are born free and equal in dignity and rights. They are endowed with reason and conscience and should act towards one another in a spirit of brotherhood. 

Article 2 
Everyone is entitled to all the rights and freedoms set forth in this Declaration, without distinction of any kind, such as race, colour, sex, language, religion, political or other opinion, national or social origin, property, birth or other status. Furthermore, no distinction shall be made on the basis of the political, jurisdictional or international status of the country or territory to which a person belongs, whether it be independent, trust, non-self-governing or under any other limitation of sovereignty. 

Article 3 
Everyone has the right to life, liberty and security of person. Article 4 No one shall be held in slavery or servitude; slavery and the slave trade shall be prohibited in all their forms. Article 5 No one shall be subjected to torture or to cruel, inhuman or degrading treatment or punishment.
"""

# NLTK sentence and word tokenization
sentences_nltk = sent_tokenize(text)
words_nltk = word_tokenize(text)

print(f"NLTK: {len(sentences_nltk)} sentences, {len(words_nltk)} tokens")
print(sentences_nltk[:3])
print(words_nltk[:25])

# spaCy sentence and word tokenization
doc = nlp(text)
sentences_spacy = [sent.text.strip() for sent in doc.sents]
words_spacy = [t.text for t in doc]

print(f"\nspaCy: {len(sentences_spacy)} sentences, {len(words_spacy)} tokens")
print(sentences_spacy[:3])
print(words_spacy[:25])

lower_text = text.lower()
lower_tokens = [w.lower() for w in words_nltk]
lower_tokens[:5]

from pprint import pprint

# (A) Regex-driven: remove punctuation from text, then tokenize
text_no_punct = re.sub(f"[{re.escape(string.punctuation)}]", " ", lower_text)
tokens_no_punct = word_tokenize(text_no_punct)

print("A) Regex -> tokenize")
print("   Count:", len(tokens_no_punct))
print("   Sample:")
pprint(tokens_no_punct[:25])
#Result: A clean word list, but multi-punctuation cases (like "U.S.A.") may break awkwardly into "U", "S", "A".



# (B) Token-level filter (keeps numbers & words only, drops punctuation)
# tokens that have at least one alphanumeric character
tokens_alpha = [w for w in lower_tokens if any(ch.isalnum() for ch in w)]

print("\nB) Token-level filter (alnum only)")
print("   Count:", len(tokens_alpha))
print("   Sample:")
pprint(tokens_alpha[:25])
#Result: Faster, doesn’t require re-tokenization, and keeps numbers.


# (C) spaCy token-level: drop punctuation AND whitespace, then lemmatize+lower
tokens_spacy_alpha = [
    t.lemma_.lower()
    for t in doc
    if not (t.is_punct or t.is_space)
]

print("\nC) spaCy filter (no punct/space) + lemma+lower")
print("   Count:", len(tokens_spacy_alpha))
print("   Sample:")
pprint(tokens_spacy_alpha[:25])

# Result: A clean list of lemmatized tokens without punctuation. More linguistically informed than regex or NLTK filter.

# NLTK Stopword Removal (≈179 words like the, of, is, and).
stop_en = set(stopwords.words('english'))

tokens_no_stop = [w for w in tokens_alpha if w not in stop_en]
len(tokens_alpha), len(tokens_no_stop), tokens_no_stop[:30]
print("NLTK:", len(tokens_alpha), "->", len(tokens_no_stop))
print(tokens_no_stop[:30])



# spaCy’s built-in stop words (slightly different set)
spacy_stop = nlp.Defaults.stop_words
tokens_spacy_no_stop = [t.text.lower() for t in doc if not t.is_punct and t.text.lower() not in spacy_stop]
len(tokens_spacy_alpha), len(tokens_spacy_no_stop), tokens_spacy_no_stop[:30]
print("\nspaCy:", len(tokens_spacy_alpha), "->", len(tokens_spacy_no_stop))
print(tokens_spacy_no_stop[:35])

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Texts for word cloud
tokens_with_stop = " ".join(tokens_alpha)      # before stopword removal
tokens_no_stop_text = " ".join(tokens_no_stop) # after stopword removal

# Generate word clouds
wc_with = WordCloud(width=600, height=400,
                    background_color="white",
                    colormap="viridis").generate(tokens_with_stop)

wc_no = WordCloud(width=600, height=400,
                  background_color="white",
                  colormap="viridis").generate(tokens_no_stop_text)

# Plot side by side
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

axes[0].imshow(wc_with, interpolation="bilinear")
axes[0].set_title("Before Stopword Removal", fontsize=14)
axes[0].axis("off")

axes[1].imshow(wc_no, interpolation="bilinear")
axes[1].set_title("After Stopword Removal", fontsize=14)
axes[1].axis("off")

plt.show()

import pandas as pd

sample_tokens = tokens_no_stop[:40]

porter = PorterStemmer()            # Porter stemmer → classic, aggressive, sometimes produces odd roots.
snowball = SnowballStemmer('english')       # Snowball stemmer → an improved version of Porter (English Snowball).
wnl = WordNetLemmatizer()      # WordNet lemmatizer → uses WordNet dictionary to return proper lemmas (but requires part-of-speech for best results).

rows = []
for w in sample_tokens:
    rows.append({
        "token": w,
        "porter_stem": porter.stem(w),
        "snowball_stem": snowball.stem(w),
        "wordnet_lemma": wnl.lemmatize(w)
    })

# Convert to a DataFrame for pretty display
df = pd.DataFrame(rows)
df.head(20)  # show first 20 rows

# (C) spaCy token-level: drop punctuation AND whitespace, then lemmatize+lower
doc = nlp(" ".join(sample_tokens))
df_spacy = pd.DataFrame([(t.text, t.lemma_, t.pos_) for t in doc],
                        columns=["token", "lemma", "POS"])
df_spacy.head(200)

def top_n(counter, n=20):
    return counter.most_common(n)

# Choose the pipeline you prefer; here use NLTK tokens without stopwords
freq = Counter(tokens_no_stop)
top20 = top_n(freq, 20)
top20

# Character length of words (unique or all occurrences? We'll do all occurrences)
word_lengths = [len(w) for w in tokens_no_stop if w.isalpha()]

plt.figure()
plt.hist(word_lengths, bins=range(1, 21), edgecolor='black')
plt.xlabel("Word length (characters)")
plt.ylabel("Frequency")
plt.title("Word Length Distribution (post stopword removal)")
plt.show()

# Sentence length in words (using NLTK tokenization)
sent_lengths = [len(word_tokenize(s)) for s in sentences_nltk]

plt.figure()
plt.hist(sent_lengths, bins=range(0, max(sent_lengths)+2, 2), edgecolor='black')
plt.xlabel("Sentence length (tokens)")
plt.ylabel("Count")
plt.title("Sentence Length Distribution")
plt.show()

results = {
    "nltk_clean_nostop": tokens_no_stop,
    "spacy_clean_nostop": tokens_spacy_no_stop,
    "spacy_lemma_nostop": tokens_spacy_alpha,
}

def top_n_words(tokens, n=15):
    return Counter(tokens).most_common(n)

N = 15
for name, toks in results.items():
    print(f"\n=== {name} (top {N}) ===")
    for w, c in top_n_words(toks, N):
        print(f"{w:>15s} : {c}")

from nltk.util import ngrams

def top_n_ngrams(tokens, n=2, topk=15):
    grams = list(ngrams(tokens, n))
    return Counter(grams).most_common(topk)

print("Top bigrams (NLTK clean nostop):")
for grams, c in top_n_ngrams(results["nltk_clean_nostop"], n=2, topk=15):
    print(f"{' '.join(grams)} : {c}")

print("\nTop trigrams (NLTK clean nostop):")
for grams, c in top_n_ngrams(results["nltk_clean_nostop"], n=3, topk=10):
    print(f"{' '.join(grams)} : {c}")

# Inspect spaCy linguistic features
doc = nlp(text)
sample = [(t.text, t.lemma_, t.pos_, t.tag_, t.is_stop, t.is_punct) for t in doc[:40]]
sample




