import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Download NLTK data (only first time)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

nlp = spacy.load("en_core_web_sm")

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# ---------- BASIC NLP FUNCTIONS ----------

def lowercase(text):
    return text.lower()

def remove_punct_num(text):
    return re.sub(r'[^a-zA-Z\s]', '', text)

def remove_stopwords(text):
    words = nltk.word_tokenize(text)
    return ' '.join([w for w in words if w.lower() not in stop_words])

def stemming(text):
    words = nltk.word_tokenize(text)
    return ' '.join([stemmer.stem(w) for w in words])

def lemmatization(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

def pos_tagging(text):
    words = nltk.word_tokenize(text)
    return nltk.pos_tag(words)

def bag_of_words(corpus):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([corpus])
    return dict(zip(vectorizer.get_feature_names_out(), X.toarray()[0]))

def tf_idf(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([corpus])
    return dict(zip(vectorizer.get_feature_names_out(), X.toarray()[0]))
