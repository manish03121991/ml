import os
import nltk
from ml_engine.nlp import Tokenizer, Stemmer, Lemmatizer, Ngram
from .utils import read_data_from_file
from ml_engine.config import BASE_DIR_PATH

# Get base stop words from file
base_stop_words_file_path = os.path.join(BASE_DIR_PATH, 'ml_engine', 'stopwords')

# regular exp used for punctuation removal
regex_to_remove_punctuations = [(r'[^\w\s]', '')]

# Initiate tokenize class
tokenizer = Tokenizer()

# Initiate stemming class
stemmer = Stemmer()

# Initiate lemma class
lemmatizer = Lemmatizer()

# Initiate ngram class
ngram = Ngram()

# POS tagging
pos_tagger = nltk.pos_tag

# POS taggers to be consider for further processing
part_of_speech = ['ADJ', 'ADV', 'NN', 'VERB', 'CONJ', 'DET']

# stopwords filteration

stopwords_allow_filter = ['sentiment', 'default']
