import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


nltk.download('stopwords')

stop_words = set(stopwords.words("english"))
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def summarize_text_bart(text, trained_model):
    """Generate a summary for the given text."""
    inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True).to(device)
    summary_ids = trained_model.generate(inputs["input_ids"], max_length=128, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return tokens

class Feng:
    def __init__(self, preprocessed_df: pd.DataFrame):
        self.preprocessed_df = preprocessed_df  # Store input data
        self.word2vec_model = None  # Initialize Word2Vec model
        self.all_tokens = self.preprocessed_df['text_tokens'].tolist() + self.preprocessed_df['summary_tokens'].tolist()

    def w2v_init(self):
        self.word2vec_model = Word2Vec(
            sentences=self.all_tokens,  # List of token lists
            vector_size=100,            # Dimension of vectors
            window=5,                   # Context window size
            min_count=2,                # Minimum word frequency
            workers=4                   # Number of threads
        )

        self.word2vec_model.save("word2vec.model")
        self.word2vec_model.load("word2vec.model")

    def get_w2v_vector(self, tokens):
        """
        Compute average vector for a list of tokens.
        Skip words not in Word2Vec model.
        """
        valid_vectors = [
            self.word2vec_model.wv[word]
            for word in tokens
            if word in self.word2vec_model.wv
        ]
        if valid_vectors:
            return np.mean(valid_vectors, axis=0)
        else:
            return np.zeros(self.word2vec_model.vector_size)  # Return zero vector if no valid words

    def replace_w_embeddings(self):
        """
        Replace text and summary with their Word2Vec embeddings.
        """
        self.preprocessed_df['text_vector'] = self.preprocessed_df['text_tokens'].apply(
            lambda tokens: self.get_w2v_vector(tokens)
        )
        self.preprocessed_df['summary_vector'] = self.preprocessed_df['summary_tokens'].apply(
            lambda tokens: self.get_w2v_vector(tokens)
        )


def rank_sentences(text, model):
    """
    Rank sentences in a document based on their similarity to the document vector.
    """
    sentences = sent_tokenize(text)
    sentence_tokens = [preprocess_text(sentence) for sentence in sentences]

    # Compute sentence vectors
    sentence_vectors = [model.get_w2v_vector(tokens) for tokens in sentence_tokens]

    # Compute document vector
    document_vector = np.mean(sentence_vectors, axis=0)

    # Rank sentences by similarity to document vector
    similarities = [
        cosine_similarity([sentence_vector], [document_vector])[0][0]
        if sentence_vector.any() else 0
        for sentence_vector in sentence_vectors
    ]

    ranked_sentences = sorted(
        zip(sentences, similarities), key=lambda x: x[1], reverse=True
    )

    return [sentence for sentence, _ in ranked_sentences[:3]]  # Top 3 sentences

