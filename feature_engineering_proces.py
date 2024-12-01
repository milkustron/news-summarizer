from gensim.models import Word2Vec
import pandas as pd
import numpy as np

class Feng:
    def __init__(self, preprocessed_df: pd.DataFrame):
        self.preprocessed_df = preprocessed_df  # Przechowuje dane wejściowe
        self.word2vec_model = None  # Inicjalizuje model Word2Vec jako None
  
    def w2v_init(self):
        # Łączenie tokenów z kolumn `text` i `summary`
        self.all_tokens = self.preprocessed_df['text'].tolist() + self.preprocessed_df['summary'].tolist()

        # Trenowanie modelu Word2Vec
        self.word2vec_model = Word2Vec(
            sentences=self.all_tokens,  # Listy tokenów (lista list słów)
            vector_size=512,            # Wymiar wektorów
            window=5,                   # Rozmiar okna kontekstu
            min_count=2,                # Minimalna liczba wystąpień słowa
            workers=4                   # Liczba wątków
        )
  
    def get_w2v_vector(self, tokens):
        """
        oblicza średni wektor dla listy tokenów.
        pomija słowa, które nie znajdują się w modelu Word2Vec.
        """
        valid_vectors = [
            self.word2vec_model.wv[word]
            for word in tokens
            if word in self.word2vec_model.wv
        ]
        if valid_vectors:
            return np.mean(valid_vectors, axis=0)
        else:
            return np.zeros(self.word2vec_model.vector_size)  # Zwraca wektor zerowy, jeśli brak słów w modelu
  
    def replace_w_embeddings(self):
        """
        zastępuje teksty w kolumnach 'text' i 'summary' ich wektorami Word2Vec.
        """
        self.preprocessed_df['text_vector'] = self.preprocessed_df['text'].apply(
            lambda tokens: self.get_w2v_vector(tokens)
        )
        self.preprocessed_df['summary_vector'] = self.preprocessed_df['summary'].apply(
            lambda tokens: self.get_w2v_vector(tokens)
        )

feng_instance = Feng(preprocessed_df=data_lemmatize)

# trenowanie modelu Word2Vec
feng_instance.w2v_init()

# zamiana tekstów na wektory Word2Vec
feng_instance.replace_w_embeddings()

# sprawdzenie wyników
df = feng_instance.preprocessed_df
df.head()
for el in df.iloc[:10,2]:
  print(len(el))
for el in df.iloc[:10,0]:
  print(len(el))