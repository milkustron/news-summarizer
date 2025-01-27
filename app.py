import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

import gensim
from gensim.models import KeyedVectors

from transformers import BartForConditionalGeneration, BartTokenizer, AutoConfig

@st.cache_data
def load_word2vec():
    w2v_path = "PATH"  # Tu trzeba podać ścieżkę do modelu Word2Vec
    w2v_model = KeyedVectors.load(w2v_path)
    return w2v_model


@st.cache_resource
def load_transformer_model():
    """
    Ładuje lokalny fine-tuned model BART
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = "PATH" # Tu trzeba podać ścieżkę do modelu 
        
        config = AutoConfig.from_pretrained(model_dir)
        config.forced_bos_token_id = 0
        
        tokenizer = BartTokenizer.from_pretrained(model_dir)
        model = BartForConditionalGeneration.from_pretrained(
            model_dir,
            config=config
        ).to(device)
        model.eval()
        
        return tokenizer, model
    except Exception as e:
        st.error(f"Błąd ładowania modelu: {str(e)}")
        return None, None


def generate_summary(tokenizer, model, text):
    """
    Generuje streszczenie tekstu z dynamiczną długością wyjścia
    """
    try:
        # Przygotowanie tekstu
        text = text.strip()
        if not text:
            return None
            
        # Obliczanie długości wejściowego tekstu
        input_length = len(text.split())
        
        # Dynamiczne ustawienie długości streszczenia
        if input_length < 100:  # Krótki tekst
            target_length = 50
            min_length = 30
        elif input_length < 300:  # Średni tekst
            target_length = 100
            min_length = 50
        else:  # Długi tekst
            target_length = 200
            min_length = 100
            
        # Tokenizacja
        inputs = tokenizer(
            text, 
            max_length=1024,  # Zwiększamy maksymalną długość wejścia
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(model.device)
        
        # Konfiguracja generacji
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=target_length,    # Dynamiczna długość
            min_length=min_length,       # Minimalna długość
            length_penalty=2.0,          # Zachęcamy do dłuższych streszczeń
            num_beams=4,                # Beam search
            early_stopping=True,
            no_repeat_ngram_size=3,     # Unikamy powtórzeń
            do_sample=True,             # Sampling dla różnorodności
            top_k=50,                   # Top-k sampling
            top_p=0.9,                  # Nucleus sampling
            temperature=0.8             # Temperatura dla kreatywności
        )
        
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
        
    except Exception as e:
        st.error(f"Błąd podczas generowania streszczenia: {str(e)}")
        return None


def calculate_rouge(hypothesis, reference):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores["rouge1"].fmeasure, scores["rougeL"].fmeasure


def analyze_errors(df):
    df['długość_tekstu'] = df['tekst'].str.len()
    df['długość_referencji'] = df['referencja'].str.len()
    df['długość_predykcji'] = df['pred_summary'].str.len()
    
    return {
        'średnia_długość_tekstu': df['długość_tekstu'].mean(),
        'średnia_długość_referencji': df['długość_referencji'].mean(),
        'średnia_długość_predykcji': df['długość_predykcji'].mean(),
        'najgorsze_przypadki': df.nsmallest(5, 'rouge1')
    }


def run_app():
    st.title("Tablica Analizy Streszczeń")
    
    st.sidebar.header("Ustawienia")
    model_choice = st.sidebar.selectbox("Wybierz model", ["Word2Vec", "Transformers"])
    
    if model_choice == "Word2Vec":
        st.header("Word2Vec Demo")
        w2v_model = load_word2vec()

        word = st.text_input("Wpisz słowo, by znaleźć podobne")
        if word and st.button("Znajdź podobne"):
            try:
                if hasattr(w2v_model, 'key_to_index'):
                    word_exists = word in w2v_model.key_to_index
                else:
                    word_exists = word in w2v_model.wv.vocab

                if word_exists:
                    similar = w2v_model.most_similar(word, topn=5)
                    st.write("Najbardziej podobne słowa:")
                    for (w, score) in similar:
                        st.write(f"{w}: {score:.4f}")
                else:
                    st.warning(f"Słowa '{word}' nie ma w słowniku w2v.")
            except Exception as e:
                st.error(f"Błąd podczas przetwarzania: {str(e)}")
    
    elif model_choice == "Transformers":
        st.header("Analiza Streszczeń")
        
        # Zakładki
        tab1, tab2, tab3 = st.tabs(["Test Pojedynczy", "Analiza Zbioru", "Analiza Błędów"])
        
        tokenizer, model = load_transformer_model()
        if tokenizer is None or model is None:
            st.error("Nie udało się załadować modelu.")
            return
            
        max_len = st.sidebar.slider("Maksymalna długość streszczenia", 100, 500, 400)
        
        with tab1:
            st.subheader("Test na pojedynczym tekście")
            user_input = st.text_area("Wklej tekst do streszczenia:", height=200)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Generuj streszczenie"):
                    if user_input.strip():
                        with st.spinner("Generuję streszczenie..."):
                            summary = generate_summary(tokenizer, model, user_input)
                            if summary:
                                st.success("Wygenerowane streszczenie:")
                                st.write(summary)
                    else:
                        st.warning("Proszę wpisać tekst do streszczenia.")
            
        with tab2:
            st.subheader("Analiza na zbiorze testowym")
            uploaded_file = st.file_uploader("Wgraj plik CSV (kolumny: tekst, referencja)", type=["csv"])
            
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                
                with st.spinner("Przetwarzam dane..."):
                    predictions = []
                    rouge1_scores = []
                    rougeL_scores = []
                    
                    progress_bar = st.progress(0)
                    for idx, row in df.iterrows():
                        pred_summary = generate_summary(tokenizer, model, row['tekst'])
                        predictions.append(pred_summary)
                        rouge1, rougeL = calculate_rouge(pred_summary, row['referencja'])
                        rouge1_scores.append(rouge1)
                        rougeL_scores.append(rougeL)
                        progress_bar.progress((idx + 1) / len(df))
                    
                    df['pred_summary'] = predictions
                    df['rouge1'] = rouge1_scores
                    df['rougeL'] = rougeL_scores
                    
                    # Wizualizacje
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Średni ROUGE-1", f"{np.mean(rouge1_scores):.3f}")
                    with col2:
                        st.metric("Średni ROUGE-L", f"{np.mean(rougeL_scores):.3f}")
                    
                    st.subheader("Rozkład wyników ROUGE")
                    fig = plt.figure(figsize=(10, 5))
                    plt.hist(rouge1_scores, alpha=0.5, label='ROUGE-1')
                    plt.hist(rougeL_scores, alpha=0.5, label='ROUGE-L')
                    plt.legend()
                    st.pyplot(fig)
                    
                    st.subheader("Wyniki szczegółowe")
                    st.dataframe(df[['tekst', 'referencja', 'pred_summary', 'rouge1', 'rougeL']])
        
        with tab3:
            st.subheader("Analiza błędów")
            if 'df' in locals() and len(df) > 0:
                error_analysis = analyze_errors(df)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Średnia długość tekstu", f"{error_analysis['średnia_długość_tekstu']:.0f}")
                with col2:
                    st.metric("Średnia długość referencji", f"{error_analysis['średnia_długość_referencji']:.0f}")
                with col3:
                    st.metric("Średnia długość predykcji", f"{error_analysis['średnia_długość_predykcji']:.0f}")
                
                st.subheader("Najgorsze przypadki")
                st.dataframe(error_analysis['najgorsze_przypadki'][['tekst', 'referencja', 'pred_summary', 'rouge1', 'rougeL']])
            else:
                st.info("Wgraj dane w zakładce 'Analiza Zbioru' aby zobaczyć analizę błędów.")


if __name__ == "__main__":
    run_app()
 