# NewsRoom Dataset

## Statystyki:
- Train dataset: 995,041 przykładów
- Dev/Validation dataset: 108,837 przykładów 
- Test dataset: 108,862 przykładów

## Struktura danych
Każdy rekord w datasecie zawiera:
- `text`: tekst artykułu
- `summary`: streszczenie
- `title`: tytuł artykułu
- `url`: URL artykułu
- `date`: data artykułu
- `density`: gęstość ekstrakcyjna
- `coverage`: pokrycie ekstrakcyjne
- `compression`: współczynnik kompresji
- `density_bin`: low/medium/high
- `coverage_bin`: extractive/abstractive
- `compression_bin`: low/medium/high

## Instrukcja użycia - wersja uproszczona
1. Pobrać wytrenowane dane z [Link](https://pejot-my.sharepoint.com/:u:/g/personal/s15259_pjwstk_edu_pl/EeJGYUm82pRAvKFKG5S6Y34BgJo1D5ADzxdoemXPZKMBgA?e=BJfsx2) i umieścić katalogu `./fine_tuned_bart`
2. Uruchomić dashboard za pomocą komendy `streamlit run .\app.py`

## Instrukcja użycia - pełny proces
1. Pobrac dane z: [Link](https://drive.google.com/file/d/1E6ZBbN-5PGwqWdJ3Mnik7fDBtUaISKMi/view?usp=sharing) i umieścić wszystkie pliki w katalogu pod ścieżka `./data`
2. Wywołać w konsoli komende `python prepare_downloaded_data.py` - dane zostaną złączone do jednego pliku
3. Wykonać wszystkie kroki z notatnika `text_summarization_word2vec.ipynb` - Trenowanie modelu word2vec
4. Wykonać wszystkie kroki z notatnika `text_summarizaiton_bart.ipynb` - Trenowanie modelu bart
5. Wykonać wszystkie kroki z notatnika `get_evals_w2v.ipynb` - zbieranie danych do porównania dla modelu word2vec
6. Wykonać wszystkie kroki z notatnika `get_evals_bert.ipynb` - zbieranie danych do porównania dla modelu bart
7. Wykonać wszystkie kroki z notatnika `compare_models.ipynb` - porównanie obu modeli
8. Uruchomić dashboard za pomocą komendy `streamlit run .\app.py`


## Demo
https://www.loom.com/share/ba21a31818204f4a96adce56b8f38893?sid=bf6822d2-05a3-42e5-a713-0c9ba6f8fc32