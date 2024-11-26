# NewsRoom Dataset

## Dataset Download
Ze względu na duży rozmiar plików (łącznie około 5.5GB), dataset jest dostępny do pobrania z Google Drive:
[Link do pobrania datasetu](https://drive.google.com/file/d/1E6ZBbN-5PGwqWdJ3Mnik7fDBtUaISKMi/view?usp=sharing)


### Statystyki:
- Train dataset: 995,041 przykładów
- Dev/Validation dataset: 108,837 przykładów 
- Test dataset: 108,862 przykładów

## Instrukcja użycia

### 1. Pobieranie i przygotowanie danych:
```bash
# Po pobraniu z Google Drive:
unzip newsroom_dataset.zip
cat train_part_* > train.jsonl
cat dev_part_* > dev.jsonl
cat test_part_* > test.jsonl
```

### 2. Używanie datasetu w Python:
```python
# Instalacja wymaganych bibliotek
pip install datasets

# Utworzenie folderu na dataset
import os
os.makedirs("~/.manual_dirs/newsroom", exist_ok=True)

# Przeniesienie plików .jsonl do utworzonego folderu
# Załadowanie datasetu
from datasets import load_dataset
dataset = load_dataset("newsroom", data_dir="~/.manual_dirs/newsroom")

# Sprawdzenie zawartości
print(f"Liczba przykładów w zbiorze treningowym: {len(dataset['train'])}")
print(f"Liczba przykładów w zbiorze walidacyjnym: {len(dataset['validation'])}")
print(f"Liczba przykładów w zbiorze testowym: {len(dataset['test'])}")
```

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

