# Base Project Instructions

## Environment and Language
- Zawsze używaj Pythona 3 (Python 3.10+).
- Kod powinien być w pełni zgodny ze standardami PEP 8 i nowoczesnymi praktykami Pythona 3.

## Architektura i Modularyzacja
- Kod projektu musi być modularny: każda odpowiedzialność (konfiguracja, przetwarzanie sygnałów, architektura modeli, trening/ewaluacja, wizualizacja) ma znajdować się w osobnym module w pakiecie `src/`.
- Notebooki Jupyter (`.ipynb`):
  - Komórki notebooka pełnią wyłącznie rolę orkiestracji i demonstracji.
  - Komórki nie mogą zawierać długich definicji funkcji, pętli treningowych ani skryptowych bloków kodu.
  - Komórka powinna składać się z 2-6 zwięzłych linii wywołujących funkcje z modułów `src/`.
- Testowanie i prototypowanie:
  - Domyślnie pipeline powinien umożliwiać uruchomienie na małym podzbiorze danych (np. `SAMPLE_RECORDS`) w celu szybkiej weryfikacji całego procesu przed pełnym treningiem.

## Zasady Pisania Kodu i Komunikacji
- Nie pisz komentarzy w plikach z kodem w miejscach oczywistych.
- Utrzymuj zwięzły, rzeczowy ton wypowiedzi bez hiperbolizacji.
