# EKG Signal Processing: Klasyfikacja arytmii w bazie MIT-BIH

Projekt implementuje potoki przetwarzania i klasyfikacji sygnałów elektrokardiograficznych (EKG) z bazy [MIT-BIH Arrhythmia Database](https://archive.physionet.org/physiobank/database/mitdb/) z podziałem na pobudzenia normalne oraz arytmiczne.

Głównym założeniem metodologicznym jest rygorystyczna **walidacja międzyosobnicza (inter-patient division)** za pomocą `GroupKFold(groups=patient_ids)`, zapobiegająca wyciekowi danych (patient leakage) i memorizacji indywidualnego „podpisu elektrycznego” pacjentów.

---

## Zastosowane technologie

- **Język**: Python 3.11
- **Dane biomedyczne**: `wfdb` (PhysioNet MIT-BIH Arrhythmia Database)
- **Przetwarzanie sygnałów**: `scipy.signal` (filtr pasmowy Butterwortha 0.5–40 Hz), `numpy`
- **Analiza czasowo-częstotliwościowa**: `pywavelets` (ciągła transformata falkowa CWT: Morlet, Mexican Hat)
- **Uczenie maszynowe i głębokie**: `tensorflow` / `keras` (1D CNN, 2D CNN), `scikit-learn` (GroupKFold, balansowanie klas, metryki)
- **Wizualizacja**: `matplotlib`

---

## Architektura i zaimplementowane procesy

Projekt udostępnia dwa równoległe podejścia badawcze:

```
EKG-Signal-Processing/
│
├── EKG.ipynb              # Proces 1: 1D CNN + SVM (CS-Net) + multimodalne cechy
├── EKG_CWT.ipynb          # Proces 2: 2D CNN na skalogramach CWT (Morlet vs Mexican Hat)
│
├── src/                   # Zmodularyzowany kod źródłowy
│   ├── config.py          # Konfiguracja, rekordy, mapowanie klas AAMI EC57, de Chazal DS1/DS2
│   ├── preprocessing.py   # Filtracja 0.5-40 Hz, segmentacja, per-record Z-score, balancing
│   ├── features.py        # Ekstrakcja 14 cech biomedycznych + 4 cechy relacyjne QRS, ANOVA F
│   ├── models.py          # Modele 1D CNN (CS-Net, hybrydowy CS-Net fuzji cech)
│   ├── training.py        # Ewaluacja GroupKFold, out-of-fold threshold tuning, protokół de Chazala
│   ├── evaluation.py      # Metryki: Accuracy, Bal-Acc, Precision, Recall, F1, MCC, macierz per-pacjent
│   ├── cwt.py             # Obliczanie skalogramów 2D CWT, wizualizacja czasowo-częstotliwościowa
│   ├── cwt_models.py      # Architektura 2D CNN dla skalogramów czasowo-częstotliwościowych (opcja Focal Loss)
│   ├── cwt_training.py    # Walidacja GroupKFold dla CWT i porównanie falek
│   ├── modwt.py           # Dekompozycja wielopodpasmowa MODWT/SWT (translation-invariant)
│   ├── wavelet_models.py  # Architektura WaveletNet (WaveletConv1D, 1x1 subband weighting, BLSTM, Focal Loss)
│   ├── wavelet_training.py# Walidacja GroupKFold dla WaveletNet i porównanie falek dyskretnych
│   ├── metadata.py        # Analiza demograficzna i rozkłady klas AAMI EC57
│   ├── signal_stats.py    # Statystyki sygnałów EKG (amplituda, wariancja, skośność, kurtoza)
│   └── visualization.py   # Wykresy przebiegów, widma QRS i macierze pomyłek
│
└── requirements.txt       # Zależności projektu
```

---

### Proces 1: 1D CNN + SVM (CS-Net) i cechy relacyjne ([EKG.ipynb](file:///e:/Github/EKG-Signal-Processing/EKG.ipynb))

1. **Segmentacja i normalizacja**:
   - Asymetryczne okno QRS ($-50\text{ ms} \dots +100\text{ ms}$, 54 próbki przy $360\text{ Hz}$) skupione na depolaryzacji komór.
   - Normalizacja Per-Record Z-score zachowująca kontrast energetyczny $V_{pp}$ między pobudzeniami.
2. **Inżynieria cech**:
   - 14 cech czasowo-widmowych oraz 4 relacyjne cechy geometryczne QRS: $\text{Var}(dV/dt)$, asymetria zbocza, szerokość FWHM i pole powierzchni.
3. **Audyt wycieku danych**:
   - Test permutacyjny i permutacja blokowa pacjentów udowodniły, że anomalia $F > 15\,000$ dla $\text{Var}(dV/dt)$ wynikała ze specyfiki aparatury pacjentów ze stymulatorem (rekordy 102/104), a nie patologii fizjologicznej.
   - Strojenie progu decyzyjnego Out-Of-Fold ($T = -0.2$) dało stabilny wzrost czułości przy zachowaniu specyficzności.

---

### Proces 2: CWT + 2D CNN ([EKG_CWT.ipynb](file:///e:/Github/EKG-Signal-Processing/EKG_CWT.ipynb))

1. **Reprezentacja czasowo-częstotliwościowa**:
   - Odcinki pobudzeń przekształcane są ciągłą transformatą falkową (CWT) na dwuwymiarowe skalogramy o wymiarach $32 \times 32$.
   - Porównanie dwóch odmiennych typów falek:
     - **Morlet (`MORL`)**: oscylacyjna falka zespolona o wysokiej selektywności częstotliwościowej.
     - **Mexican Hat (`MEXH`)**: zlokalizowana falka impulsowa (druga pochodna Gaussa), dopasowana do ostrej morfologii załamka R.
2. **Sieć 2D CNN**:
   - Warstwy `Conv2D` ($16, 32$), `MaxPooling2D`, `Flatten`, `Dense(32)`, `Dropout(0.2)` oraz wyjście sigmoidalne z opcjonalną funkcją straty `BinaryFocalCrossentropy`.
3. **Walidacja międzyosobnicza**:
   - `GroupKFold(n_splits=4)` na rekordach pacjentów bez stymulatora (`100`, `106`, `109`, `119`).

---

### Proces 3: MODWT + WaveletNet (Inspiracja WEFormer / A. Kobus 2025)

1. **Dekompozycja wielopodpasmowa MODWT (Maximal Overlap Discrete Wavelet Transform)**:
   - W odróżnieniu od klasycznej DWT, MODWT nie stosuje decymacji (downsamplingu), dzięki czemu każde podpasmo zachowuje pełną rozdzielczość czasową ($180$ próbek) i cechuje się niezmienniczością względem przesunięć (translation invariance).
   - Rozbicie sygnału na poziom 3 generuje 4 składowe: $A_3$ (niskoczęstotliwościowy trend i fala T) oraz $D_3, D_2, D_1$ (detale o narastających częstotliwościach: zbocza QRS i szum EMG).
   - Przetestowano falki o zwartym nośniku: Daubechies (`DB4`) oraz Symlet (`SYM4`).
2. **Architektura WaveletNet i WaveletConv1D**:
   - Warstwa splotowa $1 \times 1$ Conv1D na kanałach falkowych automatycznie uczy się wag ważności podpasm, wyciszając szumy w kanale $D_1$ i wzmacniając sygnatury komorowe w $D_2$ i $D_3$.
   - Bloki 1D CNN z Batch Normalization, opcjonalnym kodowaniem sekwencyjnym BLSTM oraz warstwą klasyfikującą.
3. **Focal Loss**:
   - Zastosowanie `BinaryFocalCrossentropy` ($\gamma = 2.0$) redukuje wpływ dominującej klasy pobudzeń normalnych i skupia gradient na trudnych przypadkach brzegowych.

---

## Wyniki walidacji międzyosobniczej

Wszystkie wyniki obliczono w trybie walidacji krzyżowej bez wycieku pacjentów (`GroupKFold` na rekordach `100`, `106`, `109`, `119`):

| Podejście | Reprezentacja wejściowa | Model | Accuracy | Precision (macro) | Recall (macro) | F1-Score (macro) | MCC (Absolute) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Proces 1 (OOF)** | Asymetryczne okno 1D (54 próbki) | CS-Net (1D CNN + SVM) | 88.66% | 88.10% | 87.36% | 0.8715 | 0.7818 |
| **Proces 2 (CWT MORL)** | Skalogram 2D Morlet ($32 \times 32$) | 2D CNN | 67.37% | 70.43% | 62.28% | 0.5656 | 0.5900 |
| **Proces 2 (CWT MEXH)** | Skalogram 2D Mexican Hat ($32 \times 32$) | 2D CNN | 70.81% | 65.47% | 70.77% | 0.6021 | 0.4455 |
| **Proces 3 (MODWT DB4)** | Multi-subband 1D ($180 \times 4$) | WaveletNet (1x1 Conv1D + 1D CNN) | **74.51%** | 71.69% | **73.56%** | **0.6339** | 0.5798 |
| **Proces 3 (MODWT SYM4)** | Multi-subband 1D ($180 \times 4$) | WaveletNet (1x1 Conv1D + 1D CNN) | 74.12% | 68.84% | 71.49% | 0.6225 | **0.5923** |
| **Proces 3 (MODWT DB4 + Focal)** | Multi-subband 1D ($180 \times 4$) | WaveletNet + Focal Loss ($\gamma=2$) | 73.40% | **78.96%** | 70.38% | 0.6118 | 0.5737 |

### Szczegółowa analiza techniczna wyników:

1. **Wpływ filtracji wstępnej i weryfikacja poziomu szumów**:
   - Przed podaniem sygnału do modeli (zarówno CWT, jak i MODWT) zastosowano cyfrowy filtr pasmowoprzepustowy Butterwortha 3. rzędu w paśmie 0.5–40 Hz.
   - Pomiary gęstości widmowej mocy (Welch PSD) na przefiltrowanych rekordach wykazały, że pasmo diagnostyczne 0.5–40 Hz skupia **99.0% – 99.9% całkowitej energii sygnału**. Pasmo zakłóceń linii izoelektrycznej (<0.5 Hz) stanowi 0.1% – 1.0%, a szumy wysokoczęstotliwościowe (>40 Hz, w tym szum sieciowy 60 Hz oraz szum mięśniowy EMG) stanowią **poniżej 0.1% energii**.
   - **Wniosek techniczny**: W sygnale wejściowym nie występował istotny szum zewnętrzny (>40 Hz). W związku z tym różnice w wynikach nie wynikają z „lepszego odszumiania” przez dany algorytm, lecz z reprezentacji geometryczno-czasowej i sposobu kodowania sygnału.

2. **Ocena różnicy między MODWT a CWT (Czy to rzeczywista przewaga?)**:
   - Średnia dokładność MODWT (74.51%) jest wyższa od CWT MEXH (70.81%) i CWT MORL (67.37%), a F1-Score wynosi 0.6339 vs 0.6021. Jednak przy odchyleniu standardowym między foldami wynoszącym odpowiednio $+/- 0.47$ i $+/- 0.41$, różnica rzędu 3.7 p.p. mieści się w granicach błędu walidacji międzyosobniczej.
   - Przypisywanie MODWT bezwarunkowej „przewagi algorytmicznej” byłoby nadinterpretacją. Źródłem obserwowanej rozbieżności są specyficzne ograniczenia transformacji 2D:
     - **Artefakt decymacji przestrzennej (Spatial Resampling Artifact)**: Aby podać skalogram do standardowej sieci 2D CNN, macierz CWT jest zmniejszana do wymiaru $32 \times 32$ za pomocą interpolacji powierzchniowej (`INTER_AREA`). Wymusza to 5.6-krotną kompresję w osi czasu (ze 180 do 32 próbek). Zespół prawidłowy o szerokości ~80 ms (28 próbek) kurczy się do zaledwie 5 pikseli, co rozmywa krawędzie załamków i utrudnia sieci odróżnienie wąskiego zespołu od poszerzonego pobudzenia komorowego.
     - **Utrata znaku i fazy w CWT**: Skalogram reprezentuje wyłącznie moduł współczynników $|W(a,b)|$, co eliminuje informację o biegunowości załamka R (dodatni vs odwrócony). MODWT operuje na współczynnikach zachowujących znak i pełną rozdzielczość 180 próbek w każdym z 4 podpasm.
     - **Morfologia falki**: Mexican Hat (druga pochodna Gaussa) lepiej lokalizuje ostry impuls załamka R niż oscylacyjna falka Morlet, która tworzy wstęgi interferencyjne wokół iglicy QRS.

3. **Dlaczego Proces 1 (CS-Net) uzyskał najwyższy wynik (88.66%)**:
   - Wyższy wynik Procesu 1 nie wynika z przewagi samej sieci 1D CNN nad transformatami falkowymi, lecz z odmiennej definicji okna i inżynierii cech:
     - **Wąskie asymetryczne okno QRS (54 próbki = 150 ms)**: Proces 1 wycina wyłącznie zespół depolaryzacji komorowej ($-50\text{ ms} \dots +100\text{ ms}$), eliminując załamki P i T. W oknie 180 próbek (500 ms) stosowanym w CWT i MODWT obecna jest fala T, której morfologia u różnych pacjentów wykazuje silną zmienność fizjologiczną, działającą na model jak szum cechowy.
     - **Jawne cechy relacyjne QRS**: Parametry $\text{Var}(dV/dt)$, szerokość FWHM i asymetria zboczy bezpośrednio podają sieci informację o geometrii zespołu, bez konieczności jej aproksymacji przez sploty.
     - **Strojenie progu Out-Of-Fold ($T = -0.2$)**: Skorygowanie progu decyzyjnego kompensuje przesunięcie bazowe między pacjentami, czego nie stosowano w CWT i MODWT (gdzie próg wynosił sztywne $P = 0.5$).

4. **Szum fizjologiczny i odmienność pacjentów (Rekord 109)**:
   - Wysoka wariancja w walidacji `GroupKFold` wynika z patologii przewodzenia w rekordzie 109 (blok lewej odnogi pęczka Hisa – LBBB). U tego pacjenta prawidłowe pobudzenia zatokowe są poszerzone i odwrócone.
   - Modele falkowe klasyfikujące całe okno 180 próbek przy sztywnym progu $P = 0.5$ traktują tę odmienność jako patologię, co powoduje spadek specyficzności na tym foldzie. Z kolei na rekordach 106 i 119 modele osiągają F1 rzędu 82–100%. Wariancja ta odzwierciedla realne wyzwania uogólniania modeli EKG na nieznanych pacjentów.

---

## Uruchomienie projektu

1. Zainstalowanie zależności:
   ```bash
   pip install -r requirements.txt
   ```
2. Otwarcie i uruchomienie notatników:
   - [EKG.ipynb](file:///e:/Github/EKG-Signal-Processing/EKG.ipynb) – potok 1D CNN + SVM i inżynieria cech.
   - [EKG_CWT.ipynb](file:///e:/Github/EKG-Signal-Processing/EKG_CWT.ipynb) – potok CWT + 2D CNN (obliczone wyniki w pliku).
