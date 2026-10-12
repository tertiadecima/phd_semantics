# Автоматическая семантическая разметка русского текста

Исследовательский проект по автоматическому присвоению семантических тегов словам в контексте. В репозитории сравниваются словарные, обучаемые и гибридные подходы на основе Word2Vec и RuBERT. Эксперименты оформлены как Jupyter-ноутбуки и работают с корпусами в формате CoNLL-U.

## Цель экспериментов

Основная задача — проверить, насколько разные способы представления слова и его контекста подходят для многометочной семантической классификации русских слов.

В проекте исследуются:

- zero-shot-разметка через близость контекста, словарного определения и описаний семантических классов;
- обучаемая классификация по Word2Vec-признакам слова, контекста и части речи;
- гибрид словарных и supervised-признаков;
- контекстная классификация с RuBERT без дообучения и с fine-tuning;
- переносимость моделей между жанрами: художественными текстами, новостями, поэзией, социальными медиа и Википедией.

Для supervised-экспериментов автоматически размеченные корпуса используются для обучения, ручной fiction-корпус — для валидации и выбора порогов, а ручные корпуса `news`, `poetry`, `social` и `wiki` — только для итоговой проверки.

## Эксперименты

| Ноутбук | Подход | Назначение |
| --- | --- | --- |
| `01_MiniLM (не закончен).ipynb` | Sentence Transformers | Незавершённый прототип: контекст → словарное значение → семантический класс |
| `02_01_Word2Vec_Ruscorpora_Dictionary.ipynb` | Word2Vec, zero-shot | Контекстное окно → ближайшее словарное определение → теги по сходству |
| `02_02_Word2Vec_Ruscorpora_Supervised.ipynb` | Word2Vec, supervised | One-vs-Rest-классификация по признакам слова, контекста и POS |
| `02_03_Word2Vec_Ruscorpora_Hybrid.ipynb` | Word2Vec, hybrid | Сравнение словарного, supervised и объединённого представлений на одном разбиении |
| `03_01_RuBERT_ZeroShot.ipynb` | RuBERT, zero-shot | Пакетная разметка CoNLL-U с checkpoints и добавлением предсказаний в `MISC` |
| `03_02_RuBERT_Supervised.ipynb` | RuBERT, supervised | Многометочный fine-tuning, компенсация дисбаланса и жанровая оценка |

## Структура проекта

```text
phd_semantics/
├── scripts/                 # экспериментальные Jupyter-ноутбуки
├── data/                    # корпуса и кэш словаря; не хранится в Git
│   ├── conllu/
│   │   ├── bot/
│   │   ├── cobald/
│   │   ├── syntagrus/
│   │   └── handlabeled/
│   └── dictionary_dataset/  # локальный кэш словаря определений
├── models/                  # локальные модели и Hugging Face cache; не хранится в Git
├── outputs/                 # метрики, checkpoints и Trackio; не хранится в Git
├── requirements.txt
└── README.md
```

## Подготовка окружения

Клонируйте репозиторий и создайте виртуальное окружение:

```bash
git clone <repository-url>
cd phd_semantics
python -m venv .venv
```

Активируйте окружение.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Зарегистрируйте kernel и запустите Jupyter из корня репозитория:

```bash
python -m ipykernel install --user --name phd-semantics --display-name "Python (phd-semantics)"
jupyter lab
```

Запуск именно из корня важен: ноутбуки находят проект по каталогам `.git` и `scripts`.

## Подготовка данных и моделей

Каталоги `data/`, `models/` и `outputs/` перечислены в `.gitignore`, поэтому не появляются после клонирования. Создайте следующую структуру:

```text
data/conllu/
├── bot/*.conllu
├── cobald/*.conllu
├── syntagrus/*.conllu
└── handlabeled/
    ├── GramEval2020-test-fiction.conllu
    ├── GramEval2020-test-news.conllu
    ├── GramEval2020-test-poetry.conllu
    ├── GramEval2020-test-social.conllu
    └── GramEval2020-test-wiki.conllu
```

Семантические метки должны находиться в поле `MISC` токена CoNLL-U в виде `sem=...`. Точный набор классов строится из встречающихся в корпусах меток.

Для серии `02_*` положите модель НКРЯ `main_POS_V300_W5_C5` в каталог:

```text
models/main_POS_V300_W5_C5/
├── main_POS_V300_W5_C5.model
├── main_POS_V300_W5_C5.model.wv.vectors.npy
└── main_POS_V300_W5_C5.model.syn1neg.npy
```

Словарь `snagbreac/russian-reverse-dictionary-train-data` загружается через Hugging Face Datasets при первом запуске и затем сохраняется в `data/dictionary_dataset/`.

Для серии `03_*` модель `DeepPavlov/rubert-base-cased` загружается автоматически. Если локальная копия с `config.json` находится в `models/rubert-base-cased/`, используется она. Прототип `01_*` аналогично использует `paraphrase-multilingual-MiniLM-L12-v2`.

Перед публикацией или распространением данных и весов отдельно проверьте условия лицензий каждого исходного корпуса, словаря и модели: они намеренно не включены в репозиторий.

### Google Colab

В Colab сначала клонируйте репозиторий в `/content`, перейдите в его корень и только затем откройте ноутбук. Код подключает Google Drive и ожидает входные ресурсы в:

```text
/content/drive/MyDrive/Colab Notebooks/Аспирантура/data/
/content/drive/MyDrive/Colab Notebooks/Аспирантура/models/
```

Если каталога `data/` нет, поддерживается прежняя схема, в которой `conllu/` лежит непосредственно в папке `Аспирантура`. Результаты всё равно записываются в локальный каталог `outputs/` клонированного проекта; перед завершением Colab-сессии их нужно скачать или скопировать в постоянное хранилище.

## Пример задачи

Для предложения:

```text
Солдат держал в руках автомат.
```

пайплайн рассматривает слово `автомат` вместе с контекстом, выбирает подходящее словарное значение (оружие, а не устройство или телефон-автомат) и возвращает несколько подходящих классов, например:

```text
instr | tool | weapon
```