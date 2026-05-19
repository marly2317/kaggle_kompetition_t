## Текущее рабочее состояние

- активный эксперимент: `001_best_solution`
- локально (5-fold): `0.83501 ± 0.00858`
- OOF c подбором порога: `0.84287`
- порог: `0.58`
- Kaggle public: `0.82057`

## Быстрый handover-check (5 минут)

1) Установить зависимости:

```bash
python -m venv .venv
source .venv/Scripts/activate   
pip install -e ".[dev]"
```

2) Положить данные соревнования в `data/raw/`:

- `train.csv`
- `test.csv`
- `gender_submission.csv`

3) Проверить проект:

```bash
python -m src.main all 
```


Примечание по tuning: `timeout` ограничивает запуск новых trial, но уже начатый trial Optuna завершает до конца.

## Структура (минимум, который важно знать)

- `src/main.py` — CLI entrypoint (`fit`, `submit`, `all`, `tune`)
- `src/train_functions.py` — CV-обучение, OOF, подбор порога, сохранение артефактов
- `src/features.py` — feature engineering и fold-level preprocessing artifacts
- `src/inference.py` — инференс по fold-моделям и сборка submission
- `configs/project.yaml` — общие настройки проекта
- `configs/experiments/*.yaml` — эксперименты
- `docs/notes.md` — рабочий журнал экспериментов и решений

## Ноутбуки

- `notebooks/01_eda.ipynb` — EDA 
- `notebooks/03_training_analysis.ipynb` — пост-тренировочный анализ: learning curves, OOF metrics, permutation, SHAP, feature selection candidates
