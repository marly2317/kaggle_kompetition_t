
## Результаты

Рабочее решение — `001_best_solution` (CatBoost). Все построенные модели на одной 5-fold кросс-валидации:

| Эксперимент | Модель | CV accuracy | Kaggle public |
|-------------|--------|-------------|---------------|
| `000_baseline_logreg` | Логистическая регрессия — бейзлайн | 0.798 ± 0.014 | 0.778 |
| `001_best_solution` | **CatBoost — рабочее решение** | 0.835 ± 0.009 | **0.821** |
| `032_catboost_no_drift_features` | CatBoost без дрейфящих фич | 0.829 ± 0.006 | 0.775 |
| `033_lightgbm` | LightGBM — нетюненный | 0.852 ± 0.018 | 0.778 |
| `034_lightgbm_regularized` | LightGBM — регуляризованный | 0.845 ± 0.015 | 0.794 |
| `035_mlp` | MLP (PyTorch, entity embeddings) | 0.815 ± 0.037 | 0.785 |
| `036_mlp_tuned` | MLP — гиперпараметры подобраны Optuna | 0.836 ± 0.016 | 0.797 |
| `037_mlp_tuned_adamw` | MLP — AdamW, перетюнен Optuna | 0.829 ± 0.013 | 0.778 |
 
Журнал экспериментов и разбор — `docs/notes.md`.


1) Установить зависимости:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # PowerShell (Windows)
# source .venv/Scripts/activate   # bash / Git Bash
pip install -e .
```

2) Положить данные соревнования в `data/raw/`:

- `train.csv`
- `test.csv`
- `gender_submission.csv`

3) Проверить проект:

```bash
python -m src.main all 
```



## Структура 

- `src/main.py` — CLI entrypoint (`fit`, `submit`, `all`, `tune`)
- `src/train_functions.py` — CV-обучение, OOF, подбор порога, сохранение артефактов
- `src/features.py` — feature engineering и fold-level preprocessing artifacts
- `src/model.py` — фабрика моделей по `model.family` (catboost / logreg / lightgbm / mlp)
- `src/torch_mlp.py` — PyTorch MLP с entity embeddings (DL-модель)
- `src/inference.py` — инференс по fold-моделям и сборка submission
- `configs/project.yaml` — общие настройки проекта
- `configs/experiments/*.yaml` — эксперименты
- `docs/notes.md` — рабочий журнал экспериментов и решений

## Ноутбуки

- `notebooks/01_eda.ipynb` — EDA 
- `notebooks/03_training_analysis.ipynb` — пост-тренировочный анализ: learning curves, OOF metrics, permutation, SHAP, feature selection candidates
