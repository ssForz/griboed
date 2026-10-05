## Установка и запуск

Создание окружения:

```powershell
poetry install
```

Подготовка данных:

```powershell
poetry run python src\data\prepare_dataset.py --overwrite
```

Обучение baseline:

```powershell
poetry run python src\training\train_baseline.py
```

Тесты    
```powershell
poetry run pytest tests -q
```

Линтер   
```powershell
poetry run ruff check .
```

Форматирование   
```powershell
poetry run ruff format .
```

Запуск FastAPI-сервиса локально:

```powershell
poetry run uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```
