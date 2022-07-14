# Quick start

## Ручной запуск парсинга

Для запуска парсинга необходимо запустить скрипт [краулера](src/crawler.py) из проекта.

```shell
# shell
python src/crawler.py

# python3
>>> crawler.py
```

Ход выполнения парсинга будет отображаться в консоли

## Запуск WEB-интерфейса

1. Подготовка conda окружения

```shell
conda env create -f environment.yml -p /path/to/your/envs/env_name
```

3. Запуск

```shell
streamlit run market_recommendations/src/web_app.py
```

## Docker

В планах
