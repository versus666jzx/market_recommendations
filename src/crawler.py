from queue import Queue
import pandas as pd
from threading import Thread

from tools import get_category_data, parse_category, save_to_pd_dataframe


if __name__ == '__main__':
	# формируем очередь для передачи данных между потоками
	queue = Queue()
	# датафрейм для сохранения данных
	df = pd.DataFrame()
	categories_name = get_category_data(type='cat_names')

	# создаем поток для каждой категории
	threads = [Thread(target=parse_category, args=(queue, category_name)) for category_name in categories_name]
	# создаем поток для сохранения результатов парсинга в df
	threads.append(Thread(target=save_to_pd_dataframe, args=(queue, df)))
	# стартуем потоки
	[t.start() for t in threads]
	# ждем завершения парсинга и сохранения результатов
	[t.join() for t in threads]
	exit(0)
