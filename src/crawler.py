from queue import Queue
import pandas as pd
from threading import Thread

from tools import get_category_data, parse_category, save_to_pd_dataframe, download_and_save_image


if __name__ == '__main__':
	# очередь для передачи данных между потоками
	queue = Queue()
	# очередь для загрузки изображений
	img_queue = Queue()
	# датафрейм для сохранения данных по продуктам
	df = pd.DataFrame()
	# датафрейм для данных по изображениям
	df_images = pd.DataFrame()
	# получим имена категорий
	categories_name = get_category_data(type='cat_names')
	# создаем поток для каждой категории
	threads = [Thread(target=parse_category, args=(queue, category_name)) for category_name in categories_name]
	# создаем поток для сохранения результатов парсинга в df
	threads.append(Thread(target=save_to_pd_dataframe, args=(queue, img_queue, df)))
	# создаем поток для загрузки изображений
	threads.append(Thread(target=download_and_save_image, args=(img_queue, df_images)))
	# стартуем потоки
	[t.start() for t in threads]
	# print("Threads started")
	# ждем завершения парсинга и сохранения результатов
	[t.join() for t in threads]
