from __future__ import annotations

import re
from typing import Tuple, Any, List

import numpy
import requests
from fake_useragent import UserAgent
from requests import get, exceptions
from lxml.etree import HTML
from queue import Queue
from time import sleep
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd

import torch
import transformers

import faiss

import streamlit as st

def get_category_data(type: str, cat_name: str = None) -> int | list:
	"""
	Returns the requested data by category

	:param type: Type returned data. Should be one of: id, ru_name or cat_names
	:param cat_name: category name
	:return: returns the requested data
	"""

	cat = {
		'makijazh':              {'id': 3, 'name': 'макияж'},
		'uhod':                  {'id': 4, 'name': 'уход'},
		'volosy':                {'id': 6, 'name': 'волосы'},
		'parfjumerija':          {'id': 7, 'name': 'парфюмерия'},
		'zdorov-e-i-apteka':     {'id': 3747, 'name': 'здоровье и аптека'},
		'sexual-wellness':       {'id': 5962, 'name': 'sexual wellness'},
		'azija':                 {'id': 10, 'name': 'азия'},
		'organika':              {'id': 12, 'name': 'органика'},
		'dlja-muzhchin':         {'id': 3887, 'name': 'для мужчин'},
		'dlja-detej':            {'id': 4357, 'name': 'для детей'},
		'tehnika':               {'id': 3870, 'name': 'техника'},
		'dlja-doma':             {'id': 8202, 'name': 'для дома'},
		'odezhda-i-aksessuary':  {'id': 8529, 'name': 'одежда и аксессуары'},
		'nizhnee-bel-jo':        {'id': 8563, 'name': 'нижнее бельё'},
		'ukrashenija':           {'id': 5746, 'name': 'украшения'},
		'lajfstajl':             {'id': 8579, 'name': 'лайфстайл'},
		'ini-formaty':           {'id': 5159, 'name': 'тревел-форматы'},
		'tovary-dlja-zhivotnyh': {'id': 7638, 'name': 'товары для животных'}
	}

	if type not in ['cat_names', 'id', 'ru_name']:
		raise ValueError('Parameter type should by "cat_names","id" or "ru_name"')

	if type == 'cat_names':
		return [cat for cat in cat.keys()]

	if cat_name is None:
		raise ValueError('Value cat_name should be not None')

	try:
		cat[cat_name]
	except KeyError:
		raise ValueError(f'Wrong category name: {cat_name}. \n'
		                 f'Possible cat_names: {", ".join([name for name in cat.keys()])}')

	if type == 'id':
		return cat[cat_name]['id']
	elif type == 'ru_name':
		return cat[cat_name]['name']
	else:
		raise ValueError('type should be "cat_names", "id" or "ru_name"')


def get_product_data_by_url(url: str) -> tuple[Any, Any, Any]:
	"""
    Get additional data by product URL.
    Additional data uncludes:
        - description
        - product usage
        - product composition

    """
	user_agent = UserAgent().random
	try:
		res = get(url, timeout=5, headers={'User-Agent': user_agent}).text
		res = HTML(res)
	except:
		return None, None, None

	try:
		description: str or None = ' '.join(res.xpath('/html/body/div[1]/main/div/div/section/section[3]/section[2]/section/section[1]/div/ul/li[1]/article/div/section[1]')[0].text.replace('\n\n', '').split()) \
        if len(res.xpath('/html/body/div[1]/main/div/div/section/section[3]/section[2]/section/section[1]/div/ul/li[1]/article/div/section[1]')) > 0 \
           and res.xpath('/html/body/div[1]/main/div/div/section/section[3]/section[2]/section/section[1]/div/ul/li[1]/article/div/section[1]')[0].text is not None\
        else None
	except:
		description = None

	try:
		product_usage: str or None = ' '.join(res.xpath('/html/body/div[1]/main/div/div/section/section[3]/section[2]/section/section[1]/div/ul/li[2]/article/div/section')[0].text.replace('\n\n', '').split()) \
        if len(res.xpath('/html/body/div[1]/main/div/div/section/section[3]/section[2]/section/section[1]/div/ul/li[2]/article/div/section')) > 0 \
        else None
	except:
		product_usage = None

	try:
		product_composition: str or None = ' '.join(res.xpath('/html/body/div[1]/main/div/div/section/section[3]/section[2]/section/section[1]/div/ul/li[3]/article/div/section')[0].text.replace('\n\n', '').split()) \
        if len(res.xpath('/html/body/div[1]/main/div/div/section/section[3]/section[2]/section/section[1]/div/ul/li[3]/article/div/section')) > 0 \
        else None
	except:
		product_composition = None

	return description, product_usage, product_composition


def parse_category(queue: Queue, cat_name: str):
	category_id = get_category_data(cat_name=cat_name, type='id')
	category_ru_name = get_category_data(cat_name=cat_name, type='ru_name')
	url = 'https://goldapple.ru/web_scripts/discover/category/products/'

	fields = [
        'id',
        'sku',
        'name',
        'brand',
        'brand_type',
        'dimension17',
        'dimension18',
        'dimension19',
        'dimension20',
        'country',
        'price',
        'currency',
        'old_price',
        'category_type',
        'url',
        'images',
        'type',
        'volume',
        'main_product_sku',
        'main_product_id',
        'best_loyalty_price',
        'dimension29',
        'dimension28',
        'description',
        'product_usage',
        'product_composition',
        'category',
        'category_ru'
        ]

	for page in range(1, 10000):
		try:
			# ждем рандомный момент времени
			sleep(np.round(np.random.uniform(10, 20), 1))
			# генерим рандомный юезерагент
			user_agent = UserAgent().random
			# пытаемся получить данные по странице с товарами
			res = get(url, params={'cat': category_id, 'page': page}, timeout=5, headers={'User-Agent': user_agent}).json()['products']
		except:
			break
		# в случае количества страниц, кратным 20 можем получить страницу с 0 товарами
		if len(res) == 0:
			break

		# если добрались до последней страницы, собираем с нее остатки и возвращаем результат
		elif len(res) < 20:
			for product in res:
				# фильтруем только необходимые поля, если поля нет, то None
				product_new = {your_key: product[your_key] if your_key in product.keys() else None  for your_key in fields}
				# получаем данные из html странички, которые отсутствуют в запросе (исключив это можно значительно сократить время парсинга)
				description, product_usage, product_composition = get_product_data_by_url(product['url'])
				product_new['description'] = description
				product_new['product_usage'] = product_usage
				product_new['product_composition'] = product_composition
				product_new['category'] = cat_name
				product_new['category_ru'] = category_ru_name
				# кладем объект в очередь
				queue.put(product_new)
				queue.task_done()
			break
		else:
			for product in res:
				# фильтруем только необходимые поля, если поля нет, то None
				product_new = {your_key: product[your_key] if your_key in product.keys() else None  for your_key in fields}
				# получаем данные из html странички, которые отсутствуют в запросе (исключив это можно значительно сократить время парсинга)
				description, product_usage, product_composition = get_product_data_by_url(product['url'])
				product_new['description'] = description
				product_new['product_usage'] = product_usage
				product_new['product_composition'] = product_composition
				product_new['category'] = cat_name
				product_new['category_ru'] = category_ru_name
				# кладем объект в очередь
				queue.put(product_new)
				queue.task_done()


def save_to_pd_dataframe(queue: Queue, df: pd.DataFrame):
	while True:
		try:
			# sleep + timeout в requests не должны превышать данный timeout иначе парсинг может закончится раньше времени
			product_data = queue.get(timeout=30)
			df = pd.concat([
				df,
                pd.DataFrame([product_data])
            ])
			print(f'Count crowled data: {len(df)}', end='\r')
		except:
			# если за 30 секунд в очереди не появилось данных, полагаем что парсинг завершен
			df.to_csv('../data/products.csv', index=False)
			break


def get_sitemats_list(url: str='https://goldapple.ru/sitemap.xml') -> list[str]:
	"""
    Get sitemaps list from goldapple or another url.
    :param url: url to sitemap.xml
    :return: list of sitemaps urls
    """
	try:
		res = get(url, timeout=30).text
	except exceptions.ConnectTimeout as err:
		raise exceptions.ConnectTimeout(f'connection timeout, {err}')
	except exceptions.ConnectionError:
		raise exceptions.ConnectionError(f'check url: {url}')
	except exceptions.HTTPError as err:
		raise exceptions.HTTPError(err)

	try:
		soup = BeautifulSoup(res, 'xml')
		res = [x.text for x in soup.find_all('loc')]
	except AttributeError as err:
		raise AttributeError(f'could not find attribute "loc", {err}')
	except Exception as err:
		raise f'Error parsing xml data: {err}'

	return res


def get_product_urls(sitemaps: list[str]) -> Tuple[list[str], list[str]]:
	"""
    Get products urls from sitemaps.

    :param sitemaps: list of sitemaps urls
    :return: list of product links
    """
	cat_urls = []
	prod_urls = []
	pattern = re.compile(r'\d{10,12}')
	for sitemap in sitemaps:
		xml_products = get(sitemap).text
		soup_products = BeautifulSoup(xml_products, 'xml')
		product_urls = [x.text for x in soup_products.find_all('loc')]
		for index, x in enumerate(product_urls):
			product = re.findall(pattern, x)
			if product:
				prod_urls.append(product_urls[index])
			else:
				cat_urls.append(product_urls[index])

	return cat_urls, prod_urls


def get_sku_and_product_id_from_url(url: str) -> tuple[Any, Any]:
	"""
    Return sku_id and product_id from product URL
    :param url: url to product
    :return: [sku_id, prod_id]
    """
	pattern = re.compile(r'\d{4,20}')
	res_sku = re.findall(pattern, url)
	if len(res_sku) == 1:
		prod_id = None
		sku_id = res_sku[0]
	elif len(res_sku) > 1:
		prod_id = res_sku[0]
		sku_id = res_sku[1]
	else:
		prod_id = sku_id = None

	return sku_id, prod_id


@st.experimental_memo
def _get_bert_model() -> transformers.BertModel:
	config = transformers.BertConfig.from_json_file(
		'model/bert_config.json')
	model = transformers.BertModel.from_pretrained(
		'model/pytorch_model.bin', config=config)
	return model


@st.experimental_memo
def _get_tokenizer() -> transformers.BertTokenizer:
	return transformers.BertTokenizer('model/vocab.txt')


@st.experimental_memo
def get_faiss_description_index() -> faiss.IndexFlatL2:
	return faiss.read_index('data/faiss_description_index.index')


@st.experimental_memo
def get_products_data() -> pd.DataFrame:
	return pd.read_csv('data/products.csv')


@st.experimental_memo
def get_description_embeddings() -> pd.DataFrame:
	return pd.read_csv('data/embedded_description')


def find_url(string: str) -> list[str]:
	regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
	url = re.findall(regex, string)
	res = [x[0] for x in url]
	if len(res) > 0:
		return res
	else:
		# если нет изображений у товара, возвращаем картинку No image available
		return ['https://ih0.redbubble.net/image.343726250.4611/flat,800x800,075,f.jpg']


def get_text_embedding(text: str) -> numpy.ndarray:
	try:
		tokenizer = _get_tokenizer()
		model = _get_bert_model()
	except Exception as err:
		raise Exception(f'Failed to initialize model: {err}')

	vector = torch.LongTensor(
		np.array(
			tokenizer.encode(
				text, add_special_tokens=True, max_length=512, padding=False, truncation=True
			)
		)
	).resize((-1, 1))

	attention_mask = torch.LongTensor(
		np.ones(
			shape=(1, len(vector))
		).astype('int'))

	with torch.no_grad():
		embedings = model(vector, attention_mask=attention_mask)

	text_embedding = embedings[0][:, 0, :].numpy()
	return text_embedding


def return_n_neighbors_by_description(text: str, n_neighbors: int):
	if text is None or text.strip() == '':
		return 'No text input'
	text_embedding = get_text_embedding(text)
	index = get_faiss_description_index()
	products = get_products_data()
	description_embeddings = get_description_embeddings()
	distances, indexes = index.search(
		np.ascontiguousarray(
			text_embedding
			.astype('float32')
			.reshape((1, -1))
		), n_neighbors)

	return products[indexes[0]]


def get_random_product() -> pd.Series:
	data = get_products_data()
	return data.loc[np.random.randint(len(data))]


def get_product_image(url: str) -> np.ndarray:
	try:
		requests.get(url, timeout=3).content