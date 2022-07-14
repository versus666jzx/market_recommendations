import numpy as np

import streamlit as st

import tools

# st.set_page_config(layout="wide")

# получаем датафрейм со всеми данными
all_products = tools.get_products_data()
# получаем необходимые эмбеддинги
description_embeddings = tools.get_description_embeddings()
product_usage_embeddings = tools.get_product_usage_embeddings()
product_composition_embeddings = tools.get_product_composition_embeddings()
# получаем необходимые faiss индексы
faiss_description_index = tools.get_faiss_description_index()
faiss_product_usage_index = tools.get_faiss_product_usage_index()
faiss_product_composition_index = tools.get_faiss_product_composition_index()

st.title('My first rec app')

selectbox = st.sidebar.selectbox(
	"What prediction are you want?",
	("Get prediction for random product", "Selected product")
)

if selectbox == "Get prediction for random product":
	get_prediction_for_random_product = st.button('Get prediction for random product')
	if get_prediction_for_random_product:
		random_product = tools.get_random_product()
		product_index = random_product.name
		product_sku = random_product['sku']
		# формируем контейнер с выбранным рандомно продуктом
		with st.container():
			image = tools.get_image_by_sku(product_sku)
			st.image(
				image,
				caption=f"{random_product['dimension17']} {random_product['name']}, {random_product['price']} RUB"
			)
			st.write(f"Описание: {random_product['description']}")
			st.write(f"Применение: {random_product['product_usage']}")
			st.write(f"Состав: {random_product['product_composition']}")
		# блок рекомендаций по описанию
		st.title('Recommendations by description')
		# получаем ближайшие эмбединги по описанию
		distances, indexes = faiss_description_index.search(
			np.ascontiguousarray(
				description_embeddings
				.to_numpy()
				.astype('float32')[product_index]
				.reshape((1, -1))
			),
			10  # выбираем 10 ближайших
		)
		# создаем 4 колонки по 2 товара из рекомендаций
		# в первую колонку попадают ближайшие эмбеддинги с нечетными индексами
		# во вторую колонку попадают ближайшие эмбеддинги с четными индексами
		for col, index_1, index_2 in zip(st.columns(4), range(1, 8, 2), range(2, 9, 2)):
			# первая колонка с рекомендацией
			same = all_products.loc[indexes[0][index_1]]
			image = tools.get_image_by_sku(same['sku'])
			col.image(
				image,
				caption=f"{same['dimension17']} {same['name']} - {same['price']} RUB"
			)
			# вторая колонка с рекомендацией
			same = all_products.loc[indexes[0][index_2]]
			image = tools.get_image_by_sku(same['sku'])
			col.image(
				image,
				caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB"
			)

		# блок рекомендаций по описанию применения продукта
		st.title('Recommendations by product usage')
		# получаем ближайшие эмбеддинги по описанию применения
		distances, indexes = faiss_product_usage_index.search(
			np.ascontiguousarray(
				product_usage_embeddings
				.to_numpy()
				.astype('float32')[product_index]
				.reshape((1, -1))
			),
			10  # выбираем 10 ближайших
		)

		for col, index_1, index_2 in zip(st.columns(4), range(1, 8, 2), range(2, 9, 2)):
			# первая колонка с рекомендацией
			same = all_products.loc[indexes[0][index_1]]
			image = tools.get_image_by_sku(same['sku'])
			col.image(
				image,
				caption=f"{same['dimension17']} {same['name']} - {same['price']} RUB"
			)
			# вторая колонка с рекомендацией
			same = all_products.loc[indexes[0][index_2]]
			image = tools.get_image_by_sku(same['sku'])
			col.image(
				image,
				caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB"
			)

		# блок рекомендаций по составку продукта
		st.title('Recommendations by product composition')
		# получаем ближайшие эмбеддинги по составу продукта
		distances, indexes = faiss_product_composition_index.search(
			np.ascontiguousarray(
				product_composition_embeddings
				.to_numpy()
				.astype('float32')[product_index]
				.reshape((1, -1))
			),
			10  # выбираем 10 ближайших
		)

		for col, index_1, index_2 in zip(st.columns(4), range(1, 8, 2), range(2, 9, 2)):
			# первая колонка с рекомендацией
			same = all_products.loc[indexes[0][index_1]]
			image = tools.get_image_by_sku(same['sku'])
			col.image(
				image,
				caption=f"{same['dimension17']} {same['name']} - {same['price']} RUB"
			)
			# вторая колонка с рекомендацией
			same = all_products.loc[indexes[0][index_2]]
			image = tools.get_image_by_sku(same['sku'])
			col.image(
				image,
				caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB"
			)


if selectbox == 'Selected product':
	# получаем список категорий с русским названием
	categories: list = st.multiselect('Выберете категорию продукта', options=tools.get_category_options())
	# конвертируем список категорий в вид, в котором они содержатся в датафрейме
	eng_categories = [tools.get_category_data('ru_to_eng', cat) for cat in categories]
	# если выбрана хотя бы одна категория
	if len(categories) > 0:
		# получаем все бренды из данной категории
		brands = st.multiselect(
			'Выберете бренд',
			options=list(all_products[all_products['category'].isin(eng_categories)]['brand'].value_counts().index)
		)
		# если выбран хотя бы один бренд
		if len(brands) > 0:
			# создаем кнопку выбора нового продукта
			change_product = st.button('Change product')
			# отфильтровываем продукты по выбранным пользователем критериям
			selected_products = all_products[all_products['category'].isin(eng_categories) & all_products['brand'].isin(brands)]
			# получаем рандомный продукт
			product = selected_products.loc[np.random.choice(selected_products.index)]
			# если юзер жмакнул кнопку сменить продукт
			if change_product:
				# выбираем новый продукт
				product = selected_products.loc[np.random.choice(selected_products.index)]

			# блок карточки с выбранным продуктом
			with st.container():
				image = tools.get_image_by_sku(product['sku'])
				st.image(
					image,
					caption=f"{product['dimension17']} {product['name']}, {product['price']} RUB"
				)
				st.write(f"Описание: {product['description']}")
				st.write(f"Применение: {product['product_usage']}")
				st.write(f"Состав: {product['product_composition']}")

			# блок с рекомендациями по описанию
			st.title('Recommendations by description')
			# берем ближайшие эмбеддинги по описанию
			distances, indexes = faiss_description_index.search(
				np.ascontiguousarray(
					description_embeddings
					.to_numpy()
					.astype('float32')[product.name]
					.reshape((1, -1))
				),
				10  # выбираем 10 ближайших
			)

			for col, index_1, index_2 in zip(st.columns(4), range(1, 8, 2), range(2, 9, 2)):
				# первая колонка с рекомендацией
				same = all_products.loc[indexes[0][index_1]]
				image = tools.get_image_by_sku(same['sku'])
				col.image(
					image,
					caption=f"{same['dimension17']} {same['name']} - {same['price']} RUB"
				)
				# вторая колонка с рекомендацией
				same = all_products.loc[indexes[0][index_2]]
				image = tools.get_image_by_sku(same['sku'])
				col.image(
					image,
					caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB"
				)

			# блокрекомендаций по описанию применения продукта
			st.title('Recommendations by product usage')

			distances, indexes = faiss_product_usage_index.search(
				np.ascontiguousarray(
					product_usage_embeddings
					.to_numpy()
					.astype('float32')[product.name]
					.reshape((1, -1))
				),
				10  # выбираем 10 ближайших
			)

			for col, index_1, index_2 in zip(st.columns(4), range(1, 8, 2), range(2, 9, 2)):
				# первая колонка с рекомендацией
				same = all_products.loc[indexes[0][index_1]]
				image = tools.get_image_by_sku(same['sku'])
				col.image(
					image,
					caption=f"{same['dimension17']} {same['name']} - {same['price']} RUB"
				)
				# вторая колонка с рекомендацией
				same = all_products.loc[indexes[0][index_2]]
				image = tools.get_image_by_sku(same['sku'])
				col.image(
					image,
					caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB"
				)

			# блок рекомендаций по составу продукта
			st.title('Recommendations by product composition')

			distances, indexes = faiss_product_composition_index.search(
				np.ascontiguousarray(
					product_composition_embeddings
					.to_numpy()
					.astype('float32')[product.name]
					.reshape((1, -1))
				),
				10  # выбираем 10 ближайших
			)

			for col, index_1, index_2 in zip(st.columns(4), range(1, 8, 2), range(2, 9, 2)):
				# первая колонка с рекомендацией
				same = all_products.loc[indexes[0][index_1]]
				image = tools.get_image_by_sku(same['sku'])
				col.image(
					image,
					caption=f"{same['dimension17']} {same['name']} - {same['price']} RUB"
				)
				# вторая колонка с рекомендацией
				same = all_products.loc[indexes[0][index_2]]
				image = tools.get_image_by_sku(same['sku'])
				col.image(
					image,
					caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB"
				)
