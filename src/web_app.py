import numpy as np
import requests

import streamlit as st

import tools

# st.set_page_config(layout="wide")

st.title('My first rec app')

selectbox = st.sidebar.selectbox(
	"What prediction are you want?",
	("Get prediction for random product", "Get prediction selected product")
)

if selectbox == "Get prediction for random product":
	get_prediction_for_random_product = st.button('Get prediction for random product')
	if get_prediction_for_random_product:
		all_products = tools.get_products_data()
		random_product = tools.get_random_product()
		product_index = random_product.name
		product_sku = random_product['sku']
		description_embeddings = tools.get_description_embeddings()
		product_usage_embeddings = tools.get_product_usage_embeddings()
		product_composition_embeddings = tools.get_product_composition_embeddings()
		faiss_description_index = tools.get_faiss_description_index()
		faiss_product_usage_index = tools.get_faiss_product_usage_index()
		faiss_product_composition_index = tools.get_faiss_product_composition_index()

		with st.container():
			image = tools.get_image_by_sku(product_sku)
			st.write('Your random product')
			st.image(image, use_column_width=True, caption=f"{random_product['dimension17']} {random_product['name']}, {random_product['price']} RUB")
			st.write(f"Описание: {random_product['description']}")
			st.write(f"Применение: {random_product['product_usage']}")
			st.write(f"Состав: {random_product['product_composition']}")

		st.title('Recommendations by description')
		col1, col2, col3, col4 = st.columns(4)
		distances, indexes = faiss_description_index.search(np.ascontiguousarray(description_embeddings.to_numpy().astype('float32')[product_index].reshape((1, -1))), 10)

		with col1:
			same = all_products.loc[indexes[0][1]]
			image = tools.get_image_by_sku(same['sku'])
			st.image(image, caption=f"{same['dimension17']} {same['name']} - {same['price']} RUB")

			same = all_products.loc[indexes[0][2]]
			image = tools.get_image_by_sku(same['sku'])
			st.image(image, use_column_width=True, caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB")

		with col2:
			same = all_products.loc[indexes[0][3]]
			image = tools.get_image_by_sku(same['sku'])
			st.image(image, use_column_width=True, caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB")

			same = all_products.loc[indexes[0][4]]
			image = tools.get_image_by_sku(same['sku'])
			st.image(image, use_column_width=True, caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB")

		with col3:
			same = all_products.loc[indexes[0][5]]
			image = tools.get_image_by_sku(same['sku'])
			st.image(image, use_column_width=True, caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB")

			same = all_products.loc[indexes[0][6]]
			image = tools.get_image_by_sku(same['sku'])
			st.image(image, use_column_width=True, caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB")

		with col4:
			same = all_products.loc[indexes[0][7]]
			image = tools.get_image_by_sku(same['sku'])
			st.image(image, use_column_width=True, caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB")

			same = all_products.loc[indexes[0][8]]
			image = tools.get_image_by_sku(same['sku'])
			st.image(image, use_column_width=True, caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB")

		st.title('Recommendations by product usage')
		col1, col2, col3, col4 = st.columns(4)
		distances, indexes = faiss_product_usage_index.search(np.ascontiguousarray(product_usage_embeddings.to_numpy().astype('float32')[product_index].reshape((1, -1))), 10)

		with col1:
			same = all_products.loc[indexes[0][1]]
			image = tools.get_image_by_sku(same['sku'])
			st.image(image, caption=f"{same['dimension17']} {same['name']} - {same['price']} RUB")

			same = all_products.loc[indexes[0][2]]
			image = tools.get_image_by_sku(same['sku'])
			st.image(image, use_column_width=True, caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB")

		with col2:
			same = all_products.loc[indexes[0][3]]
			image = tools.get_image_by_sku(same['sku'])
			st.image(image, use_column_width=True, caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB")

			same = all_products.loc[indexes[0][4]]
			image = tools.get_image_by_sku(same['sku'])
			st.image(image, use_column_width=True, caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB")

		with col3:
			same = all_products.loc[indexes[0][5]]
			image = tools.get_image_by_sku(same['sku'])
			st.image(image, use_column_width=True, caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB")

			same = all_products.loc[indexes[0][6]]
			image = tools.get_image_by_sku(same['sku'])
			st.image(image, use_column_width=True, caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB")

		with col4:
			same = all_products.loc[indexes[0][7]]
			image = tools.get_image_by_sku(same['sku'])
			st.image(image, use_column_width=True, caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB")

			same = all_products.loc[indexes[0][8]]
			image = tools.get_image_by_sku(same['sku'])
			st.image(image, use_column_width=True, caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB")

		st.title('Recommendations by product composition')
		col1, col2, col3, col4 = st.columns(4)
		distances, indexes = faiss_product_composition_index.search(np.ascontiguousarray(product_composition_embeddings.to_numpy().astype('float32')[product_index].reshape((1, -1))), 10)

		with col1:
			same = all_products.loc[indexes[0][1]]
			image = tools.get_image_by_sku(same['sku'])
			st.image(image, caption=f"{same['dimension17']} {same['name']} - {same['price']} RUB")

			same = all_products.loc[indexes[0][2]]
			image = tools.get_image_by_sku(same['sku'])
			st.image(image, use_column_width=True, caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB")

		with col2:
			same = all_products.loc[indexes[0][3]]
			image = tools.get_image_by_sku(same['sku'])
			st.image(image, use_column_width=True, caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB")

			same = all_products.loc[indexes[0][4]]
			image = tools.get_image_by_sku(same['sku'])
			st.image(image, use_column_width=True, caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB")

		with col3:
			same = all_products.loc[indexes[0][5]]
			image = tools.get_image_by_sku(same['sku'])
			st.image(image, use_column_width=True, caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB")

			same = all_products.loc[indexes[0][6]]
			image = tools.get_image_by_sku(same['sku'])
			st.image(image, use_column_width=True, caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB")

		with col4:
			same = all_products.loc[indexes[0][7]]
			image = tools.get_image_by_sku(same['sku'])
			st.image(image, use_column_width=True, caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB")

			same = all_products.loc[indexes[0][8]]
			image = tools.get_image_by_sku(same['sku'])
			st.image(image, use_column_width=True, caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB")

if selectbox == 'Get prediction selected product':
	pass











# img_url = 'https://goldapple.ru/media/catalog/product/4/0/4000498042748_2_hulopldzoheprlwm.jpg'
# image = requests.get(img_url).content
# st.image(image, caption='Это имэдж!')
