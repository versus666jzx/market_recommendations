import numpy as np
import requests

import streamlit as st

import tools

# st.set_page_config(layout="wide")

st.title('My first rec app')

add_selectbox = st.sidebar.selectbox(
    "What prediction are you want?",
    ("Get prediction for random product", "Get prediction by description")
)

if add_selectbox == "Get prediction for random product":
	get_prediction_for_random_product = st.button('Get prediction for random product')
	if get_prediction_for_random_product:
		all_products = tools.get_products_data()
		random_product = tools.get_random_product()
		product_index = random_product.name
		description_embeddings = tools.get_description_embeddings()
		faiss_description_index = tools.get_faiss_description_index()

		with st.container():
			imgs_url = tools.find_url(random_product['images'])
			image = requests.get(imgs_url[0]).content
			st.write('Your random product')
			st.image(image, use_column_width=True, caption=f"{random_product['dimension17']} {random_product['name']}, {random_product['price']} RUB")
			st.write(random_product['description'])

		st.title('Recommendations')
		col1, col2, col3, col4 = st.columns(4)
		distances, indexes = faiss_description_index.search(np.ascontiguousarray(description_embeddings.to_numpy().astype('float32')[product_index].reshape((1, -1))), 10)

		with col1:
			same = all_products.loc[indexes[0][1]]
			imgs_url = tools.find_url(same['images'])
			image = requests.get(imgs_url[0]).content
			st.image(image, caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB")

			same = all_products.loc[indexes[0][2]]
			imgs_url = tools.find_url(same['images'])
			image = requests.get(imgs_url[0]).content
			st.image(image, use_column_width=True, caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB")

		with col2:
			same = all_products.loc[indexes[0][3]]
			imgs_url = tools.find_url(same['images'])
			image = requests.get(imgs_url[0]).content
			st.image(image, use_column_width=True, caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB")

			same = all_products.loc[indexes[0][4]]
			imgs_url = tools.find_url(same['images'])
			image = requests.get(imgs_url[0]).content
			st.image(image, use_column_width=True, caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB")

		with col3:
			same = all_products.loc[indexes[0][5]]
			imgs_url = tools.find_url(same['images'])
			image = requests.get(imgs_url[0]).content
			st.image(image, use_column_width=True, caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB")

			same = all_products.loc[indexes[0][6]]
			imgs_url = tools.find_url(same['images'])
			image = requests.get(imgs_url[0]).content
			st.image(image, use_column_width=True, caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB")

		with col4:
			same = all_products.loc[indexes[0][7]]
			imgs_url = tools.find_url(same['images'])
			image = requests.get(imgs_url[0]).content
			st.image(image, use_column_width=True, caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB")

			same = all_products.loc[indexes[0][8]]
			imgs_url = tools.find_url(same['images'])
			image = requests.get(imgs_url[0]).content
			st.image(image, use_column_width=True, caption=f"{same['dimension17']} {same['name']}, {same['price']} RUB")














# img_url = 'https://goldapple.ru/media/catalog/product/4/0/4000498042748_2_hulopldzoheprlwm.jpg'
# image = requests.get(img_url).content
# st.image(image, caption='Это имэдж!')
