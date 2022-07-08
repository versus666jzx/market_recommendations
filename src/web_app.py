import pandas as pd
import numpy as np

import faiss


import streamlit as st


def new():
	return st.write(pd.DataFrame(data={
		'Col_1': np.random.random(5),
		'Col_2': np.random.random(5)
	}))


st.title('My first st app!')
push_btn = st.button('Update', on_click=new)
