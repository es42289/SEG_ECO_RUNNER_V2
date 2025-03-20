import random
from streamlit_d3_demo import d3_line
import streamlit as st

def generate_random_data(x_r, y_r):
    return list(zip(range(x_r), [random.randint(0, y_r) for _ in range(x_r)]))
midx = st.slider('middle x', 5, 8)
d3_line([
        (0, 9),
        (1, 6),
        (2, 1),
        (3, 10),
        (4, 9),
        (midx, 0),
        (9, 7)
], circle_radius=15, circle_color="#6495ed")