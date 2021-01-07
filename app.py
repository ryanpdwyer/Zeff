from periodic_table import periodic_table
from iso_electronic import iso_electronic
import streamlit as st

st.set_page_config(layout="wide")

iso_electronic_ions = st.checkbox("View isoelectronic ions")

if iso_electronic_ions:
    iso_electronic()
else:
    periodic_table()