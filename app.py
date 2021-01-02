import pandas as pd
import numpy as np
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import streamlit as st


def write_e_config(protons, e_config):
    el_name = element_keys[protons-1]
    charge = protons - sum(e_config)
    charge_str = ""
    if charge > 0:
        charge_str = "$^{"+f"+{charge}"+"}$"
    elif charge < 0:
        charge_str = "$^{"+f"{charge}"+"}$"
    if e_config[1] > 2:
        e2s = 2
        e2p = e_config[1] - 2
    else:
        e2s = e_config[1]
        e2p = 0
    
    e1s  = e_config[0]
    string = f"{el_name}{charge_str} : 1s$^{e1s}$ "
    if e2s > 0:
        string += f"2s$^{e2s}$ "
    if e2p > 0:
        string += f"2p$^{e2p}$ "
    return string

def Zeff(protons, n_lower, n_current):
    return protons - n_lower - (n_current - 1)/2

def n_in_shell(n):
    return 2 * n**2

def shell_r(n, Z_shell=1):
    return  (0.5*n + (n-1)**2 * 0.02)/Z_shell**0.5

def draw_electron(circles, shell, n_electron, Z_shell):
    n = n_in_shell(shell)
    interval = 2*np.pi/n
    half = np.arange(n/2)*interval
    other_half = half+np.pi
    angles = np.c_[half, other_half].flatten()
    theta = angles[n_electron]
    to_draw = np.array([np.cos(theta), np.sin(theta)])*shell_r(shell, Z_shell)
    circles.append(plt.Circle(to_draw, 0.04))
    ax.text(to_draw[0], to_draw[1], '-', fontsize=14, ha='center', va='center', color='1')



df = pd.read_csv("elements.csv")

elements = {"H": (1, [1,0]),
            "He": (2, [2,0]),
            "Li": (3, [2,1]),
            "Mg": (4, [2, 2]),
            "B": (5, [2, 3]),
            "C": (6, [2, 4]),
            "N": (7, [2, 5]),
            "O": (8, [2, 6]),
            "F": (9, [2, 7]),
            "Ne": (10, [2, 8])}



element_keys = list(elements.keys())
# element = st.selectbox('Element', options=list(elements.keys()))


cols = st.beta_columns(8)
el_fired = []
el_fired.append(cols[0].button('H'))
for col in cols[1:7]:
    col.markdown('<span style="color:white; font-size: 22px;">XX</span>', unsafe_allow_html=True)
el_fired.append(cols[7].button('He'))
for col, el in zip(cols, list(elements.keys())[2:]):
    el_fired.append(col.button(el))

charge = st.slider('charge', -2, 2, 0, 1)

for i, button in enumerate(el_fired):
    if button:
        el = element_keys[i]
        with open('el.json', 'w') as f:
            json.dump(dict(el=el), f)


with open('el.json', 'r') as f:
    el = json.load(f)['el']

protons = elements[el][0]
n1, n2 = elements[el][1]
if charge > 0:
    if n2 >= charge:
        n2 = n2 - charge
    else:
        n1 = n1 - (charge-n2)
        n2 = 0
elif charge < 0:
    elec = -charge
    if n1 + elec <= 2:
        n1 = n1 + elec
        elec = 0
    elif n1 + elec > 2:
        elec = elec - (2-n1)
        n1 = 2
    
    if n2 + elec <= 8:
        n2 = n2 + elec
    else:
        n3 = elec - (n2-8)
        n2 = 8

# protons = st.slider('protons', 0, 10, 1)
# n1 = st.slider('n=1', 0, 2, 1)
# n2 = st.slider('n=2', 0, 8, 1)


e_config = [n1, n2]

st.markdown(write_e_config(protons, e_config))


fig, ax = plt.subplots(figsize=(4,4))
lims = 1.4
ax.set_xlim(-lims,lims)
ax.set_ylim(-lims,lims)
ax.set_axis_off()
circles = []
circles.append(plt.Circle((0, 0), 0.1, color='g'))
# circles.append(plt.Circle((0, 0), 0.2, fc='none', linewidth=0.7, ec='0'))

n_electron = 0
shell = 1
n_lower = 0
for e_in_shell in e_config:
    Z_shell = Zeff(protons, n_lower, e_in_shell)
    n_electron = 0

    if e_in_shell > 0:
        circles.append(plt.Circle((0, 0), shell_r(shell, Z_shell), fc='none', linewidth=0.7, ec='0'))
    for electron in range(e_in_shell):
        draw_electron(circles, shell, electron, Z_shell)
    shell +=1
    n_lower += e_in_shell

for circle in circles:
    ax.add_artist(circle)

ax.text(0,0,'+'+str(protons), ha='center', va='center', color='1', fontsize=8)
st.write(fig)