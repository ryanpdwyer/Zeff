import pandas as pd
import numpy as np
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import streamlit as st
from collections import defaultdict
from string import Template
# st.set_page_config(layout="wide")

H = dict(Z=1,e1s=1)
df = pd.read_csv("elements.csv")
element_symbols = dict(df.loc[:, ['AtomicNumber', 'Symbol']].values)
symbol_Z = dict(df.loc[:, ['Symbol', 'AtomicNumber']].values)

st.title("Isoelectronic ions")
st.markdown("<br/>", unsafe_allow_html=True)
def max_electrons(subshell):
    max_e = dict(s=2, p=6, d=10, f=14)
    return max_e[subshell[1]]

def get_electrons(e_config):
    return {key[1:]: val for key, val in H.items() if key != 'Z'}

def get_e_in_shell(e_config, n):
    return sum(val for key, val in e_config.items() if n in key)

shells_filling = ['1s', '2s', '2p', '3s', '3p', '4s', '3d', '4p', '5s', '4d', '5p', '6s']


def subshell_sort_func(subshell):
    ell = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
    return int(subshell[0])*10 + ell[subshell[1]]

def get_e_config(protons, charge):
    if charge > 0:
        electrons_left = protons
    else:
        electrons_left = protons - charge
    i_shell = 0
    ec = defaultdict(lambda : 0)
    while electrons_left > 0:
        subshell = shells_filling[i_shell]
        if ec[subshell] < max_electrons(subshell):
            ec[subshell] += 1
            electrons_left -= 1
        else:
            i_shell += 1
    
    if charge > 0: # We need to remove electrons from orbitals with highest n first
        dkeys = list(ec.keys())
        dkeys.sort(key=subshell_sort_func, reverse=True)
        charge_left = charge
        i = 0
        while charge_left > 0:
            if ec[dkeys[i]] > 0:
                ec[dkeys[i]] -= 1
                charge_left -= 1
            else:
                i += 1

    return ec


def write_ion(protons, e_config):
    el_name = element_symbols[protons]
    charge = protons - sum(list(e_config.values()))
    
    charge_str = ""
    if charge > 1:
        charge_str = "$^{"+f"{charge}+"+"}$"
    elif charge < -1:
        charge_str = "$^{"+f"{-charge}-"+"}$"
    elif charge == 1:
        charge_str = "$^{+}$"
    elif charge == -1:
        charge_str = "$^{-}$"

    return f"{el_name}{charge_str}"

def write_e_config(protons, e_config):
    el_name = element_symbols[protons]
    charge = protons - sum(list(e_config.values()))
    
    charge_str = ""
    if charge > 1:
        charge_str = "$^{"+f"{charge}+"+"}$"
    elif charge < -1:
        charge_str = "$^{"+f"{-charge}-"+"}$"
    elif charge == 1:
        charge_str = "$^{+}$"
    elif charge == -1:
        charge_str = "$^{-}$"

    el_string = f"{el_name}{charge_str} : "
    
    t = Template("$subshell$$^{$n}$$")
    e_config_list = [t.substitute(subshell=subshell, n=n) for subshell, n in e_config.items()]

    return el_string+" ".join(e_config_list)

def Zeff(protons, n_lower, n_current):
    return protons - n_lower*0.9 - (n_current - 1)*0.4

def max_n_in_shell(n):
    return 2 * n**2

def shell_r(n, Z_shell=1):
    return  (0.5*n + (n-1)**2 * 0.02)/Z_shell**0.5

def n_in_shell(e_config):
    """e_config is a dictionary with keys like 1s, 2s, etc"""
    out = defaultdict(lambda : 0)
    for key, val in e_config.items():
        out[int(key[0])] += val
    
    return out

def draw_electron(circles, shell, n_electron, Z_shell, elec):
    n = max_n_in_shell(shell)
    interval = 2*np.pi/n
    half = np.arange(n/2)*interval
    other_half = half+np.pi
    angles = np.c_[half, other_half].flatten()
    theta = angles[n_electron]
    to_draw = np.array([np.cos(theta), np.sin(theta)])*shell_r(shell, Z_shell)
    colors = dict(s='#1f77b4', p='#ff7f0e', d='#2ca02c', f='#d62728')
    circles.append(plt.Circle(to_draw, 0.04, color=colors[elec]))
    ax.text(to_draw[0], to_draw[1], '-', fontsize=14, ha='center', va='center', color='1')



charges = dict(H=-1, He=0, Li=1, Be=2, N=-3, O=-2, F=-1, Ne=0, Na=1, Mg=2,
                                       P=-3, S=-2, Cl=-1, Ar=0, K=1, Ca=2)
symbols = list(symbol_Z.keys())
cols = st.beta_columns(8)

def make_periodic_table():
    el_fired = []

    for col, label in zip(cols, [1, 2, 13, 14, 15, 16, 17, 18]):
        col.markdown(label)


    el_fired.append(cols[0].button('H-'))
    for col in cols[1:-1]:
        col.markdown('<span style="color:white; font-size: 22px;">XX</span>', unsafe_allow_html=True)

    el_fired.append(cols[-1].button('He'))
    el_fired.append(cols[0].button('Li+'))
    el_fired.append(cols[1].button('Be2+'))

    el_fired.append(cols[-4].button('N3-'))
    el_fired.append(cols[-3].button('O2-'))
    el_fired.append(cols[-2].button('F-'))
    el_fired.append(cols[-1].button('Ne'))
    el_fired.append(cols[0].button('Na+'))
    el_fired.append(cols[1].button('Mg2+'))

    el_fired.append(cols[-4].button('P3-'))
    el_fired.append(cols[-3].button('S2-'))
    el_fired.append(cols[-2].button('Cl-'))
    el_fired.append(cols[-1].button('Ar'))
    el_fired.append(cols[0].button('K+'))
    el_fired.append(cols[1].button('Ca2+'))

    return el_fired

el_fired = make_periodic_table()

for i, button in enumerate(el_fired):
    if button:
        element = list(charges.keys())[i]
        with open('el-2.json', 'w') as f:
            json.dump(dict(el=element), f)

with open('el-2.json', 'r') as f:
    element = json.load(f)['el']
    charge = charges[element]


protons = symbol_Z[element]
e_config = get_e_config(protons, charge)
shells = n_in_shell(e_config)


show_e_config = st.checkbox('Show e⁻ configuration')
if show_e_config:
    st.markdown(write_e_config(protons, e_config))
else:
    st.markdown(write_ion(protons, e_config))


fig, ax = plt.subplots(figsize=(4,4))
lims = 1.4
ax.set_xlim(-lims,lims)
ax.set_ylim(-lims,lims)
ax.set_axis_off()
circles = []
circles.append(plt.Circle((0, 0), 0.1, color='g'))


for shell, e_in_shell in shells.items():
    n_lower = sum(val for key, val in shells.items() if key < shell)
    Z_shell = Zeff(protons, n_lower, e_in_shell)
    n_electron = 0
    if e_in_shell > 0:
        circles.append(plt.Circle((0, 0), shell_r(shell, Z_shell), fc='none', linewidth=0.7, ec='0'))
    subshells = {key[1]: val for key, val in e_config.items() if str(shell) in key}
    arrs = np.concatenate([np.full(val, key) for key, val in subshells.items()])
    for i, elec in enumerate(arrs):
        draw_electron(circles, shell, i, Z_shell, elec)

for circle in circles:
    ax.add_artist(circle)

ax.text(0,0,'+'+str(protons), ha='center', va='center', color='1', fontsize=8)
st.write(fig)