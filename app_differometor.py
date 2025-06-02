#!/usr/bin/env python
# app.py – Streamlit interface for scanning multiple Finesse constants

# ---------------------------------------------------- imports
import re
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import differometor as df
from differometor.setups import voyager
from differometor.utils import set_value
from differometor.components import demodulate_signal_power
import jax.numpy as jnp
from jax import jit
import jax

parameters = [
    ("l0", "power", 52.56458199755427),
    ("l0", "phase", -90.93817853634775),
    ("prm", "reflectivity", 0.9956487497038801),
    ("prm", "tuning", 89.99999999976836),
    ("bs", "reflectivity", 0.5005995284442057),
    ("bs", "tuning", 0.0001407613844328216),
    ("itmy", "reflectivity", 0.9963864053343761),
    ("itmy", "tuning", 1.8534643552942951e-06),
    ("etmy", "reflectivity", 0.9999949980012444),
    ("etmy", "tuning", 6.408020040377285e-10),
    ("itmx", "reflectivity", 0.9963978121469557),
    ("itmx", "tuning", 89.99999684528466),
    ("etmx", "reflectivity", 0.9999949978778255),
    ("etmx", "tuning", 89.99999267836236),
    ("srm", "reflectivity", 0.9485160310380369),
    ("srm", "tuning", 89.99945189895219),
    ("sq", "db", 19.999998541809596),
    ("sq", "angle", 179.99376765630092),
    ("fm1", "reflectivity", 0.991852564864244),
    ("fm1", "tuning", 89.99999034326207),
    ("fm2", "reflectivity", 0.999994999999954),
    ("fm2", "tuning", 89.87728357123646),
    ("bhbs", "reflectivity", 0.4996475292129385),
    ("bhbs", "tuning", 0.08119084683693283),
    ("lo", "power", 0.018686138753791454),
    ("lo", "phase", 179.99996719468135),
    ("prmsus", "mass", 0.01013675615482243),
    ("bssus", "mass", 199.9902636621568),
    ("itmysus", "mass", 199.99999982037156),
    ("etmysus", "mass", 199.9996173624414),
    ("itmxsus", "mass", 199.99945554163367),
    ("etmxsus", "mass", 199.99999961701326),
    ("srmsus", "mass", 46.063071416511995),
    ("l0_prm", "length", 1.0044463258372085),
    ("prm_bs", "length", 3999.9059066543723),
    ("bs_itmy", "length", 34.08124896071303),
    ("itmy_etmy", "length", 3999.9999988390236),
    ("bs_itmx", "length", 33.62722068330888),
    ("itmx_etmx", "length", 3999.9841586013963),
    ("bs_srm", "length", 52.180394229565195),
    ("srm_dbs1", "length", 1.0048981538149586),
    ("sq_dbs2", "length", 1.4691154380038842),
    ("dbs1_dbs2", "length", 8.264218849593902),
    ("dbs2_fm1", "length", 3785.2276893505177),
    ("fm1_fm2","length", 3999.9990722811535),
    ("dbs1_bhbs", "length", 1.0002555729364446),
    ("lo_bhbs", "length", 3999.9982122306124),
    ("noise", "phase", -179.99995109271595)]

setups = [voyager()[0], voyager("amplitude_modulation")[0], voyager("frequency_modulation")[0]]

for setup in setups:
    for name, prop, value in parameters:
        set_value(name, prop, value, setup)

optimization_pairs = [(name, prop) for name, prop, _ in parameters]
parameter_names = ["_".join((name, prop)) for name, prop, _ in parameters]
parameters = [value for _, _, value in parameters]
frequencies = jnp.logspace(jnp.log10(20), jnp.log10(5000), 100)

built_setups = []
for setup in setups:
    simulation_arrays, detector_ports, *_ = df.run_build_step(
        setup,
        [("f", "frequency")],
        frequencies,
        optimization_pairs,
    )
    built_setups.append((simulation_arrays, detector_ports))


def calculate_sensitivity(parameters):
    print("Time1")
    carrier, signal, noise = df.simulate_in_parallel(parameters, *built_setups[0][0][1:])
    powers = demodulate_signal_power(carrier, signal)
    powers = powers[built_setups[0][1]]
    powers = jnp.abs(powers[0] - powers[1])

    print("Time2")
    amplitude_carrier, amplitude_signal, _ = df.simulate_in_parallel(parameters, *built_setups[1][0][1:])
    amplitude_powers = demodulate_signal_power(amplitude_carrier, amplitude_signal)
    amplitude_powers = amplitude_powers[built_setups[1][1]]
    amplitude_powers = 4e-9 * jnp.abs(amplitude_powers[0] - amplitude_powers[1])

    print("Time3")
    frequency_carrier, frequency_signal, _ = df.simulate_in_parallel(parameters, *built_setups[2][0][1:])
    frequency_powers = demodulate_signal_power(frequency_carrier, frequency_signal)
    frequency_powers = frequency_powers[built_setups[2][1]]
    frequency_powers = 1e-8 * jnp.abs(frequency_powers[0] - frequency_powers[1])

    print("Time4")
    return jnp.sqrt(noise ** 2 + amplitude_powers ** 2 + frequency_powers ** 2) / powers


def loss_function(parameters):
    sensitivities = calculate_sensitivity(parameters)
    return jnp.sum(jnp.log(sensitivities))


# ────────────────────────── Streamlit initial layout
st.set_page_config(page_title="Interferoboard", layout="wide")

s = st.session_state  # shorthand

# persistent state
s.setdefault("picked", [])     # list of constants with sliders
s.setdefault("factors", {})     # {const: factor}
s.setdefault("need_run", True)
s.setdefault("current", None)  # last (freq,sens) from Finesse

# ---------------------------------------------------- helper to trigger recompute

def mark_run():
    s.need_run = True

# ---------------------------------------------------- button callbacks

def queue_add(const):
    """Store a constant name that should be added to the multiselect on next run."""
    s["_add_const"] = const


def queue_reset(const):
    """Remove factor and flag recompute."""
    s.factors.pop(const, None)
    s.need_run = True

# ---------------------------------------------------- process queued widget updates BEFORE widgets are created
if "_add_const" in s:
    const_to_add = s.pop("_add_const")
    if const_to_add not in s.picked:
        s.picked.append(const_to_add)
        # Update the widget's value *before* it is instantiated in this run.
        st.session_state["picked_widget"] = s.picked

# ────────────────────────── reference curve (cached once)
@st.cache_data(show_spinner=False)
def reference():
    return calculate_sensitivity(jnp.array(parameters))

sens_ref = reference()

# ────────────────────────── one-time spider chart of individual impacts
@st.cache_data(show_spinner=True)
def spider_plot():
    grad_fn = jax.grad(loss_function)
    gradients = grad_fn(jnp.array(parameters))
    impacts = []
    for ix, gradient in enumerate(gradients.tolist()):
        impacts.append((parameter_names[ix], gradient))

    impacts.sort(key=lambda x: x[1], reverse=True)
    labels = [g[0] for g in impacts]
    radii = [g[1] if g[1] > 0 else 1e-20 for g in impacts]

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    radii += radii[:1]; angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.plot(angles, radii, lw=1)
    ax.fill(angles, radii, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_xticklabels([])

    offset = 50 * ax.get_rmax()
    for ang, lab in zip(angles[:-1], labels):
        rot = np.degrees(ang)
        ha = 'right' if 90 < rot < 270 else 'left'
        if 90 < rot < 270:
            rot += 180
        ax.text(ang, offset, lab, rotation=rot, rotation_mode='anchor',
                ha=ha, va='center', fontsize=5)

    ax.tick_params(axis='y', labelsize=5)
    ax.grid(True)
    ax.set_rscale('log')
    return fig

fig_spider = spider_plot()

# ────────────────────────── UI – controls, plots, radar
s.picked = st.multiselect(
    "Choose constants", parameter_names, default=s.picked, key="picked_widget")

# sliders
for name in s.picked:
    factor = st.slider(
        name, 0.0, 5.0, s.factors.get(name, 1.0), 0.01,
        format="%.2f", key=f"f_{name}", on_change=mark_run)
    if factor != 1.0:
        s.factors[name] = factor
    elif name in s.factors:
        del s.factors[name]

# diff table + buttons
changes = {k: parameters[parameter_names.index(k)] * f for k, f in s.factors.items()}
if changes:
    st.subheader("Modified constants")
    for n, new_val in changes.items():
        c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 1, 1])
        c1.write(n)
        c2.write(f"old = {parameters[parameter_names.index(n)]:.6g}")
        c3.write(f"× = {s.factors[n]:.2f}")
        c4.button("Show slider", key=f"show_{n}", on_click=queue_add, args=(n,))
        c5.button("Reset", key=f"reset_{n}", on_click=queue_reset, args=(n,))

# recomputation if flagged
if s.need_run or s.current is None:
    new_parameters = parameters.copy()
    for name, value in changes.items():
        if name in s.factors:
            ix = parameter_names.index(name)
            new_parameters[ix] = value

    @st.cache_data(show_spinner=True)
    def current(parameters):
        return calculate_sensitivity(jnp.array(parameters))

    s.current = current(new_parameters)
    s.need_run = False

sens_cur = s.current

# sensitivity plot
fig, ax = plt.subplots(figsize=(5, 3))
ax.loglog(frequencies, sens_ref, label="reference")
ax.loglog(frequencies, sens_cur, "--", label="modified")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Sensitivity")
ax.grid(True, which="both")
ax.legend()
st.pyplot(fig, use_container_width=False)

st.caption(f"Sliders recompute on release.  Radar chart shows impact of a differential decrease in each constant on the sum of logarithmic sensitivities.") 

# spider plot below sensitivity plot
st.pyplot(fig_spider, use_container_width=False)