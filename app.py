#!/usr/bin/env python
# app.py – Streamlit interface for scanning multiple Finesse constants

# ---------------------------------------------------- imports
import re
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from pykat import finesse

# ---------------------------------------------------- original kat-script
KATSCRIPT = r"""
const l0_power 52.56458199755427
const l0_phase -90.93817853634775
const prm_reflectivity 0.9956487497038801
const prm_tuning 89.99999999976836
const bs_reflectivity 0.5005995284442057
const bs_tuning 0.0001407613844328216
const itmy_reflectivity 0.9963864053343761
const itmy_tuning 1.8534643552942951e-06
const etmy_reflectivity 0.9999949980012444
const etmy_tuning 6.408020040377285e-10
const itmx_reflectivity 0.9963978121469557
const itmx_tuning 89.99999684528466
const etmx_reflectivity 0.9999949978778255
const etmx_tuning 89.99999267836236
const srm_reflectivity 0.9485160310380369
const srm_tuning 89.99945189895219
const sq_db 19.999998541809596
const sq_angle 179.99376765630092
const fm1_reflectivity 0.991852564864244
const fm1_tuning 89.99999034326207
const fm2_reflectivity 0.999994999999954
const fm2_tuning 89.87728357123646
const bhbs_reflectivity 0.4996475292129385
const bhbs_tuning 0.08119084683693283
const lo_power 0.018686138753791454
const lo_phase 179.99996719468135
const prmsus_mass 0.01013675615482243
const bssus_mass 199.9902636621568
const itmysus_mass 199.99999982037156
const etmysus_mass 199.9996173624414
const itmxsus_mass 199.99945554163367
const etmxsus_mass 199.99999961701326
const srmsus_mass 46.063071416511995
const l0_prm_length 1.0044463258372085
const prm_bs_length 3999.9059066543723
const bs_itmy_length 34.08124896071303
const itmy_etmy_length 3999.9999988390236
const bs_itmx_length 33.62722068330888
const itmx_etmx_length 3999.9841586013963
const bs_srm_length 52.180394229565195
const srm_dbs1_length 1.0048981538149586
const sq_dbs2_length 1.4691154380038842
const dbs1_dbs2_length 8.264218849593902
const dbs2_fm1_length 3785.2276893505177
const fm1_fm2_length 3999.9990722811535
const dbs1_bhbs_length 1.0002555729364446
const lo_bhbs_length 3999.9982122306124
const noise_phase -179.99995109271595

% Laser
l L00_03 $l0_power 0.0 $l0_phase n0
s SX00_03__01_03 $l0_prm_length n0 nPRMs2
fsig mRL_0_xsig SX00_03__01_03 1 0 1
% PRM
m2 M01_03 $prm_reflectivity 5e-06 $prm_tuning nPRMs2 nPRM2
attr M01_03 mass $prmsus_mass
s SX01_03__02_03 $prm_bs_length nPRM2 nPRBS
fsig mRL_1_xsig SX01_03__02_03 1 0 1
% BS
bs2 B02_03 $bs_reflectivity 5e-06 $bs_tuning 45.0 nPRBS nYBS nXBS nSRBS
attr B02_03 mass $bssus_mass
% Y arm
s SY02_03__02_05 $bs_itmy_length nYBS nITMYs2
fsig mUD_y_1sig SY02_03__02_05 1 180 1
m2 M02_05 $itmy_reflectivity 5e-06 $itmy_tuning nITMYs2 nITMY2
attr M02_05 mass $itmysus_mass
s SY02_05__02_06 $itmy_etmy_length nITMY2 nETMY1
fsig mUD_y_0sig SY02_05__02_06 1 180 1
m2 M02_06 $etmy_reflectivity 5e-06 $etmy_tuning nETMY1 nETMYs1
attr M02_06 mass $etmysus_mass
% X arm
s SX02_03__03_03 $bs_itmx_length nXBS nITMXs2
fsig mRL_2_xsig SX02_03__03_03 1 0 1
m2 M03_03 $itmx_reflectivity 5e-06 $itmx_tuning nITMXs2 nITMX2
attr M03_03 mass $itmxsus_mass
s SX03_03__07_03 $itmx_etmx_length nITMX2 nETMX1
fsig mRL_3_xsig SX03_03__07_03 1 0 1
m2 M07_03 $etmx_reflectivity 5e-06 $etmx_tuning nETMX1 nETMXs1
attr M07_03 mass $etmxsus_mass
% down
s SY02_03__02_02 $bs_srm_length nSRBS nSRM1
fsig mUD_y_2sig SY02_03__02_02 1 180 1
m2 M02_02 $srm_reflectivity 5e-06 $srm_tuning nSRM1 nSRM2
attr M02_02 mass $srmsus_mass
s SY02_02__02_01 $srm_dbs1_length nSRM2 nFI2a

dbs F02_01 nFI2a nFI2b nFI2c nFI2d
s SY02_01__02_00 $dbs1_bhbs_length nFI2c nBHDBS_AS

bs2 B02_00 $bhbs_reflectivity 5e-06 $bhbs_tuning 45.0 nBHDBS_AS ToPD1 ToPD2 nBHDBS_LO

l L03_00 $lo_power 0.0 $lo_phase nLO
s SX03_00__02_00 $lo_bhbs_length nLO nBHDBS_LO

sq Q04_02 0 $sq_db $sq_angle nsqFinNode
s SY04_02__04_01 $sq_dbs2_length nsqFinNode nFI3b
dbs F04_01 nFI3a nFI3b nFI3c nFI3d
s SX04_01__02_01 $dbs1_dbs2_length nFI3c nFI2b
s SX04_01__05_01 $dbs2_fm1_length nFI3a FilterM1n1
m2 M05_01 $fm1_reflectivity 5e-06 $fm1_tuning FilterM1n1 FilterM1n2
s SX05_01__06_01 $fm1_fm2_length FilterM1n2 FilterM2n1
m2 M06_01 $fm2_reflectivity 5e-06 $fm2_tuning FilterM2n1 FilterM2n2
pd1 poutf1 $fs AtPD1
pd1 poutf2 $fs AtPD2
s stoPD1 1.0 ToPD1 AtPD1
s stoPD2 1.0 ToPD2 AtPD2
qnoised nodeFinalDet1 1 $fs 0 AtPD1
qnoised nodeFinalDet2 1 $fs 0 AtPD2
qhd nodeFinalDet $noise_phase AtPD1 AtPD2
pd0 poutdc1 AtPD1
pd0 poutdc2 AtPD2
phase 2
maxtem off
yaxis re:im
xaxis mRL_2_xsig f log 20 5000 100
"""

# ---------------------------------------------------- helpers
CONST_RE = re.compile(r"^const\s+(\S+)\s+([-+\d.eE]+)", re.M)
get_constants = lambda s: {k: float(v) for k, v in CONST_RE.findall(s)}

def update_katscript(txt, changes):
    def repl(m):
        n = m.group(1)
        return f"const {n} {changes[n]}" if n in changes else m.group(0)
    return CONST_RE.sub(repl, txt)


def run_model(script):
    """Return (freq, sensitivity) arrays for one Finesse run."""
    kat = finesse.kat(); kat.parse(script); out = kat.run()
    f, pd, qn = out.x, np.abs(out["poutf1"] - out["poutf2"]), np.abs(out["nodeFinalDet"])
    # amplitude RIN
    k1 = kat.deepcopy(); k1.signals.remove(); k1.xaxis.remove()
    for n, c in k1.components.items():
        if 'laser' in str(type(c)):
            s = f"{n}_amp"; k1.parse(f"fsig {s} {n} amp 1 0 {np.sqrt(c.P.value)}")
    k1.parse(f"xaxis {s} f log 20 5000 100")
    rin = 4e-9 * np.abs(k1.run()['poutf1'] - k1.run()['poutf2'])
    # frequency RIN
    k2 = kat.deepcopy(); k2.signals.remove(); k2.xaxis.remove()
    for n, c in k2.components.items():
        if 'laser' in str(type(c)):
            s = f"{n}_freq"; k2.parse(f"fsig {s} {n} freq 1 0 1")
    k2.parse(f"xaxis {s} f log 20 5000 100")
    fr = k2.run(); frn = fr.x * 1e-8 * np.abs(fr['poutf1'] - fr['poutf2'])
    return f, np.sqrt(qn ** 2 + rin ** 2 + frn ** 2) / pd

# ────────────────────────── Streamlit initial layout
st.set_page_config(page_title="Voyager sensitivity scan", layout="wide")

consts = get_constants(KATSCRIPT)
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
    return run_model(KATSCRIPT)

f_ref, sens_ref = reference()

# ────────────────────────── one-time spider chart of individual impacts
@st.cache_data(show_spinner=True)
def spider_plot(eps=0.01):
    int_sens_ref = np.sum(np.log(sens_ref))

    impacts = []
    for name, val in consts.items():
        delta = eps * val
        if delta == 0:
            impacts.append((name, 0.0))
            continue
        dn_script = update_katscript(KATSCRIPT, {name: val - delta})
        _, sens_dn = run_model(dn_script)
        impact = np.sum(np.log(sens_dn)) - int_sens_ref
        impacts.append((name, impact))

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
    "Choose constants", sorted(consts.keys()),
    default=s.picked, key="picked_widget")

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
changes = {k: consts[k] * f for k, f in s.factors.items()}
if changes:
    st.subheader("Modified constants")
    for n, new_val in changes.items():
        c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 1, 1])
        c1.write(n)
        c2.write(f"old = {consts[n]:.6g}")
        c3.write(f"× = {s.factors[n]:.2f}")
        c4.button("Show slider", key=f"show_{n}", on_click=queue_add, args=(n,))
        c5.button("Reset", key=f"reset_{n}", on_click=queue_reset, args=(n,))

# recomputation if flagged
if s.need_run or s.current is None:
    new_script = update_katscript(KATSCRIPT, changes)

    @st.cache_data(show_spinner=True)
    def current(script):
        return run_model(script)

    s.current = current(new_script)
    s.need_run = False

f_cur, sens_cur = s.current

# sensitivity plot
fig, ax = plt.subplots(figsize=(5, 3))
ax.loglog(f_ref, sens_ref, label="reference")
ax.loglog(f_cur, sens_cur, "--", label="modified")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Sensitivity")
ax.grid(True, which="both")
ax.legend()
st.pyplot(fig, use_container_width=False)

st.caption(f"Sliders recompute on release.  Radar chart shows impact of a differential decrease in each constant on the sum of logarithmic sensitivities.") 

# spider plot below sensitivity plot
st.pyplot(fig_spider, use_container_width=False)