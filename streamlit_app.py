import numpy as np
import streamlit as st  # type: ignore[import-not-found]
import matplotlib.pyplot as plt

from optics import (
    multi_layer_stack,
    default_mirrors,
    generate_rays,
    RayTracer,
    peak_slope,
    fwhm,
    optimize_system,
)
from optics.materials import gold_n_complex_scalar
from optics.elements import MetalFilm
from optics.optimizer import collect_hits, spot_uniformity_metric
from optics.utils import export_results, export_results_kafka

st.set_page_config(page_title="SPR & Beam Demo", layout="wide")
st.title("SPR + Beam Shaping Interactive Demo")

# Sidebar controls
st.sidebar.header("Parameters")
num_mirrors = st.sidebar.slider("Mirror count", 1, 6, 3)
theta_spread = st.sidebar.slider("Source spread (deg)", 2.0, 30.0, 10.0)
film_thickness = st.sidebar.slider("Au thickness (nm)", 30.0, 70.0, 50.0)

run_opt = st.sidebar.button("Run Optuna optimization (quick demo)")
record = st.sidebar.checkbox("Record this run")

# Build optical elements
mirrors = default_mirrors(num_mirrors)
film = MetalFilm(thickness_nm=film_thickness, n_metal=gold_n_complex_scalar)

def reflect_curve():
    angles = np.deg2rad(np.linspace(40, 75, 300))
    R = [film._tm_reflectance(632.8, th) for th in angles]
    return np.rad2deg(angles), R

# SPR curve plot
with st.expander("SPR reflectance curve", expanded=True):
    ang_deg, R_vals = reflect_curve()
    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(ang_deg, R_vals, 'b-')
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Reflectance")
    ax.set_title("SPR curve")
    st.pyplot(fig)
    st.write(f"Peak slope: {peak_slope(list(ang_deg), list(R_vals)):.3f} /deg")

# Beam spot
with st.expander("Beam spot heatmap", expanded=True):
    rays = generate_rays(400, theta_spread)
    tracer = RayTracer(mirrors)
    paths = tracer.trace(rays, max_interactions=3)
    hits = collect_hits(paths, [r.direction for r in rays])
    if hits.size == 0:
        st.write("No hits computed.")
    else:
        fig2, ax2 = plt.subplots(figsize=(4,3))
        hb = ax2.hexbin(hits[:,0], hits[:,1], gridsize=40, cmap='inferno')
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_title("Spot heatmap")
        st.pyplot(fig2)
        st.write(f"FWHM: {fwhm(list(hits[:,0])):.3f}")
        uni = spot_uniformity_metric(hits[:,0])
        st.write(f"Uniformity metric: {uni:.3f}")

        if record:
            record_dict = {
                "num_mirrors": num_mirrors,
                "theta_spread": theta_spread,
                "film_thickness": film_thickness,
                "peak_slope": peak_slope(list(ang_deg), list(R_vals)),
                "fwhm": fwhm(list(hits[:,0])),
                "uniformity": uni,
            }
            export_results(record_dict, "record.csv")
            export_results_kafka("spr_run", record_dict)
            fig.savefig("spot.png", dpi=200)
            st.success("Run recorded → record.csv / spot.png / Kafka")

# Optimization demo
if run_opt:
    with st.spinner("Running optimization (20 trials)..."):
        res = optimize_system(n_trials=20)
    st.success("Done"); st.json(res) 