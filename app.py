import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="AstroClassify",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="assets/logo.png"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600&family=Space+Grotesk:wght@500;700&display=swap');

/* streamlit ui overides */

.element-container :is(h1,h2,h3,h4,h5,h6) a {
    display: none !important;
}

[data-testid="stDecoration"] {
    display: none !important;
}

#MainMenu {
    display: none !important;
}

[data-testid="stStatusWidget"], 
[data-testid="stStopButton"] {
    display: none !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}

html, body { color-scheme: dark !important; }

.modebar-btn[data-val="select"],
.modebar-btn[data-val="lasso"] { display: none !important; }

/* Font changes */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.3px;
}

p, label, input, select,
.stMarkdown, [data-testid="stCaptionContainer"],
[data-testid="stText"], [class*="caption"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

[data-testid="stHeader"],
[data-testid="stBottomBlockContainer"],
[data-testid="stDecoration"],
.main {
    background-color: #080B12 !important;
    color: #E8ECF4 !important;
}
html, body, div.stApp, [data-testid="stAppViewContainer"] {
    color: #E8ECF4 !important; 
    background-color: #080B12 !important;
    background-image: 
        linear-gradient(rgba(255, 255, 255, 0.015) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.015) 1px, transparent 1px),
        radial-gradient(circle at 20% 20%, rgba(99, 102, 241, 0.09) 0%, transparent 50%) !important;
    background-size: 40px 40px, 40px 40px, 100% 100% !important;
}

/* Sidebar */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {
    background-color: #1A1C24 !important;
    border-right: 1px solid #1E2535 !important;
}

/* dropdown */
[data-baseweb="select"] > div,
[data-baseweb="select"] div[class*="ValueContainer"],
[data-baseweb="select"] div[class*="control"] {
    background-color: #1A1E2A !important;
    color: #E8ECF4 !important;
    border-color: #2D313E !important;
}
[data-baseweb="popover"] > div,
[data-baseweb="menu"],
[role="listbox"] {
    background-color: #1A1E2A !important;
    color: #E8ECF4 !important;
    border-color: #2D313E !important;
}
[role="option"]:hover,
[role="option"][aria-selected="true"] { background-color: #252A38 !important; }

/* Input fields */
[data-testid="stNumberInput"] input {
    background-color: #1A1E2A !important;
    color: #E8ECF4 !important;
    border-color: #2D313E !important;
}

/* Buttons */
button[kind="secondary"], button[kind="primary"],
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-primary"] {
    background-color: #1A1E2A !important;
    color: #E8ECF4 !important;
    border-color: #2D313E !important;
}

hr { border-color: #2D313E !important; }

.main .block-container { position: relative; z-index: 1; }

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
.main,
.block-container {
    background: transparent !important;
}

.block-container {
    position: relative;
    z-index: 2;
}

.stApp::before {
    z-index: 0 !important;
}
</style>
""", unsafe_allow_html=True)


COLOR_INDICES = ["u - g", "g - r", "r - i", "i - z"]
ALL_FILTERS   = ["u (UV)", "g (Green)", "r (Red)", "i (Near-IR)", "z (Infrared)"]
CHART_HEIGHT  = 420
PALETTE = {"GALAXY": "#8A2BE2", "QSO":    "#00FFFF",  "STAR":   "#FF9933"}
MODEBAR_CFG   = dict(remove=["select", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"])


# Data & model loading
@st.cache_resource
def load_stellar_model():
    try:
        return joblib.load("models/celestial_model.pkl")
    except:
        return None

@st.cache_data
def load_sample_data():
    try:
        df = pd.read_csv("data/star_classification.csv")
        df['u - g'] = df['u'] - df['g']
        df['g - r'] = df['g'] - df['r']
        df['r - i'] = df['r'] - df['i']
        df['i - z'] = df['i'] - df['z']
        return df[['u', 'g', 'r', 'i', 'z', 'redshift', 'class',
                   'u - g', 'g - r', 'r - i', 'i - z']]
    except:
        return None

rf_model  = load_stellar_model()
sample_df = load_sample_data()

DEFAULTS = dict(u_val=21.8858, g_val=21.0019, r_val=20.8359,
                i_val=20.7851, z_val=20.6269, redshift_val=2.435862)
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

st.markdown("""
<h1 style="margin:0 0 4px 0; font-family:'Space Grotesk',sans-serif;
           font-weight:700; letter-spacing:-0.5px; color:#E8ECF4;">
    SDSS Astronomical Object Classification Dashboard
</h1>
""", unsafe_allow_html=True)
st.caption("Interactive machine learning framework for multi-class astronomical object classification using Sloan Digital Sky Survey (SDSS) photometric observations.")
st.markdown("---")


with st.sidebar:
    st.sidebar.title("AstroClassify")
    st.sidebar.markdown("<hr style='margin-top: -10px; margin-bottom: 15px;'>", unsafe_allow_html=True)    
    st.header("Observational Parameters")
    st.write("Adjust the photometric magnitudes and redshift values used for astronomical object classification.")

    u = st.number_input("Ultraviolet Band Magnitude (u)", min_value=-0.1, max_value=40.0, value=st.session_state.u_val, format="%.4f")
    g = st.number_input("Green Band Magnitude (g)", min_value=-0.1, max_value=40.0, value=st.session_state.g_val, format="%.4f")
    r = st.number_input("Red Band Magnitude (r)", min_value=-0.1, max_value=40.0, value=st.session_state.r_val, format="%.4f")
    i = st.number_input("Near-Infrared Band Magnitude (i)", min_value=-0.1, max_value=40.0, value=st.session_state.i_val,format="%.4f")
    z = st.number_input("Infrared Band Magnitude (z)", min_value=-0.1, max_value=40.0, value=st.session_state.z_val,format="%.4f")
    redshift = st.number_input("Cosmological Redshift (z)", min_value=-0.1, max_value=10.0, value=st.session_state.redshift_val, format="%.6f")

    st.markdown("---")

    if st.button("Load Random Observation", width='stretch'):
        if sample_df is not None:
            row = sample_df.sample(n=1).iloc[0]
            st.session_state.u_val        = float(row['u'])
            st.session_state.g_val        = float(row['g'])
            st.session_state.r_val        = float(row['r'])
            st.session_state.i_val        = float(row['i'])
            st.session_state.z_val        = float(row['z'])
            st.session_state.redshift_val = float(row['redshift'])
            st.rerun()
        else:
            st.sidebar.error("Database unavailable.")


if rf_model is None:
    st.error("The trained classification model could not be loaded.")
    st.info("Please verify that 'celestial_model.pkl' exists in the models directory.")
    st.stop()
else:
    input_data           = pd.DataFrame([[u, g, r, i, z, redshift]], columns=['u', 'g', 'r', 'i', 'z', 'redshift'])
    encoded_prediction   = rf_model.predict(input_data)[0]
    prediction_probs     = rf_model.predict_proba(input_data)[0]
    predicted_class_name = {0: "GALAXY", 1: "QSO", 2: "STAR"}.get(encoded_prediction, "UNKNOWN")
    confidence_score     = float(np.max(prediction_probs) * 100)

accent_color = PALETTE.get(predicted_class_name, "#9b59b6")


# Classification result card 

st.markdown(f"""
<div style="background-color:#14171F; padding:20px 28px; border-radius:8px;
            border:1px solid #2D313E; border-left:5px solid {accent_color}; margin-bottom:28px;
            display:flex; align-items:center; gap:48px;">
    <div>
        <span style="color:#8A90A6; text-transform:uppercase; font-size:11px;
                     font-weight:600; letter-spacing:1px;">Predicted Object Class</span>
        <h2 style="margin:6px 0 0 0; color:white; font-weight:700; font-size:2rem;">{predicted_class_name}</h2>
    </div>
    <div style="width:1px; background:#2D313E; align-self:stretch;"></div>
    <div>
        <span style="color:#8A90A6; text-transform:uppercase; font-size:11px;
                     font-weight:600; letter-spacing:1px;">Model Confidence</span>
        <h2 style="margin:6px 0 0 0; color:{accent_color}; font-weight:700; font-size:2rem;">{confidence_score:.2f}%</h2>
    </div>
</div>
""", unsafe_allow_html=True)


hdr_col1, hdr_col2 = st.columns([1, 1], gap="large")

with hdr_col1:
    st.markdown(
        "<h3 style='margin:0 0 6px 0; font-size:1.25rem; font-weight:600; "
        "font-family:\"Space Grotesk\",sans-serif; letter-spacing:-0.3px;'>"
        "Spectral Energy Distribution</h3>",
        unsafe_allow_html=True
    )
    selected_filters = st.multiselect(
        "Filter Selection",
        options=ALL_FILTERS,
        default=ALL_FILTERS,
        label_visibility="collapsed"
    )

with hdr_col2:
    st.markdown(
        "<h3 style='margin:0 0 6px 0; font-size:1.25rem; font-weight:600; "
        "font-family:\"Space Grotesk\",sans-serif; letter-spacing:-0.3px;'>"
        "Color-Color Diagram</h3>",
        unsafe_allow_html=True
    )
    
    lbl_x, dd1, lbl_y, dd2 = st.columns([0.15, 0.85, 0.15, 0.85])
    
    with lbl_x:
        st.markdown("<p style='line-height:2.5rem; font-weight:600; margin:0; text-align:right;'>X:</p>", unsafe_allow_html=True)
    with dd1:
        x_index = st.selectbox("X-Axis", COLOR_INDICES, index=1, label_visibility="collapsed")
        
    with lbl_y:
        st.markdown("<p style='line-height:2.5rem; font-weight:600; margin:0; text-align:right;'>Y:</p>", unsafe_allow_html=True)
    with dd2:
        # This prevents users from plotting the exact same feature on both axes 
        y_options = [c for c in COLOR_INDICES if c != x_index]
        y_index   = st.selectbox("Y-Axis", y_options, index=0, label_visibility="collapsed")

st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)


# Charts
col1, col2 = st.columns([1, 1], gap="large")
with col1:
    source_magnitude_map = {
        "u (UV)": u, "g (Green)": g, "r (Red)": r,
        "i (Near-IR)": i, "z (Infrared)": z
    }
    filtered_x = [f for f in ALL_FILTERS if f in selected_filters]
    filtered_y = [source_magnitude_map[f] for f in filtered_x]

    fig_sed = go.Figure()
    if filtered_x:
        fig_sed.add_trace(go.Scatter(
            x=filtered_x, y=filtered_y,
            mode='lines+markers' if len(filtered_x) > 1 else 'markers',
            line=dict(color=accent_color, width=3),
            marker=dict(size=8, symbol='circle')
        ))
    fig_sed.update_layout(
        template="plotly_dark", dragmode="pan",
        margin=dict(l=40, r=20, t=20, b=40), height=CHART_HEIGHT,
        xaxis_title="Photometric Filter Channels",
        yaxis_title="Apparent Magnitude",
        yaxis=dict(autorange="reversed"),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_sed, width='stretch', config=MODEBAR_CFG)

with col2:
    if sample_df is None:
        st.info("stellar mapping clusters unavailable; check data path.")
    else:
        source_vals = {'u': u, 'g': g, 'r': r, 'i': i, 'z': z}

        def compute_index(s):
            a, b = [p.strip() for p in s.split('-')]
            return source_vals[a] - source_vals[b]

        plot_df  = sample_df.sample(n=1200, random_state=42).copy() 
        target_x = compute_index(x_index)
        target_y = compute_index(y_index)

        fig_cc = px.scatter(
            plot_df, x=x_index, y=y_index, color='class',
            color_discrete_map=PALETTE, opacity=0.4,
            labels={x_index: f'Color Index ({x_index})', y_index: f'Color Index ({y_index})'}
        )
        fig_cc.add_trace(go.Scatter(
            x=[target_x], y=[target_y], mode='markers',
            marker=dict(color='white', size=14, symbol='x', line=dict(color='black', width=2)),
            name='Selected Source'
        ))
        fig_cc.update_layout(
            template="plotly_dark", dragmode="pan",
            margin=dict(l=40, r=20, t=20, b=40), height=CHART_HEIGHT,
            legend=dict(
                bgcolor="#14171F", bordercolor="#2D313E", borderwidth=1,
                font=dict(size=12, color="#C8CDD8"), itemsizing="constant",
                x=0.99, y=0.99, xanchor="right", yanchor="top"
            ),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_cc, width='stretch', config=MODEBAR_CFG)