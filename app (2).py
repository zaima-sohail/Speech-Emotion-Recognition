import io, json
import numpy as np, pandas as pd, librosa, joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Speech Emotion Dashboard", page_icon="🎙️", layout="wide")
MAX_LEN = 100
COLORS = {"SVM": "#F59E0B", "CNN": "#6366F1"}

@st.cache_resource
def load_models():
    return (load_model("emotion_cnn.keras"), joblib.load("svm.joblib"),
            np.load("mean.npy"), np.load("std.npy"))

@st.cache_data
def load_metrics():
    return json.load(open("metrics.json"))

cnn, svm, mean, std = load_models()
M = load_metrics()
classes = M["classes"]

def analyse(data):
    y, sr = librosa.load(io.BytesIO(data), sr=16000, duration=3)
    if len(y) < 1600:
        return None
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    svm_feat = np.hstack([mfcc.mean(axis=1), mfcc.std(axis=1)])[None]
    seq = mfcc[:, :MAX_LEN]
    if seq.shape[1] < MAX_LEN:
        seq = np.pad(seq, ((0, 0), (0, MAX_LEN - seq.shape[1])))
    x = ((seq.T - mean) / std)[None]
    return {"y": y, "sr": sr, "mfcc": mfcc,
            "SVM": svm.predict_proba(svm_feat)[0],
            "CNN": cnn.predict(x, verbose=0)[0]}

# ---------- Sidebar ----------
with st.sidebar:
    st.title("🎙️ SER Dashboard")
    st.caption("Speech Emotion Recognition")
    st.write(f"**Train clips:** {M['n_train']}")
    st.write(f"**Test clips:** {M['n_test']}")
    st.write(f"**Emotions:** {len(classes)}")
    st.info("Test set mein woh speakers hain jo training mein nahi the.")

st.title("Speech Emotion Recognition")
tab1, tab2, tab3, tab4 = st.tabs(
    ["🎤 Live Prediction", "📊 Model Comparison", "🔍 Per-Emotion Analysis", "🧩 Confusion Matrix"])

# ---------- Tab 1: Live prediction ----------
with tab1:
    c1, c2 = st.columns(2)
    data = None
    with c1:
        up = st.file_uploader("WAV file upload karo", type=["wav"])
        if up is not None:
            data = up.getvalue()
    with c2:
        if hasattr(st, "audio_input"):
            rec = st.audio_input("Ya apni awaaz record karo")
            if rec is not None:
                data = rec.getvalue()
        else:
            st.warning("Is Streamlit version mein mic recording nahi hai. "
                       "`pip install -U streamlit` chalao.")

    if data:
        st.audio(data)
        try:
            r = analyse(data)
        except Exception as e:
            r = None
            st.error(f"Audio process nahi hui: {e}")
        if r is None:
            st.warning("Audio bahut chhoti hai, dobara record karo (kam az kam 1 second).")
        else:
            m1, m2 = st.columns(2)
            for col, name in zip((m1, m2), ("SVM", "CNN")):
                p = r[name]; i = int(np.argmax(p))
                col.metric(f"{name} prediction", classes[i].upper(), f"{p[i]*100:.1f}% confidence")

            df_p = pd.DataFrame({"emotion": classes * 2,
                                 "probability": np.concatenate([r["SVM"], r["CNN"]]),
                                 "model": ["SVM"] * len(classes) + ["CNN"] * len(classes)})
            fig = px.bar(df_p, x="emotion", y="probability", color="model", barmode="group",
                         color_discrete_map=COLORS, title="Probability per emotion")
            st.plotly_chart(fig, use_container_width=True)

            w1, w2 = st.columns(2)
            t = np.arange(len(r["y"])) / r["sr"]
            fw = go.Figure(go.Scatter(x=t[::20], y=r["y"][::20], line=dict(color="#6366F1")))
            fw.update_layout(title="Waveform", xaxis_title="seconds", height=320)
            w1.plotly_chart(fw, use_container_width=True)
            fm = px.imshow(r["mfcc"], aspect="auto", origin="lower",
                           color_continuous_scale="Viridis", title="MFCC heatmap")
            fm.update_layout(height=320)
            w2.plotly_chart(fm, use_container_width=True)

# ---------- Tab 2: Model comparison ----------
with tab2:
    rows = []
    for name, d in M["models"].items():
        rep = d["report"]
        rows.append({"Model": name, "Accuracy": rep["accuracy"],
                     "Macro F1": rep["macro avg"]["f1-score"],
                     "Macro Precision": rep["macro avg"]["precision"],
                     "Macro Recall": rep["macro avg"]["recall"]})
    comp = pd.DataFrame(rows)

    cols = st.columns(len(comp))
    for col, (_, row) in zip(cols, comp.iterrows()):
        col.metric(f"{row['Model']} accuracy", f"{row['Accuracy']*100:.1f}%")
    if len(comp) == 2:
        diff = (comp.loc[1, "Accuracy"] - comp.loc[0, "Accuracy"]) * 100
        st.success(f"{comp.loc[1,'Model']} ki accuracy {comp.loc[0,'Model']} se {diff:+.1f} points ka farq rakhti hai.")

    long = comp.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig = px.bar(long, x="Metric", y="Score", color="Model", barmode="group",
                 color_discrete_map=COLORS, title="Overall metrics")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(comp.style.format({c: "{:.3f}" for c in comp.columns[1:]}),
                 use_container_width=True)

    h = M.get("cnn_history", {})
    if "accuracy" in h:
        fh = go.Figure()
        fh.add_scatter(y=h["accuracy"], name="train accuracy", line=dict(color="#6366F1"))
        if "val_accuracy" in h:
            fh.add_scatter(y=h["val_accuracy"], name="test accuracy", line=dict(color="#F59E0B"))
        fh.update_layout(title="CNN training curve", xaxis_title="epoch", yaxis_title="accuracy")
        st.plotly_chart(fh, use_container_width=True)

# ---------- Tab 3: Per-emotion ----------
with tab3:
    rows = []
    for name, d in M["models"].items():
        for c in classes:
            rows.append({"Model": name, "Emotion": c, "F1": d["report"][c]["f1-score"],
                         "Support": d["report"][c]["support"]})
    pe = pd.DataFrame(rows)
    fig = px.bar(pe, x="Emotion", y="F1", color="Model", barmode="group",
                 color_discrete_map=COLORS, title="F1-score per emotion")
    st.plotly_chart(fig, use_container_width=True)
    sup = pe[pe["Model"] == pe["Model"].iloc[0]][["Emotion", "Support"]]
    st.plotly_chart(px.bar(sup, x="Emotion", y="Support", title="Test clips per emotion"),
                    use_container_width=True)
    st.caption("Jis emotion ke test clips kam hon, uska F1 kam bharosemand hota hai.")

# ---------- Tab 4: Confusion matrix ----------
with tab4:
    pick = st.radio("Model", list(M["models"].keys()), horizontal=True)
    cm = np.array(M["models"][pick]["cm"], dtype=float)
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig = px.imshow(cmn, x=classes, y=classes, text_auto=".2f",
                    color_continuous_scale="Blues", aspect="auto",
                    labels=dict(x="Predicted", y="Actual", color="Fraction"),
                    title=f"{pick} confusion matrix (row-normalized)")
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)


# ---------- Resources ----------
with st.expander("📚 Resources & Dataset Information"):
    st.markdown("""
    ### Speech Emotion Recognition (SER)
    This application evaluates two models (Classical Machine Learning with SVM and Deep Learning with a 1D CNN) using MFCC features.
    
    **Datasets Included:**
    * **CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset)
    * **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
    * **TESS** (Toronto Emotional Speech Set)
    * **SAVEE** (Surrey Audio-Visual Ecstatically Expressive)
    
    **References:**
    * Dataset source: [Kaggle Speech Emotion Recognition EN](https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en)
    * Developed using TensorFlow/Keras, Scikit-Learn, Librosa, and Streamlit.
    """)
