import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
import os
import tempfile
import pickle
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'

st.set_page_config(
    page_title="Vocal Disorder Detection",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .main-title {
        text-align: center; font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .subtitle { text-align: center; color: #8892b0; font-size: 1rem; margin-bottom: 2rem; }
    .result-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 2px solid #0f3460; border-radius: 16px;
        padding: 2rem; text-align: center; margin: 1rem 0;
    }
    .healthy-card { border-color: #4caf50 !important; }
    .disease-card { border-color: #f44336 !important; }
    .pred-label { font-size: 2.2rem; font-weight: 800; margin: 0.5rem 0; }
    .healthy-label { color: #4caf50; }
    .laryngo-label { color: #ff9800; }
    .vox-label { color: #f44336; }
    .conf-text { font-size: 1.1rem; color: #8892b0; margin-top: 0.4rem; }
    .info-box {
        background: #1a1a2e; border-left: 4px solid #667eea;
        border-radius: 8px; padding: 1rem 1.2rem;
        margin: 0.8rem 0; color: #ccd6f6; font-size: 0.92rem;
    }
    .warning-box {
        background: #2d1b00; border-left: 4px solid #ff9800;
        border-radius: 8px; padding: 1rem 1.2rem;
        margin: 0.8rem 0; color: #ffcc80; font-size: 0.92rem;
    }
    .section-header {
        font-size: 1.2rem; font-weight: 700; color: #ccd6f6;
        border-bottom: 2px solid #0f3460; padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

SAMPLE_RATE = 16000
DURATION    = 3
SAMPLES     = SAMPLE_RATE * DURATION
N_MFCC      = 40
MAX_PAD_LEN = 174
HAND_DIM    = 390
MULTI_CLASSES  = ['Healthy', 'Laryngozele', 'Vox senilis']
BINARY_CLASSES = ['Healthy', 'Diseased']

def get_custom_objects():
    import tensorflow as tf
    import tensorflow.keras.backend as K
    from tensorflow.keras.layers import Layer
    class AttentionLayer(Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        def build(self, input_shape):
            self.W = self.add_weight(name='att_W',
                shape=(input_shape[-1], input_shape[-1]),
                initializer='glorot_uniform', trainable=True)
            self.V = self.add_weight(name='att_V',
                shape=(input_shape[-1], 1),
                initializer='glorot_uniform', trainable=True)
            self.b = self.add_weight(name='att_b',
                shape=(input_shape[1], 1),
                initializer='zeros', trainable=True)
            super().build(input_shape)
        def call(self, x):
            score = K.tanh(K.dot(x, self.W))
            score = K.dot(score, self.V) + self.b
            alpha = K.softmax(score, axis=1)
            return K.sum(alpha * x, axis=1)
        def compute_output_shape(self, input_shape):
            return (input_shape[0], input_shape[-1])
        def get_config(self):
            return super().get_config()
    def focal_loss(gamma=2.0, alpha=0.25):
        def focal_loss_fn(y_true, y_pred):
            y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0)
            ce     = -y_true * tf.math.log(y_pred)
            weight = alpha * y_true * tf.pow(1 - y_pred, gamma)
            return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=1))
        return focal_loss_fn
    return {'AttentionLayer': AttentionLayer, 'focal_loss_fn': focal_loss()}

@st.cache_resource(show_spinner=False)
def load_multi_model():
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    from tensorflow.keras.models import load_model
    return load_model("best_dual_branch_overall.h5", custom_objects=get_custom_objects())

@st.cache_resource(show_spinner=False)
def load_binary_model():
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    from tensorflow.keras.models import load_model
    return load_model("best_binary_overall.h5", custom_objects=get_custom_objects())

@st.cache_resource(show_spinner=False)
def load_scaler():
    with open("scaler.pkl", "rb") as f:
        return pickle.load(f)

def preprocess_audio(file_bytes):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        audio, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
    if len(audio) > SAMPLES:
        audio = audio[:SAMPLES]
    else:
        audio = np.pad(audio, (0, SAMPLES - len(audio)), mode='constant')
    mx = np.max(np.abs(audio))
    if mx > 0:
        audio = audio / mx
    return audio

def extract_mfcc(signal):
    mfcc = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    if mfcc.shape[1] < MAX_PAD_LEN:
        mfcc = np.pad(mfcc, ((0,0),(0, MAX_PAD_LEN - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_PAD_LEN]
    return mfcc[..., np.newaxis].astype(np.float32)

def extract_handcrafted(signal):
    sr = SAMPLE_RATE
    features = []
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    features.extend(np.mean(mfcc, axis=1)); features.extend(np.std(mfcc, axis=1))
    mel    = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features.extend(np.mean(mel_db, axis=1)); features.extend(np.std(mel_db, axis=1))
    chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
    features.extend(np.mean(chroma, axis=1)); features.extend(np.std(chroma, axis=1))
    harmonic = librosa.effects.harmonic(signal)
    tonnetz  = librosa.feature.tonnetz(y=harmonic, sr=sr)
    features.extend(np.mean(tonnetz, axis=1)); features.extend(np.std(tonnetz, axis=1))
    contrast = librosa.feature.spectral_contrast(y=signal, sr=sr)
    features.extend(np.mean(contrast, axis=1)); features.extend(np.std(contrast, axis=1))
    zcr = librosa.feature.zero_crossing_rate(signal)
    features.append(float(np.mean(zcr))); features.append(float(np.std(zcr)))
    rms = librosa.feature.rms(y=signal)
    features.append(float(np.mean(rms))); features.append(float(np.std(rms)))
    return np.array(features, dtype=np.float32)

def plot_waveform(signal):
    fig, ax = plt.subplots(figsize=(8, 2.5))
    fig.patch.set_facecolor('#0e1117'); ax.set_facecolor('#1a1a2e')
    t = np.linspace(0, DURATION, len(signal))
    ax.plot(t, signal, color='#667eea', lw=0.8, alpha=0.9)
    ax.fill_between(t, signal, alpha=0.2, color='#764ba2')
    ax.set_title('Audio Waveform', color='white', fontweight='bold')
    ax.set_xlabel('Time (s)', color='white'); ax.set_ylabel('Amplitude', color='white')
    ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_edgecolor('#0f3460')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#0e1117')
    buf.seek(0); plt.close(); return buf

def plot_mfcc(signal):
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor('#0e1117'); ax.set_facecolor('#1a1a2e')
    mfcc = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    img  = librosa.display.specshow(mfcc, sr=SAMPLE_RATE, x_axis='time', ax=ax, cmap='magma')
    fig.colorbar(img, ax=ax)
    ax.set_title('MFCC Spectrogram', color='white', fontweight='bold')
    ax.tick_params(colors='white')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#0e1117')
    buf.seek(0); plt.close(); return buf

def plot_confidence(proba, class_names):
    COLORS = ['#4caf50', '#ff9800', '#f44336']
    fig, ax = plt.subplots(figsize=(6, 2.5))
    fig.patch.set_facecolor('#0e1117'); ax.set_facecolor('#1a1a2e')
    bars = ax.barh(class_names, proba*100, color=COLORS[:len(class_names)], alpha=0.85, height=0.5)
    for bar, val in zip(bars, proba):
        ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                f'{val*100:.1f}%', va='center', color='white', fontsize=10)
    ax.set_xlim(0, 115)
    ax.set_xlabel('Confidence (%)', color='white')
    ax.set_title('Prediction Confidence', color='white', fontweight='bold')
    ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_edgecolor('#0f3460')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#0e1117')
    buf.seek(0); plt.close(); return buf

st.markdown('<h1 class="main-title">🎙️ Vocal Disorder Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered · Dual-Branch CNN-BiLSTM-Attention · SVD Dataset</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    mode = st.radio("Detection Mode",
        ["🔬 Multi-class (3 classes)", "🔵 Binary (Healthy vs Diseased)"], index=0)
    st.markdown("---")
    st.markdown("### 📋 Classes")
    st.markdown("**Multi-class:** 🟢 Healthy · 🟠 Laryngozele · 🔴 Vox Senilis\n\n**Binary:** 🟢 Healthy · 🔴 Diseased")
    st.markdown("---")
    st.markdown("### 🧠 Model Info")
    st.markdown("- Architecture: Dual-Branch CNN-BiLSTM\n- Training: 5-Fold CV\n- Accuracy: ~97.73%")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="section-header">📂 Upload Voice File</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a .wav or .mp3 file", type=["wav", "mp3"])
    if uploaded_file is not None:
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
        if file_bytes and len(file_bytes) > 0:
            st.audio(file_bytes, format="audio/wav")
            st.markdown(f'<div class="info-box">📁 <b>{uploaded_file.name}</b><br>📦 {len(file_bytes)/1024:.1f} KB</div>', unsafe_allow_html=True)
            analyze_btn = st.button("🔍 Analyze Voice", use_container_width=True, type="primary")
        else:
            st.error("❌ File empty. Try again.")
            analyze_btn = False
            file_bytes  = None
    else:
        st.markdown('<div class="info-box">👆 Upload a .wav or .mp3 voice file to begin.</div>', unsafe_allow_html=True)
        analyze_btn = False
        file_bytes  = None

with col2:
    st.markdown('<div class="section-header">📊 Analysis Results</div>', unsafe_allow_html=True)
    if uploaded_file is not None and analyze_btn and file_bytes:
        is_multi = "Multi" in mode
        with st.spinner("⏳ Loading model... (first time ~60 seconds)"):
            try:
                model  = load_multi_model() if is_multi else load_binary_model()
                scaler = load_scaler()
                model_ok = True
            except Exception as e:
                st.error(f"❌ Model load failed: {e}")
                model_ok = False
        if model_ok:
            with st.spinner("🔬 Analyzing..."):
                try:
                    signal     = preprocess_audio(file_bytes)
                    X_mfcc_in  = extract_mfcc(signal)[np.newaxis, ...]
                    X_hand_raw = extract_handcrafted(signal).reshape(1, -1)
                    X_hand_in  = scaler.transform(X_hand_raw).astype(np.float32)
                    proba      = model.predict([X_mfcc_in, X_hand_in], verbose=0)[0]
                    pred_idx   = int(np.argmax(proba))
                    conf       = float(proba[pred_idx])
                    classes    = MULTI_CLASSES if is_multi else BINARY_CLASSES
                    pred       = classes[pred_idx]
                    is_healthy = pred == "Healthy"
                    card_cls   = "healthy-card" if is_healthy else "disease-card"
                    icon       = "✅" if is_healthy else "⚠️"
                    lbl_cls    = "healthy-label" if pred=="Healthy" else ("laryngo-label" if pred=="Laryngozele" else "vox-label")
                    st.markdown(f"""
                    <div class="result-card {card_cls}">
                        <div style="font-size:3rem">{icon}</div>
                        <div class="pred-label {lbl_cls}">{pred}</div>
                        <div class="conf-text">Confidence: <b>{conf*100:.1f}%</b></div>
                        <div class="conf-text" style="font-size:0.85rem">{"Multi-class" if is_multi else "Binary"} mode</div>
                    </div>""", unsafe_allow_html=True)
                    if not is_healthy:
                        st.markdown('<div class="warning-box">⚕️ AI screening only. Please consult a doctor.</div>', unsafe_allow_html=True)
                    st.image(plot_confidence(proba, classes), use_container_width=True)
                    st.session_state['signal']   = signal
                    st.session_state['analyzed'] = True
                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")
                    st.exception(e)
    else:
        st.markdown('<div class="info-box" style="text-align:center;padding:2rem;">🎙️ Upload a voice file and click <b>Analyze Voice</b></div>', unsafe_allow_html=True)

if st.session_state.get('analyzed') and st.session_state.get('signal') is not None:
    signal = st.session_state['signal']
    st.markdown("---")
    st.markdown('<div class="section-header">🔬 Visualizations</div>', unsafe_allow_html=True)
    v1, v2 = st.columns(2)
    with v1:
        st.image(plot_waveform(signal), caption="Waveform", use_container_width=True)
    with v2:
        st.image(plot_mfcc(signal), caption="MFCC Spectrogram", use_container_width=True)
    hand = extract_handcrafted(signal)
    st.markdown('<div class="section-header">📈 Acoustic Features</div>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("🎵 MFCC Mean", f"{np.mean(hand[:40]):.2f}")
    f2.metric("📉 ZCR Mean",  f"{hand[386]:.4f}")
    f3.metric("🔊 RMS",       f"{hand[388]:.4f}")
    f4.metric("📊 Features",  str(HAND_DIM))

st.markdown("---")
st.markdown("<div style='text-align:center;color:#4a5568;font-size:0.85rem'>🧠 Dual-Branch CNN-BiLSTM · SVD Dataset · Streamlit</div>", unsafe_allow_html=True)
