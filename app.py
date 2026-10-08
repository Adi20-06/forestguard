import base64, datetime, html as _html, json, os, random, uuid
from functools import lru_cache
from urllib.parse import quote
from zoneinfo import ZoneInfo

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps

BASE = os.path.dirname(os.path.abspath(__file__))
HIST_FILE = f'{BASE}/history.json'
THUMB_DIR = f'{BASE}/history_imgs'
BG_DIR = f'{BASE}/backgrounds'
os.makedirs(THUMB_DIR, exist_ok=True)
IST = ZoneInfo("Asia/Kolkata")

st.set_page_config(page_title="ForestGuard", page_icon="🌲", layout="wide",
                   initial_sidebar_state="expanded")

# ------------------------------------------------------------------ model
@st.cache_resource
def load_everything():
    m = keras.models.load_model(f'{BASE}/fire_cnn_final.keras')
    with open(f'{BASE}/config.json') as f:
        c = json.load(f)
    return m, c

model, cfg = load_everything()
IMG_SIZE, THRESHOLD = cfg['img_size'], float(cfg['threshold'])

def predict(pil_img):
    arr = tf.image.resize(np.array(pil_img.convert('RGB')), (IMG_SIZE, IMG_SIZE))
    return float(model.predict(tf.expand_dims(arr, 0), verbose=0)[0][0])

# ------------------------------------------------------------------ state
ss = st.session_state
ss.setdefault('page', 'splash')
ss.setdefault('cam_open', False)
ss.setdefault('last', None)
ss.setdefault('last_img', None)

def go(p):
    ss.page = p
    ss.cam_open = False

def toggle_cam():
    ss.cam_open = not ss.cam_open

# ------------------------------------------------------------------ helpers
def H(s):
    # Streamlit's markdown parser ends an HTML block at a blank line and treats
    # 4-space indents as code, so strip both before rendering.
    st.markdown("\n".join(l.strip() for l in s.splitlines() if l.strip()), unsafe_allow_html=True)

def load_history():
    try:
        with open(HIST_FILE) as f:
            return json.load(f)
    except Exception:
        return []

def save_history(h):
    with open(HIST_FILE, 'w') as f:
        json.dump(h, f)

@lru_cache(maxsize=256)
def thumb_uri(name):
    try:
        with open(os.path.join(THUMB_DIR, name), 'rb') as f:
            return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
    except Exception:
        return ""

def level_of(prob, thr):
    if prob >= max(0.7, thr):
        return "ALERT"
    if prob >= thr:
        return "WARNING"
    return "SAFE"

def meta(level):
    return {"ALERT": ("FIRE DETECTED", "#dc2626", "🔥"),
            "WARNING": ("POSSIBLE FIRE / SMOKE", "#d97706", "⚠️"),
            "SAFE": ("NO FIRE DETECTED", "#1f7a47", "✅")}[level]

def conf_of(e):
    return (e['prob'] if e['level'] != 'SAFE' else 1 - e['prob']) * 100

def ago(iso):
    s = int((datetime.datetime.now(IST) - datetime.datetime.fromisoformat(iso)).total_seconds())
    if s < 60: return "just now"
    if s < 3600: return f"{s // 60} min ago"
    if s < 86400: return f"{s // 3600} hours ago"
    return f"{s // 86400} days ago"

def fmt(iso):
    return datetime.datetime.fromisoformat(iso).strftime("%d %b %Y • %I:%M %p")

def badge(level):
    cls = {"ALERT": "b-alert", "WARNING": "b-warn", "SAFE": "b-safe"}[level]
    return f'<span class="badge {cls}">{level}</span>'

def row(e, relative=True):
    title, _, _ = meta(e['level'])
    when = ago(e['ts']) if relative else fmt(e['ts'])
    return (f'<div class="row"><img src="{thumb_uri(e["thumb"])}"/>'
            f'<div class="rt"><div class="rtitle">{title.title()}</div>'
            f'<div class="rsub">{conf_of(e):.0f}% confidence · {when}</div></div>{badge(e["level"])}</div>')

# ------------------------------------------------------------------ artwork (original SVG)
PAL = {
    'sunset': dict(sky=['#9fc0dc', '#f3dcc0', '#f7c79a'], far='#b6c4c9', mid='#86a692', near='#4a7d5c',
                   p1='#2f6144', p2='#1b4d32', p3='#0e3321', sun='#fff4d6'),
    'forest': dict(sky=['#86ad9f', '#4f7e6b', '#2b5a45'], far='#6d9484', mid='#3f7059', near='#245a3f',
                   p1='#17482f', p2='#0f3724', p3='#0a2a1b', sun=None),
}
PAL.update({
    'dawn':   dict(sky=['#bcd3e0', '#e9e4cf', '#f4d9b0'], far='#aebfc4', mid='#7fa28d', near='#4a7d5c',
                   p1='#2f6144', p2='#1b4d32', p3='#0e3321', sun='#fff1c9'),
    'valley': dict(sky=['#9cc7dc', '#cfe3dc', '#e8eedd'], far='#a9c4c0', mid='#6f9f87', near='#3f7a58',
                   p1='#2a6446', p2='#1b4d32', p3='#0e3321', sun=None),
    'dusk':   dict(sky=['#6d7fa6', '#c9a3a0', '#f0b48a'], far='#8e91a8', mid='#5f7a79', near='#2f5d4a',
                   p1='#1f4a37', p2='#143626', p3='#0b261a', sun='#ffe0b0'),
    'ember':  dict(sky=['#7a5a52', '#e08a52', '#f6b56a'], far='#a07a6a', mid='#5f5a45', near='#3a4a35',
                   p1='#27382a', p2='#1a2a1f', p3='#101c15', sun='#ffd27a'),
})

def landscape(uid, W, H_, pal, cls=""):
    p = PAL[pal]
    r = random.Random(7)
    def poly(pts, color):
        return '<polygon points="' + " ".join(f"{x*W:.0f},{y*H_:.0f}" for x, y in pts) + f'" fill="{color}"/>'
    def pines(base, n, hmin, hmax, color):
        s = ""
        for _ in range(n):
            x = r.uniform(0, W); h = r.uniform(hmin, hmax) * H_; w = h * .36; b = base * H_
            s += (f'<polygon points="{x:.0f},{b-h:.0f} {x-w*.7:.0f},{b-h*.45:.0f} {x+w*.7:.0f},{b-h*.45:.0f}" fill="{color}"/>'
                  f'<polygon points="{x:.0f},{b-h*.72:.0f} {x-w:.0f},{b:.0f} {x+w:.0f},{b:.0f}" fill="{color}"/>')
        return s
    sun = ""
    if p['sun']:
        sun = (f'<circle cx="{W*.78:.0f}" cy="{H_*.36:.0f}" r="{H_*.075:.0f}" fill="{p["sun"]}" opacity=".25"/>'
               f'<circle cx="{W*.78:.0f}" cy="{H_*.36:.0f}" r="{H_*.035:.0f}" fill="{p["sun"]}" opacity=".95"/>')
    far = [(0,.50),(.12,.40),(.25,.48),(.4,.34),(.55,.46),(.7,.38),(.85,.47),(1,.40),(1,1),(0,1)]
    mid = [(0,.60),(.15,.52),(.3,.60),(.48,.50),(.65,.60),(.82,.52),(1,.58),(1,1),(0,1)]
    near = [(0,.70),(.2,.64),(.4,.70),(.6,.63),(.8,.69),(1,.65),(1,1),(0,1)]
    return (f'<svg xmlns="http://www.w3.org/2000/svg" class="{cls}" viewBox="0 0 {W} {H_}" '
            f'preserveAspectRatio="xMidYMax slice"><defs>'
            f'<linearGradient id="sky{uid}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0" stop-color="{p["sky"][0]}"/><stop offset=".55" stop-color="{p["sky"][1]}"/>'
            f'<stop offset="1" stop-color="{p["sky"][2]}"/></linearGradient>'
            f'<linearGradient id="fog{uid}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0" stop-color="#fff" stop-opacity="0"/><stop offset=".5" stop-color="#fff" stop-opacity=".3"/>'
            f'<stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient></defs>'
            f'<rect width="{W}" height="{H_}" fill="url(#sky{uid})"/>{sun}'
            f'{poly(far, p["far"])}{poly(mid, p["mid"])}'
            f'<rect y="{H_*.55:.0f}" width="{W}" height="{H_*.16:.0f}" fill="url(#fog{uid})"/>'
            f'{poly(near, p["near"])}{pines(.80, int(W/16), .09, .15, p["p1"])}'
            f'{pines(.90, int(W/20), .13, .21, p["p2"])}{pines(1.0, int(W/30), .20, .34, p["p3"])}</svg>')

def logo(color, size=56):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 64 64">'
            f'<path d="M32 4 L48 26 H40 L52 44 H38 L38 58 H26 V44 H12 L24 26 H16 Z" fill="{color}"/></svg>')

# ------------------------------------------------------------------ styling
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp, button, input, textarea { font-family: 'Inter', sans-serif; }
.stApp {
  background:
    radial-gradient(700px 500px at 8% 0%, rgba(52,211,153,.30), transparent 60%),
    radial-gradient(600px 500px at 100% 18%, rgba(250,204,120,.30), transparent 60%),
    radial-gradient(800px 600px at 60% 100%, rgba(31,122,71,.20), transparent 60%),
    #f1f0e8;
  background-attachment: fixed;
  color:#14281d;
}
header[data-testid="stHeader"] { background:transparent; }
[data-testid="stToolbar"], #MainMenu, footer { display:none; }
.block-container { padding-top:2.2rem; max-width:1000px; }

/* ---------- sidebar: full-bleed, big tabs ---------- */
section[data-testid="stSidebar"] {
  background:linear-gradient(180deg,#0e3a27,#07201a); border-right:none;
  width:300px !important; min-width:300px !important; }
section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] { display:none; }
section[data-testid="stSidebar"] [data-testid="stSidebarContent"],
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding:0 !important; }
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap:0 !important; }
section[data-testid="stSidebar"] .stElementContainer,
section[data-testid="stSidebar"] .stButton { width:100% !important; }
section[data-testid="stSidebar"] .stButton>button {
  width:100% !important; height:68px; border-radius:0 !important; border:none !important;
  background:transparent !important; box-shadow:none !important; outline:none !important;
  color:#cfe5d6 !important; justify-content:flex-start; padding:0 0 0 30px; transition:all .25s ease; }
section[data-testid="stSidebar"] .stButton>button div { justify-content:flex-start; }
section[data-testid="stSidebar"] .stButton>button p { font-size:1.12rem !important; font-weight:600; letter-spacing:.01em; }
section[data-testid="stSidebar"] .stButton>button:hover {
  background:rgba(255,255,255,.10) !important; color:#fff !important; padding-left:38px; }
.brand { display:flex; align-items:center; gap:12px; color:#fff; font-weight:800; font-size:2.05rem;
  letter-spacing:-.02em; padding:30px 28px 34px; }
.sway { display:inline-block; animation:sway 5s ease-in-out infinite; filter:drop-shadow(0 6px 10px rgba(0,0,0,.35)); }
@keyframes sway { 0%,100% { transform:perspective(500px) rotateY(-28deg); }
                  50% { transform:perspective(500px) rotateY(28deg); } }
.sysok { color:#9fc7ad; font-size:.8rem; padding:40px 30px; }
.dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:#34d399; margin-right:8px;
  box-shadow:0 0 0 0 rgba(52,211,153,.7); animation:pulse 2s infinite; }
@keyframes pulse { 70% { box-shadow:0 0 0 9px rgba(52,211,153,0); } 100% { box-shadow:0 0 0 0 rgba(52,211,153,0); } }

/* ---------- buttons ---------- */
.stButton>button, .stLinkButton>a { border-radius:14px; font-weight:600; padding:.65rem 1.1rem; transition:all .25s ease; }
.stButton button p { color:inherit; margin:0; }
.stButton>button[kind="primary"], .stButton>button[data-testid="stBaseButton-primary"] {
  background:linear-gradient(135deg,#238a52,#17653a); border:none; color:#fff;
  box-shadow:0 8px 22px rgba(23,101,58,.32); }
.stButton>button[kind="primary"]:hover, .stButton>button[data-testid="stBaseButton-primary"]:hover {
  transform:translateY(-2px); box-shadow:0 12px 28px rgba(23,101,58,.4); color:#fff; }
.stButton>button[kind="secondary"], .stButton>button[data-testid="stBaseButton-secondary"] {
  background:rgba(255,255,255,.6); border:1px solid rgba(255,255,255,.9); color:#14532d; backdrop-filter:blur(10px); }

/* ---------- glass ---------- */
.tile, .row, .card, [class*="st-key-card"] {
  background:linear-gradient(135deg, rgba(255,255,255,.74), rgba(255,255,255,.36));
  backdrop-filter:blur(18px) saturate(150%); -webkit-backdrop-filter:blur(18px) saturate(150%);
  border:1px solid rgba(255,255,255,.85);
  box-shadow:0 12px 36px rgba(15,61,38,.13), inset 0 1px 0 rgba(255,255,255,.95); }
[class*="st-key-card"] { border-radius:22px; padding:22px 24px; }
.card { border-radius:22px; padding:24px; animation:rise .7s cubic-bezier(.2,.8,.2,1) backwards; }

@keyframes rise { from { opacity:0; transform:perspective(900px) rotateX(-14deg) translateY(22px); } }
@keyframes floaty { 50% { translate:0 -9px; } }

.ptitle { font-size:1.7rem; font-weight:800; color:#0f3d26; }
.psub { color:#5f7467; font-size:.95rem; margin:2px 0 20px; }
.ctitle { font-weight:700; margin-bottom:12px; color:#0f3d26; }
.gt { font-size:1.7rem; font-weight:800; color:#0f3d26; }
.gs { color:#5f7467; font-size:.95rem; margin-bottom:18px; }

/* ---------- hero ---------- */
.banner { position:relative; height:220px; border-radius:26px; overflow:hidden; margin-bottom:16px;
  box-shadow:0 22px 50px rgba(15,61,38,.28); animation:rise .8s cubic-bezier(.2,.8,.2,1) backwards; }
.bsvg { position:absolute; inset:0; width:100%; height:100%; animation:drift 22s ease-in-out infinite alternate; }
@keyframes drift { from { transform:scale(1); } to { transform:scale(1.1) translateX(-14px); } }
.mist { position:absolute; left:-30%; bottom:18%; width:160%; height:70px;
  background:linear-gradient(90deg, transparent, rgba(255,255,255,.35), transparent);
  filter:blur(14px); animation:mist 14s linear infinite; }
@keyframes mist { from { transform:translateX(-10%); } to { transform:translateX(22%); } }
.bshade { position:absolute; inset:0; background:linear-gradient(90deg, rgba(6,38,24,.85), rgba(6,38,24,.1)); }
.bcontent { position:relative; padding:30px 32px; color:#fff; max-width:480px; }
.bt { font-size:1.6rem; font-weight:800; }
.bd { opacity:.92; margin-top:8px; font-size:.97rem; }
.st-key-detect_now button { padding:.7rem 1.6rem; }

/* ---------- tiles / rows ---------- */
.tiles { display:grid; grid-template-columns:repeat(3,1fr); gap:18px; margin:24px 0; perspective:900px; }
.tile { display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center;
  min-height:130px; border-radius:22px; padding:18px;
  transition:transform .4s cubic-bezier(.2,.8,.2,1), box-shadow .4s; }
.tile:hover { transform:perspective(700px) rotateX(6deg) translateY(-6px); box-shadow:0 20px 44px rgba(15,61,38,.22); }
.ti { font-size:1.6rem; } .tv { font-size:1.9rem; font-weight:800; color:#0f3d26; margin-top:2px; }
.tl { color:#5f7467; font-size:.78rem; letter-spacing:.04em; text-transform:uppercase; }

.row { display:flex; align-items:center; gap:16px; border-radius:18px; padding:12px 16px; margin-bottom:12px;
  animation:rise .6s cubic-bezier(.2,.8,.2,1) backwards; transition:transform .35s ease; }
.row:nth-child(2) { animation-delay:.08s; } .row:nth-child(3) { animation-delay:.16s; }
.row:nth-child(4) { animation-delay:.24s; } .row:nth-child(5) { animation-delay:.32s; }
.row:hover { transform:perspective(900px) rotateX(2deg) translateX(6px) scale(1.01); }
.row img { width:64px; height:56px; border-radius:12px; object-fit:cover; background:#e5ebe6; }
.rt { flex:1; } .rtitle { font-weight:600; } .rsub { color:#5f7467; font-size:.8rem; }
[class*="st-key-card"] .row { background:rgba(255,255,255,.62); backdrop-filter:none; -webkit-backdrop-filter:none;
  box-shadow:0 4px 14px rgba(15,61,38,.08); }
[class*="st-key-card"] .row:last-child { margin-bottom:0; }
.badge { padding:6px 15px; border-radius:99px; font-size:.7rem; font-weight:700; letter-spacing:.07em; }
.b-alert { background:#fde2e2; color:#c62828; } .b-warn { background:#fdecc8; color:#b7791f; } .b-safe { background:#d7efe0; color:#1f7a47; }
.st-key-viewall { display:flex; justify-content:flex-end; }
.st-key-viewall button { background:transparent !important; border:none !important; box-shadow:none !important; color:#14532d !important; }

/* ---------- detect + result ---------- */
.or { text-align:center; color:#8a9a90; font-size:.8rem; margin:10px 0; }
[data-testid="stFileUploaderDropzone"] { background:rgba(255,255,255,.5); border:1.5px dashed #6fb48a; border-radius:16px; }
[data-testid="stImage"] img { border-radius:16px; box-shadow:0 14px 34px rgba(15,61,38,.22); }
.ph { height:200px; border:1.5px dashed #a7c9b4; border-radius:16px; display:grid; place-items:center;
  text-align:center; color:#7d8f84; margin-bottom:14px; font-size:.9rem; background:rgba(255,255,255,.35); }
.ph b { font-size:2.2rem; display:block; animation:floaty 4s ease-in-out infinite; }
.vh { display:flex; align-items:center; gap:14px; }
.vic { width:52px; height:52px; border-radius:50%; display:grid; place-items:center; font-size:1.5rem;
  background:rgba(255,255,255,.8); box-shadow:0 8px 20px rgba(0,0,0,.1); animation:floaty 3.5s ease-in-out infinite; }
.vt { font-weight:800; font-size:1.2rem; }
.cs { color:#5f7467; font-size:.8rem; margin-top:18px; }
.cv { font-size:2.1rem; font-weight:800; }
.bar { height:9px; border-radius:99px; background:rgba(0,0,0,.08); overflow:hidden; margin-top:6px; }
.fill { height:100%; border-radius:99px; }
.info { display:flex; gap:12px; margin-top:16px; align-items:flex-start; }
.ik { color:#5f7467; font-size:.74rem; } .iv { font-weight:600; font-size:.92rem; word-break:break-word; }
.alertbox { border-radius:16px; padding:16px 18px; margin-top:18px; display:flex; gap:14px; align-items:center;
  backdrop-filter:blur(12px); animation:rise .8s cubic-bezier(.2,.8,.2,1) backwards; }
.alertbox b { display:block; } .alertbox small { opacity:.85; }
.ab-fire { background:rgba(253,232,230,.8); color:#b42318; } .ab-safe { background:rgba(227,243,233,.8); color:#1f7a47; }
.note { color:#7d8f84; font-size:.76rem; margin-top:14px; }
.empty { text-align:center; color:#7d8f84; padding:50px 0; }
</style>
"""
H(CSS)

# ------------------------------------------------------------------ page backgrounds
@lru_cache(maxsize=16)
def bg_uri(page, pal):
    # Put your own photo at <fire_project>/backgrounds/<page>.jpg (home, detect, result, history)
    # and it is used instead of the drawn landscape.
    for ext, mime in (("jpg", "jpeg"), ("jpeg", "jpeg"), ("png", "png")):
        f = f'{BG_DIR}/{page}.{ext}'
        if os.path.exists(f):
            with open(f, 'rb') as fh:
                return f"data:image/{mime};base64," + base64.b64encode(fh.read()).decode()
    svg = landscape("bg" + page, 1600, 900, pal)
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()

def page_bg(page):
    pal = {"home": "dawn", "detect": "valley", "history": "dusk"}.get(page, "valley")
    if page == "result":
        pal = "ember" if (ss.last and ss.last["level"] != "SAFE") else "valley"
    uri = bg_uri(page, pal)
    H(f"""<style>
    .stApp {{ background:
        linear-gradient(180deg, rgba(243,242,236,.78) 0%, rgba(243,242,236,.38) 45%, rgba(243,242,236,.62) 100%),
        url("{uri}") center bottom / cover no-repeat fixed; }}
    </style>""")

# ------------------------------------------------------------------ sidebar
def sidebar():
    active = "detect" if ss.page == "result" else ss.page
    with st.sidebar:
        H(f'<div class="brand"><span class="sway">{logo("#ffffff", 46)}</span><span>ForestGuard</span></div>')
        for k, icon, label in [("home", "🏠", "Home"), ("detect", "🔍", "Detect Fire"), ("history", "🕘", "History")]:
            st.button(f"{icon}   {label}", key=f"nav_{k}", on_click=go, args=(k,))
        H('<div class="sysok"><span class="dot"></span>System online</div>')
    H(f'<style>.st-key-nav_{active} button {{ background:rgba(255,255,255,.16) !important; color:#fff !important; '
      f'box-shadow:inset 5px 0 0 #34d399 !important; }}</style>')

# ------------------------------------------------------------------ pages
def page_splash():
    svg = landscape("s", 1600, 800, "sunset")
    uri = base64.b64encode(svg.encode()).decode()
    H(f"""<style>
    section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {{ display:none; }}
    .stApp {{ background:url("data:image/svg+xml;base64,{uri}") center bottom / cover no-repeat; }}
    .block-container {{ max-width:460px; text-align:center; }}
    </style>""")
    H(f'<div style="padding-top:6vh"><span class="sway" style="filter:drop-shadow(0 10px 14px rgba(15,61,38,.35))">{logo("#0f3d26", 72)}</span>'
      '<div style="font-size:2.8rem;font-weight:800;color:#0f3d26;margin-top:6px">ForestGuard</div>'
      '<div style="font-size:1.35rem;font-weight:600;color:#14281d;margin-top:18px;line-height:1.3">Smarter Detection.<br>Safer Forests.</div>'
      '<div style="color:#3d5547;margin-top:10px">AI-powered early forest fire detection<br>and alert system.</div>'
      '<div style="height:36vh"></div></div>')
    st.button("Get Started  →", type="primary", use_container_width=True, key="start", on_click=go, args=("home",))

def page_home():
    hist = load_history()
    h = datetime.datetime.now(IST).hour
    greet = "Good Morning" if h < 12 else ("Good Afternoon" if h < 17 else "Good Evening")
    H(f'<div class="gt">{greet},</div><div class="gs">Let’s keep the forests safe today.</div>')
    H('<div class="banner">' + landscape("b", 800, 300, "forest", "bsvg") + '<div class="mist"></div><div class="bshade"></div>'
      '<div class="bcontent"><div class="bt">AI Forest Fire Detection</div>'
      '<div class="bd">Upload an image or use your camera to check for fire or smoke.</div></div></div>')
    st.button("Detect Now  →", type="primary", key="detect_now", on_click=go, args=("detect",))

    week_ago = datetime.datetime.now(IST) - datetime.timedelta(days=7)
    alerts = sum(1 for e in hist if e['level'] != 'SAFE' and datetime.datetime.fromisoformat(e['ts']) >= week_ago)
    if hist:
        lv = hist[0]['level']
        last_icon = meta(lv)[2]
        last_val = {"ALERT": "Alert", "WARNING": "Warning", "SAFE": "Clear"}[lv]
    else:
        last_icon, last_val = "🌲", "—"
    def tile(i, v, l):
        return f'<div class="tile"><div class="ti">{i}</div><div class="tv">{v}</div><div class="tl">{l}</div></div>'
    H('<div class="tiles">' + tile("🔥", alerts, "Alerts · 7 days") + tile("🔍", len(hist), "Images analysed")
      + tile(last_icon, last_val, "Last scan") + '</div>')

    with st.container(key="card_recent"):
        c1, c2 = st.columns([4, 1])
        with c1:
            H('<div class="ctitle" style="font-size:1.15rem;margin:6px 0 0">Recent Activity</div>')
        with c2:
            st.button("View All →", key="viewall", on_click=go, args=("history",))
        if hist:
            H("".join(row(e) for e in hist[:3]))
        else:
            H('<div class="empty">No detections yet.</div>')

def page_detect():
    H('<div class="ptitle">Detect Fire</div>'
      '<div class="psub">Add a forest image, check the preview, then run the analysis.</div>')
    c1, c2 = st.columns(2, gap="large")
    with c1:
        with st.container(key="card_up"):
            H('<div class="ctitle">1 · Select image</div>')
            up = st.file_uploader("Upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            H('<div class="or">or</div>')
            st.button("✖  Close Camera" if ss.cam_open else "📷  Open Camera", key="cam_toggle",
                      on_click=toggle_cam, use_container_width=True)
            cam = st.camera_input("Camera", label_visibility="collapsed") if ss.cam_open else None
    src = up or cam
    img = ImageOps.exif_transpose(Image.open(src)).convert("RGB") if src else None

    with c2:
        with st.container(key="card_prev"):
            H('<div class="ctitle">2 · Preview &amp; details</div>')
            if img is not None:
                st.image(img, use_container_width=True)
            else:
                H('<div class="ph"><div><b>🌲</b>No image selected</div></div>')
            loc = st.text_input("Location / GPS (optional)", placeholder="12.9716, 77.5946")
            cid = st.text_input("Camera ID (optional)", placeholder="CAM-07")
            run = st.button("Analyze image", type="primary", disabled=img is None, use_container_width=True)

    if run and img is not None:
        with st.spinner("Analysing image..."):
            prob = predict(img)
        eid = uuid.uuid4().hex[:10]
        tname = f"{eid}.jpg"
        t = img.copy(); t.thumbnail((240, 240)); t.save(os.path.join(THUMB_DIR, tname), "JPEG", quality=80)
        entry = dict(id=eid, ts=datetime.datetime.now(IST).isoformat(), prob=prob, thr=THRESHOLD,
                     level=level_of(prob, THRESHOLD), thumb=tname, location=loc.strip(), camera=cid.strip())
        hist = load_history(); hist.insert(0, entry); save_history(hist)
        disp = img.copy(); disp.thumbnail((1000, 1000))
        ss.last, ss.last_img = entry, disp
        go("result")
        st.rerun()

def page_result():
    e = ss.last
    if e is None:
        go("detect"); st.rerun()
    title, color, icon = meta(e['level'])
    fire = e['level'] != 'SAFE'
    esc = _html.escape
    H('<div class="ptitle">Analysis Result</div><div class="psub">&nbsp;</div>')
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.image(ss.last_img, use_container_width=True)
    with c2:
        cam_html = (f'<div class="info"><span>📷</span><div><div class="ik">Camera ID</div>'
                    f'<div class="iv">{esc(e["camera"])}</div></div></div>') if e["camera"] else ""
        H(f'<div class="card"><div class="vh"><div class="vic">{icon}</div>'
          f'<div class="vt" style="color:{color}">{title}</div></div>'
          f'<div class="cs">Confidence Score</div><div class="cv">{conf_of(e):.1f}%</div>'
          f'<div class="bar"><div class="fill" style="width:{conf_of(e):.1f}%;background:{color}"></div></div>'
          f'<div class="cs" style="margin-top:10px">Fire probability: <b>{e["prob"]*100:.1f}%</b></div>'
          f'<div class="info"><span>📍</span><div><div class="ik">Location</div>'
          f'<div class="iv">{esc(e["location"]) or "Not provided"}</div></div></div>'
          f'<div class="info"><span>🕒</span><div><div class="ik">Timestamp</div>'
          f'<div class="iv">{fmt(e["ts"])} IST</div></div></div>{cam_html}</div>')
    if fire:
        H('<div class="alertbox ab-fire"><span style="font-size:1.5rem">🔔</span><div><b>Alert triggered!</b>'
          '<small>Notify the nearest forest ranger and verify the location.</small></div></div>')
    else:
        H('<div class="alertbox ab-safe"><span style="font-size:1.5rem">🌲</span><div><b>All clear</b>'
          '<small>No fire or smoke found in this image. Keep monitoring.</small></div></div>')
    H('<div class="note">Very small or distant fires and smoke may be missed. '
      'A “No Fire” result is not a guarantee that no fire is present.</div>')
    st.write("")
    b1, b2 = st.columns(2)
    with b1:
        if fire:
            url = ("https://www.google.com/maps/search/?api=1&query=" + quote(e["location"])) if e["location"] else "https://maps.google.com"
            st.link_button("📍  View on Map", url, type="primary", use_container_width=True, disabled=not e["location"])
    with b2:
        st.button("↻  Analyze Another Image", key="again", on_click=go, args=("detect",), use_container_width=True)

def clear_history():
    save_history([])
    thumb_uri.cache_clear()

def page_history():
    H('<div class="ptitle">Detection History</div><div class="psub">Past detections and alerts.</div>')
    c1, c2 = st.columns([5, 1])
    opts = ["All", "Alert", "Warning", "Safe"]
    with c1:
        if hasattr(st, "segmented_control"):
            f = st.segmented_control("Filter", opts, default="All", label_visibility="collapsed") or "All"
        else:
            f = st.radio("Filter", opts, horizontal=True, label_visibility="collapsed")
    with c2:
        st.button("🗑 Clear", key="clear_hist", on_click=clear_history, use_container_width=True)
    hist = load_history()
    if f != "All":
        hist = [e for e in hist if e['level'] == f.upper()]
    if hist:
        H("".join(row(e, relative=False) for e in hist[:50]))
    else:
        H('<div class="empty">Nothing here yet.</div>')

# ------------------------------------------------------------------ router
if ss.page != "splash":
    page_bg(ss.page)
    sidebar()
{"splash": page_splash, "home": page_home, "detect": page_detect,
 "result": page_result, "history": page_history}.get(ss.page, page_home)()
