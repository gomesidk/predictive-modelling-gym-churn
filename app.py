import pandas as pd
import numpy as np
import streamlit as st
import joblib
import base64
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="Element",
    page_icon="icons/gym_icon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
            
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined');
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
            
:root {
    --bg:       #f8f9fb;
    --bg2:      #ffffff;
    --bg3:      #f1f3f7;
    --border:   #e2e6ed;
    --accent:   #2563eb;
    --accent2:  #4f46e5;
    --red:      #dc2626;
    --orange:   #ea580c;
    --green:    #16a34a;
    --text:     #0f172a;
    --muted:    #64748b;
    --muted2:   #94a3b8;
    --sidebar:  #ffffff;
}

* {
    box-sizing: border-box;
}

/* fonte principal */
html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif !important;
}
            
.material-symbols-outlined {
    font-family: 'Material Symbols Outlined' !important;
    font-weight: normal !important;
    font-style: normal !important;
    font-size: 24px !important;
    line-height: 1 !important;
    display: inline-block !important;
    white-space: nowrap !important;
}

/* ── App background ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
[data-testid="stHeader"]          { background: transparent !important; }
[data-testid="stMainBlockContainer"] { background: var(--bg) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--sidebar) !important;
    width: 340px !important;
    min-width: 340px !important;
    max-width: 340px !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Sidebar brand ── */
.sb-brand {
    margin: 0px 5px 4px 5px;
}
.sb-brand h2 {
    font-size: 24px;
    font-weight: 700;
    color: var(--text) !important;
    letter-spacing: -0.5px;
}
.sb-brand p {
    font-size: 11px;
    color: var(--muted) !important;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* ── Sidebar section headers ── */
.sb-section {
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 0 5px 6px 5px;
}
.sb-section-line {
    flex: 1;
    height: 1px;
    background: var(--border);
}
.sb-section-label {
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: var(--accent) !important;
    white-space: nowrap;
}
            
[data-testid="stSidebar"] .stSlider,
[data-testid="stSidebar"] .stSelectbox,
[data-testid="stSidebar"] .stNumberInput {
    margin: 1px 8px 1px 8px !important;
    padding: 0 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"],
[data-testid="stSidebar"] [data-baseweb="input"],
[data-testid="stSidebar"] [data-baseweb="slider"] {
    margin-left: 0 !important;
    margin-right: 0 !important;
}

/* ── Predict button ── */
            
/* botão principal */
div[data-testid="stButton"] > button[kind="primary"] {
    background: var(--accent) !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    letter-spacing: 0.3px !important;
    padding: 2px 0 !important;
    box-shadow: 0 2px 12px rgba(37,99,235,.3) !important;
    transition: all .15s !important;

    color: white !important;
}

div[data-testid="stButton"] > button[kind="primary"] * {
    color: white !important;
}
        
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #1d4ed8 !important;
    box-shadow: 0 4px 20px rgba(37,99,235,.4) !important;
}

/* ── Tabs: just a clean underline row, no box ── */
[data-testid="stTabs"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

[data-testid="stTabs"] > div:first-child {
    border-bottom: 2px solid var(--border) !important;
    padding-bottom: 20px !important;
    background: transparent !important;
}
[data-testid="stTabs"] [data-testid="stTab"] {
    font-size: 14px !important;
    font-weight: 600 !important;
    color: var(--muted) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -2px !important;
    padding: 10px 18px !important;
    border-radius: 0 !important;
}
[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}

[data-testid="stTabsContent"] {
    padding-top: 28px !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* ── Cards ── */
.card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 22px 24px;
    margin-bottom: 14px;
}
.card-sm {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 10px;
}

/* ── KPI ── */
.kpi-label { font-size: 11px; font-weight: 600; letter-spacing: 0.8px; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }
.kpi-value { font-size: 36px; font-weight: 800; line-height: 1; margin-bottom: 4px; }
.kpi-sub   { font-size: 11px; color: var(--muted); }
.c-blue   { color: #2563eb; }
.c-red    { color: #dc2626; }
.c-orange { color: #ea580c; }
.c-green  { color: #16a34a; }

/* ── Risk ring ── */
.risk-ring {
    text-align: center;
    padding: 50px 28px;
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 14px;
}
.risk-pct { font-size: 60px; font-weight: 800; line-height: 1; }
.risk-lbl { font-size: 16px; font-weight: 700; margin-top: 8px; }
.risk-sub { font-size: 12px; color: var(--muted); margin-top: 6px; }

/* ── Profile rows ── */
.prof-row { display: flex; justify-content: space-between; padding: 9px 0; border-bottom: 1px solid var(--border); font-size: 13px; }
.prof-row:last-child {
    border-bottom: none;
}
.prof-key { color: var(--muted); font-weight: 500; }
.prof-val { color: var(--text); font-weight: 600; }

/* ── Rec card ── */
.rec-card { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 20px 22px; margin-bottom: 12px; }
.rec-tag  { display: inline-block; font-size: 10px; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase; padding: 3px 10px; border-radius: 20px; margin-bottom: 10px; }
.rec-card h4 { font-size: 14px; font-weight: 700; margin: 0 0 5px; color: var(--text); }
.rec-card p  { font-size: 13px; color: var(--muted); margin: 0; line-height: 1.65; }

/* ── SHAP ── */
.shap-row { display: flex; align-items: center; gap: 12px; margin-bottom: 6px; padding: 7px 0; border-bottom: 1px solid var(--border); }
.shap-name { font-size: 12px; color: var(--muted); width: 240px; flex-shrink: 0; font-weight: 500; }
.shap-bar-wrap { flex: 1; height: 14px; background: var(--bg3); border-radius: 4px; overflow: hidden; }
.shap-bar-pos { height: 100%; background: linear-gradient(90deg,#ef4444,#f97316); border-radius: 4px; }
.shap-bar-neg { height: 100%; background: linear-gradient(90deg,#3b82f6,#6366f1); border-radius: 4px; }
.shap-val { font-size: 11px; font-weight: 600; color: var(--muted); width: 55px; text-align: right; }

/* ── Empty state ── */
.empty-state { text-align: center; padding: 80px 40px; }
.empty-state .icon {width: 90px; height: 90px; object-fit: contain; margin-bottom: 20px; opacity: 0.95;}  
.empty-state h3 { font-size: 17px; font-weight: 700; color: var(--text); margin-bottom: 8px; }
.empty-state p  { font-size: 14px; color: var(--muted); line-height: 1.65; }


#MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
ICON_DIR = Path(__file__).parent / "icons"	
MODEL_PATH  = Path(__file__).parent / "models/final_xgb_model.pkl"
SCALER_PATH = Path(__file__).parent / "models/scaler.pkl"
EXPECTED_FEATURES = [
    "gender","near_location","partner","promo_friends","phone",
    "contract_period","group_visits","age","avg_additional_charges_total",
    "month_to_end_contract","lifetime","avg_class_frequency_total",
    "avg_class_frequency_current_month",
]
FEATURE_LABELS = {
    "gender":                            "Gender",
    "near_location":                     "Close to the gym",
    "partner":                           "Partner company employee",
    "promo_friends":                     "Signed up via friend promo",
    "phone":                             "Registered phone number",
    "contract_period":                   "Contract duration (months)",
    "group_visits":                      "Group classes",
    "age":                               "Age",
    "avg_additional_charges_total":      "Avg. additional expenditure (€)",
    "month_to_end_contract":             "Months until contract end",
    "lifetime":                          "Time as member (months)",
    "avg_class_frequency_total":         "Avg. historical attendance (×/week)",
    "avg_class_frequency_current_month": "Weekly attendance — current month",
}

# ── Model helpers ──────────────────────────────────────────────────────────────
def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("gym_churn_us.csv")
    except:
        return None

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists(): return None
    try: return joblib.load(MODEL_PATH)
    except: return None

@st.cache_resource
def load_scaler():
    if not SCALER_PATH.exists(): return None
    try: return joblib.load(SCALER_PATH)
    except: return None

def classify_risk(score):
    if score >= 60: return "High Risk",   "high",   "#dc2626"
    if score >= 35: return "Medium Risk", "medium", "#ea580c"
    return               "Low Risk",    "low",    "#16a34a"

def predict_score(inp, model, scaler):
    df = pd.DataFrame([inp], columns=EXPECTED_FEATURES)
    sc = scaler.transform(df)
    if not isinstance(sc, pd.DataFrame):
        sc = pd.DataFrame(sc, columns=EXPECTED_FEATURES)
    try:    return float(model.predict_proba(sc)[0, 1]) * 100
    except: return float(model.predict(sc)[0]) * 100

@st.cache_resource
def load_shap_explainer(_model, _scaler):
    try:
        import shap
        # Usa TreeExplainer — otimizado para XGBoost
        explainer = shap.TreeExplainer(_model)
        return explainer
    except:
        return None

def get_shap_values(inp, model, scaler, explainer):
    try:
        import shap
        df = pd.DataFrame([inp], columns=EXPECTED_FEATURES)
        sc = scaler.transform(df)
        if not isinstance(sc, pd.DataFrame):
            sc = pd.DataFrame(sc, columns=EXPECTED_FEATURES)
        shap_vals = explainer.shap_values(sc)
        # Para classificação binária o XGBoost devolve array direto
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        return dict(zip(EXPECTED_FEATURES, shap_vals[0]))
    except:
        return get_shap_approx(inp, model, scaler)  # fallback
    
def get_shap_approx(inp, model, scaler):
    try:
        importances = model.feature_importances_
        base = np.array([float(inp[f]) for f in EXPECTED_FEATURES])
        norm = (base - base.mean()) / (base.std() + 1e-8)
        return dict(zip(EXPECTED_FEATURES, importances * norm))
    except:
        return {f: 0.0 for f in EXPECTED_FEATURES}

def generate_explanation(inp, score):
    reasons = []
    if inp["avg_class_frequency_current_month"] < 1.5:
        reasons.append("very low training frequency this month")
    if inp["month_to_end_contract"] <= 1:
        reasons.append("contract ending within 1 month")
    if inp["lifetime"] < 3:
        reasons.append("new member with less than 3 months of history")
    if inp["partner"] == 0:
        reasons.append("not associated with a partner company")
    if inp["promo_friends"] == 0:
        reasons.append("did not join through a friend referral")
    if inp["near_location"] == 0:
        reasons.append("does not live near the gym")
    if inp["group_visits"] == 0:
        reasons.append("does not attend group classes")
    if not reasons:
        return "No critical risk factors identified. Member shows a stable engagement profile."
    return "Key risk factors: " + "; ".join(reasons) + "."

def get_recommendations(risk_class, inp):
    recs = []

    # ── Flags baseadas nos valores reais do cliente ──
    low_attendance     = inp["avg_class_frequency_current_month"] < 1.5
    very_low_attendance = inp["avg_class_frequency_current_month"] < 0.5
    no_group           = inp["group_visits"] == 0
    contract_ending    = inp["month_to_end_contract"] <= 1
    contract_soon      = inp["month_to_end_contract"] <= 3
    new_member         = inp["lifetime"] < 3
    long_member        = inp["lifetime"] >= 12
    no_partner         = inp["partner"] == 0
    no_promo           = inp["promo_friends"] == 0
    no_phone           = inp["phone"] == 0
    far_from_gym       = inp["near_location"] == 0
    low_spend          = inp["avg_additional_charges_total"] < 50
    high_spend         = inp["avg_additional_charges_total"] >= 200
    short_contract     = inp["contract_period"] == 1

    if risk_class == "high":
        # Sempre presente para high risk
        recs.append({
            "priority": 1,
            "tag": "URGENT",
            "color": "#dc2626",
            "icon": f'<img src="data:image/svg+xml;base64,{siren}">',
            "title": "Immediate Personal Outreach",
            "text": (
                f"Contact this member this week via phone or SMS."
                f"{' No phone on record — reach out by email or in-person.' if no_phone else ''}"
                " Offer an early renewal with a 20–25% discount or a complimentary personal training session."
            ),
        })
        if contract_ending:
            recs.append({
                "priority": 1,
                "tag": "URGENT",
                "color": "#dc2626",
                "icon": f'<img src="data:image/svg+xml;base64,{clipboard}">',
                "title": "Contract Expires This Month",
                "text": (
                    "The contract ends within 1 month. Trigger an immediate renewal campaign. "
                    "Offer a loyalty bonus — e.g. 1 free month — for signing a 6+ month contract today."
                ),
            })
        elif contract_soon:
            recs.append({
                "priority": 2,
                "tag": "RETENTION",
                "color": "#ea580c",
                "icon": f'<img src="data:image/svg+xml;base64,{clipboard2}">',
                "title": "Contract Renewal Campaign",
                f"text": (
                    f"Contract ends in {int(inp['month_to_end_contract'])} month(s). "
                    "Launch a renewal campaign with a loyalty discount before it expires."
                ),
            })
        if very_low_attendance:
            recs.append({
                "priority": 1,
                "tag": "URGENT",
                "color": "#dc2626",
                "icon": f'<img src="data:image/svg+xml;base64,{down}">',
                "title": "Critical Drop in Attendance",
                "text": (
                    f"This member attended less than once per week this month "
                    f"({inp['avg_class_frequency_current_month']:.1f}×/week). "
                    "This is a strong churn signal. Assign a staff member to check in personally."
                ),
            })
        elif low_attendance:
            recs.append({
                "priority": 2,
                "tag": "ENGAGEMENT",
                "color": "#ea580c",
                "icon": f'<img src="data:image/svg+xml;base64,{down2}">',
                "title": "Low Attendance This Month",
                "text": (
                    f"Attendance has dropped to {inp['avg_class_frequency_current_month']:.1f}×/week. "
                    "Send a personalised message acknowledging the absence and offering a re-engagement incentive."
                ),
            })
        if no_group:
            recs.append({
                "priority": 2,
                "tag": "ENGAGEMENT",
                "color": "#ea580c",
                "icon": f'<img src="data:image/svg+xml;base64,{group2}">',
                "title": "Invite to a Group Class",
                "text": (
                    "This member has never attended group classes. Members who join group sessions churn at significantly lower rates. "
                    "Offer a free trial class this week."
                ),
            })
        if short_contract:
            recs.append({
                "priority": 2,
                "tag": "RETENTION", 
                "color": "#ea580c",
                "icon": f'<img src="data:image/svg+xml;base64,{file}">',
                "title": "Upgrade to Longer Contract",
                "text": (
                    "Member is on a month-to-month contract, which correlates strongly with higher churn. "
                    "Incentivise an upgrade to a 6 or 12-month plan with a discount."
                ),
            })

    elif risk_class == "medium":
        recs.append({
            "priority": 2,
            "tag": "PREVENTIVE",
            "color": "#ea580c",
            "icon": f'<img src="data:image/svg+xml;base64,{alert}">',
            "title": "Re-engagement Campaign",
            "text": (
                "Send a personalised email or SMS campaign. "
                "Highlight new classes, upcoming challenges, or gym improvements to rekindle interest."
            ),
        })
        recs.append({
            "priority": 2,
            "tag": "MONITOR",
            "color": "#ea580c",
            "icon": f'<img src="data:image/svg+xml;base64,{chart}">',
            "title": "Attendance Monitoring",
            "text": (
                f"Current attendance is {inp['avg_class_frequency_current_month']:.1f}×/week. "
                "Flag for weekly monitoring — if frequency drops below 1×/week, escalate to direct contact."
            ),
        })
        if contract_soon and not contract_ending:
            recs.append({
                "priority": 2,
                "tag": "RETENTION",
                "color": "#ea580c",
                "icon": f'<img src="data:image/svg+xml;base64,{clipboard2}">',
                "title": "Proactive Renewal Offer",
                "text": (
                    f"Contract ends in {int(inp['month_to_end_contract'])} month(s). "
                    "A proactive renewal offer now, before the member starts considering alternatives, significantly improves retention."
                ),
            })
        if no_group:
            recs.append({
                "priority": 3,
                "tag": "ENGAGEMENT",
                "color": "#ea580c",
                "icon": f'<img src="data:image/svg+xml;base64,{group2}">',
                "title": "Group Class Suggestion",
                "text": (
                    "Member has not tried group classes yet. "
                    "A targeted invitation to a class that matches their profile can boost engagement and loyalty."
                ),
            })
        if no_partner and no_promo:
            recs.append({
                "priority": 3,
                "tag": "COMMUNITY",
                "color": "#64748b",
                "icon": f'<img src="data:image/svg+xml;base64,{deal}">',
                "title": "Referral Programme",
                "text": (
                    "This member joined independently with no company or friend connection. "
                    "Enrolling them in a referral programme builds community ties and increases retention."
                ),
            })

    else:  # low risk
        recs.append({
            "priority": 3,
            "tag": "LOYALTY",
            "color": "#16a34a",
            "icon": f'<img src="data:image/svg+xml;base64,{check}">',
            "title": "Enrol in Loyalty Programme",
            "text": (
                "Member shows a stable engagement profile. "
                "Enrol in the referral programme — offer a free month for each friend successfully referred."
            ),
        })
        if high_spend:
            recs.append({
                "priority": 3,
                "tag": "UPSELL",
                "color": "#16a34a",
                "icon": f'<img src="data:image/svg+xml;base64,{diamond}">',
                "title": "Premium Membership Upsell",
                "text": (
                    f"This member already spends €{inp['avg_additional_charges_total']:.0f} in additional services. "
                    "They are an ideal candidate for a premium membership or personal training package."
                ),
            })
        elif not high_spend and not low_spend:
            recs.append({
                "priority": 3,
                "tag": "UPSELL",
                "color": "#16a34a",
                "icon": f'<img src="data:image/svg+xml;base64,{target3}">',
                "title": "Cross-sell Opportunity",
                "text": (
                    "High engagement makes this member a good candidate for add-on services "
                    "such as nutrition coaching, personal training, or merchandise."
                ),
            })
        if long_member:
            recs.append({
                "priority": 3,
                "tag": "LOYALTY",
                "color": "#16a34a",
                "icon": f'<img src="data:image/svg+xml;base64,{trophy}">',
                "title": "Long-term Member Recognition",
                "text": (
                    f"This member has been with the gym for {int(inp['lifetime'])} months. "
                    "A personalised thank-you — a free service, badge, or public recognition — "
                    "reinforces loyalty and word-of-mouth."
                ),
            })
        if no_group:
            recs.append({
                "priority": 3,
                "tag": "ENGAGEMENT",
                "color": "#16a34a",
                "icon": f'<img src="data:image/svg+xml;base64,{group3}">',
                "title": "Introduce to Group Classes",
                "text": (
                    "Member hasn't tried group classes. Introducing them now deepens their connection "
                    "to the gym community, further reducing future churn risk."
                ),
            })

    # Ordena por prioridade
    recs.sort(key=lambda x: x["priority"])
    return recs

# ── Load ───────────────────────────────────────────────────────────────────────
model     = load_model()
df        = load_dataset()
scaler    = load_scaler()
explainer = load_shap_explainer(model, scaler)  # adiciona esta linha
models_ok = model is not None and scaler is not None
logo_icon = img_to_base64("icons/gym_icon.png")
idea      = img_to_base64("icons/idea.svg")
task      = img_to_base64("icons/task_icon.svg")
settings  = img_to_base64("icons/settings.svg")
target    = img_to_base64("icons/target_icon.svg")
target3   = img_to_base64("icons/target3.svg")
microscope= img_to_base64("icons/microscope_icon.svg")
siren     = img_to_base64("icons/siren.svg")
group2    = img_to_base64("icons/group.svg")
group3    = img_to_base64("icons/group3.svg")
clipboard = img_to_base64("icons/clipboard-clock.svg")
clipboard2=img_to_base64("icons/clipboard-clock2.svg")
alert     = img_to_base64("icons/triangle-alert.svg")
chart     = img_to_base64("icons/chart-column-big.svg")
check     = img_to_base64("icons/square-check.svg")
trophy    = img_to_base64("icons/trophy.svg")
diamond   = img_to_base64("icons/gem.svg")
down      = img_to_base64("icons/arrow-down-narrow-wide.svg")
down2     = img_to_base64("icons/arrow-down-narrow-wide2.svg")
deal      = img_to_base64("icons/handshake.svg")
file      = img_to_base64("icons/file-pen-line.svg")

def sb_section(label):
    st.markdown(f"""
    <div class="sb-section">
        <span class="sb-section-label">{label}</span>
        <div class="sb-section-line"></div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
	<div class="sb-brand">
    	<div class="brand-top">
    	    <h2>Customer Features</h2>
    	</div>
    	<p>Enter the member’s details to obtain the churn probability predicted by the XGBoost model.</p>
	</div>
	""", unsafe_allow_html=True)

    if not models_ok:
        st.error("Model files not found. Place `final_xgb_model.pkl` and `scaler.pkl` in the app folder.")
        st.stop()

    # Profile
    sb_section("Profile")
    age           = st.slider("Age", 18, 80, 30)
    gender        = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x else "Female")
    near_location = st.selectbox("Close to Gym", [0, 1], index=1, format_func=lambda x: "Yes" if x else "No")
    partner       = st.selectbox("Partner Company Employee", [0, 1], format_func=lambda x: "Yes" if x else "No")
    promo_friends = st.selectbox("Signed Up via Friend Promo", [0, 1], format_func=lambda x: "Yes" if x else "No")
    phone         = st.selectbox("Registered Phone Number", [0, 1], index=1, format_func=lambda x: "Yes" if x else "No")

    # Attendance
    sb_section("Attendance")
    freq_current = st.slider("Weekly Attendance — This Month (classes/week)", 0.0, 7.0, 2.0, step=0.1)
    freq_total   = st.slider("Avg Historical Attendance (classes/week)", 0.0, 7.0, 2.0, step=0.1)
    group_visits = st.selectbox("Group Classes", [0, 1], format_func=lambda x: "Yes" if x else "No")

    # Contract
    sb_section("Contract")
    contract_period = st.selectbox("Contract Duration (months)", [1, 3, 6, 12], index=2)
    month_to_end    = st.slider("Months Until Contract End", 0, 24, 3)
    lifetime        = st.slider("Time as Member (months)", 0, 72, 6)

    # Financials
    sb_section("Financials")
    avg_additional = st.number_input("Avg Additional Expenditure (€)", 0.0, 600.0, 120.0, step=5.0)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Predict Churn", type="primary", use_container_width=True)

# ── Input dict ─────────────────────────────────────────────────────────────────
input_data = {
    "gender":                            int(gender),
    "near_location":                     int(near_location),
    "partner":                           int(partner),
    "promo_friends":                     int(promo_friends),
    "phone":                             int(phone),
    "contract_period":                   int(contract_period),
    "group_visits":                      int(group_visits),
    "age":                               float(age),
    "avg_additional_charges_total":      float(avg_additional),
    "month_to_end_contract":             float(month_to_end),
    "lifetime":                          float(lifetime),
    "avg_class_frequency_total":         float(freq_total),
    "avg_class_frequency_current_month": float(freq_current),
}

if predict_btn:
    score = predict_score(input_data, model, scaler)
    label, cls, color = classify_risk(score)
    st.session_state["result"] = {
        "score": score, "label": label, "cls": cls, "color": color,
        "input": input_data.copy(),
        "shap": get_shap_values(input_data, model, scaler, explainer),  # alterado
        "explanation": generate_explanation(input_data, score),
        "recs":  get_recommendations(cls, input_data),
    }

has_result = "result" in st.session_state
res        = st.session_state.get("result", {})
PROMPT     = "Configure the customer profile in the sidebar and click <strong> Predict Churn</strong>."

def empty(icon_base64, title, msg):
    return f"""
    <div class="empty-state">
        <img src="data:image/svg+xml;base64,{icon_base64}" class="icon">
        <h3>{title}</h3>
        <p>{msg}</p>
    </div>
    """

# ══════════════════════════════════════════════════════════════════════════════
# MAIN — 4 TABS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="margin-bottom:8px;">
    <div style="font-size:28px;font-weight:600;color:var(--text);letter-spacing:-0.4px;">Retention Dashboard</div>
    <div style="font-size:13px;color:var(--muted);margin-top:2px;">Element gym member retention dashboard powered by machine learning.</div>
</div>""", unsafe_allow_html=True)

tab_rs, tab_rec, tab_shap, tab_ds = st.tabs([
    "  Risk Score",
    "  Recommendations",
    "  SHAP Explanation",
    "  Dataset Overview",
])


# ── TAB 1: RISK SCORE ────────────────────────────────────────────────────────────
with tab_rs:
    if not has_result:
        st.markdown(empty( target, "Select the type of customer",
            "Use the sidebar on the left to configure the customer's attributes,<br>"
            "then click <strong> Predict Churn</strong> to see the full profile overview."),
            unsafe_allow_html=True)
    else:
        r   = res
        inp = r["input"]

        st.markdown(f"""
        <div class="card">
            <div style="display:flex;align-items:center;gap:18px;">
                <div style="width:100%">
                    <div style="font-size:11px;color:var(--muted);font-weight:700;letter-spacing:.8px;text-transform:uppercase;margin-bottom:4px;">Churn Risk Assessment</div>
                    <div style="font-size:24px;font-weight:800;color:{r['color']};margin-bottom:4px;">{r['score']:.1f}% · {r['label']}</div>
                    <div style="font-size:13px;color:var(--muted);margin-bottom:16px;">{r['explanation']}</div>
                    <div style="background:var(--bg3);border-radius:999px;height:10px;width:100%;overflow:hidden;">
                        <div style="width:{min(r['score'],100):.1f}%;height:100%;background:{r['color']};border-radius:999px;"></div>
                    </div>
                    <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--muted);margin-top:6px;">
                        <span>0%</span>
                        <span style="color:#16a34a;">Low &lt;35%</span>
                        <span style="color:#ea580c;">Medium 35–60%</span>
                        <span style="color:#dc2626;">High &gt;60%</span>
                        <span>100%</span>
                    </div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Member Profile**")
            rows = [
                ("Age",              f"{int(inp['age'])} years"),
                ("Gender",           "Male" if inp['gender'] else "Female"),
                ("Close to gym",     "Yes" if inp['near_location'] else "No"),
                ("Partner company",  "Yes" if inp['partner'] else "No"),
                ("Friend referral",  "Yes" if inp['promo_friends'] else "No"),
                ("Phone registered", "Yes" if inp['phone'] else "No"),
            ]
            st.markdown("".join(
                f'<div class="prof-row"><span class="prof-key">{k}</span><span class="prof-val">{v}</span></div>'
                for k, v in rows), unsafe_allow_html=True)

        with col2:
            st.markdown("**Engagement & Contract**")
            rows = [
                ("Contract duration",     f"{int(inp['contract_period'])} months"),
                ("Months until end",      f"{int(inp['month_to_end_contract'])} months"),
                ("Member since",          f"{int(inp['lifetime'])} months"),
                ("Current weekly freq.",  f"{inp['avg_class_frequency_current_month']:.1f}×/week"),
                ("Historical avg. freq.", f"{inp['avg_class_frequency_total']:.1f}×/week"),
                ("Group classes",         "Yes" if inp['group_visits'] else "No"),
                ("Extra spending",        f"€{inp['avg_additional_charges_total']:.2f}"),
            ]
            st.markdown("".join(
                    f'<div class="prof-row"><span class="prof-key">{k}</span><span class="prof-val">{v}</span></div>'
                    for k, v in rows), unsafe_allow_html=True)



# ── TAB 3: RECOMMENDATIONS ────────────────────────────────────────────────────
with tab_rec:
    if not has_result:
        st.markdown(empty(idea, "No recommendations yet", PROMPT), unsafe_allow_html=True)
    else:
        r = res
        st.markdown(
            f"Recommended actions for a member classified as "
            f"<span style='color:{r['color']};font-weight:700'>{r['label']}</span> "
            f"<span style='color:var(--muted);font-size:13px'>— based on their specific profile</span>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        for rec in r["recs"]:
            border_left = f"4px solid {rec['color']}"
            st.markdown(f"""
            <div class="card" style="border-left:{border_left};padding:20px;margin-top:5px;margin-bottom:5px">
                <span style="font-size:10px;font-weight:700;letter-spacing:.8px;text-transform:uppercase;
                      color:{rec['color']};background:{rec['color']}18;padding:3px 10px;
                      border-radius:20px;border:1px solid {rec['color']}40">{rec['tag']}</span>
                <div style="display:flex;align-items:center;gap:8px;margin:10px 0 4px 0">
                    <span style="font-size:15px">{rec['icon']}</span>
                    <span style="font-size:16px;font-weight:700;color:var(--text)">{rec['title']}</span>
                </div>
                <p style="font-size:14px;color:var(--muted);margin:0;line-height:1.6">{rec['text']}</p>
            </div>""", unsafe_allow_html=True)

# ── TAB 4: SHAP EXPLANATION ───────────────────────────────────────────────────
with tab_shap:
    if not has_result:
        st.markdown(empty(microscope, "No explanation yet", PROMPT), unsafe_allow_html=True)
    else:
        r         = res
        shap_vals = r["shap"]
        score     = r["score"]
        BASE_RATE = 26.5  # % média de churn no dataset — ajusta com o valor real

        sorted_feats = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)
        max_abs      = max(abs(v) for _, v in sorted_feats) + 1e-8

        # ── Header: base rate → score final ──
        delta      = score - BASE_RATE
        delta_sign = "+" if delta >= 0 else ""
        delta_color = "#dc2626" if delta >= 0 else "#16a34a"
        st.markdown("#### Global Feature Importance")
        st.caption("Red bars increase churn risk · Blue bars decrease it · Values are importance-weighted feature contributions.")
        st.markdown(f"""
        <div class="card" style="margin-bottom:16px">
            <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
                <div style="text-align:center;padding:0 16px">
                    <div class="kpi-label">Base Rate</div>
                    <div style="font-size:26px;font-weight:800;color:var(--muted)">{BASE_RATE:.1f}%</div>
                    <div style="font-size:11px;color:var(--muted2)">dataset avg</div>
                </div>
                <div style="font-size:22px;color:var(--muted2)">→</div>
                <div style="flex:1;font-size:12px;color:var(--muted);text-align:center">
                    SHAP contributions<br>from each feature
                </div>
                <div style="font-size:22px;color:var(--muted2)">→</div>
                <div style="text-align:center;padding:0 16px">
                    <div class="kpi-label">This Member</div>
                    <div style="font-size:26px;font-weight:800;color:{r['color']}">{score:.1f}%</div>
                    <div style="font-size:11px;color:{delta_color};font-weight:600">{delta_sign}{delta:.1f}% vs avg</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Mapeamento de valores reais para texto legível
        def fmt_value(feat, inp):
            v = inp[feat]
            if feat in ["avg_class_frequency_current_month", "avg_class_frequency_total"]:
                return f"{v:.1f}×/wk"
            if feat in ["avg_additional_charges_total"]:
                return f"€{v:.0f}"
            if feat in ["month_to_end_contract", "lifetime"]:
                return f"{int(v)} mo"
            if feat in ["contract_period"]:
                return f"{int(v)} mo"
            if feat in ["age"]:
                return f"{int(v)} yr"
            if feat in ["gender"]:
                return "Male" if v else "Female"
            if feat in ["near_location"]:
                return "Yes" if v else "No"
            if feat in ["partner"]:
                return "Yes" if v else "No"
            if feat in ["promo_friends"]:
                return "Yes" if v else "No"
            if feat in ["phone"]:
                return "Yes" if v else "No"
            if feat in ["group_visits"]:
                return "Yes" if v else "No"
            return str(v)

        rows_html = ""
        for feat, val in sorted_feats:
            label      = FEATURE_LABELS.get(feat, feat)
            raw_val    = fmt_value(feat, r["input"])
            pct        = abs(val) / max_abs * 100
            bar_cls    = "shap-bar-pos" if val >= 0 else "shap-bar-neg"
            sign       = "+" if val >= 0 else ""
            color      = "#dc2626" if val > 0 else "#2563eb"
            rows_html += f"""
            <div class="shap-row">
                <div class="shap-name">
                    {label}
                    <span style="font-size:10px;color:var(--muted2);font-weight:400;margin-left:4px">({raw_val})</span>
                </div>
                <div class="shap-bar-wrap"><div class="{bar_cls}" style="width:{pct:.1f}%"></div></div>
                <div class="shap-val" style="color:{color}">{sign}{val:.3f}</div>
            </div>"""

        st.markdown(f'<div class="card">{rows_html}</div>', unsafe_allow_html=True)

# ── TAB 4: DATASET OVERVIEW ───────────────────────────────────────────────────
with tab_ds:
    if df is None:
        st.error("Dataset not found. Place `gym_churn_us.csv` in the app folder.")
    else:
        # ── Threshold slider ──
        st.markdown("""
        <div style="font-size:13px;color:var(--muted);margin-bottom:4px;">
            Adjust the risk threshold to see how many members would be flagged for intervention.
        </div>""", unsafe_allow_html=True)
        threshold = st.slider("Risk threshold (%)", 10, 90, 60, step=5, label_visibility="collapsed")

        # ── Calcular métricas ──
        total       = len(df)
        churned     = int(df["Churn"].sum())
        retained    = total - churned
        churn_rate  = churned / total * 100

        # Prever probabilidades para todo o dataset
        @st.cache_data
        def get_all_scores(_model, _scaler):
            try:
                cols = EXPECTED_FEATURES
                X = df[cols] if all(c in df.columns for c in cols) else df.drop(columns=["Churn"], errors="ignore")
                # normaliza nomes das colunas
                X.columns = [c.lower() for c in X.columns]
                sc = _scaler.transform(X)
                return _model.predict_proba(sc)[:, 1] * 100
            except:
                return None

        all_scores   = get_all_scores(model, scaler)
        at_risk      = int((all_scores >= threshold).sum()) if all_scores is not None else None
        at_risk_pct  = at_risk / total * 100 if at_risk is not None else None
        AVG_MONTHLY_REVENUE = 40
        revenue_at_risk = at_risk * AVG_MONTHLY_REVENUE if at_risk is not None else None

        # ── KPI row ──
        k1, k2, k3, k4 = st.columns(4)
        for col, label, value, sub, css in [
            (k1, "Total Members",     f"{total:,}",           "in dataset",                        "c-blue"),
            (k2, "Churn Rate",        f"{churn_rate:.1f}%",   f"{churned:,} members left",         "c-red"),
            (k3, "At Risk",           f"{at_risk:,}" if at_risk is not None else "N/A",
                                                               f"above {threshold}% threshold",     "c-orange"),
            (k4, "Revenue at Risk",   f"€{revenue_at_risk:,.0f}" if revenue_at_risk is not None else "N/A",
                                                               f"@ €{AVG_MONTHLY_REVENUE}/member/mo","c-orange"),
        ]:
            with col:
                st.markdown(f"""
                <div class="card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value {css}">{value}</div>
                    <div class="kpi-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        gc1, gc2 = st.columns([3, 2], gap="large")

        # ── Gráfico 1: Feature Importance ──
        with gc1:
            st.markdown('<div class="kpi-label" style="margin-bottom:8px">Feature Importance — XGBoost Model</div>', unsafe_allow_html=True)
            feat_labels = {
                "avg_class_frequency_current_month": "Attendance (this month)",
                "avg_class_frequency_total":         "Attendance (historical)",
                "lifetime":                          "Membership Duration",
                "month_to_end_contract":             "Months to Contract End",
                "avg_additional_charges_total":      "Additional Spend",
                "contract_period":                   "Contract Period",
                "age":                               "Age",
                "group_visits":                      "Group Classes",
                "partner":                           "Partner Company",
                "promo_friends":                     "Friend Referral",
                "near_location":                     "Lives Near Gym",
                "gender":                            "Gender",
                "phone":                             "Phone Registered",
            }
            importances = model.feature_importances_
            labels      = [feat_labels.get(f, f) for f in EXPECTED_FEATURES]
            idx         = np.argsort(importances)

            fig, ax = plt.subplots(figsize=(7, 4.5))
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")
            ax.barh([labels[i] for i in idx], [importances[i] for i in idx],
                    color="#2563eb", alpha=0.85, height=0.6)
            ax.set_xlabel("Importance", fontsize=10, color="#64748b")
            ax.tick_params(colors="#0f172a", labelsize=9)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#e2e6ed")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # ── Gráfico 2: Donut Churn vs Retained ──
        with gc2:
            st.markdown('<div class="kpi-label" style="margin-bottom:8px">Churn Distribution</div>', unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            fig2.patch.set_alpha(0)
            ax2.set_facecolor("none")
            ax2.pie(
                [retained, churned],
                labels=["Retained", "Churned"],
                colors=["#16a34a", "#dc2626"],
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops=dict(width=0.55),
                textprops=dict(fontsize=11, color="#0f172a"),
            )
            ax2.set_aspect("equal")
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close()

        # ── Model metrics ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="kpi-label" style="margin-bottom:10px">Model Performance — XGBoost Test Set</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        for col, label, val, css in [
            (m1, "Accuracy",  "94.5%", "c-blue"),
            (m2, "Precision", "90.8%", "c-green"),
            (m3, "Recall",    "88.2%", "c-orange"),
            (m4, "F1-Score",  "89.5%", "c-blue"),
        ]:
            with col:
                st.markdown(f"""
                <div class="card-sm" style="text-align:center">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value {css}" style="font-size:28px">{val}</div>
                </div>""", unsafe_allow_html=True)