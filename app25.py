# Phishing Website Detection - Streamlit App
# Model: XGBoost | Dataset: UCI Phishing Websites (ID=327)
# To run: streamlit run app.py

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
import time
import socket
import requests
import whois
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urlparse

st.set_page_config(page_title="Phishing Detector", page_icon="🛡️", layout="wide")

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0a0e1a;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px; }

/* ── Custom header banner ── */
.custom-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #112240 50%, #0d1b2a 100%);
    border: 1px solid rgba(100, 180, 255, 0.15);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.custom-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #00d2ff, #3a7bd5, #00d2ff);
    background-size: 200% 100%;
    animation: shimmer 3s linear infinite;
}
@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
.custom-header h1 {
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(135deg, #e0f0ff, #7ec8ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem 0;
}
.custom-header p {
    color: rgba(160, 200, 255, 0.55);
    font-size: 0.8rem; margin: 0;
    letter-spacing: 0.03em;
}
.shield-icon { font-size: 2.2rem; margin-right: 0.8rem; vertical-align: middle; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(255,255,255,0.07);
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    padding: 0.5rem 1.4rem;
    color: rgba(160,200,255,0.6);
    font-weight: 500;
    font-size: 0.9rem;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1a3a5c, #1e4976) !important;
    color: #7ec8ff !important;
    box-shadow: 0 2px 12px rgba(0,180,255,0.15);
}

/* ── Input box ── */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(100,180,255,0.2) !important;
    border-radius: 10px !important;
    color: #e0f0ff !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem !important;
    padding: 0.7rem 1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(0,180,255,0.5) !important;
    box-shadow: 0 0 0 3px rgba(0,180,255,0.08) !important;
}

/* ── Analyse / Predict button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0070f3, #0050c8) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 2rem !important;
    letter-spacing: 0.04em !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 18px rgba(0,112,243,0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1a80ff, #0060dd) !important;
    box-shadow: 0 6px 24px rgba(0,112,243,0.45) !important;
    transform: translateY(-1px) !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(100,180,255,0.12);
    border-radius: 14px;
    padding: 1.2rem 1.5rem !important;
}
[data-testid="stMetricLabel"] {
    color: rgba(160,200,255,0.5) !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: #7ec8ff !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 2.2rem !important;
    font-weight: 600 !important;
}

/* ── Alert boxes ── */
/* Success — LEGITIMATE */
.stAlert[data-baseweb="notification"][kind="positive"],
div[data-testid="stNotification"].success,
.element-container .stSuccess > div {
    background: rgba(0, 200, 100, 0.07) !important;
    border: 1px solid rgba(0, 200, 100, 0.3) !important;
    border-left: 4px solid #00c864 !important;
    border-radius: 12px !important;
    color: #88ffcc !important;
}
/* Error — PHISHING */
.stAlert[data-baseweb="notification"][kind="negative"],
div[data-testid="stNotification"].error,
.element-container .stError > div {
    background: rgba(255, 50, 50, 0.07) !important;
    border: 1px solid rgba(255, 80, 80, 0.3) !important;
    border-left: 4px solid #ff3232 !important;
    border-radius: 12px !important;
    color: #ffaaaa !important;
}
/* Warning — CAUTION */
.stAlert[data-baseweb="notification"][kind="warning"],
div[data-testid="stNotification"].warning,
.element-container .stWarning > div {
    background: rgba(255, 180, 0, 0.07) !important;
    border: 1px solid rgba(255,180,0,0.3) !important;
    border-left: 4px solid #ffb400 !important;
    border-radius: 12px !important;
    color: #ffd166 !important;
}
/* Info */
.element-container .stInfo > div {
    background: rgba(0,140,255,0.07) !important;
    border: 1px solid rgba(0,140,255,0.25) !important;
    border-left: 4px solid #008cff !important;
    border-radius: 12px !important;
    color: #a0d4ff !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(100,180,255,0.1) !important;
    border-radius: 10px !important;
    color: rgba(160,200,255,0.7) !important;
    font-size: 0.85rem !important;
}
.streamlit-expanderContent {
    border: 1px solid rgba(100,180,255,0.08) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
    background: rgba(255,255,255,0.02) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid rgba(100,180,255,0.1) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1525 !important;
    border-right: 1px solid rgba(100,180,255,0.1) !important;
}
[data-testid="stSidebar"] .stMarkdown table {
    width: 100%;
    border-collapse: collapse;
}
[data-testid="stSidebar"] .stMarkdown table td,
[data-testid="stSidebar"] .stMarkdown table th {
    border: 1px solid rgba(100,180,255,0.12);
    padding: 0.4rem 0.6rem;
    font-size: 0.82rem;
    color: rgba(200,230,255,0.8);
}
[data-testid="stSidebar"] .stMarkdown table th {
    background: rgba(0,100,200,0.15);
    font-weight: 600;
    color: #7ec8ff;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #7ec8ff !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] li {
    color: rgba(180,215,255,0.7) !important;
    font-size: 0.82rem !important;
}
[data-testid="stSidebar"] strong { color: #a0d4ff !important; }
[data-testid="stSidebar"] hr { border-color: rgba(100,180,255,0.12) !important; }

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(100,180,255,0.18) !important;
    border-radius: 8px !important;
    color: #c0deff !important;
}

/* ── Section headings inside tabs ── */
.stMarkdown h3 { color: #7ec8ff !important; font-size: 1rem !important; font-weight: 600 !important; }
.stMarkdown strong { color: #a0d4ff !important; }

/* ── Caption / small text ── */
.stCaption { color: rgba(120,170,220,0.5) !important; font-size: 0.75rem !important; }

/* ── Divider ── */
hr { border-color: rgba(100,180,255,0.1) !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #3a7bd5 !important; }

/* ── Subheader ── */
h2 { color: #c0deff !important; font-weight: 600 !important; }
h3 { color: #a0cbf0 !important; }
p, .stMarkdown p { color: rgba(180,215,255,0.65) !important; }

/* ── Footer bar ── */
.footer-bar {
    text-align: center;
    color: rgba(100,160,220,0.35);
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Threat Intelligence ────────────────────────────────────────────────────────

POPULAR_DOMAINS = [
    "google", "facebook", "microsoft", "apple", "amazon", "paypal", "netflix",
    "instagram", "twitter", "linkedin", "yahoo", "dropbox", "github", "gmail",
    "outlook", "live", "office", "adobe", "bank", "chase", "wellsfargo",
    "bankofamerica", "citibank", "hsbc", "dhl", "fedex", "ups", "usps",
]

def check_phishtank(url: str) -> dict:
    """
    Query PhishTank's free checkurl API (no API key needed for basic checks).
    Returns {'is_phish': bool, 'verified': bool, 'phish_id': str|None}
    """
    try:
        resp = requests.post(
            "https://checkurl.phishtank.com/checkurl/",
            data={"url": url, "format": "json", "app_key": ""},
            headers={"User-Agent": "phishtank/guneeth-phishing-detector"},
            timeout=6,
        )
        data = resp.json()
        result = data.get("results", {})
        return {
            "is_phish": result.get("in_database", False) and result.get("valid", False),
            "verified": result.get("verified", False),
            "phish_id": result.get("phish_id"),
        }
    except Exception:
        return {"is_phish": False, "verified": False, "phish_id": None}


def check_typosquatting(domain: str) -> list:
    """
    Check if a domain looks like a typosquat of a popular brand.
    Returns list of suspicious brand matches.
    """
    import difflib
    domain_base = domain.lower().split(".")[0]
    # strip numbers/hyphens for comparison
    domain_clean = re.sub(r"[0-9\-]", "", domain_base)
    matches = []
    for brand in POPULAR_DOMAINS:
        # exact after cleaning → flag
        if domain_clean == brand and domain_base != brand:
            matches.append(brand)
            continue
        ratio = difflib.SequenceMatcher(None, domain_clean, brand).ratio()
        if ratio >= 0.85 and domain_base != brand:
            matches.append(brand)
    return matches

st.markdown("""
<div class="custom-header">
    <span class="shield-icon">🛡️</span>
    <h1 style="display:inline;">Phishing Website Detection System</h1>
    <p style="margin-top:0.5rem;">Powered by XGBoost &nbsp;·&nbsp; UCI Phishing Websites Dataset &nbsp;·&nbsp; PhishTank TI &nbsp;·&nbsp; SASTRA University</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = joblib.load("xgbModel.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


FEATURES = {
    "having_IP_Address":          {"desc": "URL contains IP address instead of domain name",      "options": {"-1 (No — Legitimate)": -1, "1 (Yes — Suspicious)": 1}},
    "URL_Length":                 {"desc": "Length of the URL",                                    "options": {"-1 (Short <54)": -1, "0 (Medium 54–75)": 0, "1 (Long >75)": 1}},
    "Shortining_Service":         {"desc": "URL uses shortening service (bit.ly, tinyurl etc.)",   "options": {"-1 (No)": -1, "1 (Yes)": 1}},
    "having_At_Symbol":           {"desc": "URL contains @ symbol",                                "options": {"-1 (No)": -1, "1 (Yes)": 1}},
    "double_slash_redirecting":   {"desc": "URL has // after protocol (redirection)",              "options": {"-1 (No)": -1, "1 (Yes)": 1}},
    "Prefix_Suffix":              {"desc": "Domain has - prefix or suffix",                        "options": {"-1 (No)": -1, "1 (Yes)": 1}},
    "having_Sub_Domain":          {"desc": "Number of sub-domains in URL",                         "options": {"-1 (Single)": -1, "0 (Double)": 0, "1 (Multiple)": 1}},
    "SSLfinal_State":             {"desc": "SSL certificate status and issuer trust",              "options": {"-1 (Trusted HTTPS)": -1, "0 (Untrusted HTTPS)": 0, "1 (HTTP only)": 1}},
    "Domain_registeration_length":{"desc": "Domain registration duration",                        "options": {"-1 (>1 year)": -1, "1 (<1 year)": 1}},
    "Favicon":                    {"desc": "Favicon loaded from external domain",                  "options": {"-1 (Same domain)": -1, "1 (External domain)": 1}},
    "port":                       {"desc": "Uses non-standard port",                               "options": {"-1 (Standard port)": -1, "1 (Non-standard port)": 1}},
    "HTTPS_token":                {"desc": "HTTPS token appears in domain part of URL",            "options": {"-1 (Not present)": -1, "1 (Present)": 1}},
    "Request_URL":                {"desc": "% of external objects loaded in webpage",              "options": {"-1 (<22% external)": -1, "0 (22–61% external)": 0, "1 (>61% external)": 1}},
    "URL_of_Anchor":              {"desc": "% of anchor tags linking to external pages",           "options": {"-1 (<31%)": -1, "0 (31–67%)": 0, "1 (>67%)": 1}},
    "Links_in_tags":              {"desc": "% of links in meta/script/link tags",                  "options": {"-1 (<17%)": -1, "0 (17–81%)": 0, "1 (>81%)": 1}},
    "SFH":                        {"desc": "Server Form Handler — where form data is sent",        "options": {"-1 (Same domain)": -1, "0 (Empty string)": 0, "1 (External/about:blank)": 1}},
    "Submitting_to_email":        {"desc": "Form submits data via email (mailto:)",                "options": {"-1 (No)": -1, "1 (Yes)": 1}},
    "Abnormal_URL":               {"desc": "Host name not part of URL",                            "options": {"-1 (Normal)": -1, "1 (Abnormal)": 1}},
    "Redirect":                   {"desc": "Number of redirects",                                  "options": {"0 (≤1 redirect)": 0, "1 (>1 redirect)": 1}},
    "on_mouseover":               {"desc": "Status bar changed on mouseover",                      "options": {"-1 (No change)": -1, "1 (Changes)": 1}},
    "RightClick":                 {"desc": "Right click disabled on page",                         "options": {"-1 (Enabled)": -1, "1 (Disabled)": 1}},
    "popUpWidnow":                {"desc": "Pop-up window with text fields appears",               "options": {"-1 (No)": -1, "1 (Yes)": 1}},
    "Iframe":                     {"desc": "Uses invisible iframe",                                "options": {"-1 (No iframe)": -1, "1 (Has iframe)": 1}},
    "age_of_domain":              {"desc": "Age of the domain",                                    "options": {"-1 (≥6 months old)": -1, "1 (<6 months old)": 1}},
    "DNSRecord":                  {"desc": "DNS record exists for domain",                         "options": {"-1 (Record found)": -1, "1 (No record)": 1}},
    "web_traffic":                {"desc": "Website traffic rank (Alexa)",                         "options": {"-1 (Top 100k)": -1, "0 (>100k rank)": 0, "1 (No traffic data)": 1}},
    "Page_Rank":                  {"desc": "Google Page Rank value",                               "options": {"-1 (PageRank < 0.2)": -1, "1 (PageRank ≥ 0.2)": 1}},
    "Google_Index":               {"desc": "Page is indexed by Google",                            "options": {"-1 (Indexed)": -1, "1 (Not indexed)": 1}},
    "Links_pointing_to_page":     {"desc": "Number of links pointing to page",                     "options": {"-1 (>2 links)": -1, "0 (1–2 links)": 0, "1 (0 links)": 1}},
    "Statistical_report":         {"desc": "Host in top phishing IPs/domains statistical report",  "options": {"-1 (Not reported)": -1, "1 (Reported)": 1}},
}

FEATURE_NAMES = list(FEATURES.keys())


def get_domain(url):
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except:
        return ""

def is_ip(domain):
    ip_pattern = re.compile(r"^(([01]?\d\d?|2[0-4]\d|25[0-5])\.){3}([01]?\d\d?|2[0-4]\d|25[0-5])$")
    return bool(ip_pattern.match(domain))

def extract_url_features(url):
    domain = get_domain(url)

    # defaults: URL features → -1 (legitimate), WHOIS/web → 1 (suspicious if unreachable)
    features = {
        "having_IP_Address": -1,
        "URL_Length": -1,
        "Shortining_Service": -1,
        "having_At_Symbol": -1,
        "double_slash_redirecting": -1,
        "Prefix_Suffix": -1,
        "having_Sub_Domain": -1,
        "SSLfinal_State": -1,
        "HTTPS_token": -1,
        "port": -1,
        "Abnormal_URL": -1,
        "Domain_registeration_length": 1,
        "age_of_domain": 1,
        "DNSRecord": 1,
        "Favicon": 1,
        "Request_URL": 1,
        "URL_of_Anchor": 1,
        "Links_in_tags": 1,
        "SFH": 1,
        "Submitting_to_email": -1,
        "Redirect": 1,
        "on_mouseover": -1,
        "RightClick": -1,
        "popUpWidnow": -1,
        "Iframe": -1,
        "web_traffic": 1,
        "Page_Rank": -1,
        "Google_Index": 1,
        "Links_pointing_to_page": 1,
        "Statistical_report": -1,
    }

    # URL string features
    features["having_IP_Address"] = 1 if is_ip(domain) else -1

    length = len(url)
    features["URL_Length"] = -1 if length < 54 else (0 if length <= 75 else 1)

    shorteners = ["bit.ly", "tinyurl", "goo.gl", "ow.ly", "t.co", "tiny.cc", "is.gd", "buff.ly", "adf.ly", "rebrand.ly"]
    features["Shortining_Service"] = 1 if any(s in url for s in shorteners) else -1

    features["having_At_Symbol"] = 1 if "@" in url else -1
    features["double_slash_redirecting"] = 1 if url.rfind("//") > 7 else -1
    features["Prefix_Suffix"] = 1 if "-" in domain else -1

    dots = domain.count(".")
    features["having_Sub_Domain"] = -1 if dots == 1 else (0 if dots == 2 else 1)
    features["SSLfinal_State"] = -1 if url.startswith("https") else 1
    features["HTTPS_token"] = 1 if "https" in domain.lower() else -1
    features["Abnormal_URL"] = -1 if domain in url else 1

    try:
        port = urlparse(url).port
        standard = [80, 443, 21, 22, 23, 25, 110, 143, 3389]
        features["port"] = 1 if (port and port not in standard) else -1
    except:
        features["port"] = -1

    # DNS check
    try:
        socket.setdefaulttimeout(3)
        socket.gethostbyname(domain)
        features["DNSRecord"] = -1
    except:
        features["DNSRecord"] = 1
        return features  # stop here if DNS fails

    # WHOIS lookup
    if not is_ip(domain):
        try:
            w = whois.whois(domain)
            try:
                exp = w.expiration_date
                cre = w.creation_date
                if isinstance(exp, list): exp = exp[0]
                if isinstance(cre, list): cre = cre[0]
                if exp and cre:
                    features["Domain_registeration_length"] = -1 if (exp - cre).days >= 365 else 1
            except: pass
            try:
                cre = w.creation_date
                if isinstance(cre, list): cre = cre[0]
                if cre:
                    features["age_of_domain"] = -1 if (datetime.now() - cre).days >= 180 else 1
            except: pass
        except: pass

    # HTML/page features
    soup = None
    html = ""
    site_reachable = False
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(url, timeout=7, headers=headers, allow_redirects=True)
        html = resp.text
        soup = BeautifulSoup(html, "html.parser")
        site_reachable = True

        try:
            fav = soup.find("link", rel=lambda r: r and "icon" in r)
            features["Favicon"] = 1 if (fav and fav.get("href", "").startswith("http") and domain not in fav.get("href", "")) else -1
        except: features["Favicon"] = -1

        try:
            tags = soup.find_all(["img", "audio", "video", "embed"], src=True)
            total = len(tags)
            if total > 0:
                pct = sum(1 for t in tags if domain not in (t.get("src", "") or "")) / total * 100
                features["Request_URL"] = -1 if pct < 22 else (0 if pct <= 61 else 1)
            else: features["Request_URL"] = -1
        except: features["Request_URL"] = -1

        try:
            anchors = soup.find_all("a", href=True)
            total = len(anchors)
            if total > 0:
                pct = sum(1 for a in anchors if domain not in (a.get("href", "") or "") and (a.get("href", "") or "").startswith("http")) / total * 100
                features["URL_of_Anchor"] = -1 if pct < 31 else (0 if pct <= 67 else 1)
            else: features["URL_of_Anchor"] = -1
        except: features["URL_of_Anchor"] = -1

        try:
            ml = soup.find_all(["meta", "script", "link"])
            total = len(ml)
            if total > 0:
                pct = sum(1 for t in ml if domain not in (t.get("src", "") or t.get("href", "") or "")) / total * 100
                features["Links_in_tags"] = -1 if pct < 17 else (0 if pct <= 81 else 1)
            else: features["Links_in_tags"] = -1
        except: features["Links_in_tags"] = -1

        try:
            forms = soup.find_all("form", action=True)
            if forms:
                action = forms[0].get("action", "").strip()
                if action in ["", "about:blank"]: features["SFH"] = 0
                elif domain not in action and action.startswith("http"): features["SFH"] = 1
                else: features["SFH"] = -1
            else: features["SFH"] = -1
        except: features["SFH"] = -1

        try:
            forms = soup.find_all("form", action=True)
            features["Submitting_to_email"] = 1 if any("mailto:" in (f.get("action", "") or "") for f in forms) else -1
        except: pass

        try: features["Redirect"] = 0 if len(resp.history) <= 1 else 1
        except: features["Redirect"] = 0

        features["on_mouseover"] = 1 if "onmouseover" in html.lower() else -1
        features["RightClick"] = 1 if ("contextmenu" in html.lower() or "event.button==2" in html) else -1
        features["popUpWidnow"] = 1 if "window.open" in html else -1
        features["Iframe"] = 1 if soup.find_all("iframe") else -1

        try:
            bl = soup.find_all("a", href=lambda h: h and domain in h)
            features["Links_pointing_to_page"] = -1 if len(bl) > 2 else (0 if len(bl) >= 1 else 1)
        except: features["Links_pointing_to_page"] = 1

    except requests.exceptions.SSLError:
        features["SSLfinal_State"] = 1
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        pass
    except Exception:
        pass

    features["web_traffic"] = 0 if site_reachable else 1
    features["Google_Index"] = 0 if site_reachable else 1
    features["Page_Rank"] = -1
    features["Statistical_report"] = -1

    return features


# sidebar
with st.sidebar:
    st.header("Model Info")
    st.markdown("""
    | Metric | Score |
    |--------|-------|
    | Accuracy | 96.99% |
    | Precision | 96.42% |
    | Recall | 98.31% |
    | F1-Score | 97.36% |
    | ROC-AUC | 0.9960 |
    """)
    st.divider()
    st.markdown("""
    This app detects phishing websites using
    **30 URL-based features** from the UCI Phishing Websites Dataset.

    **Two input modes:**
    - 🔗 Enter a URL directly
    - Input feature values manually

    **Detection layers:**
    1. 🔴 PhishTank threat intel (real-time)
    2. ⚠️ Typosquatting detection
    3. 🤖 XGBoost ML classifier

    **Model:** XGBoost Classifier
    **Dataset:** 11,055 samples
    """)
    st.divider()
    st.markdown("**Team:** N. Guneeth Sai (127004171)")

if not model_loaded:
    st.error("Model files not found! Make sure `xgbModel.pkl` and `scaler.pkl` are in the same folder as `app.py`")
    st.stop()

tab1, tab2 = st.tabs(["🔗 URL Input", "🔢 Manual Input"])

with tab1:
    st.subheader("Enter a Website URL")
    st.write("The system will automatically extract features and classify the site.")

    url_input = st.text_input("Enter URL", placeholder="https://www.example.com")

    if st.button("Analyse", key="url_btn", type="primary"):
        if not url_input.strip():
            st.warning("Please enter a URL first.")
        else:
            url_clean = url_input.strip()
            domain = get_domain(url_clean)

            # ── Layer 1: PhishTank real-time threat intel ──────────────────
            phishtank_result = None
            with st.spinner("🔴 Checking PhishTank threat database..."):
                phishtank_result = check_phishtank(url_clean)

            if phishtank_result["is_phish"]:
                st.error(
                    f"🚨 **PHISHING DETECTED (PhishTank)** — `{url_clean}` is a **confirmed phishing URL** "
                    f"in the PhishTank database (ID: {phishtank_result['phish_id']}). "
                    f"Do NOT enter any credentials."
                )
                if phishtank_result["verified"]:
                    st.warning("⚠️ This phish has been **community-verified** as valid.")
                st.info("ℹ️ ML analysis skipped — PhishTank confirmation is definitive.")
                st.stop()

            # ── Layer 2: Typosquatting check ───────────────────────────────
            typo_matches = check_typosquatting(domain)
            if typo_matches:
                st.warning(
                    f"⚠️ **TYPOSQUATTING WARNING** — `{domain}` closely resembles "
                    f"**{', '.join(typo_matches)}**. This may be an impersonation attempt. "
                    f"Proceeding with ML analysis..."
                )

            # ── Layer 3: XGBoost ML classifier ────────────────────────────
            with st.spinner("🤖 Extracting features and running ML classifier..."):
                features_dict = extract_url_features(url_clean)
                input_array = np.array([[features_dict[f] for f in FEATURE_NAMES]])
                input_scaled = scaler.transform(input_array)

                start = time.time()
                prediction = model.predict(input_scaled)[0]
                proba = model.predict_proba(input_scaled)[0]
                pred_time = time.time() - start

            confidence = proba[1] * 100 if prediction == 1 else proba[0] * 100

            col1, col2 = st.columns(2)
            col1.metric("Confidence", f"{confidence:.2f}%")
            col2.metric("Prediction Time", f"{pred_time*1000:.2f} ms")

            # Show PhishTank status even when clean
            if phishtank_result["phish_id"] is None:
                st.caption("🔴 PhishTank: Not in database (or lookup failed)")
            else:
                st.caption("🟢 PhishTank: Not flagged as phishing")

            if prediction == 1:
                st.error(f"🚨 **PHISHING DETECTED** — `{url_clean}` appears to be a phishing site. Do not enter any credentials.")
            else:
                if typo_matches:
                    st.warning(
                        f"⚠️ **PROCEED WITH CAUTION** — ML classifier says legitimate, but `{domain}` "
                        f"resembles **{', '.join(typo_matches)}**. Verify the URL carefully."
                    )
                else:
                    st.success(f"✅ **LEGITIMATE** — `{url_clean}` looks safe based on the extracted features.")

            with st.expander("View extracted features"):
                feat_df = pd.DataFrame({
                    "Feature": FEATURE_NAMES,
                    "Value": [features_dict[f] for f in FEATURE_NAMES],
                    "Description": [FEATURES[f]["desc"] for f in FEATURE_NAMES]
                })
                st.dataframe(feat_df, use_container_width=True)

with tab2:
    st.subheader("Manual Feature Input")
    st.write("Select values for all 30 features manually. `-1` = Legitimate, `0` = Suspicious, `1` = Phishing indicator")
    st.divider()

    input_values = {}
    feature_list = list(FEATURES.items())

    groups = [
        ("🔗 URL-Based Features", feature_list[0:10]),
        ("📄 Content-Based Features", feature_list[10:22]),
        ("🌐 External-Based Features", feature_list[22:30]),
    ]

    for heading, group in groups:
        st.markdown(f"**{heading}**")
        cols = st.columns(2)
        for i, (fname, finfo) in enumerate(group):
            with cols[i % 2]:
                selected = st.selectbox(
                    f"{fname.replace('_', ' ')} — {finfo['desc']}",
                    options=list(finfo["options"].keys()),
                    key=f"feat_{fname}"
                )
                input_values[fname] = finfo["options"][selected]
        st.divider()

    if st.button("Predict", key="manual_btn", type="primary"):
        with st.spinner("Running prediction..."):
            time.sleep(0.3)
            input_array = np.array([[input_values[f] for f in FEATURE_NAMES]])
            input_scaled = scaler.transform(input_array)

            start = time.time()
            prediction = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0]
            pred_time = time.time() - start

        confidence = proba[1] * 100 if prediction == 1 else proba[0] * 100

        col1, col2 = st.columns(2)
        col1.metric("Confidence", f"{confidence:.2f}%")
        col2.metric("Prediction Time", f"{pred_time*1000:.2f} ms")

        if prediction == 1:
            st.error("🚨 **PHISHING DETECTED** — Based on the entered features, this site is classified as phishing.")
        else:
            st.success("✅ **LEGITIMATE** — No phishing indicators detected based on the entered features.")

        with st.expander("View feature summary"):
            summary_df = pd.DataFrame({
                "Feature": FEATURE_NAMES,
                "Value": [input_values[f] for f in FEATURE_NAMES],
                "Description": [FEATURES[f]["desc"] for f in FEATURE_NAMES]
            })
            st.dataframe(summary_df, use_container_width=True)

st.markdown("""
<div class="footer-bar">
    🛡️ &nbsp; PHISHING WEBSITE DETECTION SYSTEM &nbsp;·&nbsp; N. GUNEETH SAI (127004171) &nbsp;·&nbsp; SASTRA UNIVERSITY &nbsp;·&nbsp; XGBOOST + UCI DATASET
</div>
""", unsafe_allow_html=True)
