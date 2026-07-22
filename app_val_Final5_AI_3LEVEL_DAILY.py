import re
import os
import json
import time
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

st.set_page_config(page_title="Meta Ads Team Dashboard", layout="wide")

def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if st.session_state["password_correct"]:
        return True

    st.title("🔐 Login Required")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if password == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("Wrong password")

    return False


if not check_password():
    st.stop()

BASE_URL = "https://graph.facebook.com"
API_VERSION = st.secrets.get("META_API_VERSION", "v17.0")
ACCESS_TOKEN = st.secrets["META_ACCESS_TOKEN"]
BUSINESS_IDS = ["751488620224306", "1178859133269743"]
FETCH_CAMPAIGNS = True  # Needed to fetch campaign status (Active / Not Active)

# OpenAI integration is optional. The dashboard still works without it.
# Keep the key in Streamlit Secrets; never hardcode it in this file.
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", "gpt-5.6-sol")
OPENAI_REASONING_EFFORT = st.secrets.get("OPENAI_REASONING_EFFORT", "high")
OPENAI_REASONING_MODE = st.secrets.get("OPENAI_REASONING_MODE", "standard").lower()
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_TIMEOUT_SECONDS = int(st.secrets.get("OPENAI_TIMEOUT_SECONDS", 180))
META_TIME_INCREMENT = st.secrets.get("META_TIME_INCREMENT", 1)  # Keep 1 for calendar-day comparisons.
AI_WEEK_WINDOW_DAYS = int(st.secrets.get("AI_WEEK_WINDOW_DAYS", 7))
AI_DAILY_HISTORY_DAYS = int(st.secrets.get("AI_DAILY_HISTORY_DAYS", 14))
DECISION_WINDOW_OPTIONS = [
    "Since Launch",
    "Last 2 Calendar Days",
    "Last 3 Calendar Days",
    "Last 7 Calendar Days",
    "Week vs Previous Week",
    "Last 2 Active Spend Days",
]
REFRESH_LOCK_MAX_AGE_SECONDS = 10 * 60

DATA_DIR = Path("app_data")
DATA_DIR.mkdir(exist_ok=True)

FACT_FILE = DATA_DIR / "fact_snapshot.parquet"
ACCOUNTS_FILE = DATA_DIR / "accounts_snapshot.parquet"
RAW_ACCOUNTS_FILE = DATA_DIR / "raw_accounts_snapshot.parquet"
META_FILE = DATA_DIR / "meta_snapshot.json"
GENDER_FILE = DATA_DIR / "gender_snapshot.parquet"
AGE_FILE = DATA_DIR / "age_snapshot.parquet"
BALANCE_FILE = DATA_DIR / "balance_snapshot.parquet"
LOCK_FILE = DATA_DIR / "refresh.lock"

TMP_FACT_FILE = DATA_DIR / "fact_snapshot.tmp.parquet"
TMP_ACCOUNTS_FILE = DATA_DIR / "accounts_snapshot.tmp.parquet"
TMP_RAW_ACCOUNTS_FILE = DATA_DIR / "raw_accounts_snapshot.tmp.parquet"
TMP_META_FILE = DATA_DIR / "meta_snapshot.tmp.json"
TMP_GENDER_FILE = DATA_DIR / "gender_snapshot.tmp.parquet"
TMP_AGE_FILE = DATA_DIR / "age_snapshot.tmp.parquet"
TMP_BALANCE_FILE = DATA_DIR / "balance_snapshot.tmp.parquet"

MEDIA_BUYER_MAP = {
    "AA": "Abdallah Adel",
    "HM": "Ahmed Hesham",
    "BM": "Bassem Shalawy",
    "EK": "Esraa Kamal",
    "MA": "Mahmoud",
    "AF": "Amr Fathy",
    "SQ": "(R)Ahmed Sharkawy",
    "OS": "(R)Osama Serwe",
    "MM": "(R)Mohamed Mahmoud",
    "NB": "(R)Mohamed Nabih",
}

OBJECTIVE_ORDER = [
    "Lead generation",
    "Lead - Message",
    "Whatsapp Message",
    "Conversion",
    "Unknown",
]

# -----------------------------
# Utils
# -----------------------------
def to_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default

def safe_div(a, b):
    try:
        if b and b != 0:
            return a / b
    except Exception:
        pass
    return None

def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip().upper()

def extract_buyer_code(account_name: str) -> str:
    text = normalize_text(account_name)
    for code in MEDIA_BUYER_MAP.keys():
        pattern = rf"(?<![A-Z0-9]){re.escape(code)}(?![A-Z0-9])"
        if re.search(pattern, text):
            return code
    return "UNKNOWN"

def normalize_account_id(account_id):
    if account_id is None:
        return ""
    return str(account_id).replace("act_", "")

def is_relevant_account_name(account_name: str) -> bool:
    text = normalize_text(account_name)
    return (
        "OK-FB-HR-" in text
        or "OK-FB-NF-" in text
        or "US-FB-HR-" in text
        or "US-BO-HR-" in text
    )

def source_is_okaby(source: str) -> bool:
    return "751488620224306" in str(source)

def source_is_val(source: str) -> bool:
    return "1178859133269743" in str(source)

def classify_objective_from_campaign_name(campaign_name: str) -> str:
    text = normalize_text(campaign_name)

    if re.search(r"(?<![A-Z0-9])CONVL(?![A-Z0-9])", text):
        return "Conversion"
    if re.search(r"(?<![A-Z0-9])CONVS(?![A-Z0-9])", text):
        return "Conversion"
    if re.search(r"(?<![A-Z0-9])CONV(?![A-Z0-9])", text):
        return "Conversion"

    if re.search(r"(?<![A-Z0-9])WA(?![A-Z0-9])", text):
        return "Whatsapp Message"
    if re.search(r"(?<![A-Z0-9])LG(?![A-Z0-9])", text):
        return "Lead generation"
    if re.search(r"(?<![A-Z0-9])LM(?![A-Z0-9])", text):
        return "Lead - Message"

    return "Unknown"

def campaign_status_label(status=None, effective_status=None):
    raw = normalize_text(effective_status) or normalize_text(status)
    if raw == "ACTIVE":
        return "Active"
    if raw:
        return "Not Active"
    return "Unknown"

def flatten_actions(actions):
    result = {}
    if not isinstance(actions, list):
        return result

    for item in actions:
        action_type = str(item.get("action_type", "")).strip().lower()
        value = to_float(item.get("value", 0))
        if action_type:
            result[action_type] = result.get(action_type, 0.0) + value
    return result

def get_result_by_objective(objective_label, actions_map):
    if objective_label in {"Lead generation", "Lead - Message"}:
        keys = [
            "offsite_conversion.fb_pixel_lead",
            "onsite_conversion.lead_grouped",
            "offsite_conversion.custom",
        ]
        return sum(actions_map.get(k, 0.0) for k in keys)

    if objective_label == "Conversion":
        return actions_map.get("purchase", 0.0)

    if objective_label == "Whatsapp Message":
        keys = [
            "onsite_conversion.messaging_conversation_started",
            "onsite_conversion.messaging_conversation_started_7d",
        ]
        return sum(actions_map.get(k, 0.0) for k in keys)

    return 0.0

def format_display_df(df):
    out = df.copy()

    money_cols = ["spend", "cpl", "cpc"]
    pct_cols = ["ctr"]
    float_cols = ["frequency"]
    int_like_cols = ["results", "campaigns", "impressions", "clicks"]

    for col in money_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(2)

    for col in pct_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(2)

    for col in float_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(2)

    for col in int_like_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).round(0)

    return out

# -----------------------------
# Persistence
# -----------------------------
def snapshot_exists():
    return FACT_FILE.exists() and META_FILE.exists()

def save_snapshot_atomic(fact, accounts_dedup, accounts_raw, meta, gender_df=None, age_df=None, balance_df=None):
    fact.to_parquet(TMP_FACT_FILE, index=False)
    accounts_dedup.to_parquet(TMP_ACCOUNTS_FILE, index=False)
    accounts_raw.to_parquet(TMP_RAW_ACCOUNTS_FILE, index=False)

    if gender_df is None:
        gender_df = pd.DataFrame()
    if age_df is None:
        age_df = pd.DataFrame()
    if balance_df is None:
        balance_df = pd.DataFrame()

    gender_df.to_parquet(TMP_GENDER_FILE, index=False)
    age_df.to_parquet(TMP_AGE_FILE, index=False)
    balance_df.to_parquet(TMP_BALANCE_FILE, index=False)

    with open(TMP_META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    os.replace(TMP_FACT_FILE, FACT_FILE)
    os.replace(TMP_ACCOUNTS_FILE, ACCOUNTS_FILE)
    os.replace(TMP_RAW_ACCOUNTS_FILE, RAW_ACCOUNTS_FILE)
    os.replace(TMP_GENDER_FILE, GENDER_FILE)
    os.replace(TMP_AGE_FILE, AGE_FILE)
    os.replace(TMP_BALANCE_FILE, BALANCE_FILE)
    os.replace(TMP_META_FILE, META_FILE)

def safe_read_parquet(file_path):
    try:
        if file_path.exists():
            return pd.read_parquet(file_path)
    except Exception:
        try:
            file_path.unlink()
        except Exception:
            pass
    return pd.DataFrame()

def load_snapshot():
    if not snapshot_exists():
        return None

    fact = safe_read_parquet(FACT_FILE)
    accounts_dedup = safe_read_parquet(ACCOUNTS_FILE)
    accounts_raw = safe_read_parquet(RAW_ACCOUNTS_FILE)
    gender_df = safe_read_parquet(GENDER_FILE)
    age_df = safe_read_parquet(AGE_FILE)
    balance_df = safe_read_parquet(BALANCE_FILE)

    if fact.empty:
        return None

    try:
        with open(META_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        meta = {}

    return {
        "fact": fact,
        "accounts_dedup": accounts_dedup,
        "accounts_raw": accounts_raw,
        "gender_df": gender_df,
        "age_df": age_df,
        "balance_df": balance_df,
        "meta": meta,
    }

# -----------------------------
# API
# -----------------------------
@st.cache_data(ttl=1800)
def fetch_all_pages(url, params=None):
    all_rows = []

    while True:
        response = requests.get(url, params=params, timeout=90)
        response.raise_for_status()
        data = response.json()

        all_rows.extend(data.get("data", []))

        paging = data.get("paging", {})
        next_url = paging.get("next")
        if not next_url:
            break

        url = next_url
        params = None

    return all_rows

@st.cache_data(ttl=1800)
def get_ad_accounts():
    all_dfs = []

    sources = [
        ("me/adaccounts", f"{BASE_URL}/{API_VERSION}/me/adaccounts"),
    ]
    for business_id in BUSINESS_IDS:
        sources.extend([
            (f"business/{business_id}/owned_ad_accounts", f"{BASE_URL}/{API_VERSION}/{business_id}/owned_ad_accounts"),
            (f"business/{business_id}/client_ad_accounts", f"{BASE_URL}/{API_VERSION}/{business_id}/client_ad_accounts"),
        ])

    for source_name, url in sources:
        try:
            params = {
                "fields": "id,account_id,name,account_status,currency,funding_source_details",
                "access_token": ACCESS_TOKEN,
                "limit": 500,
            }
            rows = fetch_all_pages(url, params)
            df = pd.DataFrame(rows)
            if not df.empty:
                df["source"] = source_name
                all_dfs.append(df)
        except Exception:
            pass

    if not all_dfs:
        return pd.DataFrame(), pd.DataFrame()

    raw_accounts = pd.concat(all_dfs, ignore_index=True)

    # Speed optimization with safe Unknown handling:
    # - Fetch all El-Okaby business accounts, even if naming is not coded correctly.
    #   These will appear under Media Buyer = Unknown.
    # - Fetch VAL accounts only when they match VAL Hair / VAL Booty naming.
    # - Keep matching accounts from /me/adaccounts as well.
    if "name" in raw_accounts.columns:
        name_match = raw_accounts["name"].apply(is_relevant_account_name)
        okaby_source = raw_accounts["source"].apply(source_is_okaby) if "source" in raw_accounts.columns else False
        raw_accounts = raw_accounts[name_match | okaby_source].copy()

    dedup = raw_accounts.copy()
    if "id" in dedup.columns:
        dedup = dedup.sort_values(["name", "source"]).drop_duplicates(subset=["id"], keep="first")
    elif "account_id" in dedup.columns:
        dedup = dedup.sort_values(["name", "source"]).drop_duplicates(subset=["account_id"], keep="first")

    return dedup.reset_index(drop=True), raw_accounts.reset_index(drop=True)

@st.cache_data(ttl=1800)
def get_campaigns(account_id):
    clean_id = str(account_id).replace("act_", "")
    url = f"{BASE_URL}/{API_VERSION}/act_{clean_id}/campaigns"
    params = {
        "fields": "id,name,status,effective_status",
        "access_token": ACCESS_TOKEN,
        "limit": 1000,
    }
    rows = fetch_all_pages(url, params)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["account_id"] = f"act_{clean_id}"
    return df


@st.cache_data(ttl=1800)
def get_adsets(account_id):
    clean_id = str(account_id).replace("act_", "")
    url = f"{BASE_URL}/{API_VERSION}/act_{clean_id}/adsets"
    params = {
        "fields": "id,name,campaign_id,status,effective_status",
        "access_token": ACCESS_TOKEN,
        "limit": 1000,
    }
    rows = fetch_all_pages(url, params)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["account_id"] = f"act_{clean_id}"
    return df


@st.cache_data(ttl=1800)
def get_ads(account_id):
    clean_id = str(account_id).replace("act_", "")
    url = f"{BASE_URL}/{API_VERSION}/act_{clean_id}/ads"
    params = {
        "fields": "id,name,campaign_id,adset_id,status,effective_status",
        "access_token": ACCESS_TOKEN,
        "limit": 1000,
    }
    rows = fetch_all_pages(url, params)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["account_id"] = f"act_{clean_id}"
    return df


@st.cache_data(ttl=1800)
def get_insights_for_account(account_id, since, until, level="campaign"):
    """Fetch daily Meta insights for one hierarchy level."""
    if level not in {"campaign", "adset", "ad"}:
        raise ValueError(f"Unsupported insights level: {level}")

    clean_id = str(account_id).replace("act_", "")
    url = f"{BASE_URL}/{API_VERSION}/act_{clean_id}/insights"

    hierarchy_fields = {
        "campaign": ["campaign_id", "campaign_name"],
        "adset": ["campaign_id", "campaign_name", "adset_id", "adset_name"],
        "ad": [
            "campaign_id", "campaign_name", "adset_id", "adset_name",
            "ad_id", "ad_name",
        ],
    }
    metric_fields = [
        "account_id", "account_name", "spend", "impressions", "reach",
        "clicks", "inline_link_clicks", "ctr", "cpc", "cpm", "frequency",
        "actions", "date_start", "date_stop",
    ]
    params = {
        "fields": ",".join(hierarchy_fields[level] + metric_fields),
        "level": level,
        "time_increment": META_TIME_INCREMENT,
        "time_range": f'{{"since":"{since}","until":"{until}"}}',
        "access_token": ACCESS_TOKEN,
        "limit": 1000,
    }

    rows = fetch_all_pages(url, params)
    df = pd.DataFrame(rows)
    if "actions" not in df.columns:
        df["actions"] = None
    df["entity_level"] = level
    return df


@st.cache_data(ttl=1800)
def get_gender_spend(account_id, since, until):
    clean_id = str(account_id).replace("act_", "")
    url = f"{BASE_URL}/{API_VERSION}/act_{clean_id}/insights"
    params = {
        "fields": "spend",
        "breakdowns": "gender",
        "time_range": f'{{"since":"{since}","until":"{until}"}}',
        "access_token": ACCESS_TOKEN,
        "limit": 1000,
    }

    rows = fetch_all_pages(url, params)
    df = pd.DataFrame(rows)

    if not df.empty:
        df["account_id"] = f"act_{clean_id}"

    if "spend" not in df.columns:
        df["spend"] = 0
    if "gender" not in df.columns:
        df["gender"] = "unknown"

    return df

@st.cache_data(ttl=1800)
def get_age_spend(account_id, since, until):
    clean_id = str(account_id).replace("act_", "")
    url = f"{BASE_URL}/{API_VERSION}/act_{clean_id}/insights"
    params = {
        "fields": "spend",
        "breakdowns": "age",
        "time_range": f'{{"since":"{since}","until":"{until}"}}',
        "access_token": ACCESS_TOKEN,
        "limit": 1000,
    }

    rows = fetch_all_pages(url, params)
    df = pd.DataFrame(rows)

    if not df.empty:
        df["account_id"] = f"act_{clean_id}"

    if "spend" not in df.columns:
        df["spend"] = 0
    if "age" not in df.columns:
        df["age"] = "unknown"

    return df

@st.cache_data(ttl=1800)
def get_age_gender_spend(account_id, since, until):
    clean_id = str(account_id).replace("act_", "")
    url = f"{BASE_URL}/{API_VERSION}/act_{clean_id}/insights"
    params = {
        "fields": "spend",
        "breakdowns": "age,gender",
        "time_range": f'{{"since":"{since}","until":"{until}"}}',
        "access_token": ACCESS_TOKEN,
        "limit": 1000,
    }

    rows = fetch_all_pages(url, params)
    df = pd.DataFrame(rows)

    if not df.empty:
        df["account_id"] = f"act_{clean_id}"

    if "spend" not in df.columns:
        df["spend"] = 0
    if "gender" not in df.columns:
        df["gender"] = "unknown"
    if "age" not in df.columns:
        df["age"] = "unknown"

    return df

def split_age_gender_breakdown(age_gender_df):
    if age_gender_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = age_gender_df.copy()
    df["spend"] = pd.to_numeric(df["spend"], errors="coerce").fillna(0)

    gender_df = (
        df.groupby(["account_id", "gender"], dropna=False)
        .agg(spend=("spend", "sum"))
        .reset_index()
    )

    age_df = (
        df.groupby(["account_id", "age"], dropna=False)
        .agg(spend=("spend", "sum"))
        .reset_index()
    )

    return gender_df, age_df

def parse_balance_from_display_string(display_string):
    if not display_string:
        return None

    match = re.search(r"([\d,.]+)", str(display_string))
    if not match:
        return None

    return to_float(match.group(1).replace(",", ""), default=None)

@st.cache_data(ttl=1800)
def get_account_balance(account_id):
    clean_id = str(account_id).replace("act_", "")
    url = f"{BASE_URL}/{API_VERSION}/act_{clean_id}"
    params = {
        "fields": "name,funding_source_details",
        "access_token": ACCESS_TOKEN,
    }

    response = requests.get(url, params=params, timeout=90)
    response.raise_for_status()
    data = response.json()

    display_string = None
    if isinstance(data.get("funding_source_details"), dict):
        display_string = data["funding_source_details"].get("display_string")

    balance = parse_balance_from_display_string(display_string)

    return {
        "account_id": f"act_{clean_id}",
        "account_name": data.get("name", "Unknown"),
        "balance": balance,
        "balance_display_string": display_string or "N/A",
    }

def fetch_one_account(row, since, until):
    account_id = row["id"]
    account_name = row.get("name", account_id)

    campaigns_df = pd.DataFrame()
    adsets_df = pd.DataFrame()
    ads_df = pd.DataFrame()
    insight_frames = []
    gender_df = pd.DataFrame()
    age_df = pd.DataFrame()
    balance_row = {}
    errors = []

    if FETCH_CAMPAIGNS:
        try:
            campaigns_df = get_campaigns(account_id)
        except Exception as e:
            errors.append(f"campaign metadata: {e}")
        try:
            adsets_df = get_adsets(account_id)
        except Exception as e:
            errors.append(f"ad set metadata: {e}")
        try:
            ads_df = get_ads(account_id)
        except Exception as e:
            errors.append(f"ad metadata: {e}")

    for level in ["campaign", "adset", "ad"]:
        try:
            level_df = get_insights_for_account(account_id, str(since), str(until), level)
            if not level_df.empty:
                insight_frames.append(level_df)
        except Exception as e:
            errors.append(f"{level} insights: {e}")

    try:
        age_gender_df = get_age_gender_spend(account_id, str(since), str(until))
        gender_df, age_df = split_age_gender_breakdown(age_gender_df)
    except Exception:
        try:
            gender_df = get_gender_spend(account_id, str(since), str(until))
            age_df = get_age_spend(account_id, str(since), str(until))
        except Exception as e:
            errors.append(f"audience breakdown: {e}")

    insights_df = pd.concat(insight_frames, ignore_index=True) if insight_frames else pd.DataFrame()
    error = f"{account_name}: " + " | ".join(errors) if errors else None

    return {
        "account_id": account_id,
        "account_name": account_name,
        "campaigns_df": campaigns_df,
        "adsets_df": adsets_df,
        "ads_df": ads_df,
        "insights_df": insights_df,
        "gender_df": gender_df,
        "age_df": age_df,
        "balance_row": balance_row,
        "error": error,
    }

# -----------------------------
# Transform
# -----------------------------
def _prepare_entity_metadata_map(df, entity_level):
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["_account_id_clean"] = out["account_id"].apply(normalize_account_id)
    rename = {
        "id": f"{entity_level}_id",
        "name": f"{entity_level}_name_master",
        "status": f"{entity_level}_status_raw",
        "effective_status": f"{entity_level}_effective_status_raw",
    }
    return out.rename(columns=rename)


def prepare_data(all_campaigns_df, all_adsets_df, all_ads_df, all_insights_df):
    """Normalize campaign, ad-set and ad rows into one hierarchy-aware fact table."""
    if all_insights_df.empty:
        return pd.DataFrame()

    fact = all_insights_df.copy()
    if "entity_level" not in fact.columns:
        fact["entity_level"] = "campaign"

    required_text_cols = [
        "account_id", "account_name", "campaign_id", "campaign_name",
        "adset_id", "adset_name", "ad_id", "ad_name",
    ]
    for col in required_text_cols:
        if col not in fact.columns:
            fact[col] = None

    if "actions" not in fact.columns:
        fact["actions"] = None

    numeric_cols = [
        "spend", "clicks", "inline_link_clicks", "impressions", "reach",
        "ctr", "cpc", "cpm", "frequency",
    ]
    for col in numeric_cols:
        if col in fact.columns:
            fact[col] = pd.to_numeric(fact[col], errors="coerce").fillna(0)
        else:
            fact[col] = 0

    fact["_account_id_clean"] = fact["account_id"].apply(normalize_account_id)

    campaign_map = _prepare_entity_metadata_map(all_campaigns_df, "campaign")
    if not campaign_map.empty:
        keep = [
            "_account_id_clean", "campaign_id", "campaign_name_master",
            "campaign_status_raw", "campaign_effective_status_raw",
        ]
        fact = fact.merge(campaign_map[[c for c in keep if c in campaign_map.columns]],
                          on=["_account_id_clean", "campaign_id"], how="left")
    else:
        for col in ["campaign_name_master", "campaign_status_raw", "campaign_effective_status_raw"]:
            fact[col] = None

    adset_map = _prepare_entity_metadata_map(all_adsets_df, "adset")
    if not adset_map.empty:
        keep = [
            "_account_id_clean", "adset_id", "adset_name_master",
            "adset_status_raw", "adset_effective_status_raw",
        ]
        fact = fact.merge(adset_map[[c for c in keep if c in adset_map.columns]],
                          on=["_account_id_clean", "adset_id"], how="left")
    else:
        for col in ["adset_name_master", "adset_status_raw", "adset_effective_status_raw"]:
            fact[col] = None

    ad_map = _prepare_entity_metadata_map(all_ads_df, "ad")
    if not ad_map.empty:
        keep = [
            "_account_id_clean", "ad_id", "ad_name_master",
            "ad_status_raw", "ad_effective_status_raw",
        ]
        fact = fact.merge(ad_map[[c for c in keep if c in ad_map.columns]],
                          on=["_account_id_clean", "ad_id"], how="left")
    else:
        for col in ["ad_name_master", "ad_status_raw", "ad_effective_status_raw"]:
            fact[col] = None

    fact["campaign_name"] = fact["campaign_name"].fillna(fact["campaign_name_master"]).fillna("Unknown")
    fact["adset_name"] = fact["adset_name"].fillna(fact["adset_name_master"]).fillna("Unknown")
    fact["ad_name"] = fact["ad_name"].fillna(fact["ad_name_master"]).fillna("Unknown")
    fact["account_name"] = fact["account_name"].fillna("Unknown")

    fact["campaign_status"] = fact.apply(
        lambda r: campaign_status_label(r.get("campaign_status_raw"), r.get("campaign_effective_status_raw")),
        axis=1,
    )
    fact["adset_status"] = fact.apply(
        lambda r: campaign_status_label(r.get("adset_status_raw"), r.get("adset_effective_status_raw")),
        axis=1,
    )
    fact["ad_status"] = fact.apply(
        lambda r: campaign_status_label(r.get("ad_status_raw"), r.get("ad_effective_status_raw")),
        axis=1,
    )

    def resolve_entity_status(row):
        if row.get("entity_level") == "ad":
            return row.get("ad_status", "Unknown")
        if row.get("entity_level") == "adset":
            return row.get("adset_status", "Unknown")
        return row.get("campaign_status", "Unknown")

    fact["entity_status"] = fact.apply(resolve_entity_status, axis=1)
    fact["entity_id"] = fact.apply(
        lambda r: r.get("ad_id") if r.get("entity_level") == "ad"
        else r.get("adset_id") if r.get("entity_level") == "adset"
        else r.get("campaign_id"), axis=1,
    )
    fact["entity_name"] = fact.apply(
        lambda r: r.get("ad_name") if r.get("entity_level") == "ad"
        else r.get("adset_name") if r.get("entity_level") == "adset"
        else r.get("campaign_name"), axis=1,
    )

    fact["buyer_code"] = fact["account_name"].apply(extract_buyer_code)
    fact["media_buyer"] = fact["buyer_code"].map(MEDIA_BUYER_MAP).fillna("Unknown")
    fact["objective_label"] = fact["campaign_name"].apply(classify_objective_from_campaign_name)
    fact["actions_map"] = fact["actions"].apply(flatten_actions)
    fact["results"] = fact.apply(
        lambda r: get_result_by_objective(r["objective_label"], r["actions_map"]), axis=1
    )
    fact["cpl"] = fact.apply(lambda r: safe_div(r["spend"], r["results"]), axis=1)
    fact = fact.drop(columns=["_account_id_clean"], errors="ignore")
    return fact

# -----------------------------
# Summaries
# -----------------------------
def build_overall_summary(fact):
    total_spend = fact["spend"].sum()
    total_results = fact["results"].sum()
    return {
        "total_spend": total_spend,
        "total_results": total_results,
        "total_cpl": safe_div(total_spend, total_results),
    }

def build_objective_summary(fact):
    out = (
        fact.groupby("objective_label", dropna=False)
        .agg(
            spend=("spend", "sum"),
            results=("results", "sum"),
            campaigns=("campaign_id", "nunique"),
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
        )
        .reset_index()
    )
    out["cpl"] = out.apply(lambda r: safe_div(r["spend"], r["results"]), axis=1)
    out["ctr"] = out.apply(lambda r: safe_div(r["clicks"], r["impressions"]) * 100 if r["impressions"] > 0 else None, axis=1)
    out["cpc"] = out.apply(lambda r: safe_div(r["spend"], r["clicks"]), axis=1)
    out["objective_label"] = pd.Categorical(out["objective_label"], OBJECTIVE_ORDER)
    return out.sort_values(["objective_label", "spend"], ascending=[True, False]).reset_index(drop=True)

def build_buyer_summary(fact):
    out = (
        fact.groupby(["buyer_code", "media_buyer"], dropna=False)
        .agg(
            spend=("spend", "sum"),
            results=("results", "sum"),
            campaigns=("campaign_id", "nunique"),
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
        )
        .reset_index()
    )
    out["cpl"] = out.apply(lambda r: safe_div(r["spend"], r["results"]), axis=1)
    out["ctr"] = out.apply(lambda r: safe_div(r["clicks"], r["impressions"]) * 100 if r["impressions"] > 0 else None, axis=1)
    out["cpc"] = out.apply(lambda r: safe_div(r["spend"], r["clicks"]), axis=1)
    return out.sort_values("spend", ascending=False).reset_index(drop=True)

def add_overall_row_to_buyer_summary(buyer_summary_df):
    if buyer_summary_df.empty:
        return buyer_summary_df

    df = buyer_summary_df.copy()

    total_spend = df["spend"].sum()
    total_results = df["results"].sum()
    total_campaigns = df["campaigns"].sum()
    total_impressions = df["impressions"].sum()
    total_clicks = df["clicks"].sum()

    overall_row = {
        "buyer_code": "All",
        "media_buyer": "Overall All Agents",
        "spend": total_spend,
        "results": total_results,
        "campaigns": total_campaigns,
        "impressions": total_impressions,
        "clicks": total_clicks,
        "cpl": safe_div(total_spend, total_results),
        "ctr": safe_div(total_clicks, total_impressions) * 100 if total_impressions > 0 else None,
        "cpc": safe_div(total_spend, total_clicks),
    }

    return pd.concat([df, pd.DataFrame([overall_row])], ignore_index=True)




def detect_business_unit(account_name="", campaign_name="", source=""):
    acc = normalize_text(account_name)
    camp = normalize_text(campaign_name)
    src = str(source)

    # IMPORTANT:
    # Account-name rules must come first and must win over campaign-name fallback.
    # El - Okaby account examples:
    # OK-FB-HR-AA
    # OK-FB-NF-AA
    # OK-FB-NF-NB-004
    # (R) OK-FB-HR-OS
    if "OK-FB-HR-" in acc or "OK-FB-NF-" in acc:
        return "El - Okaby"

    # VAL is split into two separate business units.
    # Val Hair account example: US-FB-HR-AA
    # Do not classify any OK-* account as Val Hair.
    if "US-FB-HR-" in acc:
        return "Val Hair"

    # VAL Booty account example: US-BO-HR-AA
    if "US-BO-HR-" in acc:
        return "VAL Booty"

    # Any account coming from El-Okaby business but not matching the naming convention
    # should still be fetched and shown as Unknown media buyer under El - Okaby.
    if source_is_okaby(src):
        return "El - Okaby"

    # Fallback from campaign naming ONLY when the account name is not enough.
    # Val Hair campaign example: AA-FB-HR _ LG
    if re.search(r"(?<![A-Z0-9])(?:AA|HM|BM|EK|MA|AF|SQ|OS|MM|NB)-FB-HR(?![A-Z0-9])", camp):
        return "Val Hair"

    # VAL Booty campaign example: AA-FB-BO _ Convs
    if re.search(r"(?<![A-Z0-9])(?:AA|HM|BM|EK|MA|AF|SQ|OS|MM|NB)-FB-BO(?![A-Z0-9])", camp):
        return "VAL Booty"

    return "Unknown"

def add_business_unit_columns(fact):
    if fact.empty:
        return fact
    out = fact.copy()
    out["business_unit"] = out.apply(
        lambda r: detect_business_unit(r.get("account_name", ""), r.get("campaign_name", "")),
        axis=1,
    )
    return out


def add_business_unit_to_accounts(accounts_df):
    if accounts_df.empty:
        return accounts_df.copy()
    out = accounts_df.copy()
    name_col = "name" if "name" in out.columns else "account_name" if "account_name" in out.columns else None
    source_col = "source" if "source" in out.columns else None
    if name_col is None:
        out["business_unit"] = "Unknown"
    else:
        out["business_unit"] = out.apply(
            lambda r: detect_business_unit(r.get(name_col, ""), "", r.get(source_col, "") if source_col else ""),
            axis=1,
        )
    return out

def assign_fact_business_unit_from_accounts(fact, accounts_df):
    if fact.empty:
        return fact
    out = add_business_unit_columns(fact)
    if accounts_df.empty or "id" not in accounts_df.columns:
        return out
    acc_units = add_business_unit_to_accounts(accounts_df)
    acc_units = acc_units[["id", "business_unit"]].drop_duplicates().rename(
        columns={"id": "account_id", "business_unit": "account_business_unit"}
    )
    out = out.merge(acc_units, on="account_id", how="left")

    # Account classification is more reliable than campaign fallback.
    # If the account has a known unit, it overrides the campaign-based unit.
    out["business_unit"] = out.apply(
        lambda r: r["account_business_unit"]
        if pd.notna(r.get("account_business_unit")) and str(r.get("account_business_unit")) != "Unknown"
        else r.get("business_unit", "Unknown"),
        axis=1,
    )
    out["business_unit"] = out["business_unit"].fillna("Unknown")
    out = out.drop(columns=["account_business_unit"], errors="ignore")
    return out

def filter_related_by_accounts(df, accounts_df):
    if df.empty or accounts_df.empty or "account_id" not in df.columns:
        return df.copy()
    id_col = "id" if "id" in accounts_df.columns else "account_id" if "account_id" in accounts_df.columns else None
    if id_col is None:
        return df.copy()
    account_ids = set(accounts_df[id_col].dropna().astype(str).unique())
    return df[df["account_id"].astype(str).isin(account_ids)].copy()

def filter_accounts_by_business_unit(accounts_df, business_unit):
    if accounts_df.empty or business_unit == "All":
        return accounts_df.copy()
    out = add_business_unit_to_accounts(accounts_df)
    return out[out["business_unit"] == business_unit].copy()

def filter_by_business_unit(fact, business_unit):
    if fact.empty or business_unit == "All":
        return fact.copy()
    if "business_unit" not in fact.columns:
        fact = add_business_unit_columns(fact)
    return fact[fact["business_unit"] == business_unit].copy()

def filter_related_by_fact_accounts(df, fact):
    if df.empty or fact.empty or "account_id" not in df.columns or "account_id" not in fact.columns:
        return df.copy()
    account_ids = set(fact["account_id"].dropna().astype(str).unique())
    return df[df["account_id"].astype(str).isin(account_ids)].copy()

def build_agent_objective_report(fact, objective_label=None):
    if fact.empty:
        return pd.DataFrame(columns=["Media Buyer", "Spent (EGP)", "Results", "CPL"])
    df = fact.copy()
    if objective_label is not None:
        df = df[df["objective_label"] == objective_label]
    if df.empty:
        return pd.DataFrame(columns=["Media Buyer", "Spent (EGP)", "Results", "CPL"])
    out = (
        df.groupby("media_buyer", dropna=False)
        .agg(**{"Spent (EGP)": ("spend", "sum"), "Results": ("results", "sum")})
        .reset_index()
        .rename(columns={"media_buyer": "Media Buyer"})
    )
    out["CPL"] = out.apply(lambda r: safe_div(r["Spent (EGP)"], r["Results"]), axis=1)
    total = pd.DataFrame([{
        "Media Buyer": "🔵 Overall",
        "Spent (EGP)": out["Spent (EGP)"].sum(),
        "Results": out["Results"].sum(),
        "CPL": safe_div(out["Spent (EGP)"].sum(), out["Results"].sum()),
    }])
    out = out.sort_values("Spent (EGP)", ascending=False)
    return pd.concat([out, total], ignore_index=True)

def build_campaign_details_for_agent(fact, media_buyer, objective_label=None):
    if fact.empty or media_buyer in [None, ""]:
        return pd.DataFrame()
    df = fact[fact["media_buyer"] == media_buyer].copy()
    if objective_label is not None:
        df = df[df["objective_label"] == objective_label]
    if df.empty:
        return pd.DataFrame()
    out = build_campaign_summary(df)
    return out.rename(columns={
        "account_name": "Ad Account Name",
        "objective_label": "Objective",
        "campaign_name": "Campaign",
        "campaign_status": "Campaign Status",
        "spend": "Spent",
        "results": "Results",
        "cpl": "CPL",
        "ctr": "CTR",
        "cpc": "CPC",
        "frequency": "Frequency",
    })

def build_unified_campaign_details(fact, media_buyer="🔵 Overall", objective_label="All", campaign_status="All"):
    if fact.empty:
        return pd.DataFrame()

    df = fact.copy()

    if media_buyer not in [None, "", "🔵 Overall", "All"]:
        df = df[df["media_buyer"] == media_buyer]

    if objective_label not in [None, "", "All"]:
        df = df[df["objective_label"] == objective_label]

    if campaign_status not in [None, "", "All"] and "campaign_status" in df.columns:
        df = df[df["campaign_status"] == campaign_status]

    if df.empty:
        return pd.DataFrame()

    out = build_campaign_summary(df)
    return out.rename(columns={
        "media_buyer": "Media Buyer",
        "objective_label": "Objective",
        "account_name": "Ad Account Name",
        "campaign_name": "Campaign",
        "campaign_status": "Campaign Status",
        "spend": "Spent",
        "results": "Results",
        "cpl": "CPL",
        "ctr": "CTR",
        "cpc": "CPC",
        "frequency": "Frequency",
    })

def render_objective_agent_section(fact, objective_label, title, icon, key_prefix):
    st.subheader(f"{icon} {title}")
    report_df = build_agent_objective_report(fact, objective_label)
    if report_df.empty:
        st.info("No data for this section.")
        return
    st.dataframe(format_display_df(report_df), use_container_width=True, hide_index=True)

def render_media_buyer_campaign_details(fact):
    st.markdown("### 📋 Media Buyer Campaign Details")

    if fact.empty:
        st.info("No campaigns found.")
        return

    buyer_report = build_agent_objective_report(fact, None)
    agents = [x for x in buyer_report["Media Buyer"].tolist() if x != "🔵 Overall"] if not buyer_report.empty else []
    agents = ["🔵 Overall"] + agents

    objective_options = ["All"] + [obj for obj in OBJECTIVE_ORDER if obj in fact["objective_label"].dropna().astype(str).unique().tolist()]

    if "campaign_status" in fact.columns:
        status_values = fact["campaign_status"].dropna().astype(str).unique().tolist()
    else:
        status_values = []
    preferred_status_order = ["Active", "Not Active", "Unknown"]
    status_options = ["All"] + [s for s in preferred_status_order if s in status_values]

    f1, f2, f3 = st.columns(3)
    with f1:
        selected_agent = st.selectbox("اختر Media Buyer", agents, key="campaign_details_agent")
    with f2:
        selected_objective = st.selectbox("اختر Objective", objective_options, key="campaign_details_objective")
    with f3:
        selected_status = st.selectbox("Campaign Status", status_options, key="campaign_details_status")

    campaign_df = build_unified_campaign_details(
        fact,
        media_buyer=selected_agent,
        objective_label=selected_objective,
        campaign_status=selected_status,
    )

    if campaign_df.empty:
        st.info("No campaigns found for the selected filters.")
    else:
        cols = [
            "Ad Account Name",
            "Media Buyer",
            "Objective",
            "Campaign",
            "Campaign Status",
            "Spent",
            "Results",
            "CPL",
            "CTR",
            "CPC",
            "Frequency",
            "impressions",
            "clicks",
        ]
        cols = [c for c in cols if c in campaign_df.columns]
        display_df = format_display_df(campaign_df[cols])

        sticky_cols = [c for c in ["Ad Account Name", "Media Buyer"] if c in display_df.columns]
        if sticky_cols:
            display_df = display_df.set_index(sticky_cols)
            st.dataframe(display_df, use_container_width=True)
        else:
            st.dataframe(display_df, use_container_width=True, hide_index=True)

def render_overall_agent_section(fact):
    st.subheader("👥 Overall Agent")
    report_df = build_agent_objective_report(fact, None)
    st.dataframe(format_display_df(report_df), use_container_width=True, hide_index=True)
    agents = [x for x in report_df["Media Buyer"].tolist() if x != "🔵 Overall"]
    if not agents:
        return
    selected_agent = st.selectbox("اختر Media Buyer", agents, key="overall_agent_select")
    agent_obj = build_buyer_objective_summary(fact[fact["media_buyer"] == selected_agent])
    if not agent_obj.empty:
        st.markdown(f"### {selected_agent} - Objectives")
        st.dataframe(format_display_df(agent_obj[["objective_label", "spend", "results", "campaigns", "cpl", "ctr", "cpc"]]), use_container_width=True, hide_index=True)

def build_buyer_objective_summary(fact):
    out = (
        fact.groupby(["media_buyer", "objective_label"], dropna=False)
        .agg(
            spend=("spend", "sum"),
            results=("results", "sum"),
            campaigns=("campaign_id", "nunique"),
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
        )
        .reset_index()
    )
    out["cpl"] = out.apply(lambda r: safe_div(r["spend"], r["results"]), axis=1)
    out["ctr"] = out.apply(lambda r: safe_div(r["clicks"], r["impressions"]) * 100 if r["impressions"] > 0 else None, axis=1)
    out["cpc"] = out.apply(lambda r: safe_div(r["spend"], r["clicks"]), axis=1)
    out["objective_label"] = pd.Categorical(out["objective_label"], OBJECTIVE_ORDER)
    return out.sort_values(["media_buyer", "objective_label", "spend"], ascending=[True, True, False]).reset_index(drop=True)


# -----------------------------
# AI hierarchy analyst: Campaign -> Ad Set -> Ad
# -----------------------------
def _json_safe_value(value):
    """Convert pandas / numpy values to JSON-safe native Python values."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _json_safe_nested(value):
    if isinstance(value, dict):
        return {str(k): _json_safe_nested(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_nested(v) for v in value]
    return _json_safe_value(value)


def _records_for_json(df):
    if df is None or df.empty:
        return []
    return [
        {str(k): _json_safe_nested(v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]


def _weighted_metric(df, numerator_col, denominator_col, multiplier=1.0):
    if df.empty or numerator_col not in df.columns or denominator_col not in df.columns:
        return None
    numerator = pd.to_numeric(df[numerator_col], errors="coerce").fillna(0).sum()
    denominator = pd.to_numeric(df[denominator_col], errors="coerce").fillna(0).sum()
    return safe_div(numerator, denominator) * multiplier if denominator > 0 else None


def _fact_for_level(fact, level):
    if fact.empty:
        return fact.copy()
    if "entity_level" not in fact.columns:
        return fact.copy() if level == "campaign" else pd.DataFrame(columns=fact.columns)
    return fact[fact["entity_level"] == level].copy()


def _entity_group_cols(level):
    common = ["account_id", "account_name", "media_buyer", "objective_label", "campaign_id", "campaign_name"]
    if level == "campaign":
        return common
    if level == "adset":
        return common + ["adset_id", "adset_name"]
    if level == "ad":
        return common + ["adset_id", "adset_name", "ad_id", "ad_name"]
    raise ValueError(f"Unsupported level: {level}")


def build_entity_summary(fact, level):
    """Aggregate one hierarchy level without double-counting other levels."""
    df = _fact_for_level(fact, level)
    if df.empty:
        return pd.DataFrame()

    group_cols = _entity_group_cols(level)
    rows = []
    for keys, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        spend = pd.to_numeric(grp["spend"], errors="coerce").fillna(0).sum()
        results = pd.to_numeric(grp["results"], errors="coerce").fillna(0).sum()
        impressions = pd.to_numeric(grp["impressions"], errors="coerce").fillna(0).sum()
        reach = pd.to_numeric(grp.get("reach", 0), errors="coerce").fillna(0).sum()
        clicks = pd.to_numeric(grp["clicks"], errors="coerce").fillna(0).sum()
        link_clicks = pd.to_numeric(grp.get("inline_link_clicks", 0), errors="coerce").fillna(0).sum()

        status_values = grp.get("entity_status", pd.Series(dtype=str)).dropna().astype(str)
        row.update({
            "entity_level": level,
            "entity_id": row.get("ad_id") if level == "ad" else row.get("adset_id") if level == "adset" else row.get("campaign_id"),
            "entity_name": row.get("ad_name") if level == "ad" else row.get("adset_name") if level == "adset" else row.get("campaign_name"),
            "entity_status": status_values.iloc[0] if not status_values.empty else "Unknown",
            "campaign_status": grp.get("campaign_status", pd.Series(dtype=str)).dropna().astype(str).iloc[0]
                if "campaign_status" in grp.columns and not grp["campaign_status"].dropna().empty else "Unknown",
            "spend": spend,
            "results": results,
            "cpl": safe_div(spend, results),
            "impressions": impressions,
            "reach": reach,
            "clicks": clicks,
            "inline_link_clicks": link_clicks,
            "ctr": safe_div(clicks, impressions) * 100 if impressions > 0 else None,
            "cpc": safe_div(spend, clicks),
            "cpm": safe_div(spend, impressions) * 1000 if impressions > 0 else None,
            "frequency": (
                safe_div(
                    (
                        pd.to_numeric(grp["frequency"], errors="coerce").fillna(0)
                        * pd.to_numeric(grp["impressions"], errors="coerce").fillna(0)
                    ).sum(),
                    impressions,
                ) if impressions > 0 and "frequency" in grp.columns else None
            ),
        })
        rows.append(row)

    return pd.DataFrame(rows).sort_values("spend", ascending=False).reset_index(drop=True)


def _parse_period_date(value, fallback=None):
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return fallback
    return parsed.normalize()


def _pct_change(current, old):
    if current is None or old in [None, 0]:
        return None
    try:
        if pd.isna(current) or pd.isna(old):
            return None
    except Exception:
        pass
    return safe_div(current - old, old) * 100


def _daily_metrics(frame):
    """Aggregate a hierarchy entity to one row per calendar day."""
    if frame is None or frame.empty or "date_start" not in frame.columns:
        return pd.DataFrame(columns=[
            "date_start", "spend", "results", "impressions", "reach", "clicks",
            "inline_link_clicks", "frequency", "cpl", "ctr", "cpc", "cpm",
        ])

    df = frame.copy()
    df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date_start"])
    if df.empty:
        return pd.DataFrame()

    numeric_cols = [
        "spend", "results", "impressions", "reach", "clicks",
        "inline_link_clicks", "frequency",
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["_frequency_weight"] = df["frequency"] * df["impressions"]
    daily = (
        df.groupby("date_start", dropna=False)
        .agg(
            spend=("spend", "sum"),
            results=("results", "sum"),
            impressions=("impressions", "sum"),
            reach=("reach", "sum"),
            clicks=("clicks", "sum"),
            inline_link_clicks=("inline_link_clicks", "sum"),
            frequency_weight=("_frequency_weight", "sum"),
        )
        .sort_index()
    )
    daily["frequency"] = daily.apply(
        lambda r: safe_div(r["frequency_weight"], r["impressions"]) if r["impressions"] > 0 else None,
        axis=1,
    )
    daily = daily.drop(columns=["frequency_weight"], errors="ignore")
    daily["cpl"] = daily.apply(lambda r: safe_div(r["spend"], r["results"]), axis=1)
    daily["ctr"] = daily.apply(
        lambda r: safe_div(r["clicks"], r["impressions"]) * 100 if r["impressions"] > 0 else None,
        axis=1,
    )
    daily["cpc"] = daily.apply(lambda r: safe_div(r["spend"], r["clicks"]), axis=1)
    daily["cpm"] = daily.apply(
        lambda r: safe_div(r["spend"], r["impressions"]) * 1000 if r["impressions"] > 0 else None,
        axis=1,
    )
    return daily.reset_index()


def _summary_from_daily(daily):
    if daily is None or daily.empty:
        return {
            "spend": 0.0, "results": 0.0, "cpl": None,
            "impressions": 0.0, "reach": 0.0, "clicks": 0.0,
            "inline_link_clicks": 0.0, "ctr": None, "cpc": None,
            "cpm": None, "frequency": None, "active_spend_days": 0,
            "calendar_days": 0,
        }

    spend = pd.to_numeric(daily["spend"], errors="coerce").fillna(0).sum()
    results = pd.to_numeric(daily["results"], errors="coerce").fillna(0).sum()
    impressions = pd.to_numeric(daily["impressions"], errors="coerce").fillna(0).sum()
    reach = pd.to_numeric(daily.get("reach", 0), errors="coerce").fillna(0).sum()
    clicks = pd.to_numeric(daily["clicks"], errors="coerce").fillna(0).sum()
    link_clicks = pd.to_numeric(daily.get("inline_link_clicks", 0), errors="coerce").fillna(0).sum()
    frequency_weight = (
        pd.to_numeric(daily.get("frequency", 0), errors="coerce").fillna(0)
        * pd.to_numeric(daily["impressions"], errors="coerce").fillna(0)
    ).sum()
    return {
        "spend": float(spend),
        "results": float(results),
        "cpl": safe_div(spend, results),
        "impressions": float(impressions),
        "reach": float(reach),
        "clicks": float(clicks),
        "inline_link_clicks": float(link_clicks),
        "ctr": safe_div(clicks, impressions) * 100 if impressions > 0 else None,
        "cpc": safe_div(spend, clicks),
        "cpm": safe_div(spend, impressions) * 1000 if impressions > 0 else None,
        "frequency": safe_div(frequency_weight, impressions) if impressions > 0 else None,
        "active_spend_days": int((pd.to_numeric(daily["spend"], errors="coerce").fillna(0) > 0).sum()),
        "calendar_days": int(len(daily)),
    }


def _reindex_calendar_days(daily, start_date, end_date):
    if start_date is None or end_date is None or end_date < start_date:
        return pd.DataFrame()
    full_index = pd.date_range(start_date, end_date, freq="D")
    if daily is None or daily.empty:
        out = pd.DataFrame(index=full_index)
    else:
        out = daily.set_index("date_start").reindex(full_index)
    zero_cols = ["spend", "results", "impressions", "reach", "clicks", "inline_link_clicks"]
    for col in zero_cols:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    for col in ["frequency", "cpl", "ctr", "cpc", "cpm"]:
        if col not in out.columns:
            out[col] = None
    out.index.name = "date_start"
    return out.reset_index()


def _selected_end_date(daily, meta):
    fact_max = None
    if daily is not None and not daily.empty:
        fact_max = pd.to_datetime(daily["date_start"], errors="coerce").max()
        fact_max = fact_max.normalize() if pd.notna(fact_max) else None
    return _parse_period_date(meta.get("date_to"), fact_max) or fact_max


def _first_spend_date(daily):
    if daily is None or daily.empty:
        return None
    active = daily[pd.to_numeric(daily["spend"], errors="coerce").fillna(0) > 0]
    if not active.empty:
        return pd.to_datetime(active["date_start"], errors="coerce").min().normalize()
    parsed = pd.to_datetime(daily["date_start"], errors="coerce").dropna()
    return parsed.min().normalize() if not parsed.empty else None


def _window_context(daily, meta, decision_window):
    """Resolve the selected decision window for one entity or the overall scope."""
    end_date = _selected_end_date(daily, meta)
    launch_date = _first_spend_date(daily)
    if end_date is None:
        return {}

    context = {
        "decision_window": decision_window,
        "launch_date": launch_date,
        "current_start": None,
        "current_end": end_date,
        "previous_start": None,
        "previous_end": None,
        "current_dates": None,
        "previous_dates": None,
        "comparison_type": "current_window_only",
    }

    calendar_days_map = {
        "Last 2 Calendar Days": 2,
        "Last 3 Calendar Days": 3,
        "Last 7 Calendar Days": 7,
        "Week vs Previous Week": 7,
    }

    if decision_window == "Since Launch":
        context["current_start"] = launch_date or _parse_period_date(meta.get("date_from"), end_date)
        context["comparison_type"] = "first_2_active_vs_last_2_active"
        return context

    if decision_window in calendar_days_map:
        days = calendar_days_map[decision_window]
        context["current_start"] = end_date - pd.Timedelta(days=days - 1)
        context["previous_end"] = context["current_start"] - pd.Timedelta(days=1)
        context["previous_start"] = context["previous_end"] - pd.Timedelta(days=days - 1)
        context["comparison_type"] = "previous_equal_calendar_period"
        return context

    if decision_window == "Last 2 Active Spend Days":
        active_dates = []
        if daily is not None and not daily.empty:
            active_dates = (
                daily[pd.to_numeric(daily["spend"], errors="coerce").fillna(0) > 0]["date_start"]
                .dropna().sort_values().drop_duplicates().tolist()
            )
        active_dates = [pd.Timestamp(x).normalize() for x in active_dates if pd.Timestamp(x).normalize() <= end_date]
        current_dates = active_dates[-2:]
        context["current_dates"] = current_dates
        context["current_start"] = current_dates[0] if current_dates else None
        context["current_end"] = current_dates[-1] if current_dates else end_date
        context["comparison_type"] = "previous_active_day_vs_latest_active_day"
        return context

    raise ValueError(f"Unsupported decision window: {decision_window}")


def _select_context_daily(daily, context, part="current"):
    if not context:
        return pd.DataFrame()
    dates_key = f"{part}_dates"
    start_key = f"{part}_start"
    end_key = f"{part}_end"
    dates = context.get(dates_key)
    if dates is not None:
        if not dates or daily is None or daily.empty:
            return pd.DataFrame()
        wanted = {pd.Timestamp(x).normalize() for x in dates}
        return daily[daily["date_start"].isin(wanted)].sort_values("date_start").reset_index(drop=True)
    start_date = context.get(start_key)
    end_date = context.get(end_key)
    if start_date is None or end_date is None:
        return pd.DataFrame()
    return _reindex_calendar_days(daily, start_date, end_date)


def _active_slice(daily, count=2, from_end=False):
    if daily is None or daily.empty:
        return pd.DataFrame()
    active = daily[pd.to_numeric(daily["spend"], errors="coerce").fillna(0) > 0].sort_values("date_start")
    if active.empty:
        return pd.DataFrame()
    return (active.tail(count) if from_end else active.head(count)).reset_index(drop=True)


def _trend_status(change_pct, recent_cpl, baseline_cpl):
    if recent_cpl is None or baseline_cpl is None or change_pct is None:
        return "insufficient_data"
    if change_pct <= -15:
        return "improving"
    if change_pct >= 15:
        return "deteriorating"
    return "stable"


def _best_worst_day(current_daily):
    out = {
        "best_day_date": None, "best_day_cpl": None,
        "worst_day_date": None, "worst_day_cpl": None,
        "worst_day_zero_results": False,
    }
    if current_daily is None or current_daily.empty:
        return out
    active = current_daily[pd.to_numeric(current_daily["spend"], errors="coerce").fillna(0) > 0].copy()
    if active.empty:
        return out

    successful = active[pd.to_numeric(active["results"], errors="coerce").fillna(0) > 0].copy()
    if not successful.empty:
        best_idx = pd.to_numeric(successful["cpl"], errors="coerce").idxmin()
        best = successful.loc[best_idx]
        out["best_day_date"] = pd.Timestamp(best["date_start"]).strftime("%Y-%m-%d")
        out["best_day_cpl"] = _json_safe_value(best.get("cpl"))

    zero_result = active[pd.to_numeric(active["results"], errors="coerce").fillna(0) <= 0]
    if not zero_result.empty:
        worst_idx = pd.to_numeric(zero_result["spend"], errors="coerce").idxmax()
        worst = zero_result.loc[worst_idx]
        out["worst_day_date"] = pd.Timestamp(worst["date_start"]).strftime("%Y-%m-%d")
        out["worst_day_cpl"] = None
        out["worst_day_zero_results"] = True
    elif not successful.empty:
        worst_idx = pd.to_numeric(successful["cpl"], errors="coerce").idxmax()
        worst = successful.loc[worst_idx]
        out["worst_day_date"] = pd.Timestamp(worst["date_start"]).strftime("%Y-%m-%d")
        out["worst_day_cpl"] = _json_safe_value(worst.get("cpl"))
    return out


def _daily_history_records(current_daily, limit=None):
    if current_daily is None or current_daily.empty:
        return []
    limit = int(limit or AI_DAILY_HISTORY_DAYS)
    selected = current_daily.sort_values("date_start").tail(max(1, limit)).copy()
    selected["date_start"] = pd.to_datetime(selected["date_start"], errors="coerce").dt.strftime("%Y-%m-%d")
    cols = ["date_start", "spend", "results", "cpl", "ctr", "cpc", "cpm", "frequency"]
    cols = [c for c in cols if c in selected.columns]
    return _records_for_json(selected[cols])


def _window_metrics_from_frame(frame, meta, decision_window):
    daily = _daily_metrics(frame)
    context = _window_context(daily, meta, decision_window)
    if not context:
        return {}, pd.DataFrame(), pd.DataFrame()

    current_daily = _select_context_daily(daily, context, "current")
    previous_daily = _select_context_daily(daily, context, "previous")
    current_summary = _summary_from_daily(current_daily)
    previous_summary = _summary_from_daily(previous_daily)

    first_count = 1 if decision_window == "Last 2 Active Spend Days" else 2
    first_active = _active_slice(current_daily, first_count, from_end=False)
    last_active = _active_slice(current_daily, first_count, from_end=True)
    first_summary = _summary_from_daily(first_active)
    last_summary = _summary_from_daily(last_active)
    trend_change = _pct_change(last_summary.get("cpl"), first_summary.get("cpl"))

    if current_daily is None or current_daily.empty or "spend" not in current_daily.columns:
        active_days = pd.DataFrame()
    else:
        active_days = current_daily[pd.to_numeric(current_daily["spend"], errors="coerce").fillna(0) > 0]
    latest_day = active_days.tail(1)
    previous_active_day = active_days.tail(2).head(1) if len(active_days) >= 2 else pd.DataFrame()
    latest_summary = _summary_from_daily(latest_day)
    previous_active_summary = _summary_from_daily(previous_active_day)

    metrics = {
        "decision_window": decision_window,
        "comparison_type": context.get("comparison_type"),
        "entity_launch_date": context["launch_date"].strftime("%Y-%m-%d") if context.get("launch_date") is not None else None,
        "analysis_window_start": context["current_start"].strftime("%Y-%m-%d") if context.get("current_start") is not None else None,
        "analysis_window_end": context["current_end"].strftime("%Y-%m-%d") if context.get("current_end") is not None else None,
        "analysis_spend": current_summary["spend"],
        "analysis_results": current_summary["results"],
        "analysis_cpl": current_summary["cpl"],
        "analysis_ctr": current_summary["ctr"],
        "analysis_cpc": current_summary["cpc"],
        "analysis_cpm": current_summary["cpm"],
        "analysis_frequency": current_summary["frequency"],
        "analysis_calendar_days": current_summary["calendar_days"],
        "analysis_active_spend_days": current_summary["active_spend_days"],
        "comparison_window_start": context["previous_start"].strftime("%Y-%m-%d") if context.get("previous_start") is not None else None,
        "comparison_window_end": context["previous_end"].strftime("%Y-%m-%d") if context.get("previous_end") is not None else None,
        "comparison_spend": previous_summary["spend"],
        "comparison_results": previous_summary["results"],
        "comparison_cpl": previous_summary["cpl"],
        "comparison_ctr": previous_summary["ctr"],
        "comparison_cpc": previous_summary["cpc"],
        "comparison_active_spend_days": previous_summary["active_spend_days"],
        "period_spend_change_pct": _pct_change(current_summary["spend"], previous_summary["spend"]),
        "period_results_change_pct": _pct_change(current_summary["results"], previous_summary["results"]),
        "period_cpl_change_pct": _pct_change(current_summary["cpl"], previous_summary["cpl"]),
        "period_ctr_change_pct": _pct_change(current_summary["ctr"], previous_summary["ctr"]),
        "first_active_days_spend": first_summary["spend"],
        "first_active_days_results": first_summary["results"],
        "first_active_days_cpl": first_summary["cpl"],
        "last_active_days_spend": last_summary["spend"],
        "last_active_days_results": last_summary["results"],
        "last_active_days_cpl": last_summary["cpl"],
        "recent_vs_early_cpl_change_pct": trend_change,
        "daily_trend": _trend_status(trend_change, last_summary.get("cpl"), first_summary.get("cpl")),
        "latest_active_day_date": pd.Timestamp(latest_day.iloc[0]["date_start"]).strftime("%Y-%m-%d") if not latest_day.empty else None,
        "latest_active_day_spend": latest_summary["spend"],
        "latest_active_day_results": latest_summary["results"],
        "latest_active_day_cpl": latest_summary["cpl"],
        "previous_active_day_date": pd.Timestamp(previous_active_day.iloc[0]["date_start"]).strftime("%Y-%m-%d") if not previous_active_day.empty else None,
        "previous_active_day_spend": previous_active_summary["spend"],
        "previous_active_day_results": previous_active_summary["results"],
        "previous_active_day_cpl": previous_active_summary["cpl"],
        "daily_history": _daily_history_records(current_daily),
        **_best_worst_day(current_daily),
    }

    if decision_window == "Since Launch":
        metrics["comparison_window_start"] = first_active["date_start"].min().strftime("%Y-%m-%d") if not first_active.empty else None
        metrics["comparison_window_end"] = first_active["date_start"].max().strftime("%Y-%m-%d") if not first_active.empty else None
        metrics["comparison_spend"] = first_summary["spend"]
        metrics["comparison_results"] = first_summary["results"]
        metrics["comparison_cpl"] = first_summary["cpl"]
        metrics["comparison_ctr"] = first_summary["ctr"]
        metrics["comparison_cpc"] = first_summary["cpc"]
        metrics["comparison_active_spend_days"] = first_summary["active_spend_days"]
        metrics["period_spend_change_pct"] = _pct_change(last_summary["spend"], first_summary["spend"])
        metrics["period_results_change_pct"] = _pct_change(last_summary["results"], first_summary["results"])
        metrics["period_cpl_change_pct"] = trend_change
        metrics["period_ctr_change_pct"] = _pct_change(last_summary["ctr"], first_summary["ctr"])

    if decision_window == "Last 2 Active Spend Days":
        metrics["comparison_window_start"] = metrics["previous_active_day_date"]
        metrics["comparison_window_end"] = metrics["previous_active_day_date"]
        metrics["comparison_spend"] = previous_active_summary["spend"]
        metrics["comparison_results"] = previous_active_summary["results"]
        metrics["comparison_cpl"] = previous_active_summary["cpl"]
        metrics["comparison_ctr"] = previous_active_summary["ctr"]
        metrics["comparison_cpc"] = previous_active_summary["cpc"]
        metrics["comparison_active_spend_days"] = previous_active_summary["active_spend_days"]
        metrics["period_spend_change_pct"] = _pct_change(latest_summary["spend"], previous_active_summary["spend"])
        metrics["period_results_change_pct"] = _pct_change(latest_summary["results"], previous_active_summary["results"])
        metrics["period_cpl_change_pct"] = _pct_change(latest_summary["cpl"], previous_active_summary["cpl"])
        metrics["period_ctr_change_pct"] = _pct_change(latest_summary["ctr"], previous_active_summary["ctr"])

    return metrics, current_daily, previous_daily


def build_analysis_overview(fact, meta, decision_window):
    campaign_fact = _fact_for_level(fact, "campaign")
    metrics, current_daily, previous_daily = _window_metrics_from_frame(campaign_fact, meta, decision_window)
    if not metrics:
        return {}
    metrics["comparison_quality"] = (
        "comparison_available" if metrics.get("comparison_active_spend_days", 0) > 0
        else "current_window_only"
    )
    metrics["current_daily_rows"] = int(len(current_daily))
    metrics["previous_daily_rows"] = int(len(previous_daily))
    return metrics


def attach_decision_window_metrics(entity_table, fact, level, meta, decision_window):
    if entity_table.empty:
        return entity_table
    df = _fact_for_level(fact, level)
    if df.empty:
        return entity_table

    metric_rows = []
    for (account_id, entity_id), grp in df.groupby(["account_id", "entity_id"], dropna=False):
        metrics, _, _ = _window_metrics_from_frame(grp, meta, decision_window)
        metrics.update({"account_id": account_id, "entity_id": entity_id})
        metric_rows.append(metrics)
    metric_df = pd.DataFrame(metric_rows)
    if metric_df.empty:
        return entity_table
    return entity_table.merge(metric_df, on=["account_id", "entity_id"], how="left")


def _decision_confidence(row, target):
    spend = to_float(row.get("analysis_spend"), 0)
    results = to_float(row.get("analysis_results"), 0)
    active_days = int(to_float(row.get("analysis_active_spend_days"), 0))
    if active_days >= 4 and spend >= target * 2 and results >= 5:
        return "High"
    if active_days >= 2 and (spend >= target or results >= 2):
        return "Medium"
    return "Low"


def _entity_rule_decision(row):
    """Safety-first decision using the selected period plus its daily trend."""
    spend = to_float(row.get("analysis_spend", row.get("spend")), 0)
    results = to_float(row.get("analysis_results", row.get("results")), 0)
    cpl = row.get("analysis_cpl")
    target = row.get("target_cpl")
    ctr = row.get("analysis_ctr")
    frequency = row.get("analysis_frequency", row.get("frequency"))
    recent_spend = to_float(row.get("last_active_days_spend"), 0)
    recent_results = to_float(row.get("last_active_days_results"), 0)
    recent_cpl = row.get("last_active_days_cpl")
    trend = row.get("daily_trend", "insufficient_data")
    level = row.get("entity_level", "campaign")

    cpl = None if cpl is None or pd.isna(cpl) else to_float(cpl)
    target = None if target is None or pd.isna(target) else to_float(target)
    ctr = None if ctr is None or pd.isna(ctr) else to_float(ctr)
    frequency = None if frequency is None or pd.isna(frequency) else to_float(frequency)
    recent_cpl = None if recent_cpl is None or pd.isna(recent_cpl) else to_float(recent_cpl)

    if not target or target <= 0:
        return "REVIEW", "Medium", "Low", "Target CPL غير محدد لهذا الـObjective."

    confidence = _decision_confidence(row, target)
    fatigue_signal = frequency is not None and frequency >= 3.5 and ctr is not None and ctr < 1.0
    scope_word = {"campaign": "الحملة", "adset": "الـAd Set", "ad": "الإعلان"}.get(level, "العنصر")
    recent_decline = (
        trend == "deteriorating" and recent_cpl is not None
        and recent_cpl > target * 1.25 and recent_spend >= target * 0.75
    )
    recent_improvement = (
        trend == "improving" and recent_cpl is not None
        and recent_cpl <= target and recent_results >= 2
    )

    if results <= 0:
        if spend >= target * 1.5:
            action = "PAUSE_AD_REVIEW" if level == "ad" else "REDUCE_OR_PAUSE_REVIEW"
            reason = f"{scope_word} صرف {spend:.2f} في الفترة المختارة بدون نتائج وتجاوز 1.5× Target CPL."
            if fatigue_signal:
                reason += " توجد أيضًا إشارة محتملة لتشبع الكرييتف."
            return action, "Critical", confidence, reason
        if spend >= target * 0.75:
            return "WATCH_CLOSELY", "High", confidence, f"{scope_word} بدون نتائج والصرف اقترب من Target CPL."
        return "TESTING", "Low", confidence, "الإنفاق الحالي غير كافٍ لاتخاذ قرار إيقاف."

    if cpl is None:
        return "REVIEW", "Medium", confidence, "تعذر حساب CPL بصورة صحيحة."

    if recent_decline:
        action = "CREATIVE_OPTIMIZE_RECENT_DECLINE" if level == "ad" else "OPTIMIZE_RECENT_DECLINE"
        return action, "High", confidence, (
            f"المتوسط العام لا يكفي للحكم: آخر أيام الصرف تراجعت وبلغ CPL الحديث {recent_cpl:.2f} "
            f"مقابل Target {target:.2f}. لا تعمل Scale قبل معالجة التراجع."
        )

    if cpl > target * 1.25 and recent_improvement:
        return "KEEP_IMPROVING", "Medium", confidence, (
            f"CPL الإجمالي {cpl:.2f} أعلى من الهدف، لكن آخر أيام الصرف تحسنت إلى {recent_cpl:.2f}. "
            "استمر بالمراقبة ولا توقف العنصر الآن."
        )

    if cpl <= target * 0.75 and results >= 5:
        if recent_cpl is not None and recent_cpl > target:
            return "KEEP_NO_SCALE", "Medium", confidence, "الإجمالي Winner لكن الأداء الحديث خارج الهدف؛ استمر بدون Scale."
        if recent_results < 2 and row.get("analysis_active_spend_days", 0) <= 2:
            return "KEEP_TESTING", "Medium", confidence, "الأداء واعد لكن الداتا الحديثة ما زالت محدودة."
        if level == "ad":
            return "WINNER_CREATE_VARIATIONS", "High", confidence, f"Ad Winner: CPL أقل من 75% من الهدف مع {results:.0f} نتائج."
        if fatigue_signal:
            return "KEEP_REFRESH_CREATIVE", "Medium", confidence, "الأداء جيد لكن راقب تشبع الكرييتف."
        return "SCALE_10_15_PERCENT", "High", confidence, f"CPL أقل من 75% من الهدف مع {results:.0f} نتائج والأداء الحديث مستقر."

    if cpl <= target:
        if results < 3:
            return "KEEP_TESTING", "Low", confidence, "CPL داخل الهدف لكن عدد النتائج ما زال محدودًا."
        if fatigue_signal:
            return "KEEP_REFRESH_CREATIVE", "Medium", confidence, "CPL داخل الهدف لكن Frequency مرتفع وCTR منخفض."
        return "KEEP", "Low", confidence, "CPL داخل الهدف والترند الحديث لا يظهر تراجعًا حادًا."

    if cpl <= target * 1.25:
        action = "CREATIVE_OPTIMIZE" if level == "ad" else "OPTIMIZE"
        return action, "Medium", confidence, "CPL أعلى قليلًا من الهدف؛ يحتاج تحسين قبل قرار الإيقاف."

    if spend >= target * 2:
        action = "PAUSE_AD_REVIEW" if level == "ad" else "REDUCE_OR_PAUSE_REVIEW"
        reason = "CPL أعلى من 125% من الهدف مع إنفاق كافٍ لاتخاذ قرار."
        if fatigue_signal:
            reason += " وتوجد إشارة Creative Fatigue محتملة."
        return action, "Critical", confidence, reason

    return "WATCH_CLOSELY", "High", confidence, "CPL أعلى من الهدف لكن حجم الإنفاق يحتاج متابعة إضافية."


def build_ai_level_table(fact, level, target_cpl_by_objective, meta, decision_window):
    out = build_entity_summary(fact, level)
    if out.empty:
        return out
    out = attach_decision_window_metrics(out, fact, level, meta, decision_window)
    out["target_cpl"] = out["objective_label"].map(target_cpl_by_objective)
    out["analysis_cpl_vs_target"] = out.apply(
        lambda r: safe_div(r.get("analysis_cpl"), r.get("target_cpl"))
        if r.get("analysis_cpl") is not None and r.get("target_cpl") else None,
        axis=1,
    )
    decisions = out.apply(_entity_rule_decision, axis=1, result_type="expand")
    decisions.columns = ["rule_action", "priority", "data_confidence", "rule_reason"]
    out = pd.concat([out, decisions], axis=1)
    priority_rank = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    out["_priority_rank"] = out["priority"].map(priority_rank).fillna(9)
    out = out.sort_values(["_priority_rank", "analysis_spend"], ascending=[True, False])
    return out.drop(columns=["_priority_rank"], errors="ignore").reset_index(drop=True)


def build_daily_ai_summary(fact, meta, decision_window):
    campaign_fact = _fact_for_level(fact, "campaign")
    metrics, current_daily, previous_daily = _window_metrics_from_frame(campaign_fact, meta, decision_window)
    frames = []
    if not previous_daily.empty:
        previous_daily = previous_daily.copy()
        previous_daily["period"] = "Comparison"
        frames.append(previous_daily)
    if not current_daily.empty:
        current_daily = current_daily.copy()
        current_daily["period"] = "Current"
        frames.append(current_daily)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["date_start"] = pd.to_datetime(out["date_start"], errors="coerce").dt.strftime("%Y-%m-%d")
    return out.reset_index(drop=True)


def build_entity_daily_detail(fact, level, account_id, entity_id, meta, decision_window):
    df = _fact_for_level(fact, level)
    if df.empty:
        return pd.DataFrame()
    selected = df[(df["account_id"].astype(str) == str(account_id)) & (df["entity_id"].astype(str) == str(entity_id))]
    if selected.empty:
        return pd.DataFrame()
    _, current_daily, previous_daily = _window_metrics_from_frame(selected, meta, decision_window)
    frames = []
    if not previous_daily.empty:
        previous_daily = previous_daily.copy()
        previous_daily["period"] = "Comparison"
        frames.append(previous_daily)
    if not current_daily.empty:
        current_daily = current_daily.copy()
        current_daily["period"] = "Current"
        frames.append(current_daily)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["date_start"] = pd.to_datetime(out["date_start"], errors="coerce").dt.strftime("%Y-%m-%d")
    return out.reset_index(drop=True)


def _mask_names(table, include_names):
    if include_names or table.empty:
        return table
    out = table.copy()
    for name_col, id_col in [
        ("account_name", "account_id"), ("campaign_name", "campaign_id"),
        ("adset_name", "adset_id"), ("ad_name", "ad_id"),
        ("entity_name", "entity_id"),
    ]:
        if name_col in out.columns and id_col in out.columns:
            out[name_col] = out[id_col].astype(str)
    return out


def build_openai_analysis_payload(
    fact, target_cpl_by_objective, meta, business_unit,
    selected_agent, selected_objective, selected_status, decision_window,
    max_campaigns=40, max_adsets=70, max_ads=120, include_names=True,
):
    campaign_table = build_ai_level_table(fact, "campaign", target_cpl_by_objective, meta, decision_window).head(max_campaigns)
    adset_table = build_ai_level_table(fact, "adset", target_cpl_by_objective, meta, decision_window).head(max_adsets)
    ad_table = build_ai_level_table(fact, "ad", target_cpl_by_objective, meta, decision_window).head(max_ads)

    campaign_table = _mask_names(campaign_table, include_names)
    adset_table = _mask_names(adset_table, include_names)
    ad_table = _mask_names(ad_table, include_names)

    campaign_fact = _fact_for_level(fact, "campaign")
    objective_summary = build_objective_summary(campaign_fact) if not campaign_fact.empty else pd.DataFrame()
    buyer_summary = build_buyer_summary(campaign_fact) if not campaign_fact.empty else pd.DataFrame()
    daily_summary = build_daily_ai_summary(fact, meta, decision_window)
    analysis_overview = build_analysis_overview(fact, meta, decision_window)

    common_cols = [
        "account_id", "account_name", "media_buyer", "objective_label",
        "campaign_id", "campaign_name", "entity_level", "entity_id", "entity_name",
        "entity_status", "campaign_status", "target_cpl", "decision_window",
        "entity_launch_date", "analysis_window_start", "analysis_window_end",
        "analysis_spend", "analysis_results", "analysis_cpl", "analysis_ctr",
        "analysis_cpc", "analysis_cpm", "analysis_frequency", "analysis_calendar_days",
        "analysis_active_spend_days", "comparison_type", "comparison_window_start",
        "comparison_window_end", "comparison_spend", "comparison_results", "comparison_cpl",
        "comparison_ctr", "comparison_cpc", "comparison_active_spend_days",
        "period_spend_change_pct", "period_results_change_pct", "period_cpl_change_pct",
        "period_ctr_change_pct", "first_active_days_spend", "first_active_days_results",
        "first_active_days_cpl", "last_active_days_spend", "last_active_days_results",
        "last_active_days_cpl", "recent_vs_early_cpl_change_pct", "daily_trend",
        "latest_active_day_date", "latest_active_day_spend", "latest_active_day_results",
        "latest_active_day_cpl", "previous_active_day_date", "previous_active_day_spend",
        "previous_active_day_results", "previous_active_day_cpl", "best_day_date",
        "best_day_cpl", "worst_day_date", "worst_day_cpl", "worst_day_zero_results",
        "daily_history", "analysis_cpl_vs_target", "rule_action", "priority",
        "data_confidence", "rule_reason",
    ]
    adset_extra = ["adset_id", "adset_name"]
    ad_extra = ["adset_id", "adset_name", "ad_id", "ad_name"]

    def selected_records(table, extras):
        cols = [c for c in common_cols + extras if c in table.columns]
        return _records_for_json(table[cols]) if not table.empty else []

    return {
        "analysis_scope": {
            "business_unit": business_unit,
            "media_buyer": selected_agent,
            "objective": selected_objective,
            "campaign_status": selected_status,
            "decision_window": decision_window,
            "snapshot_date_from": meta.get("date_from"),
            "snapshot_date_to": meta.get("date_to"),
            "snapshot_last_updated": meta.get("last_fetch_ts"),
            "hierarchy": ["campaign", "adset", "ad"],
            "names_included": bool(include_names),
            "daily_history_days_per_entity_sent": int(AI_DAILY_HISTORY_DAYS),
            "rows_included": {
                "campaigns": int(len(campaign_table)),
                "adsets": int(len(adset_table)),
                "ads": int(len(ad_table)),
            },
        },
        "targets_egp": target_cpl_by_objective,
        "decision_window_overall_campaign_level_only": _json_safe_nested(analysis_overview),
        "objective_summary_snapshot_campaign_level": _records_for_json(objective_summary),
        "buyer_summary_snapshot_campaign_level": _records_for_json(buyer_summary),
        "daily_trend_campaign_level": _records_for_json(daily_summary.tail(90)),
        "campaigns": selected_records(campaign_table, []),
        "adsets": selected_records(adset_table, adset_extra),
        "ads": selected_records(ad_table, ad_extra),
    }, {"campaign": campaign_table, "adset": adset_table, "ad": ad_table}, analysis_overview, daily_summary

def extract_openai_response_text(response_json):
    text_parts = []
    for item in response_json.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                text_parts.append(content["text"])
    return "\n".join(text_parts).strip()


def test_openai_connection():
    """Verify API key and model access without sending campaign data."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing from Streamlit secrets.")
    response = requests.get(
        f"https://api.openai.com/v1/models/{OPENAI_MODEL}",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        timeout=30,
    )
    if not response.ok:
        try:
            message = response.json().get("error", {}).get("message", response.text)
        except Exception:
            message = response.text
        raise RuntimeError(f"OpenAI connection failed {response.status_code}: {message}")
    return response.json().get("id", OPENAI_MODEL)


def call_openai_campaign_analysis(analysis_payload, analyst_notes=""):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing from Streamlit secrets.")

    system_prompt = """
أنت Senior Performance Marketing Analyst متخصص في Meta Ads.
حلل البيانات المرسلة فقط ولا تخترع أرقامًا أو أسبابًا غير مدعومة.
البيانات هرمية: Campaign ثم Ad Set ثم Ad. لا تجمع صرف المستويات الثلاثة معًا؛
الإجماليات الرسمية تأتي من مستوى Campaign فقط.

استخدم Decision Window المرسل داخل analysis_scope ولا تفترض دائمًا أنه أسبوع:
- Since Launch: حلل الإجمالي من أول يوم صرف داخل الـSnapshot، ثم حلل كل يوم، وقارن أول يومين صرف بآخر يومين صرف.
  أعط وزنًا أكبر للأداء الحديث. لا تسمح لمتوسط إجمالي جيد بإخفاء تراجع واضح في آخر الأيام،
  ولا تسمح لمتوسط إجمالي سيئ بإخفاء تحسن حديث مدعوم بنتائج كافية.
- Last 2/3/7 Calendar Days: حلل إجمالي الفترة وكل يوم داخلها، وقارن بالفترة التقويمية السابقة المساوية عندما تتوفر.
  الأيام التي لا يوجد بها صرف تظل جزءًا من الفترة بصرف ونتائج صفر.
- Week vs Previous Week: قارن 7 أيام تقويمية بالسبعة السابقة، مع تحليل يومي داخل كل فترة.
- Last 2 Active Spend Days: قارن آخر يومين حدث فيهما صرف فعلي حتى لو كان بينهما أيام بدون صرف.
لو المقارنة السابقة جزئية أو غير متاحة، اذكر ذلك بوضوح ولا تعرض نسبة تغير مضللة.

الـrule_action ناتج من قواعد رقمية ثابتة تجمع بين إجمالي الفترة، الـDaily Trend، وآخر أيام الصرف.
استخدم الذكاء الاصطناعي لشرح القرار، ربط الأسباب عبر المستويات، وترتيب الأولويات؛
لا تتجاوز قواعد الأمان ولا تنفذ تعديلًا تلقائيًا.
- Campaign Level: قرارات الميزانية العامة والـScale أو تقليل الإنفاق.
- Ad Set Level: الجمهور، التوزيع، الـPlacement، ومكان تسريب الميزانية.
- Ad Level: اختيار الـWinner، إيقاف/مراجعة الخاسر، وCreative Fatigue والـVariations.
افصل بين Lead Generation وLead Message وWhatsApp وConversion لأن تعريف النتيجة مختلف.
لا تعتبر CTR أو Frequency سببًا مؤكدًا وحده؛ اذكرهما كإشارة محتملة فقط.
راعِ data_confidence، ولا تعط قرارًا حاسمًا عندما تكون الداتا محدودة.
اكتب بالعربية المصرية المهنية واستخدم أسماء العناصر كما هي إن كانت موجودة.

صيغة التقرير:
1) ملخص تنفيذي للفترة المختارة بالأرقام وجودة الداتا.
2) تحليل الاتجاه اليومي: أفضل يوم، أسوأ يوم، أول أيام الصرف مقابل آخرها، وهل الأداء يتحسن أم يتراجع.
3) تشخيص هرمي: Campaign -> Ad Set -> Ad.
4) أهم 7 مشاكل مرتبة حسب التأثير المالي.
5) جدول قرارات واضح لكل مستوى: Keep / Scale / Optimize / Pause Review مع الدليل والثقة.
6) Winners وLosers على مستوى الـAds مع اقتراح Variations للوينرز.
7) تقرير مختصر لكل Media Buyer.
8) خطة عمل للـ24 ساعة القادمة ثم الأيام التالية.
9) تحذيرات جودة البيانات وما لا يمكن الجزم به.

لا توصي بزيادة الميزانية بأكثر من 15% في الخطوة الواحدة،
ولا توصي بإيقاف عنصر لم يجمع إنفاقًا كافيًا طبقًا للقواعد المرسلة.
""".strip()

    notes = analyst_notes.strip() or "لا توجد ملاحظات إضافية من المستخدم."
    user_prompt = (
        "حلل Snapshot الحملات الهرمي التالي.\n"
        f"ملاحظات المستخدم: {notes}\n\nDATA_JSON:\n"
        + json.dumps(_json_safe_nested(analysis_payload), ensure_ascii=False, separators=(",", ":"))
    )

    reasoning = {"effort": OPENAI_REASONING_EFFORT}
    if OPENAI_REASONING_MODE == "pro":
        reasoning["mode"] = "pro"
    request_payload = {
        "model": OPENAI_MODEL,
        "store": False,
        "reasoning": reasoning,
        "text": {"verbosity": "high"},
        "max_output_tokens": 6500,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    response = requests.post(
        OPENAI_RESPONSES_URL, headers=headers, json=request_payload,
        timeout=OPENAI_TIMEOUT_SECONDS,
    )
    if not response.ok:
        try:
            error_message = response.json().get("error", {}).get("message", response.text)
        except Exception:
            error_message = response.text
        raise RuntimeError(f"OpenAI API error {response.status_code}: {error_message}")

    response_json = response.json()
    output_text = extract_openai_response_text(response_json)
    if not output_text:
        raise RuntimeError("OpenAI returned no readable text output.")
    return output_text, response_json.get("usage", {})


def _render_rule_table(table, level):
    if table.empty:
        st.info(f"لا توجد بيانات على مستوى {level} في الـSnapshot الحالي.")
        return
    cols = [
        "account_name", "media_buyer", "objective_label", "campaign_name",
        "adset_name", "ad_name", "entity_status", "entity_launch_date",
        "analysis_window_start", "analysis_window_end", "analysis_spend",
        "analysis_results", "analysis_cpl", "last_active_days_cpl",
        "first_active_days_cpl", "daily_trend", "recent_vs_early_cpl_change_pct",
        "comparison_cpl", "period_cpl_change_pct", "analysis_ctr",
        "analysis_frequency", "target_cpl", "rule_action", "priority",
        "data_confidence", "rule_reason",
    ]
    cols = [c for c in cols if c in table.columns]
    display = table[cols].copy()
    for col in display.select_dtypes(include="number").columns:
        display[col] = display[col].round(2)
    st.dataframe(display, use_container_width=True, hide_index=True)


def _daily_display_table(daily_df):
    if daily_df is None or daily_df.empty:
        return pd.DataFrame()
    cols = [
        "period", "date_start", "spend", "results", "cpl", "ctr", "cpc",
        "cpm", "frequency", "impressions", "clicks",
    ]
    cols = [c for c in cols if c in daily_df.columns]
    out = daily_df[cols].copy()
    for col in out.select_dtypes(include="number").columns:
        out[col] = out[col].round(2)
    return out


def _entity_selector_label(row, level):
    level_name = {
        "campaign": row.get("campaign_name", row.get("entity_name", "-")),
        "adset": row.get("adset_name", row.get("entity_name", "-")),
        "ad": row.get("ad_name", row.get("entity_name", "-")),
    }.get(level, row.get("entity_name", "-"))
    return f"{level_name} | {row.get('account_name', '-')} | Spend {to_float(row.get('analysis_spend'), 0):.2f}"


def render_ai_campaign_analyst(fact, meta, business_unit):
    st.subheader("🤖 AI Hierarchy Analyst — Campaign → Ad Set → Ad")
    st.caption(
        "القرارات الرقمية تظهر أولًا من Rule Engine، ثم OpenAI يربط المستويات ويشرح الأولويات. "
        "كل قرار يجمع إجمالي الفترة، أداء كل يوم، وآخر أيام الصرف. النسخة Read Only."
    )
    if fact.empty:
        st.info("لا توجد بيانات في النطاق المختار للتحليل.")
        return

    available_levels = set(fact.get("entity_level", pd.Series(["campaign"])).dropna().astype(str).unique())
    missing_levels = {"campaign", "adset", "ad"} - available_levels
    if missing_levels:
        st.warning(
            "الـSnapshot الحالي لا يحتوي على كل المستويات. اعمل Refresh جديد بعد رفع النسخة؛ "
            f"المستويات الناقصة: {', '.join(sorted(missing_levels))}."
        )

    campaign_fact = _fact_for_level(fact, "campaign")
    agents = sorted(campaign_fact["media_buyer"].dropna().astype(str).unique().tolist())
    objectives = [obj for obj in OBJECTIVE_ORDER if obj in campaign_fact["objective_label"].dropna().astype(str).unique().tolist()]
    statuses = sorted(campaign_fact["campaign_status"].dropna().astype(str).unique().tolist()) if "campaign_status" in campaign_fact.columns else []

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        selected_agent = st.selectbox("AI - Media Buyer", ["All"] + agents, key="ai_agent_filter")
    with f2:
        selected_objective = st.selectbox("AI - Objective", ["All"] + objectives, key="ai_objective_filter")
    with f3:
        default_status_index = 1 if "Active" in statuses else 0
        selected_status = st.selectbox(
            "AI - Parent Campaign Status", ["All"] + statuses,
            index=default_status_index, key="ai_status_filter",
        )
    with f4:
        decision_window = st.selectbox(
            "Decision Window", DECISION_WINDOW_OPTIONS,
            index=0, key="ai_decision_window",
            help="Since Launch يبدأ من أول يوم صرف موجود داخل الـSnapshot، لذلك اختار Refresh range يغطي تاريخ الإطلاق الحقيقي.",
        )

    if decision_window == "Since Launch":
        st.caption(
            "Since Launch = من أول يوم صرف موجود داخل البيانات المحملة، مع تحليل يومي ومقارنة أول يومين صرف بآخر يومين صرف. "
            "لتحليل العمر الحقيقي للحملة، اعمل Refresh من تاريخ يسبق إطلاقها."
        )

    with st.expander("Target CPL / CPR Settings", expanded=True):
        t1, t2, t3, t4 = st.columns(4)
        with t1:
            target_lg = st.number_input("Lead Generation Target", min_value=1.0, value=20.0, step=1.0, key="ai_target_lg")
        with t2:
            target_lm = st.number_input("Lead Message Target", min_value=1.0, value=20.0, step=1.0, key="ai_target_lm")
        with t3:
            target_wa = st.number_input("WhatsApp Target", min_value=1.0, value=20.0, step=1.0, key="ai_target_wa")
        with t4:
            target_conv = st.number_input("Conversion Target", min_value=1.0, value=100.0, step=5.0, key="ai_target_conv")

    targets = {
        "Lead generation": float(target_lg), "Lead - Message": float(target_lm),
        "Whatsapp Message": float(target_wa), "Conversion": float(target_conv),
    }

    scoped = fact.copy()
    if selected_agent != "All":
        scoped = scoped[scoped["media_buyer"] == selected_agent]
    if selected_objective != "All":
        scoped = scoped[scoped["objective_label"] == selected_objective]
    if selected_status != "All" and "campaign_status" in scoped.columns:
        scoped = scoped[scoped["campaign_status"] == selected_status]
    if scoped.empty:
        st.info("لا توجد بيانات مطابقة للفلاتر المختارة.")
        return

    o1, o2, o3, o4 = st.columns(4)
    with o1:
        max_campaigns = st.slider("Campaigns sent", 10, 100, 40, 10, key="ai_max_campaigns")
    with o2:
        max_adsets = st.slider("Ad Sets sent", 10, 150, 70, 10, key="ai_max_adsets")
    with o3:
        max_ads = st.slider("Ads sent", 20, 250, 120, 10, key="ai_max_ads")
    with o4:
        include_names = st.checkbox("Send entity names", value=True, key="ai_include_names")

    analyst_notes = st.text_area(
        "ملاحظات للتحليل (اختياري)",
        placeholder="مثال: جودة الليد أهم من العدد، والحملات الجديدة لا تتوقف قبل إنفاق محدد...",
        key="ai_analyst_notes",
    )

    payload, rule_tables, overview, overall_daily = build_openai_analysis_payload(
        scoped, targets, meta, business_unit, selected_agent, selected_objective,
        selected_status, decision_window, max_campaigns, max_adsets, max_ads, include_names,
    )

    comparison_text = {
        "first_2_active_vs_last_2_active": "أول أيام الصرف مقابل آخر أيام الصرف",
        "previous_equal_calendar_period": "الفترة الحالية مقابل فترة تقويمية سابقة مساوية",
        "previous_active_day_vs_latest_active_day": "آخر يوم صرف مقابل يوم الصرف السابق",
        "current_window_only": "الفترة الحالية فقط",
    }.get(overview.get("comparison_type"), overview.get("comparison_type", "-"))
    st.info(
        f"{decision_window}: {overview.get('analysis_window_start', '-')} → {overview.get('analysis_window_end', '-')} | "
        f"Calendar days: {overview.get('analysis_calendar_days', 0)} | "
        f"Active spend days: {overview.get('analysis_active_spend_days', 0)} | "
        f"Spend: {to_float(overview.get('analysis_spend'), 0):.2f} | "
        f"Results: {to_float(overview.get('analysis_results'), 0):.0f} | "
        f"CPL: {overview.get('analysis_cpl') if overview.get('analysis_cpl') is not None else '-'} | "
        f"Trend: {overview.get('daily_trend', 'insufficient_data')} | {comparison_text}."
    )
    if overview.get("comparison_quality") == "current_window_only":
        st.warning("لا توجد داتا مقارنة كافية داخل الـSnapshot؛ القرار سيعتمد على الفترة الحالية والترند اليومي فقط.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Campaigns", len(rule_tables["campaign"]))
    m2.metric("Ad Sets", len(rule_tables["adset"]))
    m3.metric("Ads", len(rule_tables["ad"]))
    total_critical = sum(int((t["priority"] == "Critical").sum()) for t in rule_tables.values() if not t.empty)
    m4.metric("Critical Reviews", total_critical)

    tabs = st.tabs(["Campaign Decisions", "Ad Set Decisions", "Ad Decisions", "Overall Daily Trend"])
    with tabs[0]:
        _render_rule_table(rule_tables["campaign"], "Campaign")
    with tabs[1]:
        _render_rule_table(rule_tables["adset"], "Ad Set")
    with tabs[2]:
        _render_rule_table(rule_tables["ad"], "Ad")
    with tabs[3]:
        daily_display = _daily_display_table(overall_daily)
        if daily_display.empty:
            st.info("لا توجد بيانات يومية للفترة المختارة.")
        else:
            st.dataframe(daily_display, use_container_width=True, hide_index=True)
            current_chart = overall_daily[overall_daily["period"] == "Current"].copy()
            current_chart["cpl"] = pd.to_numeric(current_chart["cpl"], errors="coerce")
            current_chart = current_chart.dropna(subset=["cpl"])
            if not current_chart.empty:
                fig = px.line(current_chart, x="date_start", y="cpl", markers=True, title="Daily CPL — Current Window")
                st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔎 Daily Detail for One Campaign / Ad Set / Ad", expanded=False):
        detail_level_label = st.selectbox(
            "Level", ["Campaign", "Ad Set", "Ad"], key="ai_daily_detail_level"
        )
        detail_level = {"Campaign": "campaign", "Ad Set": "adset", "Ad": "ad"}[detail_level_label]
        detail_table = rule_tables.get(detail_level, pd.DataFrame()).reset_index(drop=True)
        if detail_table.empty:
            st.info("لا توجد عناصر في هذا المستوى.")
        else:
            options = detail_table.index.tolist()
            selected_idx = st.selectbox(
                "Entity", options,
                format_func=lambda idx: _entity_selector_label(detail_table.loc[idx], detail_level),
                key=f"ai_daily_entity_{detail_level}",
            )
            selected_row = detail_table.loc[selected_idx]
            detail_daily = build_entity_daily_detail(
                scoped, detail_level, selected_row.get("account_id"), selected_row.get("entity_id"),
                meta, decision_window,
            )
            display_detail = _daily_display_table(detail_daily)
            if display_detail.empty:
                st.info("لا توجد بيانات يومية لهذا العنصر في الفترة المختارة.")
            else:
                st.dataframe(display_detail, use_container_width=True, hide_index=True)
                current_detail = detail_daily[detail_daily["period"] == "Current"].copy()
                current_detail["cpl"] = pd.to_numeric(current_detail["cpl"], errors="coerce")
                current_detail = current_detail.dropna(subset=["cpl"])
                if not current_detail.empty:
                    fig = px.line(current_detail, x="date_start", y="cpl", markers=True, title="Entity Daily CPL")
                    target_value = selected_row.get("target_cpl")
                    if target_value is not None and not pd.isna(target_value):
                        fig.add_hline(y=float(target_value), line_dash="dash", annotation_text="Target CPL")
                    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"OpenAI model: {OPENAI_MODEL} | Reasoning effort: {OPENAI_REASONING_EFFORT} "
        f"| Mode: {OPENAI_REASONING_MODE} | Daily history/entity sent: {AI_DAILY_HISTORY_DAYS} days"
    )
    if not OPENAI_API_KEY:
        st.warning("الـRule Engine شغال، لكن تحليل ChatGPT غير مفعل. أضف OPENAI_API_KEY في Streamlit Secrets.")

    test_col, analyze_col = st.columns([1, 2])
    with test_col:
        test_clicked = st.button(
            "Test OpenAI connection", use_container_width=True,
            disabled=(not bool(OPENAI_API_KEY)), key="test_openai_connection",
        )
    with analyze_col:
        analyze_clicked = st.button(
            "Analyze hierarchy with OpenAI", type="primary", use_container_width=True,
            disabled=(not bool(OPENAI_API_KEY)), key="run_openai_analysis",
        )

    if test_clicked:
        try:
            accessible_model = test_openai_connection()
            st.success(f"OpenAI connected successfully. Model access confirmed: {accessible_model}")
        except Exception as e:
            st.error(str(e))
    if analyze_clicked:
        try:
            with st.spinner(f"OpenAI is analyzing {decision_window}, daily trend and hierarchy..."):
                report, usage = call_openai_campaign_analysis(payload, analyst_notes)
            st.session_state["latest_ai_report"] = report
            st.session_state["latest_ai_usage"] = usage
            st.session_state["latest_ai_payload"] = payload
            st.session_state["latest_ai_decision_window"] = decision_window
        except Exception as e:
            st.error(str(e))

    if st.session_state.get("latest_ai_report"):
        st.markdown("#### AI Analysis Report")
        st.caption(f"Decision Window used: {st.session_state.get('latest_ai_decision_window', decision_window)}")
        st.markdown(st.session_state["latest_ai_report"])
        usage = st.session_state.get("latest_ai_usage", {})
        if usage:
            st.caption(
                f"Model: {OPENAI_MODEL} | Reasoning: {OPENAI_REASONING_EFFORT}/{OPENAI_REASONING_MODE} | "
                f"Input tokens: {usage.get('input_tokens', '-')} | Output tokens: {usage.get('output_tokens', '-')}"
            )
        st.download_button(
            "Download AI Report", data=st.session_state["latest_ai_report"],
            file_name="meta_ads_hierarchy_daily_ai_analysis.md", mime="text/markdown",
            use_container_width=True,
        )


def build_campaign_summary(fact):
    if fact.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["account_id", "media_buyer", "objective_label", "account_name", "campaign_id", "campaign_name"]
    for keys, grp in fact.groupby(group_cols, dropna=False):
        account_id, media_buyer, objective_label, account_name, campaign_id, campaign_name = keys
        spend = grp["spend"].sum()
        results = grp["results"].sum()
        impressions = grp["impressions"].sum()
        clicks = grp["clicks"].sum()

        campaign_status = "Unknown"
        if "campaign_status" in grp.columns:
            status_values = grp["campaign_status"].dropna().astype(str)
            if not status_values.empty:
                campaign_status = status_values.iloc[0]

        rows.append({
            "account_id": account_id,
            "media_buyer": media_buyer,
            "objective_label": objective_label,
            "account_name": account_name,
            "campaign_id": campaign_id,
            "campaign_name": campaign_name,
            "campaign_status": campaign_status,
            "spend": spend,
            "results": results,
            "cpl": safe_div(spend, results),
            "ctr": safe_div(clicks, impressions) * 100 if impressions > 0 else None,
            "cpc": safe_div(spend, clicks),
            "frequency": pd.to_numeric(grp["frequency"], errors="coerce").dropna().mean() if not grp.empty else None,
            "impressions": impressions,
            "clicks": clicks,
        })

    return pd.DataFrame(rows).sort_values("spend", ascending=False).reset_index(drop=True)

def build_account_sources_table(raw_accounts):
    if raw_accounts.empty:
        return pd.DataFrame()

    out = (
        raw_accounts.groupby(["id", "account_id", "name"], dropna=False)
        .agg(
            sources=("source", lambda x: ", ".join(sorted(set(x))))
        )
        .reset_index()
        .sort_values("name")
        .reset_index(drop=True)
    )
    return out

def enrich_breakdown_spend(df, accounts_dedup, breakdown_col):
    if df.empty:
        return pd.DataFrame()

    out = df.copy()

    if "account_id" not in out.columns:
        return pd.DataFrame()

    if breakdown_col not in out.columns:
        out[breakdown_col] = "unknown"

    if "spend" in out.columns:
        out["spend"] = pd.to_numeric(out["spend"], errors="coerce").fillna(0)
    else:
        out["spend"] = 0

    account_map = accounts_dedup[["id", "name"]].drop_duplicates().rename(
        columns={"id": "account_id", "name": "account_name"}
    )

    out = out.merge(account_map, on="account_id", how="left")
    out["account_name"] = out["account_name"].fillna("Unknown")
    out["buyer_code"] = out["account_name"].apply(extract_buyer_code)
    out["media_buyer"] = out["buyer_code"].map(MEDIA_BUYER_MAP).fillna("Unknown")

    return out

def build_breakdown_by_account(df, breakdown_col):
    if df.empty:
        return pd.DataFrame()

    return (
        df.groupby(["account_id", "account_name", "media_buyer", breakdown_col], dropna=False)
        .agg(spend=("spend", "sum"))
        .reset_index()
        .sort_values(["account_name", "spend"], ascending=[True, False])
        .reset_index(drop=True)
    )

def build_breakdown_by_buyer(df, breakdown_col):
    if df.empty:
        return pd.DataFrame()

    return (
        df.groupby(["media_buyer", breakdown_col], dropna=False)
        .agg(spend=("spend", "sum"))
        .reset_index()
        .sort_values(["media_buyer", "spend"], ascending=[True, False])
        .reset_index(drop=True)
    )

def _empty_audience_columns(level="account"):
    if level == "account":
        return [
            "Ad Account", "Ad Account Name", "Media Buyer", "Balance",
            "Male Spend (This Month)", "Female Spend (This Month)",
            "18-24 Spend", "25-34 Spend", "35-44 Spend", "45-54 Spend", "55-64 Spend", "65+ Spend",
        ]
    return [
        "Media Buyer", "Balance",
        "Male Spend (This Month)", "Female Spend (This Month)",
        "18-24 Spend", "25-34 Spend", "35-44 Spend", "45-54 Spend", "55-64 Spend", "65+ Spend",
    ]

def build_audience_table_by_account(gender_df, age_df, balance_df):
    age_cols = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]

    frames = []
    for df in [gender_df, age_df]:
        if not df.empty and {"account_id", "account_name", "media_buyer"}.issubset(df.columns):
            frames.append(df[["account_id", "account_name", "media_buyer"]].drop_duplicates())

    if not balance_df.empty and {"account_id", "account_name", "media_buyer"}.issubset(balance_df.columns):
        frames.append(balance_df[["account_id", "account_name", "media_buyer"]].drop_duplicates())

    if not frames:
        return pd.DataFrame(columns=_empty_audience_columns("account"))

    base = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["account_id"])

    if not gender_df.empty:
        g = gender_df.copy()
        g["gender"] = g["gender"].astype(str).str.lower()
        g = g.groupby(["account_id", "gender"], dropna=False)["spend"].sum().unstack(fill_value=0).reset_index()
        g = g.rename(columns={"male": "Male Spend (This Month)", "female": "Female Spend (This Month)"})
        base = base.merge(g, on="account_id", how="left")

    if not age_df.empty:
        a = age_df.copy()
        a["age"] = a["age"].astype(str)
        a = a.groupby(["account_id", "age"], dropna=False)["spend"].sum().unstack(fill_value=0).reset_index()
        a = a.rename(columns={age: f"{age} Spend" for age in age_cols})
        base = base.merge(a, on="account_id", how="left")

    if not balance_df.empty and "balance" in balance_df.columns:
        b = balance_df[["account_id", "balance"]].drop_duplicates(subset=["account_id"])
        base = base.merge(b, on="account_id", how="left")
    else:
        base["balance"] = None

    base = base.rename(columns={
        "account_id": "Ad Account",
        "account_name": "Ad Account Name",
        "media_buyer": "Media Buyer",
        "balance": "Balance",
    })

    for col in ["Male Spend (This Month)", "Female Spend (This Month)"] + [f"{age} Spend" for age in age_cols]:
        if col not in base.columns:
            base[col] = 0
        base[col] = pd.to_numeric(base[col], errors="coerce").fillna(0).round(2)

    if "Balance" in base.columns:
        base["Balance"] = pd.to_numeric(base["Balance"], errors="coerce").round(2)

    cols = _empty_audience_columns("account")
    return base[cols].sort_values(["Media Buyer", "Ad Account Name"]).reset_index(drop=True)

def build_audience_table_by_buyer(account_audience_df):
    if account_audience_df.empty:
        return pd.DataFrame(columns=_empty_audience_columns("buyer"))

    value_cols = [
        "Balance", "Male Spend (This Month)", "Female Spend (This Month)",
        "18-24 Spend", "25-34 Spend", "35-44 Spend", "45-54 Spend", "55-64 Spend", "65+ Spend",
    ]

    out = account_audience_df.copy()
    for col in value_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    out = out.groupby("Media Buyer", dropna=False)[value_cols].sum().reset_index()
    for col in value_cols:
        out[col] = out[col].round(2)

    return out.sort_values("Media Buyer").reset_index(drop=True)


# -----------------------------
# Refresh lock
# -----------------------------
def get_lock_info():
    if not LOCK_FILE.exists():
        return None
    try:
        raw = LOCK_FILE.read_text(encoding="utf-8").strip()
        try:
            info = json.loads(raw)
        except Exception:
            info = {"created_at": float(raw) if raw else LOCK_FILE.stat().st_mtime}
        created_at = float(info.get("created_at", LOCK_FILE.stat().st_mtime))
        age = time.time() - created_at
        info["age_seconds"] = age
        return info
    except Exception:
        return {"created_at": LOCK_FILE.stat().st_mtime, "age_seconds": time.time() - LOCK_FILE.stat().st_mtime}

def is_refresh_locked():
    info = get_lock_info()
    if not info:
        return False
    try:
        if info.get("age_seconds", 0) > REFRESH_LOCK_MAX_AGE_SECONDS:
            LOCK_FILE.unlink(missing_ok=True)
            return False
        return True
    except Exception:
        return False

def acquire_refresh_lock():
    if is_refresh_locked():
        return False
    try:
        payload = {
            "created_at": time.time(),
            "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        }
        LOCK_FILE.write_text(json.dumps(payload), encoding="utf-8")
        return True
    except Exception:
        return False

def release_refresh_lock():
    try:
        LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass

def force_clear_refresh_lock():
    try:
        LOCK_FILE.unlink(missing_ok=True)
        return True
    except Exception:
        return False

# -----------------------------
# Session init
# -----------------------------
for key, default in {
    "filter_buyer": "All",
    "filter_objective": "All",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# -----------------------------
# Load saved snapshot first
# -----------------------------
snapshot = load_snapshot()

st.title("Meta Ads Team Dashboard")
st.caption("Snapshot-based dashboard")

with st.sidebar:
    st.header("Data Load")

    quick_range = st.selectbox(
        "Quick Range",
        ["Custom", "Today", "Yesterday", "Last 7 Days", "Last 14 Days (Week vs Week)", "This Month", "Last Month"],
        index=4,
    )

    today = pd.Timestamp.now(tz="Africa/Cairo").normalize().tz_localize(None).date()

    if quick_range == "Today":
        since = today
        until = today
    elif quick_range == "Yesterday":
        since = (pd.Timestamp(today) - pd.Timedelta(days=1)).date()
        until = since
    elif quick_range == "Last 7 Days":
        since = (pd.Timestamp(today) - pd.Timedelta(days=6)).date()
        until = today
    elif quick_range == "Last 14 Days (Week vs Week)":
        since = (pd.Timestamp(today) - pd.Timedelta(days=13)).date()
        until = today
    elif quick_range == "This Month":
        since = pd.Timestamp(today).replace(day=1).date()
        until = today
    elif quick_range == "Last Month":
        first_this_month = pd.Timestamp(today).replace(day=1)
        last_month_end = first_this_month - pd.Timedelta(days=1)
        since = last_month_end.replace(day=1).date()
        until = last_month_end.date()
    else:
        since = st.date_input("From")
        until = st.date_input("To")

    st.caption(f"Selected range: {since} → {until}")

    max_workers = st.slider("Parallel workers", min_value=2, max_value=10, value=4, step=2)
    show_account_sources = st.checkbox("Show account sources", value=True)
    refresh_clicked = st.button("Refresh Data", use_container_width=True)
    force_unlock_clicked = st.button("Clear stuck refresh lock", use_container_width=True)

if force_unlock_clicked:
    force_clear_refresh_lock()
    st.success("Refresh lock cleared. You can click Refresh Data now.")

if refresh_clicked:
    if not acquire_refresh_lock():
        info = get_lock_info() or {}
        age = info.get("age_seconds")
        if age is not None:
            st.warning(f"Refresh is already running or was recently interrupted. Lock age: {age/60:.1f} minutes. If you are sure no refresh is running, click Clear stuck refresh lock.")
        else:
            st.warning("Refresh is already running. If you are sure no refresh is running, click Clear stuck refresh lock.")
        st.stop()

    try:
        old_snapshot = load_snapshot()

        with st.status("Refreshing data in background-like flow...", expanded=True) as status:
            accounts_df, raw_accounts_df = get_ad_accounts()
            status.write(f"Loaded {len(accounts_df)} relevant ad accounts from configured businesses.")

            if accounts_df.empty:
                st.error("No matching Okaby / VAL ad accounts found.")
                st.stop()

            all_campaigns = []
            all_adsets = []
            all_ads = []
            all_insights = []
            all_gender = []
            all_age = []
            all_balances = []
            errors = []

            progress = st.progress(0)
            progress_text = st.empty()
            refresh_started_at = time.time()
            total = len(accounts_df)
            done = 0
            progress_text.info(f"Starting refresh for {total} accounts...")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(fetch_one_account, row, since, until)
                    for _, row in accounts_df.iterrows()
                ]

                for future in as_completed(futures):
                    result = future.result()

                    if result["error"]:
                        errors.append(result["error"])

                    if not result["campaigns_df"].empty:
                        all_campaigns.append(result["campaigns_df"])
                    if not result["adsets_df"].empty:
                        all_adsets.append(result["adsets_df"])
                    if not result["ads_df"].empty:
                        all_ads.append(result["ads_df"])

                    if not result["insights_df"].empty:
                        all_insights.append(result["insights_df"])

                    if not result["gender_df"].empty:
                        all_gender.append(result["gender_df"])

                    if not result["age_df"].empty:
                        all_age.append(result["age_df"])

                    if result.get("balance_row"):
                        all_balances.append(result["balance_row"])

                    done += 1
                    elapsed = time.time() - refresh_started_at
                    progress.progress(done / total)
                    progress_text.info(
                        f"Refreshing accounts: {done}/{total} | "
                        f"Last finished: {result.get('account_name', '-')} | "
                        f"Elapsed: {elapsed:.1f}s | Errors: {len(errors)}"
                    )

            all_campaigns_df = pd.concat(all_campaigns, ignore_index=True) if all_campaigns else pd.DataFrame()
            all_adsets_df = pd.concat(all_adsets, ignore_index=True) if all_adsets else pd.DataFrame()
            all_ads_df = pd.concat(all_ads, ignore_index=True) if all_ads else pd.DataFrame()
            all_insights_df = pd.concat(all_insights, ignore_index=True) if all_insights else pd.DataFrame()
            gender_df = pd.concat(all_gender, ignore_index=True) if all_gender else pd.DataFrame()
            age_df = pd.concat(all_age, ignore_index=True) if all_age else pd.DataFrame()
            balance_rows = []
            if not accounts_df.empty:
                for _, acc in accounts_df.iterrows():
                    fsd = acc.get("funding_source_details")
                    display_string = None
                    if isinstance(fsd, dict):
                        display_string = fsd.get("display_string")
                    balance_rows.append({
                        "account_id": acc.get("id"),
                        "account_name": acc.get("name", "Unknown"),
                        "balance": parse_balance_from_display_string(display_string),
                        "balance_display_string": display_string or "N/A",
                    })
            balance_df = pd.DataFrame(balance_rows)

            fact = prepare_data(all_campaigns_df, all_adsets_df, all_ads_df, all_insights_df)
            fact = assign_fact_business_unit_from_accounts(fact, accounts_df)

            if fact.empty:
                st.error("Refresh finished but no data returned.")
                st.stop()

            gender_df = enrich_breakdown_spend(gender_df, accounts_df, "gender")
            age_df = enrich_breakdown_spend(age_df, accounts_df, "age")
            if not balance_df.empty:
                if "media_buyer" not in balance_df.columns:
                    account_map = accounts_df[["id", "name"]].drop_duplicates().rename(
                        columns={"id": "account_id", "name": "account_name_from_map"}
                    )
                    balance_df = balance_df.merge(account_map, on="account_id", how="left")
                    balance_df["account_name"] = balance_df["account_name"].fillna(balance_df["account_name_from_map"]).fillna("Unknown")
                    balance_df = balance_df.drop(columns=["account_name_from_map"], errors="ignore")
                    balance_df["buyer_code"] = balance_df["account_name"].apply(extract_buyer_code)
                    balance_df["media_buyer"] = balance_df["buyer_code"].map(MEDIA_BUYER_MAP).fillna("Unknown")
                if "balance" in balance_df.columns:
                    balance_df["balance"] = pd.to_numeric(balance_df["balance"], errors="coerce")

            meta = {
                "last_fetch_ts": pd.Timestamp.utcnow().isoformat(),
                "date_from": str(since),
                "date_to": str(until),
                "accounts_count": int(accounts_df["id"].nunique()),
                "business_ids": BUSINESS_IDS,
                "rows_count": int(len(fact)),
                "rows_by_level": (
                    {str(k): int(v) for k, v in fact["entity_level"].value_counts().to_dict().items()}
                    if "entity_level" in fact.columns else {"campaign": int(len(fact))}
                ),
                "errors_count": int(len(errors)),
                "errors": errors[:100],
            }

            save_snapshot_atomic(
                fact=fact,
                accounts_dedup=accounts_df,
                accounts_raw=build_account_sources_table(raw_accounts_df),
                meta=meta,
                gender_df=gender_df,
                age_df=age_df,
                balance_df=balance_df,
            )

            status.update(label="Refresh complete. New snapshot saved.", state="complete")
            st.rerun()

    except Exception as e:
        st.error(f"Refresh failed: {e}")
        raise
    finally:
        release_refresh_lock()

if not snapshot:
    st.info("No saved snapshot yet. Choose dates and click Refresh Data.")
    st.stop()

accounts_dedup = snapshot["accounts_dedup"]
accounts_raw = snapshot["accounts_raw"]
fact_all = assign_fact_business_unit_from_accounts(snapshot["fact"], accounts_dedup)
if "entity_level" not in fact_all.columns:
    fact_all["entity_level"] = "campaign"
fact = _fact_for_level(fact_all, "campaign")
gender_df = snapshot.get("gender_df", pd.DataFrame())
age_df = snapshot.get("age_df", pd.DataFrame())
balance_df = snapshot.get("balance_df", pd.DataFrame())
meta = snapshot["meta"]

last_updated_value = meta.get("last_fetch_ts", "-")
st.caption(
    f"Last updated: {last_updated_value}"
    f" | Range: {meta.get('date_from', '-')} → {meta.get('date_to', '-')}"
    f" | Fetched accounts: {meta.get('accounts_count', 0)}"
    f" | Errors: {meta.get('errors_count', 0)}"
)

# Main business unit selector on the right
left_title_col, right_filter_col = st.columns([3, 1])
with right_filter_col:
    selected_business_unit = st.selectbox(
        "Business Unit",
        ["El - Okaby", "Val Hair", "VAL Booty", "All"],
        index=0,
        key="business_unit_selector",
    )

filtered_fact_all = filter_by_business_unit(fact_all, selected_business_unit)
filtered_fact_main = _fact_for_level(filtered_fact_all, "campaign")
# Account count should reflect fetched ad accounts, not only accounts that had spend rows in the snapshot.
filtered_accounts_dedup = filter_accounts_by_business_unit(accounts_dedup, selected_business_unit)
# Audience should be filtered by fetched accounts, not only accounts that had campaign spend rows.
filtered_gender_df = filter_related_by_accounts(gender_df, filtered_accounts_dedup)
filtered_age_df = filter_related_by_accounts(age_df, filtered_accounts_dedup)
filtered_balance_df = filter_related_by_accounts(balance_df, filtered_accounts_dedup)

overall = build_overall_summary(filtered_fact_main)
objective_summary = build_objective_summary(filtered_fact_main) if not filtered_fact_main.empty else pd.DataFrame()
buyer_summary = build_buyer_summary(filtered_fact_main) if not filtered_fact_main.empty else pd.DataFrame()
buyer_summary_with_total = add_overall_row_to_buyer_summary(buyer_summary)
buyer_objective_summary = build_buyer_objective_summary(filtered_fact_main) if not filtered_fact_main.empty else pd.DataFrame()
campaign_summary = build_campaign_summary(filtered_fact_main) if not filtered_fact_main.empty else pd.DataFrame()

audience_by_account = build_audience_table_by_account(filtered_gender_df, filtered_age_df, filtered_balance_df)
audience_by_buyer = build_audience_table_by_buyer(audience_by_account)

c1, c2, c3, c4 = st.columns(4)
fetched_accounts_count = filtered_accounts_dedup["id"].nunique() if "id" in filtered_accounts_dedup.columns else 0
c1.metric("Fetched Ad Accounts", f"{fetched_accounts_count}")
c2.metric("Total Spend", f"{overall['total_spend']:,.2f}")
c3.metric("Total Results", f"{overall['total_results']:,.0f}")
c4.metric("Overall CPL", "-" if overall["total_cpl"] is None else f"{overall['total_cpl']:,.2f}")

if show_account_sources:
    st.subheader("Loaded Ad Accounts")
    shown_accounts = filter_accounts_by_business_unit(accounts_raw, selected_business_unit)
    if not shown_accounts.empty and {"name", "sources"}.issubset(shown_accounts.columns):
        st.dataframe(shown_accounts[["name", "sources"]], use_container_width=True, hide_index=True)

st.divider()

objective_tabs = st.tabs([
    "🎯 Lead Generation",
    "💬 Lead Message",
    "📱 WhatsApp",
    "🛒 Conversion",
    "👥 Overall Agent",
])

with objective_tabs[0]:
    render_objective_agent_section(filtered_fact_main, "Lead generation", "Lead Generation — CPL (EGP)", "🎯", "lg")

with objective_tabs[1]:
    render_objective_agent_section(filtered_fact_main, "Lead - Message", "Lead Message — CPL (EGP)", "💬", "lm")

with objective_tabs[2]:
    render_objective_agent_section(filtered_fact_main, "Whatsapp Message", "WhatsApp — CPL (EGP)", "📱", "wa")

with objective_tabs[3]:
    render_objective_agent_section(filtered_fact_main, "Conversion", "Conversion — CPR (EGP)", "🛒", "conv")

with objective_tabs[4]:
    render_overall_agent_section(filtered_fact_main)

st.divider()
render_media_buyer_campaign_details(filtered_fact_main)
st.divider()
render_ai_campaign_analyst(filtered_fact_all, meta, selected_business_unit)
st.divider()

st.subheader("Audience Spend & Balance")
aud_tab1, aud_tab2 = st.tabs(["By Ad Account", "By Agent"])

with aud_tab1:
    if audience_by_account.empty:
        st.info("No gender / age spend data in the saved snapshot.")
    else:
        st.dataframe(format_display_df(audience_by_account), use_container_width=True, hide_index=True)

with aud_tab2:
    if audience_by_buyer.empty:
        st.info("No gender / age spend data by agent in the saved snapshot.")
    else:
        st.dataframe(format_display_df(audience_by_buyer), use_container_width=True, hide_index=True)
