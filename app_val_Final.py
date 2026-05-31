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

BASE_URL = "https://graph.facebook.com"
API_VERSION = st.secrets.get("META_API_VERSION", "v17.0")
ACCESS_TOKEN = st.secrets["META_ACCESS_TOKEN"]
BUSINESS_IDS = ["751488620224306", "1178859133269743"]
FETCH_CAMPAIGNS = False  # Faster: insights already returns campaign_id/campaign_name
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
        or "US-FB-HR-" in text
        or "US-BO-HR-" in text
    )

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

    # Speed optimization: fetch only accounts that match Okaby / VAL naming.
    # Okaby examples: OK-FB-HR-AA, (R) OK-FB-HR-OS
    # VAL examples: US-FB-HR-AA, US-BO-HR-AA
    if "name" in raw_accounts.columns:
        raw_accounts = raw_accounts[raw_accounts["name"].apply(is_relevant_account_name)].copy()

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
def get_insights_for_account(account_id, since, until):
    clean_id = str(account_id).replace("act_", "")
    url = f"{BASE_URL}/{API_VERSION}/act_{clean_id}/insights"
    params = {
        "fields": ",".join([
            "account_id",
            "account_name",
            "campaign_id",
            "campaign_name",
            "spend",
            "impressions",
            "clicks",
            "ctr",
            "cpc",
            "frequency",
            "actions",
            "date_start",
            "date_stop",
        ]),
        "level": "campaign",
        "time_range": f'{{"since":"{since}","until":"{until}"}}',
        "access_token": ACCESS_TOKEN,
        "limit": 1000,
    }

    rows = fetch_all_pages(url, params)
    df = pd.DataFrame(rows)

    if "actions" not in df.columns:
        df["actions"] = None

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
    insights_df = pd.DataFrame()
    gender_df = pd.DataFrame()
    age_df = pd.DataFrame()
    balance_row = {}
    error = None

    try:
        if FETCH_CAMPAIGNS:
            campaigns_df = get_campaigns(account_id)
        insights_df = get_insights_for_account(account_id, str(since), str(until))

        try:
            age_gender_df = get_age_gender_spend(account_id, str(since), str(until))
            gender_df, age_df = split_age_gender_breakdown(age_gender_df)
        except Exception:
            gender_df = get_gender_spend(account_id, str(since), str(until))
            age_df = get_age_spend(account_id, str(since), str(until))

        balance_row = {}
    except Exception as e:
        error = f"{account_name}: {e}"

    return {
        "account_id": account_id,
        "account_name": account_name,
        "campaigns_df": campaigns_df,
        "insights_df": insights_df,
        "gender_df": gender_df,
        "age_df": age_df,
        "balance_row": balance_row,
        "error": error,
    }

# -----------------------------
# Transform
# -----------------------------
def prepare_data(all_campaigns_df, all_insights_df):
    if all_insights_df.empty:
        return pd.DataFrame()

    fact = all_insights_df.copy()

    if "actions" not in fact.columns:
        fact["actions"] = None

    for col in ["spend", "clicks", "impressions", "ctr", "cpc", "frequency"]:
        if col in fact.columns:
            fact[col] = pd.to_numeric(fact[col], errors="coerce").fillna(0)
        else:
            fact[col] = 0

    if not all_campaigns_df.empty:
        campaigns_map = all_campaigns_df[["id", "name", "account_id"]].rename(
            columns={"id": "campaign_id", "name": "campaign_name_master"}
        )
        fact = fact.merge(campaigns_map, on=["campaign_id", "account_id"], how="left")
    else:
        fact["campaign_name_master"] = None

    if "campaign_name" not in fact.columns:
        fact["campaign_name"] = fact["campaign_name_master"]
    else:
        fact["campaign_name"] = fact["campaign_name"].fillna(fact["campaign_name_master"])

    if "account_name" not in fact.columns:
        fact["account_name"] = "Unknown"

    fact["campaign_name"] = fact["campaign_name"].fillna("Unknown")
    fact["account_name"] = fact["account_name"].fillna("Unknown")

    fact["buyer_code"] = fact["account_name"].apply(extract_buyer_code)
    fact["media_buyer"] = fact["buyer_code"].map(MEDIA_BUYER_MAP).fillna("Unknown")

    fact["objective_label"] = fact["campaign_name"].apply(classify_objective_from_campaign_name)
    fact["actions_map"] = fact["actions"].apply(flatten_actions)
    fact["results"] = fact.apply(lambda r: get_result_by_objective(r["objective_label"], r["actions_map"]), axis=1)
    fact["cpl"] = fact.apply(lambda r: safe_div(r["spend"], r["results"]), axis=1)

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




def detect_business_unit(account_name="", campaign_name=""):
    acc = normalize_text(account_name)
    camp = normalize_text(campaign_name)

    # Account name has priority because Okaby and VAL campaign codes can both contain FB-HR.
    # Okaby: OK-FB-HR-AA or (R) OK-FB-HR-OS
    if "OK-FB-HR-" in acc:
        return "El - Okaby"

    # VAL Hair: US-FB-HR-AA | VAL Booty: US-BO-HR-AA
    if "US-FB-HR-" in acc or "US-BO-HR-" in acc:
        return "VAL"

    # Fallback from campaign naming when account name is missing.
    if re.search(r"(?<![A-Z0-9])(?:AA|HM|BM|EK|MA|AF|SQ|OS|MM|NB)-FB-(?:HR|BO)(?![A-Z0-9])", camp):
        return "VAL"

    return "El - Okaby"

def add_business_unit_columns(fact):
    if fact.empty:
        return fact
    out = fact.copy()
    out["business_unit"] = out.apply(
        lambda r: detect_business_unit(r.get("account_name", ""), r.get("campaign_name", "")),
        axis=1,
    )
    return out

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
        "campaign_name": "Campaign",
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

    agents = [x for x in report_df["Media Buyer"].tolist() if x != "🔵 Overall"]
    if not agents:
        return
    st.markdown("### 📋 Media Buyer Campaign Details")
    selected_agent = st.selectbox("اختر Media Buyer", agents, key=f"{key_prefix}_agent")
    campaign_df = build_campaign_details_for_agent(fact, selected_agent, objective_label)
    if campaign_df.empty:
        st.info("No campaigns found for this media buyer.")
    else:
        cols = ["Campaign", "Spent", "Results", "CPL", "CTR", "CPC", "Frequency", "impressions", "clicks"]
        cols = [c for c in cols if c in campaign_df.columns]
        st.dataframe(format_display_df(campaign_df[cols]), use_container_width=True, hide_index=True)


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
    campaign_df = build_campaign_details_for_agent(fact, selected_agent, None)
    st.markdown("### Campaigns")
    if campaign_df.empty:
        st.info("No campaigns found for this media buyer.")
    else:
        cols = ["objective_label", "Campaign", "Spent", "Results", "CPL", "CTR", "CPC", "Frequency", "impressions", "clicks"]
        cols = [c for c in cols if c in campaign_df.columns]
        st.dataframe(format_display_df(campaign_df[cols]), use_container_width=True, hide_index=True)

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

def build_campaign_summary(fact):
    if fact.empty:
        return pd.DataFrame()

    rows = []
    for keys, grp in fact.groupby(["media_buyer", "objective_label", "campaign_id", "campaign_name"], dropna=False):
        media_buyer, objective_label, campaign_id, campaign_name = keys
        spend = grp["spend"].sum()
        results = grp["results"].sum()
        impressions = grp["impressions"].sum()
        clicks = grp["clicks"].sum()

        rows.append({
            "media_buyer": media_buyer,
            "objective_label": objective_label,
            "campaign_id": campaign_id,
            "campaign_name": campaign_name,
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
        ["Custom", "Today", "Yesterday", "Last 7 Days", "This Month", "Last Month"]
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

            fact = prepare_data(all_campaigns_df, all_insights_df)

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

fact = add_business_unit_columns(snapshot["fact"])
accounts_dedup = snapshot["accounts_dedup"]
accounts_raw = snapshot["accounts_raw"]
gender_df = snapshot.get("gender_df", pd.DataFrame())
age_df = snapshot.get("age_df", pd.DataFrame())
balance_df = snapshot.get("balance_df", pd.DataFrame())
meta = snapshot["meta"]

st.caption(
    f"Last fetch: {meta.get('last_fetch_ts', '-')}"
    f" | Range: {meta.get('date_from', '-')} → {meta.get('date_to', '-')}"
    f" | Accounts: {meta.get('accounts_count', 0)}"
    f" | Errors: {meta.get('errors_count', 0)}"
)

# Main business unit selector on the right
left_title_col, right_filter_col = st.columns([3, 1])
with right_filter_col:
    selected_business_unit = st.selectbox(
        "Business Unit",
        ["El - Okaby", "VAL", "All"],
        index=0,
        key="business_unit_selector",
    )

filtered_fact_main = filter_by_business_unit(fact, selected_business_unit)
filtered_gender_df = filter_related_by_fact_accounts(gender_df, filtered_fact_main)
filtered_age_df = filter_related_by_fact_accounts(age_df, filtered_fact_main)
filtered_balance_df = filter_related_by_fact_accounts(balance_df, filtered_fact_main)
filtered_accounts_dedup = accounts_dedup.copy()
if not filtered_fact_main.empty and "account_id" in filtered_fact_main.columns and "id" in filtered_accounts_dedup.columns:
    filtered_accounts_dedup = filtered_accounts_dedup[filtered_accounts_dedup["id"].astype(str).isin(set(filtered_fact_main["account_id"].astype(str).unique()))]

overall = build_overall_summary(filtered_fact_main)
objective_summary = build_objective_summary(filtered_fact_main) if not filtered_fact_main.empty else pd.DataFrame()
buyer_summary = build_buyer_summary(filtered_fact_main) if not filtered_fact_main.empty else pd.DataFrame()
buyer_summary_with_total = add_overall_row_to_buyer_summary(buyer_summary)
buyer_objective_summary = build_buyer_objective_summary(filtered_fact_main) if not filtered_fact_main.empty else pd.DataFrame()
campaign_summary = build_campaign_summary(filtered_fact_main) if not filtered_fact_main.empty else pd.DataFrame()

audience_by_account = build_audience_table_by_account(filtered_gender_df, filtered_age_df, filtered_balance_df)
audience_by_buyer = build_audience_table_by_buyer(audience_by_account)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Ad Accounts", f"{filtered_accounts_dedup['id'].nunique() if 'id' in filtered_accounts_dedup.columns else 0}")
c2.metric("Total Spend", f"{overall['total_spend']:,.2f}")
c3.metric("Total Results", f"{overall['total_results']:,.0f}")
c4.metric("Overall CPL", "-" if overall["total_cpl"] is None else f"{overall['total_cpl']:,.2f}")

if show_account_sources:
    st.subheader("Loaded Ad Accounts")
    if selected_business_unit != "All" and not filtered_fact_main.empty:
        shown_accounts = accounts_raw[accounts_raw.get("id", pd.Series(dtype=str)).astype(str).isin(set(filtered_fact_main["account_id"].astype(str).unique()))].copy()
    else:
        shown_accounts = accounts_raw.copy()
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
