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
API_VERSION = st.secrets.get("META_API_VERSION", "v25.0")
ACCESS_TOKEN = st.secrets["META_ACCESS_TOKEN"]
BUSINESS_IDS = ["751488620224306", "1178859133269743"]
FETCH_CAMPAIGNS = True  # Needed to fetch campaign status (Active / Not Active)
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

# Dedicated account cache. It is refreshed only by Sync Ad Accounts, not on every data refresh.
SYNCED_ACCOUNTS_FILE = DATA_DIR / "synced_accounts.parquet"
SYNCED_RAW_ACCOUNTS_FILE = DATA_DIR / "synced_raw_accounts.parquet"
ACCOUNT_SYNC_META_FILE = DATA_DIR / "account_sync_meta.json"
ACCOUNT_CACHE_VERSION = 2

TMP_FACT_FILE = DATA_DIR / "fact_snapshot.tmp.parquet"
TMP_ACCOUNTS_FILE = DATA_DIR / "accounts_snapshot.tmp.parquet"
TMP_RAW_ACCOUNTS_FILE = DATA_DIR / "raw_accounts_snapshot.tmp.parquet"
TMP_META_FILE = DATA_DIR / "meta_snapshot.tmp.json"
TMP_GENDER_FILE = DATA_DIR / "gender_snapshot.tmp.parquet"
TMP_AGE_FILE = DATA_DIR / "age_snapshot.tmp.parquet"
TMP_BALANCE_FILE = DATA_DIR / "balance_snapshot.tmp.parquet"
TMP_SYNCED_ACCOUNTS_FILE = DATA_DIR / "synced_accounts.tmp.parquet"
TMP_SYNCED_RAW_ACCOUNTS_FILE = DATA_DIR / "synced_raw_accounts.tmp.parquet"
TMP_ACCOUNT_SYNC_META_FILE = DATA_DIR / "account_sync_meta.tmp.json"

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
def _graph_error_message(response):
    try:
        payload = response.json()
    except Exception:
        payload = {}

    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    message = error.get("message") or response.text[:500] or "Unknown Meta API error"
    parts = [f"HTTP {response.status_code}"]
    if error.get("type"):
        parts.append(str(error["type"]))
    if error.get("code") is not None:
        parts.append(f"code={error['code']}")
    if error.get("error_subcode") is not None:
        parts.append(f"subcode={error['error_subcode']}")
    return f"{' | '.join(parts)} | {message}"


def _graph_headers():
    return {"Authorization": f"Bearer {ACCESS_TOKEN}"}


def _graph_get(url, params=None, timeout=90, retries=3):
    """GET with small retry/backoff for temporary Meta or network failures."""
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = requests.get(
                url,
                params=params or {},
                headers=_graph_headers(),
                timeout=timeout,
            )
            if response.ok:
                data = response.json()
                if not isinstance(data, dict):
                    raise RuntimeError(f"Unexpected Meta response for {url}")
                return data

            error_text = _graph_error_message(response)
            last_error = RuntimeError(error_text)

            # Retry only temporary/rate-limit/server conditions.
            retryable = response.status_code in {429, 500, 502, 503, 504}
            try:
                code = (response.json().get("error") or {}).get("code")
                retryable = retryable or code in {1, 2, 4, 17, 32, 613}
            except Exception:
                pass

            if not retryable or attempt >= retries:
                raise last_error
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc
            if attempt >= retries:
                raise RuntimeError(f"Network error for {url}: {exc}") from exc

        time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(str(last_error) if last_error else f"Meta request failed: {url}")


def fetch_all_pages(url, params=None, source_name=None, include_summary=False):
    """Fetch all Graph pages without using Streamlit cache inside worker threads."""
    rows = []
    pages = 0
    total_count = None
    visited = set()
    current_url = url
    current_params = dict(params or {})
    current_params.setdefault("limit", 100)
    if include_summary:
        current_params.setdefault("summary", "true")

    while True:
        if current_url in visited:
            raise RuntimeError(f"Pagination loop detected for {source_name or current_url}")
        visited.add(current_url)

        data = _graph_get(current_url, current_params)
        pages += 1
        page_rows = data.get("data", [])
        if isinstance(page_rows, list):
            rows.extend(page_rows)

        summary = data.get("summary") or {}
        if total_count is None and isinstance(summary, dict):
            total_count = summary.get("total_count")

        next_url = (data.get("paging") or {}).get("next")
        if not next_url:
            break
        current_url = next_url
        current_params = None

    return rows, {"count": len(rows), "pages": pages, "total_count": total_count}


def _secret_id_list(name):
    raw = st.secrets.get(name, "")
    values = raw if isinstance(raw, (list, tuple)) else re.split(r"[\s,;]+", str(raw))
    return [str(v).strip().replace("act_", "") for v in values if str(v).strip()]


def _account_sync_sources():
    """Only inspect the configured businesses; no dynamic seven-business discovery."""
    business_ids = [str(x).strip() for x in BUSINESS_IDS if str(x).strip()]
    business_ids.extend(_secret_id_list("EXTRA_BUSINESS_IDS"))
    business_ids = list(dict.fromkeys(business_ids))

    sources = [("me/adaccounts", f"{BASE_URL}/{API_VERSION}/me/adaccounts")]
    for business_id in business_ids:
        sources.extend([
            (
                f"business/{business_id}/owned_ad_accounts",
                f"{BASE_URL}/{API_VERSION}/{business_id}/owned_ad_accounts",
            ),
            (
                f"business/{business_id}/client_ad_accounts",
                f"{BASE_URL}/{API_VERSION}/{business_id}/client_ad_accounts",
            ),
        ])
    return sources


def _filter_relevant_synced_accounts(raw_accounts):
    """Keep all El-Okaby business accounts and only named VAL accounts."""
    if raw_accounts.empty:
        return raw_accounts.copy()

    df = raw_accounts.copy()
    if "name" not in df.columns:
        return df

    name_match = df["name"].apply(is_relevant_account_name)
    source = df.get("source", pd.Series("", index=df.index)).astype(str)
    okaby_source = source.apply(source_is_okaby)

    # This preserves the original intended rule:
    # all accounts belonging to El-Okaby, plus matching VAL accounts.
    return df[name_match | okaby_source].copy()


def save_synced_accounts(accounts_df, raw_accounts_df, source_report):
    accounts_df.to_parquet(TMP_SYNCED_ACCOUNTS_FILE, index=False)
    raw_accounts_df.to_parquet(TMP_SYNCED_RAW_ACCOUNTS_FILE, index=False)
    meta = {
        "cache_version": ACCOUNT_CACHE_VERSION,
        "last_sync_ts": pd.Timestamp.utcnow().isoformat(),
        "accounts_count": int(accounts_df["id"].nunique()) if "id" in accounts_df.columns else len(accounts_df),
        "raw_rows_count": int(len(raw_accounts_df)),
        "business_ids": BUSINESS_IDS,
        "source_report": source_report,
    }
    with open(TMP_ACCOUNT_SYNC_META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    os.replace(TMP_SYNCED_ACCOUNTS_FILE, SYNCED_ACCOUNTS_FILE)
    os.replace(TMP_SYNCED_RAW_ACCOUNTS_FILE, SYNCED_RAW_ACCOUNTS_FILE)
    os.replace(TMP_ACCOUNT_SYNC_META_FILE, ACCOUNT_SYNC_META_FILE)
    return meta


def load_synced_accounts():
    try:
        if not (SYNCED_ACCOUNTS_FILE.exists() and ACCOUNT_SYNC_META_FILE.exists()):
            return pd.DataFrame(), pd.DataFrame(), {}
        with open(ACCOUNT_SYNC_META_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("cache_version") != ACCOUNT_CACHE_VERSION:
            return pd.DataFrame(), pd.DataFrame(), {}

        accounts_df = safe_read_parquet(SYNCED_ACCOUNTS_FILE)
        raw_df = safe_read_parquet(SYNCED_RAW_ACCOUNTS_FILE)
        if accounts_df.empty:
            return pd.DataFrame(), pd.DataFrame(), {}
        return accounts_df, raw_df, meta
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), {}


def sync_ad_accounts():
    """Synchronize account IDs once, using lightweight fields only."""
    all_dfs = []
    source_report = []
    fields = "id,account_id,name,account_status,currency"

    for source_name, url in _account_sync_sources():
        try:
            rows, page_meta = fetch_all_pages(
                url,
                {"fields": fields, "limit": 100},
                source_name=source_name,
                include_summary=True,
            )
            source_report.append({
                "source": source_name,
                "count": page_meta["count"],
                "pages": page_meta["pages"],
                "total_count": page_meta["total_count"],
                "error": None,
            })
            print(
                f"[ACCOUNT_SYNC] OK source={source_name} rows={page_meta['count']} "
                f"pages={page_meta['pages']}",
                flush=True,
            )
            if rows:
                df = pd.DataFrame(rows)
                df["source"] = source_name
                all_dfs.append(df)
        except Exception as exc:
            source_report.append({
                "source": source_name,
                "count": 0,
                "pages": 0,
                "total_count": None,
                "error": str(exc),
            })
            print(f"[ACCOUNT_SYNC] ERROR source={source_name} error={exc}", flush=True)

    # Optional explicit IDs for accounts that are not exposed by list endpoints.
    explicit_rows = []
    for clean_id in _secret_id_list("EXTRA_AD_ACCOUNT_IDS"):
        source_name = f"explicit/act_{clean_id}"
        try:
            data = _graph_get(
                f"{BASE_URL}/{API_VERSION}/act_{clean_id}",
                {"fields": fields},
            )
            data["source"] = source_name
            explicit_rows.append(data)
            source_report.append({
                "source": source_name,
                "count": 1,
                "pages": 1,
                "total_count": 1,
                "error": None,
            })
        except Exception as exc:
            source_report.append({
                "source": source_name,
                "count": 0,
                "pages": 0,
                "total_count": None,
                "error": str(exc),
            })

    if explicit_rows:
        all_dfs.append(pd.DataFrame(explicit_rows))

    if not all_dfs:
        errors = "; ".join(
            f"{item['source']}: {item['error']}"
            for item in source_report if item.get("error")
        )
        raise RuntimeError(f"No ad accounts were returned. {errors}")

    raw_all = pd.concat(all_dfs, ignore_index=True, sort=False)
    relevant_raw = _filter_relevant_synced_accounts(raw_all)
    if relevant_raw.empty:
        raise RuntimeError("Meta returned accounts, but none matched the Okaby / VAL account rules.")

    dedup = relevant_raw.copy()
    sort_cols = [c for c in ["name", "source"] if c in dedup.columns]
    if sort_cols:
        dedup = dedup.sort_values(sort_cols)
    key = "id" if "id" in dedup.columns else "account_id"
    dedup = dedup.drop_duplicates(subset=[key], keep="first").reset_index(drop=True)

    meta = save_synced_accounts(dedup, relevant_raw.reset_index(drop=True), source_report)
    print(
        f"[ACCOUNT_SYNC] COMPLETE relevant_accounts={len(dedup)} raw_relevant_rows={len(relevant_raw)}",
        flush=True,
    )
    return dedup, relevant_raw.reset_index(drop=True), meta


def get_campaigns(account_id):
    clean_id = normalize_account_id(account_id)
    rows, _ = fetch_all_pages(
        f"{BASE_URL}/{API_VERSION}/act_{clean_id}/campaigns",
        {"fields": "id,name,status,effective_status", "limit": 1000},
        source_name=f"act_{clean_id}/campaigns",
    )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["account_id"] = f"act_{clean_id}"
    return df


def get_insights_for_account(account_id, since, until):
    clean_id = normalize_account_id(account_id)
    params = {
        "fields": ",".join([
            "account_id", "account_name", "campaign_id", "campaign_name",
            "spend", "impressions", "clicks", "ctr", "cpc", "frequency",
            "actions", "date_start", "date_stop",
        ]),
        "level": "campaign",
        "time_range": json.dumps({"since": str(since), "until": str(until)}),
        "limit": 1000,
    }
    rows, _ = fetch_all_pages(
        f"{BASE_URL}/{API_VERSION}/act_{clean_id}/insights",
        params,
        source_name=f"act_{clean_id}/insights",
    )
    df = pd.DataFrame(rows)
    if "actions" not in df.columns:
        df["actions"] = None
    return df


def get_gender_spend(account_id, since, until):
    clean_id = normalize_account_id(account_id)
    rows, _ = fetch_all_pages(
        f"{BASE_URL}/{API_VERSION}/act_{clean_id}/insights",
        {
            "fields": "spend",
            "breakdowns": "gender",
            "time_range": json.dumps({"since": str(since), "until": str(until)}),
            "limit": 1000,
        },
        source_name=f"act_{clean_id}/gender",
    )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["account_id"] = f"act_{clean_id}"
    if "spend" not in df.columns:
        df["spend"] = 0
    if "gender" not in df.columns:
        df["gender"] = "unknown"
    return df


def get_age_spend(account_id, since, until):
    clean_id = normalize_account_id(account_id)
    rows, _ = fetch_all_pages(
        f"{BASE_URL}/{API_VERSION}/act_{clean_id}/insights",
        {
            "fields": "spend",
            "breakdowns": "age",
            "time_range": json.dumps({"since": str(since), "until": str(until)}),
            "limit": 1000,
        },
        source_name=f"act_{clean_id}/age",
    )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["account_id"] = f"act_{clean_id}"
    if "spend" not in df.columns:
        df["spend"] = 0
    if "age" not in df.columns:
        df["age"] = "unknown"
    return df


def get_age_gender_spend(account_id, since, until):
    clean_id = normalize_account_id(account_id)
    rows, _ = fetch_all_pages(
        f"{BASE_URL}/{API_VERSION}/act_{clean_id}/insights",
        {
            "fields": "spend",
            "breakdowns": "age,gender",
            "time_range": json.dumps({"since": str(since), "until": str(until)}),
            "limit": 1000,
        },
        source_name=f"act_{clean_id}/age_gender",
    )
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
    gender_df = df.groupby(["account_id", "gender"], dropna=False).agg(spend=("spend", "sum")).reset_index()
    age_df = df.groupby(["account_id", "age"], dropna=False).agg(spend=("spend", "sum")).reset_index()
    return gender_df, age_df


def parse_balance_from_display_string(display_string):
    if not display_string:
        return None
    match = re.search(r"([\d,.]+)", str(display_string))
    return to_float(match.group(1).replace(",", ""), default=None) if match else None


def fetch_one_account(row, since, until, include_audience=False):
    """Insights first; skip extra calls for accounts with no rows in the date range."""
    account_id = row["id"]
    account_name = row.get("name", account_id)
    campaigns_df = pd.DataFrame()
    insights_df = pd.DataFrame()
    gender_df = pd.DataFrame()
    age_df = pd.DataFrame()
    errors = []

    try:
        insights_df = get_insights_for_account(account_id, since, until)
    except Exception as exc:
        errors.append(f"Insights: {exc}")
        print(
            f"[ACCOUNT_FETCH_ERROR] account={account_name} id={account_id} stage=insights error={exc}",
            flush=True,
        )
        return {
            "account_id": account_id,
            "account_name": account_name,
            "campaigns_df": campaigns_df,
            "insights_df": insights_df,
            "gender_df": gender_df,
            "age_df": age_df,
            "balance_row": {},
            "error": f"{account_name}: {'; '.join(errors)}",
        }

    # No spend/insight rows means no need for campaigns or audience requests.
    if insights_df.empty:
        return {
            "account_id": account_id,
            "account_name": account_name,
            "campaigns_df": campaigns_df,
            "insights_df": insights_df,
            "gender_df": gender_df,
            "age_df": age_df,
            "balance_row": {},
            "error": None,
        }

    if FETCH_CAMPAIGNS:
        try:
            campaigns_df = get_campaigns(account_id)
        except Exception as exc:
            errors.append(f"Campaigns: {exc}")
            print(
                f"[ACCOUNT_FETCH_ERROR] account={account_name} id={account_id} stage=campaigns error={exc}",
                flush=True,
            )

    if include_audience:
        try:
            age_gender_df = get_age_gender_spend(account_id, since, until)
            gender_df, age_df = split_age_gender_breakdown(age_gender_df)
        except Exception as combined_exc:
            try:
                gender_df = get_gender_spend(account_id, since, until)
                age_df = get_age_spend(account_id, since, until)
            except Exception as fallback_exc:
                errors.append(f"Audience: {fallback_exc}")
                print(
                    f"[ACCOUNT_FETCH_ERROR] account={account_name} id={account_id} "
                    f"stage=audience error={fallback_exc}; combined_error={combined_exc}",
                    flush=True,
                )

    return {
        "account_id": account_id,
        "account_name": account_name,
        "campaigns_df": campaigns_df,
        "insights_df": insights_df,
        "gender_df": gender_df,
        "age_df": age_df,
        "balance_row": {},
        "error": f"{account_name}: {'; '.join(errors)}" if errors else None,
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
        campaign_cols = ["id", "name", "account_id"]
        for col in ["status", "effective_status"]:
            if col in all_campaigns_df.columns:
                campaign_cols.append(col)

        campaigns_map = all_campaigns_df[campaign_cols].copy()
        campaigns_map["_account_id_clean"] = campaigns_map["account_id"].apply(normalize_account_id)
        campaigns_map = campaigns_map.drop(columns=["account_id"], errors="ignore").rename(
            columns={"id": "campaign_id", "name": "campaign_name_master"}
        )

        fact["_account_id_clean"] = fact["account_id"].apply(normalize_account_id)
        fact = fact.merge(campaigns_map, on=["campaign_id", "_account_id_clean"], how="left")
        fact = fact.drop(columns=["_account_id_clean"], errors="ignore")
    else:
        fact["campaign_name_master"] = None
        fact["status"] = None
        fact["effective_status"] = None

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

    if "status" not in fact.columns:
        fact["status"] = None
    if "effective_status" not in fact.columns:
        fact["effective_status"] = None

    fact["campaign_status"] = fact.apply(
        lambda r: campaign_status_label(r.get("status"), r.get("effective_status")),
        axis=1,
    )

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

def build_campaign_summary(fact):
    if fact.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["media_buyer", "objective_label", "account_name", "campaign_id", "campaign_name"]
    for keys, grp in fact.groupby(group_cols, dropna=False):
        media_buyer, objective_label, account_name, campaign_id, campaign_name = keys
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

    max_workers = st.slider("Parallel workers", min_value=2, max_value=8, value=4, step=2)
    refresh_audience = st.checkbox(
        "Refresh age/gender breakdown",
        value=False,
        help="Leave this off for a faster refresh. Existing audience data is preserved.",
    )
    show_account_sources = st.checkbox("Show account sources", value=True)
    sync_accounts_clicked = st.button("Sync Ad Accounts", use_container_width=True)
    refresh_clicked = st.button("Refresh Data", use_container_width=True, type="primary")
    force_unlock_clicked = st.button("Clear stuck refresh lock", use_container_width=True)

if force_unlock_clicked:
    force_clear_refresh_lock()
    st.success("Refresh lock cleared. You can click Refresh Data now.")

if sync_accounts_clicked:
    if not acquire_refresh_lock():
        st.warning("Another refresh or account sync is already running.")
        st.stop()
    try:
        with st.status("Syncing configured ad accounts...", expanded=True) as status:
            synced_accounts, synced_raw, sync_meta = sync_ad_accounts()
            status.write(f"Synced {len(synced_accounts)} relevant Okaby / VAL ad accounts.")
            report_df = pd.DataFrame(sync_meta.get("source_report", []))
            if not report_df.empty:
                st.dataframe(report_df, use_container_width=True, hide_index=True)
            status.update(label="Ad account sync complete.", state="complete")
        st.success(f"Saved {len(synced_accounts)} ad accounts. Refresh Data will now use this saved list.")
    except Exception as exc:
        st.error(f"Account sync failed: {exc}")
        raise
    finally:
        release_refresh_lock()

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

        with st.status("Refreshing saved ad accounts...", expanded=True) as status:
            accounts_df, raw_accounts_df, account_sync_meta = load_synced_accounts()
            if accounts_df.empty:
                status.write("No valid saved account list found. Running one-time account sync...")
                accounts_df, raw_accounts_df, account_sync_meta = sync_ad_accounts()

            status.write(
                f"Using {len(accounts_df)} saved relevant ad accounts. "
                f"Last account sync: {account_sync_meta.get('last_sync_ts', '-')}"
            )

            if accounts_df.empty:
                st.error("No matching Okaby / VAL ad accounts found. Click Sync Ad Accounts.")
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
                    executor.submit(fetch_one_account, row, since, until, refresh_audience)
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

            if refresh_audience:
                gender_df = pd.concat(all_gender, ignore_index=True) if all_gender else pd.DataFrame()
                age_df = pd.concat(all_age, ignore_index=True) if all_age else pd.DataFrame()
            else:
                gender_df = old_snapshot.get("gender_df", pd.DataFrame()) if old_snapshot else pd.DataFrame()
                age_df = old_snapshot.get("age_df", pd.DataFrame()) if old_snapshot else pd.DataFrame()
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
            fact = assign_fact_business_unit_from_accounts(fact, accounts_df)

            if fact.empty:
                print(
                    f"[REFRESH_SUMMARY] accounts={len(accounts_df)} insight_rows=0 errors={len(errors)}",
                    flush=True,
                )
                st.error(
                    "No campaign insights were returned for the selected date range. "
                    "The saved snapshot was not overwritten. Try Last 7 Days and review the errors shown below."
                )
                if errors:
                    st.code("\n".join(errors[:30]))
                st.stop()

            print(
                f"[REFRESH_SUMMARY] accounts={len(accounts_df)} "
                f"insight_rows={len(all_insights_df)} fact_rows={len(fact)} errors={len(errors)}",
                flush=True,
            )

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
                "account_sync_ts": account_sync_meta.get("last_sync_ts"),
                "audience_refreshed": bool(refresh_audience),
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
fact = assign_fact_business_unit_from_accounts(snapshot["fact"], accounts_dedup)
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
        index=3,
        key="business_unit_selector",
    )

filtered_fact_main = filter_by_business_unit(fact, selected_business_unit)
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
