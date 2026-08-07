import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import streamlit as st


# =========================================================
# App config
# =========================================================
st.set_page_config(
    page_title="Meta Governorate Spend Dashboard",
    layout="wide",
)

BASE_URL = "https://graph.facebook.com"
API_VERSION = st.secrets.get("META_API_VERSION", "v25.0")
ACCESS_TOKEN = st.secrets["META_ACCESS_TOKEN"]

REQUEST_TIMEOUT = 90
ACCESS_CACHE_TTL = 15 * 60
MAX_WORKERS = int(st.secrets.get("MAX_WORKERS", 8))
MAX_WORKERS = max(2, min(MAX_WORKERS, 20))

HEADERS = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
}


# =========================================================
# Helpers
# =========================================================
def normalize_account_id(value):
    if value is None:
        return ""
    return str(value).replace("act_", "").strip()


def safe_error_text(response):
    try:
        payload = response.json()
        error = payload.get("error", {})
        message = error.get("message") or str(payload)
        code = error.get("code")
        subcode = error.get("error_subcode")
        parts = [str(message)]
        if code is not None:
            parts.append(f"code={code}")
        if subcode is not None:
            parts.append(f"subcode={subcode}")
        return " | ".join(parts)
    except Exception:
        return (response.text or "Unknown Meta API error")[:1500]


def fetch_all_pages(url, params=None):
    rows = []
    current_url = url
    current_params = params

    while True:
        response = requests.get(
            current_url,
            params=current_params,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )

        if not response.ok:
            raise RuntimeError(
                f"Meta API {response.status_code}: {safe_error_text(response)}"
            )

        payload = response.json()
        rows.extend(payload.get("data", []))

        next_url = payload.get("paging", {}).get("next")
        if not next_url:
            break

        current_url = next_url
        current_params = None

    return rows


def get_optional_business_ids_from_secrets():
    """
    Optional fallback only.

    If /me/businesses does not expose every business for a system-user token,
    you can optionally add this to Streamlit secrets:

        BUSINESS_IDS = ["123456789", "987654321"]
    """
    raw = st.secrets.get("BUSINESS_IDS", [])

    if raw in [None, ""]:
        return []

    if isinstance(raw, (list, tuple, set)):
        values = list(raw)
    else:
        values = [x.strip() for x in str(raw).replace(";", ",").split(",")]

    output = []
    seen = set()

    for value in values:
        value = str(value).strip()
        if value and value not in seen:
            seen.add(value)
            output.append(value)

    return output


def get_business_name(business_id):
    url = f"{BASE_URL}/{API_VERSION}/{business_id}"
    params = {"fields": "id,name"}

    response = requests.get(
        url,
        params=params,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )

    if not response.ok:
        return f"Business {business_id}"

    data = response.json()
    return data.get("name") or f"Business {business_id}"


def fetch_accounts_from_endpoint(url):
    rows = fetch_all_pages(
        url,
        {
            "fields": "id,account_id,name,account_status,currency",
            "limit": 500,
        },
    )

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(
            columns=[
                "id",
                "account_id",
                "name",
                "account_status",
                "currency",
            ]
        )

    for col in ["id", "account_id", "name", "account_status", "currency"]:
        if col not in df.columns:
            df[col] = None

    df["id"] = df["id"].astype(str)

    df["account_id"] = df["account_id"].fillna(
        df["id"].apply(normalize_account_id)
    )

    df["name"] = df["name"].fillna(df["id"])
    df["currency"] = df["currency"].fillna("Unknown")

    return df[
        [
            "id",
            "account_id",
            "name",
            "account_status",
            "currency",
        ]
    ].copy()


@st.cache_data(ttl=ACCESS_CACHE_TTL, show_spinner=False)
def discover_accessible_accounts_and_businesses():
    """
    Discover all ad accounts the token can access.

    Sources:
    - /me/adaccounts
    - Every discovered business:
        /{business_id}/owned_ad_accounts
        /{business_id}/client_ad_accounts

    The business list is discovered from /me/businesses.
    Optional BUSINESS_IDS from secrets can supplement discovery.
    """
    errors = []
    business_records = {}

    # 1) Discover businesses available to the token.
    try:
        business_rows = fetch_all_pages(
            f"{BASE_URL}/{API_VERSION}/me/businesses",
            {
                "fields": "id,name",
                "limit": 500,
            },
        )

        for row in business_rows:
            business_id = str(row.get("id", "")).strip()

            if business_id:
                business_records[business_id] = (
                    row.get("name") or f"Business {business_id}"
                )

    except Exception as exc:
        errors.append(f"/me/businesses: {exc}")

    # 2) Optional fallback business IDs from Streamlit secrets.
    for business_id in get_optional_business_ids_from_secrets():
        if business_id not in business_records:
            business_records[business_id] = get_business_name(business_id)

    # 3) Pull all directly accessible accounts.
    direct_df = pd.DataFrame()

    try:
        direct_df = fetch_accounts_from_endpoint(
            f"{BASE_URL}/{API_VERSION}/me/adaccounts"
        )

    except Exception as exc:
        errors.append(f"/me/adaccounts: {exc}")

    # 4) Pull owned + client accounts for every discovered business.
    business_account_frames = []
    membership_frames = []

    for business_id, business_name in business_records.items():
        endpoints = [
            (
                "owned_ad_accounts",
                f"{BASE_URL}/{API_VERSION}/{business_id}/owned_ad_accounts",
            ),
            (
                "client_ad_accounts",
                f"{BASE_URL}/{API_VERSION}/{business_id}/client_ad_accounts",
            ),
        ]

        for source_name, url in endpoints:
            try:
                df = fetch_accounts_from_endpoint(url)

                if df.empty:
                    continue

                business_account_frames.append(df.copy())

                membership = df[
                    ["id", "name", "currency"]
                ].copy()

                membership["business_id"] = business_id
                membership["business_name"] = business_name
                membership["source"] = source_name

                membership_frames.append(membership)

            except Exception as exc:
                errors.append(
                    f"{business_name} ({business_id}) / {source_name}: {exc}"
                )

    business_accounts_df = (
        pd.concat(
            business_account_frames,
            ignore_index=True,
        )
        if business_account_frames
        else pd.DataFrame()
    )

    # 5) Build one unique account list from every source.
    master_frames = []

    if not direct_df.empty:
        master_frames.append(direct_df)

    if not business_accounts_df.empty:
        master_frames.append(business_accounts_df)

    if master_frames:
        account_master = pd.concat(
            master_frames,
            ignore_index=True,
        )

        account_master = (
            account_master.sort_values(["name", "id"])
            .drop_duplicates(subset=["id"], keep="first")
            .reset_index(drop=True)
        )

    else:
        account_master = pd.DataFrame(
            columns=[
                "id",
                "account_id",
                "name",
                "account_status",
                "currency",
            ]
        )

    # 6) Business/account membership map.
    if membership_frames:
        account_business_map = pd.concat(
            membership_frames,
            ignore_index=True,
        )

        account_business_map = (
            account_business_map.drop_duplicates(
                subset=["business_id", "id"]
            )
            .reset_index(drop=True)
        )

    else:
        account_business_map = pd.DataFrame(
            columns=[
                "id",
                "name",
                "currency",
                "business_id",
                "business_name",
                "source",
            ]
        )

    # 7) Accounts returned by /me/adaccounts but not mapped to any business.
    if not direct_df.empty:
        mapped_ids = set(
            account_business_map["id"].astype(str).tolist()
            if not account_business_map.empty
            else []
        )

        direct_only = direct_df[
            ~direct_df["id"].astype(str).isin(mapped_ids)
        ].copy()

        if not direct_only.empty:
            direct_membership = direct_only[
                ["id", "name", "currency"]
            ].copy()

            direct_membership["business_id"] = "__direct__"
            direct_membership["business_name"] = "Direct / Other Access"
            direct_membership["source"] = "me/adaccounts"

            account_business_map = pd.concat(
                [
                    account_business_map,
                    direct_membership,
                ],
                ignore_index=True,
            )

    # 8) Business selector rows.
    if not account_business_map.empty:
        business_df = (
            account_business_map[
                ["business_id", "business_name"]
            ]
            .drop_duplicates()
            .sort_values("business_name")
            .reset_index(drop=True)
        )
    else:
        business_df = pd.DataFrame(
            columns=["business_id", "business_name"]
        )

    return (
        account_master,
        account_business_map,
        business_df,
        errors,
    )


def fetch_region_spend_for_account(account_row, since, until):
    """
    Pull ONLY spend broken down by Meta region/governorate.

    Requested metric:
    - spend

    Requested dimensions:
    - account_id
    - account_name
    - region

    No impressions, clicks, leads, actions, CPL, CTR, CPC, etc.
    """
    account_id = str(account_row["id"])
    clean_id = normalize_account_id(account_id)
    account_name = str(account_row.get("name", account_id))
    currency = str(account_row.get("currency", "Unknown"))

    url = f"{BASE_URL}/{API_VERSION}/act_{clean_id}/insights"

    params = {
        "fields": "account_id,account_name,spend",
        "level": "account",
        "breakdowns": "region",
        "time_range": json.dumps(
            {
                "since": str(since),
                "until": str(until),
            }
        ),
        "limit": 5000,
    }

    try:
        rows = fetch_all_pages(url, params)
        df = pd.DataFrame(rows)

        if df.empty:
            return {
                "account_id": f"act_{clean_id}",
                "account_name": account_name,
                "data": pd.DataFrame(),
                "error": None,
            }

        if "region" not in df.columns:
            df["region"] = "Unknown"

        if "spend" not in df.columns:
            df["spend"] = 0

        df["spend"] = pd.to_numeric(
            df["spend"],
            errors="coerce",
        ).fillna(0.0)

        df["region"] = (
            df["region"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )

        df["account_id"] = f"act_{clean_id}"
        df["account_name"] = account_name
        df["currency"] = currency

        df = df[
            [
                "account_id",
                "account_name",
                "currency",
                "region",
                "spend",
            ]
        ].copy()

        return {
            "account_id": f"act_{clean_id}",
            "account_name": account_name,
            "data": df,
            "error": None,
        }

    except Exception as exc:
        return {
            "account_id": f"act_{clean_id}",
            "account_name": account_name,
            "data": pd.DataFrame(),
            "error": str(exc),
        }


def build_scope_options(account_business_map):
    options = [
        (
            "__overall__",
            "Overall - All Accessible Ad Accounts",
        )
    ]

    if account_business_map.empty:
        return options

    rows = (
        account_business_map[
            ["business_id", "business_name"]
        ]
        .drop_duplicates()
        .sort_values("business_name")
    )

    for _, row in rows.iterrows():
        business_id = str(row["business_id"])
        business_name = str(row["business_name"])

        if business_id == "__direct__":
            label = business_name
        else:
            label = f"{business_name} ({business_id})"

        options.append((business_id, label))

    return options


def get_scope_account_ids(
    selected_scope,
    account_master,
    account_business_map,
):
    if account_master.empty:
        return set()

    if selected_scope == "__overall__":
        return set(
            account_master["id"].astype(str).tolist()
        )

    if account_business_map.empty:
        return set()

    ids = account_business_map.loc[
        account_business_map["business_id"].astype(str)
        == str(selected_scope),
        "id",
    ].astype(str)

    return set(ids.tolist())


def format_money_table(df, spend_col="Spend"):
    out = df.copy()

    if spend_col in out.columns:
        out[spend_col] = pd.to_numeric(
            out[spend_col],
            errors="coerce",
        ).fillna(0.0).round(2)

    return out


# =========================================================
# Access discovery
# =========================================================
st.title("Meta Governorate Spend Dashboard")
st.caption(
    "Spend only — broken down by Meta region/governorate "
    "across every accessible ad account."
)

with st.spinner(
    "Loading accessible businesses and ad accounts..."
):
    try:
        (
            account_master,
            account_business_map,
            business_df,
            discovery_errors,
        ) = discover_accessible_accounts_and_businesses()

    except Exception as exc:
        st.error(
            f"Could not load Meta account access: {exc}"
        )
        st.stop()


if account_master.empty:
    st.error(
        "No accessible ad accounts were found for META_ACCESS_TOKEN."
    )
    st.stop()


# =========================================================
# Sidebar
# =========================================================
scope_options = build_scope_options(
    account_business_map
)

scope_keys = [
    key for key, _ in scope_options
]

scope_labels = dict(scope_options)


with st.sidebar:
    st.header("Report Filters")

    selected_scope = st.selectbox(
        "Business",
        scope_keys,
        format_func=lambda key: scope_labels.get(
            key,
            key,
        ),
        index=0,
    )

    quick_range = st.selectbox(
        "Quick Range",
        [
            "Custom",
            "Today",
            "Yesterday",
            "Last 7 Days",
            "This Month",
            "Last Month",
        ],
        index=4,
    )

    today = (
        pd.Timestamp.now(tz="Africa/Cairo")
        .normalize()
        .tz_localize(None)
        .date()
    )

    if quick_range == "Today":
        since = today
        until = today

    elif quick_range == "Yesterday":
        since = (
            pd.Timestamp(today)
            - pd.Timedelta(days=1)
        ).date()

        until = since

    elif quick_range == "Last 7 Days":
        since = (
            pd.Timestamp(today)
            - pd.Timedelta(days=6)
        ).date()

        until = today

    elif quick_range == "This Month":
        since = (
            pd.Timestamp(today)
            .replace(day=1)
            .date()
        )

        until = today

    elif quick_range == "Last Month":
        first_this_month = (
            pd.Timestamp(today)
            .replace(day=1)
        )

        last_month_end = (
            first_this_month
            - pd.Timedelta(days=1)
        )

        since = (
            last_month_end
            .replace(day=1)
            .date()
        )

        until = last_month_end.date()

    else:
        since = st.date_input(
            "From",
            value=(
                pd.Timestamp(today)
                .replace(day=1)
                .date()
            ),
        )

        until = st.date_input(
            "To",
            value=today,
        )

    st.caption(
        f"Selected range: {since} → {until}"
    )

    refresh_clicked = st.button(
        "Refresh Data",
        use_container_width=True,
        type="primary",
    )

    reload_access_clicked = st.button(
        "Reload Businesses / Accounts",
        use_container_width=True,
    )


if reload_access_clicked:
    st.cache_data.clear()
    st.rerun()


# =========================================================
# Refresh all accessible accounts
# =========================================================
if refresh_clicked:
    if since > until:
        st.error(
            "'From' date cannot be after 'To' date."
        )
        st.stop()

    unique_accounts = (
        account_master
        .sort_values(["name", "id"])
        .drop_duplicates(subset=["id"])
        .reset_index(drop=True)
    )

    frames = []
    errors = []

    total = len(unique_accounts)
    done = 0
    started_at = time.time()

    progress = st.progress(0)
    progress_text = st.empty()

    with ThreadPoolExecutor(
        max_workers=min(
            MAX_WORKERS,
            max(2, total),
        )
    ) as executor:

        futures = [
            executor.submit(
                fetch_region_spend_for_account,
                row,
                since,
                until,
            )
            for _, row in unique_accounts.iterrows()
        ]

        for future in as_completed(futures):
            result = future.result()
            done += 1

            if result["error"]:
                errors.append(
                    f"{result['account_name']}: "
                    f"{result['error']}"
                )

            if not result["data"].empty:
                frames.append(
                    result["data"]
                )

            progress.progress(
                done / total
            )

            elapsed = (
                time.time() - started_at
            )

            progress_text.info(
                f"Refreshing ad accounts: "
                f"{done}/{total} | "
                f"Last finished: "
                f"{result['account_name']} | "
                f"Elapsed: {elapsed:.1f}s | "
                f"Errors: {len(errors)}"
            )

    progress.empty()
    progress_text.empty()

    report_df = (
        pd.concat(
            frames,
            ignore_index=True,
        )
        if frames
        else pd.DataFrame(
            columns=[
                "account_id",
                "account_name",
                "currency",
                "region",
                "spend",
            ]
        )
    )

    if not report_df.empty:
        report_df["spend"] = (
            pd.to_numeric(
                report_df["spend"],
                errors="coerce",
            )
            .fillna(0.0)
        )

        report_df = (
            report_df.groupby(
                [
                    "account_id",
                    "account_name",
                    "currency",
                    "region",
                ],
                dropna=False,
                as_index=False,
            )["spend"]
            .sum()
        )

    st.session_state[
        "region_spend_report"
    ] = report_df

    st.session_state[
        "region_spend_meta"
    ] = {
        "since": str(since),
        "until": str(until),
        "refreshed_at": (
            pd.Timestamp.utcnow()
            .isoformat()
        ),
        "accounts_requested": int(total),
        "errors": errors,
    }

    if errors:
        st.warning(
            f"Refresh completed with "
            f"{len(errors)} account error(s). "
            "Successful accounts are still included."
        )

    else:
        st.success(
            f"Refresh completed for "
            f"{total} accessible ad accounts."
        )


# =========================================================
# Report display
# =========================================================
report_df = st.session_state.get(
    "region_spend_report",
    pd.DataFrame(),
)

report_meta = st.session_state.get(
    "region_spend_meta",
    {},
)


if report_df.empty:
    st.info(
        "Choose the time range, then click "
        "Refresh Data to load governorate spend."
    )

else:
    current_range = (
        str(since),
        str(until),
    )

    loaded_range = (
        str(report_meta.get("since", "")),
        str(report_meta.get("until", "")),
    )

    if current_range != loaded_range:
        st.warning(
            "The selected time range changed "
            "after the last refresh. "
            "Click Refresh Data to update the report."
        )

    scope_ids = get_scope_account_ids(
        selected_scope,
        account_master,
        account_business_map,
    )

    filtered = report_df[
        report_df["account_id"]
        .astype(str)
        .isin(scope_ids)
    ].copy()

    if filtered.empty:
        st.info(
            "No governorate spend was returned "
            "for this business/scope in the "
            "refreshed date range."
        )

    else:
        currencies = sorted(
            [
                x
                for x in filtered["currency"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
                if x
            ]
        )

        scope_title = scope_labels.get(
            selected_scope,
            "Selected Scope",
        )

        st.markdown(
            f"## {scope_title}"
        )

        st.caption(
            f"Loaded range: "
            f"{report_meta.get('since')} → "
            f"{report_meta.get('until')}"
        )

        # =================================================
        # TOP TABLE: total spend per governorate
        # =================================================
        st.markdown(
            "### 🗺️ Spend by Governorate"
        )

        if len(currencies) <= 1:
            summary_df = (
                filtered.groupby(
                    "region",
                    dropna=False,
                    as_index=False,
                )["spend"]
                .sum()
                .rename(
                    columns={
                        "region": "Governorate",
                        "spend": "Spend",
                    }
                )
                .sort_values(
                    "Spend",
                    ascending=False,
                )
                .reset_index(drop=True)
            )

            if currencies:
                st.caption(
                    f"Currency: {currencies[0]}"
                )

        else:
            st.warning(
                "Multiple ad-account currencies "
                "are present. Spend is separated "
                "by currency and is NOT converted."
            )

            summary_df = (
                filtered.groupby(
                    [
                        "currency",
                        "region",
                    ],
                    dropna=False,
                    as_index=False,
                )["spend"]
                .sum()
                .rename(
                    columns={
                        "currency": "Currency",
                        "region": "Governorate",
                        "spend": "Spend",
                    }
                )
                .sort_values(
                    [
                        "Currency",
                        "Spend",
                    ],
                    ascending=[
                        True,
                        False,
                    ],
                )
                .reset_index(drop=True)
            )

        st.dataframe(
            format_money_table(
                summary_df
            ),
            use_container_width=True,
            hide_index=True,
            height=520,
        )

        # =================================================
        # BOTTOM TABLE: spend per account + governorate
        # =================================================
        st.markdown(
            "### 📋 Governorate Spend by Ad Account"
        )

        if len(currencies) <= 1:
            detail_df = (
                filtered.groupby(
                    [
                        "account_name",
                        "account_id",
                        "region",
                    ],
                    dropna=False,
                    as_index=False,
                )["spend"]
                .sum()
                .rename(
                    columns={
                        "account_name": "Ad Account Name",
                        "account_id": "Ad Account ID",
                        "region": "Governorate",
                        "spend": "Spend",
                    }
                )
                .sort_values(
                    [
                        "Ad Account Name",
                        "Spend",
                    ],
                    ascending=[
                        True,
                        False,
                    ],
                )
                .reset_index(drop=True)
            )

        else:
            detail_df = (
                filtered.groupby(
                    [
                        "account_name",
                        "account_id",
                        "currency",
                        "region",
                    ],
                    dropna=False,
                    as_index=False,
                )["spend"]
                .sum()
                .rename(
                    columns={
                        "account_name": "Ad Account Name",
                        "account_id": "Ad Account ID",
                        "currency": "Currency",
                        "region": "Governorate",
                        "spend": "Spend",
                    }
                )
                .sort_values(
                    [
                        "Ad Account Name",
                        "Currency",
                        "Spend",
                    ],
                    ascending=[
                        True,
                        True,
                        False,
                    ],
                )
                .reset_index(drop=True)
            )

        st.dataframe(
            format_money_table(
                detail_df
            ),
            use_container_width=True,
            hide_index=True,
            height=650,
        )

        refresh_errors = (
            report_meta.get("errors")
            or []
        )

        if refresh_errors:
            with st.expander(
                f"Account errors "
                f"({len(refresh_errors)})"
            ):
                for error in refresh_errors:
                    st.error(error)


if discovery_errors:
    with st.expander(
        f"Access discovery notes "
        f"({len(discovery_errors)})"
    ):
        for error in discovery_errors:
            st.warning(error)
