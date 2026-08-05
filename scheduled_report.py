"""Hourly Meta Ads Today report sender for GitHub Actions.

This script is intentionally independent from Streamlit. It:
1) fetches today's Meta Ads data in Africa/Cairo timezone,
2) builds the same WhatsApp summary used by the dashboard,
3) sends it to one or many recipients.

Required environment variables:
- META_ACCESS_TOKEN
- WHATSAPP_ACCESS_TOKEN
- WHATSAPP_PHONE_NUMBER_ID
- WHATSAPP_REPORT_TO (comma/semicolon/newline-separated international numbers)

Optional environment variables:
- META_API_VERSION (default: v25.0)
- BUSINESS_IDS (default: the two dashboard business IDs)
- REPORT_TIMEZONE (default: Africa/Cairo)
- MAX_WORKERS (default: 4)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import requests

BASE_URL = "https://graph.facebook.com"
DEFAULT_BUSINESS_IDS = ["751488620224306", "1178859133269743"]

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


@dataclass(frozen=True)
class Config:
    meta_access_token: str
    whatsapp_access_token: str
    whatsapp_phone_number_id: str
    recipients: list[str]
    api_version: str
    business_ids: list[str]
    timezone: str
    max_workers: int


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def split_values(raw: str) -> list[str]:
    return [x.strip() for x in re.split(r"[,;\n]+", raw or "") if x.strip()]


def normalize_phone(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def load_config() -> Config:
    recipients: list[str] = []
    seen: set[str] = set()
    for value in split_values(require_env("WHATSAPP_REPORT_TO")):
        phone = normalize_phone(value)
        if phone and phone not in seen:
            seen.add(phone)
            recipients.append(phone)

    if not recipients:
        raise RuntimeError("WHATSAPP_REPORT_TO does not contain any valid numbers")

    business_ids = split_values(os.getenv("BUSINESS_IDS", ",".join(DEFAULT_BUSINESS_IDS)))
    if not business_ids:
        business_ids = DEFAULT_BUSINESS_IDS.copy()

    try:
        max_workers = max(1, min(10, int(os.getenv("MAX_WORKERS", "4"))))
    except ValueError:
        max_workers = 4

    return Config(
        meta_access_token=require_env("META_ACCESS_TOKEN"),
        whatsapp_access_token=require_env("WHATSAPP_ACCESS_TOKEN"),
        whatsapp_phone_number_id=require_env("WHATSAPP_PHONE_NUMBER_ID"),
        recipients=recipients,
        api_version=os.getenv("META_API_VERSION", "v25.0").strip() or "v25.0",
        business_ids=business_ids,
        timezone=os.getenv("REPORT_TIMEZONE", "Africa/Cairo").strip() or "Africa/Cairo",
        max_workers=max_workers,
    )


def mask_phone(phone: str) -> str:
    return f"{'*' * max(0, len(phone) - 4)}{phone[-4:]}"


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_div(a: float, b: float) -> float | None:
    return a / b if b else None


def normalize_text(value: Any) -> str:
    return str(value or "").strip().upper()


def extract_buyer_code(account_name: str) -> str:
    text = normalize_text(account_name)
    for code in MEDIA_BUYER_MAP:
        if re.search(rf"(?<![A-Z0-9]){re.escape(code)}(?![A-Z0-9])", text):
            return code
    return "UNKNOWN"


def is_relevant_account_name(account_name: str) -> bool:
    text = normalize_text(account_name)
    return any(marker in text for marker in ("OK-FB-HR-", "OK-FB-NF-", "US-FB-HR-", "US-BO-HR-"))


def source_is_okaby(source: str) -> bool:
    return "751488620224306" in str(source)


def classify_objective(campaign_name: str) -> str:
    text = normalize_text(campaign_name)
    if re.search(r"(?<![A-Z0-9])CONVL?(?![A-Z0-9])", text) or re.search(r"(?<![A-Z0-9])CONVS(?![A-Z0-9])", text):
        return "Conversion"
    if re.search(r"(?<![A-Z0-9])WA(?![A-Z0-9])", text):
        return "Whatsapp Message"
    if re.search(r"(?<![A-Z0-9])LG(?![A-Z0-9])", text):
        return "Lead generation"
    if re.search(r"(?<![A-Z0-9])LM(?![A-Z0-9])", text):
        return "Lead - Message"
    return "Unknown"


def flatten_actions(actions: Any) -> dict[str, float]:
    result: dict[str, float] = {}
    if not isinstance(actions, list):
        return result
    for item in actions:
        action_type = str(item.get("action_type", "")).strip().lower()
        if action_type:
            result[action_type] = result.get(action_type, 0.0) + to_float(item.get("value"))
    return result


def result_for_objective(objective: str, actions_map: dict[str, float]) -> float:
    if objective in {"Lead generation", "Lead - Message"}:
        return sum(actions_map.get(k, 0.0) for k in (
            "offsite_conversion.fb_pixel_lead",
            "onsite_conversion.lead_grouped",
            "offsite_conversion.custom",
        ))
    if objective == "Conversion":
        return actions_map.get("purchase", 0.0)
    if objective == "Whatsapp Message":
        return sum(actions_map.get(k, 0.0) for k in (
            "onsite_conversion.messaging_conversation_started",
            "onsite_conversion.messaging_conversation_started_7d",
        ))
    return 0.0


def request_json(method: str, url: str, *, params: dict[str, Any] | None = None,
                 headers: dict[str, str] | None = None, payload: dict[str, Any] | None = None,
                 attempts: int = 3) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.request(
                method,
                url,
                params=params,
                headers=headers,
                json=payload,
                timeout=90,
            )
            if response.ok:
                return response.json()
            try:
                detail = json.dumps(response.json(), ensure_ascii=False)
            except Exception:
                detail = response.text
            raise RuntimeError(f"HTTP {response.status_code}: {detail[:2000]}")
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(str(last_error))


def fetch_all_pages(url: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    next_url: str | None = url
    next_params: dict[str, Any] | None = params
    while next_url:
        data = request_json("GET", next_url, params=next_params)
        rows.extend(data.get("data", []))
        next_url = data.get("paging", {}).get("next")
        next_params = None
    return rows


def get_ad_accounts(config: Config) -> list[dict[str, Any]]:
    sources: list[tuple[str, str]] = [
        ("me/adaccounts", f"{BASE_URL}/{config.api_version}/me/adaccounts")
    ]
    for business_id in config.business_ids:
        sources.extend([
            (f"business/{business_id}/owned_ad_accounts", f"{BASE_URL}/{config.api_version}/{business_id}/owned_ad_accounts"),
            (f"business/{business_id}/client_ad_accounts", f"{BASE_URL}/{config.api_version}/{business_id}/client_ad_accounts"),
        ])

    collected: list[dict[str, Any]] = []
    for source_name, url in sources:
        try:
            rows = fetch_all_pages(url, {
                "fields": "id,account_id,name,account_status",
                "access_token": config.meta_access_token,
                "limit": 500,
            })
            for row in rows:
                row = dict(row)
                row["source"] = source_name
                collected.append(row)
        except Exception as exc:
            print(f"WARNING account source failed: {source_name}: {exc}", file=sys.stderr)

    filtered = [
        row for row in collected
        if is_relevant_account_name(str(row.get("name", ""))) or source_is_okaby(str(row.get("source", "")))
    ]

    dedup: dict[str, dict[str, Any]] = {}
    for row in filtered:
        account_id = str(row.get("id") or "")
        if account_id and account_id not in dedup:
            dedup[account_id] = row
    return list(dedup.values())


def get_account_data(config: Config, account: dict[str, Any], day: str) -> dict[str, Any]:
    account_id = str(account.get("id", "")).replace("act_", "")
    account_name = str(account.get("name") or account_id)

    insights_url = f"{BASE_URL}/{config.api_version}/act_{account_id}/insights"
    common = {
        "time_range": json.dumps({"since": day, "until": day}),
        "access_token": config.meta_access_token,
        "limit": 1000,
    }

    insights = fetch_all_pages(insights_url, {
        **common,
        "fields": ",".join([
            "account_id", "account_name", "campaign_id", "campaign_name", "spend", "actions"
        ]),
        "level": "campaign",
    })

    genders = fetch_all_pages(insights_url, {
        **common,
        "fields": "spend",
        "breakdowns": "gender",
    })

    return {
        "account_id": f"act_{account_id}",
        "account_name": account_name,
        "insights": insights,
        "genders": genders,
    }


def aggregate(accounts_data: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    agent_totals: dict[str, dict[str, float]] = {}
    male_totals: dict[tuple[str, str, str], float] = {}

    for account_data in accounts_data:
        account_id = account_data["account_id"]
        account_name = account_data["account_name"]
        buyer_code = extract_buyer_code(account_name)
        agent = MEDIA_BUYER_MAP.get(buyer_code, "Unknown")

        for row in account_data.get("insights", []):
            spend = to_float(row.get("spend"))
            objective = classify_objective(str(row.get("campaign_name", "")))
            results = result_for_objective(objective, flatten_actions(row.get("actions")))
            totals = agent_totals.setdefault(agent, {"spend": 0.0, "results": 0.0})
            totals["spend"] += spend
            totals["results"] += results

        for row in account_data.get("genders", []):
            if str(row.get("gender", "")).strip().lower() != "male":
                continue
            spend = to_float(row.get("spend"))
            if spend <= 0:
                continue
            key = (account_id, account_name, agent)
            male_totals[key] = male_totals.get(key, 0.0) + spend

    agents: list[dict[str, Any]] = []
    for agent, totals in agent_totals.items():
        agents.append({
            "agent": agent,
            "spend": totals["spend"],
            "results": totals["results"],
            "cpl": safe_div(totals["spend"], totals["results"]),
        })

    male_alerts = [
        {"account_id": k[0], "account_name": k[1], "agent": k[2], "spend": spend}
        for k, spend in male_totals.items()
    ]
    return agents, male_alerts


def money(value: Any) -> str:
    return f"{to_float(value):,.2f} EGP"


def leads(value: Any) -> str:
    return f"{to_float(value):,.0f}"


def build_report(day: str, agents: list[dict[str, Any]], male_alerts: list[dict[str, Any]]) -> str:
    total_spend = sum(a["spend"] for a in agents)
    total_results = sum(a["results"] for a in agents)
    total_cpl = safe_div(total_spend, total_results)

    ranked = [
        a for a in agents
        if a["agent"] != "Unknown" and a["spend"] > 0 and a["results"] > 0 and a["cpl"] is not None
    ]
    highest = max(ranked, key=lambda x: x["cpl"]) if ranked else None
    lowest = min(ranked, key=lambda x: x["cpl"]) if ranked else None

    lines = [
        "📊 Meta Ads Refresh Report",
        "Range: Today",
        f"Dates: {day} → {day}",
        "",
        f"Overall Spend: {money(total_spend)}",
        f"Overall Leads: {leads(total_results)}",
        f"Overall CPL: {money(total_cpl) if total_cpl is not None else 'N/A'}",
        "",
    ]

    if highest:
        lines += [
            f"🔴 Highest CPL ({highest['agent']})",
            f"Spend: {money(highest['spend'])}",
            f"Leads: {leads(highest['results'])}",
            f"CPL: {money(highest['cpl'])}",
        ]
    else:
        lines.append("🔴 Highest CPL: No agent with valid leads and CPL")

    lines.append("")
    if lowest:
        lines += [
            f"🟢 Lowest CPL ({lowest['agent']})",
            f"Spend: {money(lowest['spend'])}",
            f"Leads: {leads(lowest['results'])}",
            f"CPL: {money(lowest['cpl'])}",
        ]
    else:
        lines.append("🟢 Lowest CPL: No agent with valid leads and CPL")

    lines += ["", "👥 Agents Details"]
    visible_agents = sorted(
        [a for a in agents if a["agent"] != "Unknown"],
        key=lambda x: (-x["spend"], x["agent"]),
    )
    if not visible_agents:
        lines.append("No agent data found for this range")
    else:
        for index, agent in enumerate(visible_agents, 1):
            lines += [
                "",
                f"Agent #{index}",
                f"Agent name: {agent['agent']}",
                f"Spend: {money(agent['spend'])}",
                f"Leads: {leads(agent['results'])}",
                f"CPL: {money(agent['cpl']) if agent['cpl'] is not None else 'N/A'}",
            ]

    lines.append("")
    male_alerts = sorted(male_alerts, key=lambda x: x["spend"], reverse=True)
    if not male_alerts:
        lines.append("✅ Gender Alert: No Male Spend detected")
    else:
        lines.append(f"⚠️ Gender Alert: Male Spend detected in {len(male_alerts)} ad account(s)")
        for index, row in enumerate(male_alerts, 1):
            lines += [
                "",
                f"Gender Alert #{index}",
                f"Male Spend: {money(row['spend'])}",
                f"Agent name: {row['agent']}",
                f"Ad account name: {row['account_name']}",
                f"Ad account ID: {row['account_id']}",
            ]

    return "\n".join(lines).strip()


def split_message(message: str, max_chars: int = 3500) -> list[str]:
    if len(message) <= max_chars:
        return [message]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in message.splitlines():
        added = len(line) + 1
        if current and size + added > max_chars:
            chunks.append("\n".join(current).strip())
            current = []
            size = 0
        current.append(line)
        size += added
    if current:
        chunks.append("\n".join(current).strip())
    return chunks


def send_whatsapp(config: Config, recipient: str, body: str) -> None:
    url = f"{BASE_URL}/{config.api_version}/{config.whatsapp_phone_number_id}/messages"
    request_json(
        "POST",
        url,
        headers={
            "Authorization": f"Bearer {config.whatsapp_access_token}",
            "Content-Type": "application/json",
        },
        payload={
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": recipient,
            "type": "text",
            "text": {"preview_url": False, "body": body},
        },
    )


def main() -> int:
    config = load_config()
    now = datetime.now(ZoneInfo(config.timezone))
    day = now.date().isoformat()
    print(f"Starting Today report for {day} ({config.timezone})")
    print(f"API version: {config.api_version}")

    accounts = get_ad_accounts(config)
    if not accounts:
        raise RuntimeError("No matching ad accounts were found")
    print(f"Found {len(accounts)} relevant ad accounts")

    account_results: list[dict[str, Any]] = []
    fetch_errors: list[str] = []
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        future_map = {executor.submit(get_account_data, config, account, day): account for account in accounts}
        for future in as_completed(future_map):
            account = future_map[future]
            try:
                result = future.result()
                account_results.append(result)
                print(f"Fetched: {result['account_name']}")
            except Exception as exc:
                name = str(account.get("name") or account.get("id") or "Unknown")
                fetch_errors.append(f"{name}: {exc}")
                print(f"ERROR fetch {name}: {exc}", file=sys.stderr)

    if not account_results:
        raise RuntimeError("All account fetches failed")

    agents, male_alerts = aggregate(account_results)
    report = build_report(day, agents, male_alerts)
    chunks = split_message(report)

    print("\n--- REPORT PREVIEW ---\n")
    print(report)
    print("\n--- END REPORT ---\n")

    success = 0
    failures: list[str] = []
    for recipient in config.recipients:
        try:
            for chunk in chunks:
                send_whatsapp(config, recipient, chunk)
            success += 1
            print(f"Sent successfully to {mask_phone(recipient)}")
        except Exception as exc:
            safe_error = str(exc).replace(config.whatsapp_access_token, "[REDACTED]")
            failures.append(f"{mask_phone(recipient)}: {safe_error}")
            print(f"ERROR sending to {mask_phone(recipient)}: {safe_error}", file=sys.stderr)

    print(f"Completed. Sent to {success}/{len(config.recipients)} recipients. Fetch errors: {len(fetch_errors)}")
    if fetch_errors:
        print("Fetch warnings:", file=sys.stderr)
        for error in fetch_errors:
            print(f"- {error}", file=sys.stderr)
    if failures:
        print("Send failures:", file=sys.stderr)
        for error in failures:
            print(f"- {error}", file=sys.stderr)

    return 0 if success > 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(1)
