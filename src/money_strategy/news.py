from __future__ import annotations

import json
import re
import hashlib
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests


GOV_PUSHINFO_URL = "https://www.gov.cn/pushinfo/v150203/pushinfo.jsonp"
SENTIMENT_COLUMNS = [
    "date",
    "news_score",
    "policy_score",
    "theme_score_159819",
    "theme_score_512480",
    "theme_score_159770",
    "theme_score_516160",
    "theme_score_561560",
    "theme_score_563530",
    "note",
    "source_url",
]


@dataclass(frozen=True)
class NewsCandidate:
    date: str
    news_score: int
    policy_score: int
    theme_score_159819: int
    theme_score_512480: int
    theme_score_159770: int
    theme_score_516160: int
    theme_score_561560: int
    theme_score_563530: int
    note: str
    source_url: str
    source: str
    title: str


def refresh_hotspot_candidates(
    *,
    days: int = 30,
    min_policy_score: int = 58,
    timeout: int = 20,
    include_akshare: bool = True,
    include_policy: bool = True,
) -> pd.DataFrame:
    cutoff = date.today() - timedelta(days=days)
    rows = []
    if include_akshare:
        rows.extend(asdict(candidate) for candidate in fetch_akshare_candidates(cutoff=cutoff, min_policy_score=min_policy_score))

    if include_policy:
        for item in fetch_gov_pushinfo(timeout=timeout):
            pub_date = _parse_date(item.get("pubDate", ""))
            if pub_date is None or pub_date < cutoff:
                continue
            candidate = score_news_item(
                pub_date.isoformat(),
                str(item.get("title") or ""),
                str(item.get("description") or ""),
                str(item.get("link") or ""),
                source=str(item.get("author") or "中国政府网"),
            )
            if candidate and candidate.policy_score >= min_policy_score:
                rows.append(asdict(candidate))

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=[*SENTIMENT_COLUMNS, "source", "title"])
    return frame.drop_duplicates(subset=["source_url"]).sort_values(["date", "policy_score"], ascending=[False, False])


def refresh_news_candidates(**kwargs) -> pd.DataFrame:
    return refresh_hotspot_candidates(**kwargs)


def fetch_akshare_candidates(*, cutoff: date, min_policy_score: int = 58, ak=None) -> list[NewsCandidate]:
    candidates: list[NewsCandidate] = []
    if ak is None:
        try:
            import akshare as ak
        except ImportError:
            return candidates

    candidates.extend(_akshare_cls_candidates(ak, cutoff=cutoff, min_policy_score=min_policy_score))
    candidates.extend(_akshare_hot_keyword_candidates(ak, min_policy_score=min_policy_score))
    candidates.extend(_akshare_hot_rank_candidates(ak, min_policy_score=min_policy_score))
    return candidates


def _akshare_cls_candidates(ak, *, cutoff: date, min_policy_score: int) -> list[NewsCandidate]:
    rows: list[NewsCandidate] = []
    try:
        frame = ak.stock_info_global_cls()
    except Exception:
        return rows
    if frame is None or frame.empty:
        return rows

    for _, row in frame.head(200).iterrows():
        pub_date = _parse_date(str(row.get("发布日期", "")))
        if pub_date is None or pub_date < cutoff:
            continue
        title = str(row.get("标题") or "")
        content = str(row.get("内容") or "")
        publish_time = str(row.get("发布时间") or "")
        source_url = f"akshare://stock_info_global_cls/{pub_date.isoformat()}/{publish_time}/{_stable_id(title + content)}"
        candidate = score_news_item(
            pub_date.isoformat(),
            title,
            content,
            source_url,
            source="AkShare-财联社",
        )
        if candidate and candidate.policy_score >= min_policy_score and _is_a_share_relevant(title + content, candidate):
            rows.append(candidate)
    return rows


def _akshare_hot_keyword_candidates(ak, *, min_policy_score: int) -> list[NewsCandidate]:
    try:
        frame = ak.stock_hot_keyword_em()
    except Exception:
        return []
    if frame is None or frame.empty or "概念名称" not in frame.columns:
        return []

    date_text = _latest_date_text(frame.get("时间"))
    concepts = frame["概念名称"].dropna().astype(str).tolist()
    counter = Counter(concepts)
    rows: list[NewsCandidate] = []
    for concept, count in counter.most_common(20):
        heat = pd.to_numeric(frame.loc[frame["概念名称"] == concept, "热度"], errors="coerce").fillna(0).sum()
        description = f"东方财富热度概念 {concept} 出现 {count} 次，合计热度 {int(heat)}"
        candidate = score_news_item(
            date_text,
            f"市场热点：{concept}",
            description,
            f"akshare://stock_hot_keyword_em/{date_text}/{concept}",
            source="AkShare-东方财富热词",
        )
        if candidate and candidate.policy_score >= min_policy_score:
            rows.append(candidate)
    return rows


def _akshare_hot_rank_candidates(ak, *, min_policy_score: int) -> list[NewsCandidate]:
    try:
        frame = ak.stock_hot_rank_em()
    except Exception:
        return []
    if frame is None or frame.empty or "股票名称" not in frame.columns:
        return []

    date_text = date.today().isoformat()
    text = "、".join(frame.head(30)["股票名称"].dropna().astype(str).tolist())
    candidate = score_news_item(
        date_text,
        "东方财富个股人气榜 TOP30",
        text,
        f"akshare://stock_hot_rank_em/{date_text}",
        source="AkShare-东方财富人气榜",
    )
    return [candidate] if candidate and candidate.policy_score >= min_policy_score else []


def fetch_gov_pushinfo(*, timeout: int = 20) -> list[dict[str, str]]:
    response = requests.get(GOV_PUSHINFO_URL, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    text = response.text.strip()
    match = re.search(r"^[^(]+\((.*)\)\s*;?$", text, flags=re.S)
    if not match:
        raise ValueError("unexpected gov pushinfo response")
    return json.loads(match.group(1))


def score_news_item(date_text: str, title: str, description: str, url: str, *, source: str) -> NewsCandidate | None:
    text = f"{title} {description}"
    if _has_any(text, ["风险提示", "高于行业平均", "澄清公告", "异常波动公告", "监管函", "问询函"]):
        return None

    policy_score = 50
    news_score = 50
    themes = {
        "theme_score_159819": 50,
        "theme_score_512480": 50,
        "theme_score_159770": 50,
        "theme_score_516160": 50,
        "theme_score_561560": 50,
        "theme_score_563530": 50,
    }

    broad_policy_hits = _count_hits(
        text,
        [
            "资本市场",
            "股票市场",
            "中长期资金",
            "回购",
            "增持",
            "降准",
            "降息",
            "货币政策",
            "金融支持",
            "流动性",
            "新质生产力",
            "战略性新兴产业",
            "未来产业",
            "扩大内需",
        ],
    )
    if broad_policy_hits:
        policy_score += min(28, 7 * broad_policy_hits)
        news_score += min(20, 5 * broad_policy_hits)

    if _has_any(text, ["人工智能", "大模型", "算力", "数字经济"]):
        themes["theme_score_159819"] = 75
    if _has_any(text, ["集成电路", "半导体", "芯片"]):
        themes["theme_score_512480"] = 75
    if _has_any(text, ["机器人", "智能终端", "智能制造装备"]):
        themes["theme_score_159770"] = 75
    if _has_any(text, ["新能源汽车", "新能源", "储能", "光伏", "风电"]):
        themes["theme_score_516160"] = 72
    if _has_any(text, ["电力", "能源保供", "电网", "电力市场"]):
        themes["theme_score_561560"] = 68
    if _has_any(text, ["商业航天", "卫星", "北斗", "低空经济"]):
        themes["theme_score_563530"] = 70

    theme_hit = any(value > 50 for value in themes.values())
    if theme_hit:
        policy_score = max(policy_score, 62)
        news_score = max(news_score, 58)

    policy_score = min(policy_score, 85)
    news_score = min(news_score, 78)
    if policy_score < 58 and not theme_hit:
        return None

    note = _compact_note(title or description)
    return NewsCandidate(
        date=date_text,
        news_score=news_score,
        policy_score=policy_score,
        note=note,
        source_url=url,
        source=source,
        title=title,
        **themes,
    )


def write_news_candidates(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def append_candidates_to_sentiment(candidates: pd.DataFrame, sentiment_path: Path) -> pd.DataFrame:
    sentiment_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_events = candidates[[column for column in SENTIMENT_COLUMNS if column in candidates.columns]].copy()
    if candidate_events.empty:
        return _read_or_empty_sentiment(sentiment_path)

    existing = _read_or_empty_sentiment(sentiment_path)
    combined = pd.concat([existing, candidate_events], ignore_index=True)
    combined = combined.drop_duplicates(subset=["source_url"], keep="first")
    combined["date"] = pd.to_datetime(combined["date"]).dt.date.astype(str)
    combined = combined.reindex(columns=SENTIMENT_COLUMNS).sort_values(["date", "source_url"])
    combined.to_csv(sentiment_path, index=False)
    return combined


def _read_or_empty_sentiment(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=SENTIMENT_COLUMNS)
    frame = pd.read_csv(path)
    for column in SENTIMENT_COLUMNS:
        if column not in frame.columns:
            frame[column] = 50 if column.endswith("_score") or column in {"news_score", "policy_score"} else ""
    return frame[SENTIMENT_COLUMNS]


def _parse_date(value: str) -> date | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _latest_date_text(values) -> str:
    if values is None:
        return date.today().isoformat()
    parsed = pd.to_datetime(values, errors="coerce")
    parsed = parsed.dropna() if hasattr(parsed, "dropna") else parsed
    if len(parsed) == 0:
        return date.today().isoformat()
    return parsed.max().date().isoformat()


def _has_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _count_hits(text: str, keywords: list[str]) -> int:
    return sum(1 for keyword in keywords if keyword in text)


def _is_a_share_relevant(text: str, candidate: NewsCandidate) -> bool:
    if any(
        getattr(candidate, column) > 50
        for column in (
            "theme_score_159819",
            "theme_score_512480",
            "theme_score_159770",
            "theme_score_516160",
            "theme_score_561560",
            "theme_score_563530",
        )
    ):
        return True
    return _has_any(
        text,
        [
            "A股",
            "沪深",
            "创业板",
            "科创板",
            "资本市场",
            "股票市场",
            "证监会",
            "中国人民银行",
            "国务院",
            "发改委",
            "中长期资金",
        ],
    )


def _stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _compact_note(text: str, limit: int = 80) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]
