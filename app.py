"""
Long-Duration Spotify Tracks (2014–2024) - Interactive Streamlit Dashboard
========================================================================

This Streamlit app loads a cleaned dataset of Spotify tracks and provides an
interactive, portfolio-friendly dashboard:

Core features
- Sidebar filters (duration range, artist, search)
- KPI cards (tracks, artists, median duration, max duration)
- Interactive charts (duration distribution, top artists, duration tiers)
- Longest-tracks table
- Sortable, searchable data table
- Download filtered dataset

Premium features
- INSIGHTS tab: top keywords, theme inference, concentration metrics, outliers
- RECOMMENDATIONS: track-to-track similarity with explainability
- Map-style insight views (no geo needed): treemap, sunburst, heatmap, theme map

Expected input file: data_clean.csv with columns:
- id
- name_clean
- duration_minutes
- artists
- artist_list (stringified list is OK)
- primary_artist
- artist_count

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------
# Page config + global styling
# -----------------------------
st.set_page_config(
    page_title="Long Spotify Tracks Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
[data-testid="stMetricLabel"] { opacity: 0.75; }
div[data-testid="stPlotlyChart"] > div { border-radius: 12px; }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
section[data-testid="stSidebar"] { min-width: 320px; }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Data + utilities
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data(csv_path: str = "data_clean.csv") -> pd.DataFrame:
    """
    Load and lightly validate the cleaned dataset.

    Notes:
    - Handles `artist_list` stored as a Python-like string (e.g. "['A', 'B']").
    - Ensures expected columns exist.
    - Coerces duration to numeric and drops rows with missing critical fields.
    """
    df = pd.read_csv(csv_path)

    expected = {
        "id",
        "name_clean",
        "duration_minutes",
        "artists",
        "primary_artist",
        "artist_count",
    }
    missing_cols = expected - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing columns in {csv_path}: {sorted(missing_cols)}. "
            "Re-export data_clean.csv from your notebook."
        )

    df["duration_minutes"] = pd.to_numeric(df["duration_minutes"], errors="coerce")
    df = df.dropna(subset=["id", "name_clean", "duration_minutes", "primary_artist"])
    df["duration_minutes"] = df["duration_minutes"].astype(float)

    if "artist_list" in df.columns:
        def _parse_list(x):
            if isinstance(x, list):
                return x
            if pd.isna(x):
                return []
            if isinstance(x, str):
                x = x.strip()
                try:
                    v = ast.literal_eval(x)
                    return v if isinstance(v, list) else [str(v)]
                except Exception:
                    return [p.strip() for p in x.split(",") if p.strip()]
            return [str(x)]

        df["artist_list"] = df["artist_list"].apply(_parse_list)

    df["artist_count"] = pd.to_numeric(df["artist_count"], errors="coerce").fillna(1).astype(int)

    df["duration_bucket"] = pd.cut(
        df["duration_minutes"],
        bins=[0, 6, 10, 20, 40, 60, 120, np.inf],
        labels=["≤6", "6–10", "10–20", "20–40", "40–60", "60–120", "120+"],
        right=True,
        include_lowest=True,
    )

    df["track_label"] = df["name_clean"].astype(str).str.slice(0, 55)

    # Tokenize title for insights + recommendations
    df["title_tokens"] = df["name_clean"].astype(str).apply(tokenize_title)

    return df.reset_index(drop=True)


def fmt_minutes(x: float) -> str:
    """Format minutes with no trailing .0 for clean display."""
    if pd.isna(x):
        return ""
    if float(x).is_integer():
        return f"{int(x)}"
    return f"{x:.1f}"


@dataclass
class Filters:
    duration_range: Tuple[float, float]
    artists: list[str]
    search: str
    multi_artist_only: bool


def apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    """Apply user-selected filters to the dataset."""
    lo, hi = f.duration_range
    out = df[(df["duration_minutes"] >= lo) & (df["duration_minutes"] <= hi)].copy()

    if f.artists:
        out = out[out["primary_artist"].isin(f.artists)].copy()

    if f.multi_artist_only:
        out = out[out["artist_count"] > 1].copy()

    q = (f.search or "").strip().lower()
    if q:
        out = out[
            out["name_clean"].str.lower().str.contains(q, na=False)
            | out["artists"].str.lower().str.contains(q, na=False)
            | out["primary_artist"].str.lower().str.contains(q, na=False)
        ].copy()

    return out.reset_index(drop=True)


def kpi_row(df: pd.DataFrame) -> None:
    """Render KPI metrics for the filtered dataset."""
    total_tracks = len(df)
    unique_artists = df["primary_artist"].nunique()

    median_dur = float(df["duration_minutes"].median()) if total_tracks else 0.0
    p90_dur = float(df["duration_minutes"].quantile(0.90)) if total_tracks else 0.0
    max_dur = float(df["duration_minutes"].max()) if total_tracks else 0.0

    c1, c2, c3, c4, c5 = st.columns([1.2, 1.2, 1.2, 1.2, 1.2])
    c1.metric("Tracks", f"{total_tracks:,}")
    c2.metric("Unique primary artists", f"{unique_artists:,}")
    c3.metric("Median duration (min)", fmt_minutes(median_dur))
    c4.metric("90th percentile (min)", fmt_minutes(p90_dur))
    c5.metric("Max duration (min)", fmt_minutes(max_dur))


# -----------------------------
# Text processing (Insights + Recs)
# -----------------------------
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into", "is", "it",
    "of", "on", "or", "the", "to", "with", "without", "vol", "volume", "part", "pt", "mix",
    "session", "sessions", "set", "feat", "ft", "remix", "edit", "version", "sounds", "sound",
    "music", "track"
}


def tokenize_title(title: str) -> list[str]:
    """
    Tokenize a track title into clean tokens for lightweight NLP tasks.
    - Lowercases
    - Removes punctuation
    - Splits on whitespace
    - Filters stopwords and very short tokens
    """
    title = (title or "").lower()
    title = re.sub(r"[^a-z0-9\s]+", " ", title)
    tokens = [t.strip() for t in title.split() if t.strip()]
    tokens = [t for t in tokens if len(t) >= 3 and t not in STOPWORDS]
    return tokens


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# -----------------------------
# Charts (core)
# -----------------------------
def duration_histogram(df: pd.DataFrame) -> None:
    n = len(df)
    if n == 0:
        st.info("No data to plot for current filters.")
        return

    iqr = df["duration_minutes"].quantile(0.75) - df["duration_minutes"].quantile(0.25)
    bin_width = 2 * iqr / (n ** (1 / 3)) if iqr > 0 else 1.0
    bins = int((df["duration_minutes"].max() - df["duration_minutes"].min()) / max(bin_width, 1.0))
    bins = int(np.clip(bins, 8, 40))

    fig = px.histogram(
        df,
        x="duration_minutes",
        nbins=bins,
        title="Duration distribution (minutes)",
        labels={"duration_minutes": "Duration (minutes)"},
        marginal="box",
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)


def top_artists_bar(df: pd.DataFrame, top_n: int = 15) -> None:
    if len(df) == 0:
        st.info("No data to plot for current filters.")
        return

    counts = (
        df["primary_artist"]
        .value_counts()
        .head(top_n)
        .rename_axis("primary_artist")
        .reset_index(name="track_count")
    )

    fig = px.bar(
        counts,
        x="track_count",
        y="primary_artist",
        orientation="h",
        title=f"Top {top_n} primary artists (by track count)",
        labels={"track_count": "Tracks", "primary_artist": "Artist"},
    )
    fig.update_layout(
        height=500,
        margin=dict(l=10, r=10, t=60, b=10),
        yaxis={"categoryorder": "total ascending"},
    )
    st.plotly_chart(fig, use_container_width=True)


def tier_pie(df: pd.DataFrame) -> None:
    if len(df) == 0:
        st.info("No data to plot for current filters.")
        return

    vc = df["duration_bucket"].value_counts(dropna=False).rename_axis("bucket").reset_index(name="count")
    vc["bucket"] = vc["bucket"].astype(str)

    fig = px.pie(
        vc,
        names="bucket",
        values="count",
        title="Track duration tiers",
        hole=0.45,
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)


def longest_tracks(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    cols = ["name_clean", "primary_artist", "duration_minutes", "artists", "id", "artist_count"]
    out = df.sort_values("duration_minutes", ascending=False)[cols].head(n).copy()
    out["duration_minutes"] = out["duration_minutes"].map(fmt_minutes)
    return out


def render_table(df: pd.DataFrame) -> None:
    show = df.copy()
    show = show.rename(columns={"name_clean": "track_name"})
    show["duration_minutes"] = show["duration_minutes"].map(fmt_minutes)

    cols = ["track_name", "primary_artist", "duration_minutes", "artists", "artist_count", "id"]
    show = show[cols]

    st.dataframe(
        show,
        use_container_width=True,
        hide_index=True,
        column_config={
            "track_name": st.column_config.TextColumn("Track", width="large"),
            "primary_artist": st.column_config.TextColumn("Primary artist", width="medium"),
            "duration_minutes": st.column_config.NumberColumn("Duration (min)", width="small"),
            "artists": st.column_config.TextColumn("Artists (raw)", width="large"),
            "artist_count": st.column_config.NumberColumn("Artist count", width="small"),
            "id": st.column_config.TextColumn("Track ID", width="medium"),
        },
    )


# -----------------------------
# “Map-style” views
# -----------------------------
def content_treemap(df: pd.DataFrame) -> None:
    if len(df) == 0:
        st.info("No data to plot for current filters.")
        return

    fig = px.treemap(
        df,
        path=["primary_artist", "track_label"],
        values="duration_minutes",
        color="duration_minutes",
        title="Content landscape (Treemap): size = duration",
        hover_data={"duration_minutes": ":.1f"},
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)


def duration_sunburst(df: pd.DataFrame) -> None:
    if len(df) == 0:
        st.info("No data to plot for current filters.")
        return

    grp = (
        df.groupby(["duration_bucket", "primary_artist"], dropna=False)
        .size()
        .reset_index(name="track_count")
    )
    grp["duration_bucket"] = grp["duration_bucket"].astype(str)

    fig = px.sunburst(
        grp,
        path=["duration_bucket", "primary_artist"],
        values="track_count",
        title="Hierarchy map (Sunburst): duration tier → artist",
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)


def keyword_map(df: pd.DataFrame, theme_dict: dict[str, list[str]]) -> None:
    if len(df) == 0:
        st.info("No data to plot for current filters.")
        return

    titles = df["name_clean"].astype(str).str.lower().fillna("")

    rows = []
    for theme, kws in theme_dict.items():
        mask = np.zeros(len(df), dtype=bool)
        for kw in kws:
            mask |= titles.str.contains(re.escape(kw.lower()), na=False)
        subset = df[mask]
        if len(subset):
            rows.append(
                {
                    "theme": theme,
                    "tracks": int(len(subset)),
                    "avg_duration": float(subset["duration_minutes"].mean()),
                    "median_duration": float(subset["duration_minutes"].median()),
                }
            )

    if not rows:
        st.info("No themed keywords matched your current filtered titles.")
        return

    theme_df = pd.DataFrame(rows).sort_values("tracks", ascending=True)

    fig = px.bar(
        theme_df,
        x="tracks",
        y="theme",
        orientation="h",
        title="Theme map (from titles): frequency and typical duration",
        hover_data={"avg_duration": ":.1f", "median_duration": ":.1f"},
        labels={"tracks": "Tracks", "theme": "Theme"},
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)


def artist_duration_heatmap(df: pd.DataFrame, top_artists: int = 20) -> None:
    if len(df) == 0:
        st.info("No data to plot for current filters.")
        return

    top = df["primary_artist"].value_counts().head(top_artists).index
    tmp = df[df["primary_artist"].isin(top)].copy()
    tmp["duration_bucket"] = tmp["duration_bucket"].astype(str)

    pivot = tmp.pivot_table(
        index="primary_artist",
        columns="duration_bucket",
        values="id",
        aggfunc="count",
        fill_value=0,
    )

    bucket_order = ["≤6", "6–10", "10–20", "20–40", "40–60", "60–120", "120+"]
    existing = [b for b in bucket_order if b in pivot.columns]
    pivot = pivot.reindex(columns=existing)

    melted = pivot.reset_index().melt(
        id_vars="primary_artist",
        var_name="duration_bucket",
        value_name="tracks",
    )

    fig = px.density_heatmap(
        melted,
        x="duration_bucket",
        y="primary_artist",
        z="tracks",
        title=f"Heatmap: top {top_artists} artists × duration tier (track counts)",
        labels={"duration_bucket": "Duration tier", "primary_artist": "Artist", "tracks": "Tracks"},
    )
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# INSIGHTS (polished)
# -----------------------------
@st.cache_data(show_spinner=False)
def compute_insights(df: pd.DataFrame) -> dict:
    """
    Compute dataset-level insights (on the current filtered df).
    Cached to keep the app snappy.
    """
    if len(df) == 0:
        return {"empty": True}

    # Concentration: share of tracks by top artists
    artist_counts = df["primary_artist"].value_counts()
    total = artist_counts.sum()
    top5_share = float(artist_counts.head(5).sum() / total) if total else 0.0
    top10_share = float(artist_counts.head(10).sum() / total) if total else 0.0

    # Outliers
    p90 = float(df["duration_minutes"].quantile(0.90))
    outliers = df[df["duration_minutes"] >= p90].sort_values("duration_minutes", ascending=False).head(15)

    # Keywords (from title_tokens)
    token_counts = {}
    for tokens in df["title_tokens"]:
        for t in tokens:
            token_counts[t] = token_counts.get(t, 0) + 1
    top_keywords = (
        pd.DataFrame(sorted(token_counts.items(), key=lambda x: x[1], reverse=True), columns=["keyword", "count"])
        .head(25)
    )

    # Keywords by longness: compare top tokens in long vs short-ish within filtered
    median = float(df["duration_minutes"].median())
    long_df = df[df["duration_minutes"] >= median]
    short_df = df[df["duration_minutes"] < median]

    def token_freq(sub):
        d = {}
        for tokens in sub["title_tokens"]:
            for t in tokens:
                d[t] = d.get(t, 0) + 1
        return d

    long_tokens = token_freq(long_df)
    short_tokens = token_freq(short_df)

    # "lift": token appears more in long than short (normalized)
    def norm(d):
        s = sum(d.values()) or 1
        return {k: v / s for k, v in d.items()}

    long_norm = norm(long_tokens)
    short_norm = norm(short_tokens)

    lift_rows = []
    for tok, ln in long_norm.items():
        sn = short_norm.get(tok, 0.000001)
        lift = ln / sn if sn > 0 else ln / 0.000001
        # keep only meaningful tokens
        if (long_tokens.get(tok, 0) >= 3) and (lift >= 1.5):
            lift_rows.append((tok, lift, long_tokens.get(tok, 0), short_tokens.get(tok, 0)))

    lift_df = pd.DataFrame(lift_rows, columns=["keyword", "lift_long_vs_short", "count_long", "count_short"])
    lift_df = lift_df.sort_values("lift_long_vs_short", ascending=False).head(20)

    return {
        "empty": False,
        "top5_share": top5_share,
        "top10_share": top10_share,
        "outliers": outliers[["name_clean", "primary_artist", "duration_minutes", "artists", "id"]],
        "top_keywords": top_keywords,
        "lift_df": lift_df,
    }


def insights_tab(df: pd.DataFrame) -> None:
    """Render a polished Insights tab with narrative-friendly outputs."""
    st.subheader("Insights")
    st.caption("Auto-generated findings for the current filtered selection.")

    res = compute_insights(df)
    if res.get("empty"):
        st.info("No data available for the current filters.")
        return

    # Concentration cards
    c1, c2, c3 = st.columns([1, 1, 1])
    c1.metric("Top 5 artists share", f"{res['top5_share']*100:.1f}%")
    c2.metric("Top 10 artists share", f"{res['top10_share']*100:.1f}%")
    c3.metric("Outlier threshold (P90)", fmt_minutes(float(df["duration_minutes"].quantile(0.90))))

    st.divider()

    # Keywords + lift
    k1, k2 = st.columns(2)
    with k1:
        st.markdown("#### Top keywords in titles")
        fig = px.bar(
            res["top_keywords"].sort_values("count", ascending=True),
            x="count",
            y="keyword",
            orientation="h",
            title="Most common title keywords",
        )
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with k2:
        st.markdown("#### Keywords associated with longer tracks")
        if len(res["lift_df"]) == 0:
            st.info("Not enough signal to compute keyword ‘long vs short’ lift for this selection.")
        else:
            fig = px.bar(
                res["lift_df"].sort_values("lift_long_vs_short", ascending=True),
                x="lift_long_vs_short",
                y="keyword",
                orientation="h",
                title="Keyword lift (long vs short, within filters)",
                hover_data={"count_long": True, "count_short": True},
            )
            fig.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.markdown("#### Long-track outliers (top of the distribution)")
    out = res["outliers"].copy()
    out["duration_minutes"] = out["duration_minutes"].map(fmt_minutes)
    st.dataframe(out, use_container_width=True, hide_index=True)


# -----------------------------
# RECOMMENDATIONS (explainable similarity)
# -----------------------------
@st.cache_data(show_spinner=False)
def prepare_reco_frame(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a stable frame for recommendations:
    - keep needed columns
    - precompute token sets
    """
    df = df_all.copy()
    df["token_set"] = df["title_tokens"].apply(lambda t: set(t) if isinstance(t, list) else set())
    return df


def recommend_tracks(
    df_base: pd.DataFrame,
    track_id: str,
    k: int = 10,
    w_duration: float = 0.55,
    w_keywords: float = 0.35,
    w_artist: float = 0.10,
    limit_pool_to_filtered: bool = True,
    df_pool: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Recommend similar tracks with explainability.
    Similarity = weighted combination of:
    - duration similarity (closer duration -> higher)
    - keyword similarity (Jaccard on title tokens)
    - same primary artist bonus

    Parameters
    ----------
    df_base : filtered dataframe used for selection context
    track_id : id of selected track
    k : number of recommendations
    w_duration, w_keywords, w_artist : weights (sum doesn't have to be 1; normalized internally)
    limit_pool_to_filtered : if True, recommend from filtered pool; else from df_pool (full dataset)
    df_pool : full dataset frame (required if limit_pool_to_filtered=False)
    """
    if limit_pool_to_filtered:
        pool = df_base.copy()
    else:
        if df_pool is None:
            raise ValueError("df_pool must be provided when limit_pool_to_filtered=False")
        pool = df_pool.copy()

    if len(pool) == 0:
        return pd.DataFrame()

    if track_id not in set(pool["id"]):
        # If selected track not in pool (e.g., pool is filtered), locate in df_pool instead
        if df_pool is not None and track_id in set(df_pool["id"]):
            target_row = df_pool[df_pool["id"] == track_id].iloc[0]
        else:
            return pd.DataFrame()
    else:
        target_row = pool[pool["id"] == track_id].iloc[0]

    target_dur = float(target_row["duration_minutes"])
    target_artist = str(target_row["primary_artist"])
    target_tokens = set(target_row.get("token_set", set()))

    # Duration similarity scaling: convert abs diff into 0-1 similarity
    # Use a robust scale based on IQR or range to stay stable
    dur_series = pool["duration_minutes"]
    iqr = float(dur_series.quantile(0.75) - dur_series.quantile(0.25))
    scale = max(iqr, 3.0)  # minutes
    # sim_duration = exp(-|diff|/scale) (smoothly decays)
    def sim_duration(d: float) -> float:
        return float(np.exp(-abs(d - target_dur) / scale))

    # Normalize weights
    w_sum = w_duration + w_keywords + w_artist
    w_duration_n, w_keywords_n, w_artist_n = w_duration / w_sum, w_keywords / w_sum, w_artist / w_sum

    rows = []
    for _, r in pool.iterrows():
        if str(r["id"]) == str(track_id):
            continue

        d = float(r["duration_minutes"])
        tokens = set(r.get("token_set", set()))
        kw_sim = jaccard(target_tokens, tokens)
        dur_sim = sim_duration(d)
        same_artist = 1.0 if str(r["primary_artist"]) == target_artist else 0.0

        score = w_duration_n * dur_sim + w_keywords_n * kw_sim + w_artist_n * same_artist

        shared = sorted(list(target_tokens & tokens))[:8]
        rows.append(
            {
                "id": r["id"],
                "track": r["name_clean"],
                "artist": r["primary_artist"],
                "duration_minutes": r["duration_minutes"],
                "score": score,
                "why_duration": abs(d - target_dur),
                "why_shared_keywords": ", ".join(shared) if shared else "",
                "why_same_artist": "Yes" if same_artist > 0 else "No",
                "kw_sim": kw_sim,
                "dur_sim": dur_sim,
            }
        )

    recs = pd.DataFrame(rows).sort_values("score", ascending=False).head(k).copy()
    if len(recs) == 0:
        return recs

    recs["duration_minutes"] = recs["duration_minutes"].map(fmt_minutes)
    recs["why_duration"] = recs["why_duration"].map(lambda x: f"{x:.1f} min")
    recs["score"] = recs["score"].map(lambda x: f"{x:.3f}")
    recs["kw_sim"] = recs["kw_sim"].map(lambda x: f"{x:.3f}")
    recs["dur_sim"] = recs["dur_sim"].map(lambda x: f"{x:.3f}")

    return recs[
        ["track", "artist", "duration_minutes", "score", "why_duration", "why_shared_keywords", "why_same_artist", "kw_sim", "dur_sim", "id"]
    ]


def recommendations_tab(df_filtered: pd.DataFrame, df_full: pd.DataFrame) -> None:
    st.subheader("Recommendations")
    st.caption("Pick a track and get similar tracks with explainable reasons (duration + keywords + artist affinity).")

    if len(df_filtered) == 0:
        st.info("No data available for current filters.")
        return

    # Selector shows track name + artist + duration for UX
    df_pick = df_filtered.copy()
    df_pick["label"] = (
        df_pick["name_clean"].astype(str)
        + " - "
        + df_pick["primary_artist"].astype(str)
        + " ("
        + df_pick["duration_minutes"].map(fmt_minutes)
        + " min)"
    )

    selected_label = st.selectbox("Select a track", options=df_pick["label"].tolist(), index=0)
    selected_id = df_pick.loc[df_pick["label"] == selected_label, "id"].iloc[0]

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    k = c1.slider("Results", 5, 30, 10, 1)
    pool_choice = c2.selectbox("Recommend from", ["Filtered selection", "Entire dataset"], index=0)

    w_duration = c3.slider("Weight: duration", 0.0, 1.0, 0.55, 0.05)
    w_keywords = c4.slider("Weight: keywords", 0.0, 1.0, 0.35, 0.05)
    w_artist = st.slider("Weight: same-artist bonus", 0.0, 1.0, 0.10, 0.05)

    # Explainability legend
    with st.expander("How similarity works"):
        st.markdown(
            """
- **Duration similarity**: closer duration → higher score (smooth exponential decay).
- **Keyword similarity**: Jaccard overlap of cleaned title tokens.
- **Same-artist bonus**: small boost if the primary artist matches.

The final score is a weighted combination of the above.
"""
        )

    limit_pool = pool_choice == "Filtered selection"

    df_full_reco = prepare_reco_frame(df_full)
    df_filtered_reco = prepare_reco_frame(df_filtered)

    recs = recommend_tracks(
        df_base=df_filtered_reco,
        track_id=selected_id,
        k=k,
        w_duration=w_duration,
        w_keywords=w_keywords,
        w_artist=w_artist,
        limit_pool_to_filtered=limit_pool,
        df_pool=df_full_reco,
    )

    if len(recs) == 0:
        st.info("No recommendations found (try expanding filters or selecting the entire dataset pool).")
        return

    st.markdown("#### Recommended tracks")
    st.dataframe(recs, use_container_width=True, hide_index=True)


# -----------------------------
# Sidebar (filters + controls)
# -----------------------------
df_all_raw = load_data("data_clean.csv")  # includes title_tokens
df_all = df_all_raw.copy()

st.title(" Long-Duration Spotify Tracks (2014–2024)")
st.caption(
    "Explore long-form Spotify tracks by duration and artist. "
    "Filter dynamically, inspect patterns, and export your selection."
)

with st.sidebar:
    st.header("Filters")

    min_d = float(df_all["duration_minutes"].min())
    max_d = float(df_all["duration_minutes"].max())

    duration_range = st.slider(
        "Duration range (minutes)",
        min_value=float(math.floor(min_d)),
        max_value=float(math.ceil(max_d)),
        value=(float(math.floor(min_d)), float(math.ceil(max_d))),
        step=1.0,
    )

    artist_options = sorted(df_all["primary_artist"].unique().tolist())
    artists = st.multiselect(
        "Primary artist",
        options=artist_options,
        default=[],
        help="Select one or more artists. Leave empty to include all.",
    )

    multi_artist_only = st.toggle(
        "Only multi-artist tracks",
        value=False,
        help="Tracks where the artist field contains multiple artists (artist_count > 1).",
    )

    search = st.text_input(
        "Search (track or artist)",
        value="",
        placeholder="e.g., rain, meditation, tibetan",
    )

    st.divider()

    st.subheader("Charts")
    top_n = st.slider("Top artists (bar)", 5, 30, 15, 1)
    heat_top = st.slider("Heatmap: top artists", 10, 50, 20, 5)

    st.divider()

    st.subheader("Table Options")
    sort_mode = st.selectbox(
        "Sort table by",
        ["Duration (desc)", "Duration (asc)", "Artist (A→Z)", "Track (A→Z)"],
        index=0,
    )

    st.divider()

    st.subheader("Theme map (keywords)")
    st.caption("Edit these keywords to match your dataset content.")
    theme_defaults = {
        "Sleep": ["sleep", "insomnia", "dream"],
        "Meditation": ["meditation", "mindfulness", "zen"],
        "Study / Focus": ["study", "focus", "concentration", "deep work"],
        "Nature": ["rain", "forest", "waves", "ocean", "wind", "thunder", "river"],
        "Healing": ["healing", "chakra", "reiki", "therapy"],
        "Singing bowls": ["singing bowl", "tibetan"],
        "Binaural / Frequency": ["binaural", "frequency", "hz"],
        "Ambient": ["ambient", "drone", "soundscape"],
        "Yoga": ["yoga", "pranayama"],
    }

    theme_blob = st.text_area(
        "Themes (one per line: Theme: kw1, kw2, kw3)",
        value="\n".join([f"{k}: {', '.join(v)}" for k, v in theme_defaults.items()]),
        height=220,
    )

    st.caption("Tip: Use filters + search to craft a story for your portfolio.")


def parse_theme_blob(blob: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for line in blob.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        theme, kws = line.split(":", 1)
        theme = theme.strip()
        kw_list = [k.strip() for k in kws.split(",") if k.strip()]
        if theme and kw_list:
            out[theme] = kw_list
    return out


themes = parse_theme_blob(theme_blob)

filters = Filters(
    duration_range=duration_range,
    artists=artists,
    search=search,
    multi_artist_only=multi_artist_only,
)

df_f = apply_filters(df_all, filters)

# KPIs
kpi_row(df_f)

# Tabs (now includes Insights + Recommendations)
tab_map, tab_trends, tab_insights, tab_recs, tab_data = st.tabs(
    ["Map Views", "Trends", "Insights", "Recommendations", "Data Explorer"]
)

with tab_map:
    st.subheader("Map views (pattern & hierarchy)")
    m1, m2 = st.columns(2)
    with m1:
        content_treemap(df_f)
    with m2:
        duration_sunburst(df_f)

    st.divider()
    m3, m4 = st.columns([1.05, 0.95])
    with m3:
        keyword_map(df_f, themes)
    with m4:
        artist_duration_heatmap(df_f, top_artists=heat_top)

with tab_trends:
    left, right = st.columns([1.35, 1])
    with left:
        duration_histogram(df_f)
    with right:
        tier_pie(df_f)

    c1, c2 = st.columns([1.2, 1])
    with c1:
        top_artists_bar(df_f, top_n=top_n)
    with c2:
        st.subheader("Longest tracks (current filters)")
        st.caption("A quick view of the extreme end of the dataset.")
        st.dataframe(
            longest_tracks(df_f, n=20),
            use_container_width=True,
            hide_index=True,
        )

with tab_insights:
    insights_tab(df_f)

with tab_recs:
    recommendations_tab(df_filtered=df_f, df_full=df_all)

with tab_data:
    df_table = df_f.copy()
    if sort_mode == "Duration (desc)":
        df_table = df_table.sort_values("duration_minutes", ascending=False)
    elif sort_mode == "Duration (asc)":
        df_table = df_table.sort_values("duration_minutes", ascending=True)
    elif sort_mode == "Artist (A→Z)":
        df_table = df_table.sort_values(["primary_artist", "duration_minutes"], ascending=[True, False])
    else:
        df_table = df_table.sort_values(["name_clean", "duration_minutes"], ascending=[True, False])

    st.subheader("Browse tracks")
    render_table(df_table.reset_index(drop=True))

    csv_bytes = df_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered dataset (CSV)",
        data=csv_bytes,
        file_name="spotify_long_tracks_filtered.csv",
        mime="text/csv",
    )

st.caption("Built with Streamlit + Plotly. Dataset via Spotify public API (ODC-By license).")
