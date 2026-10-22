import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="ACCT 3343 Quantitative Stock Screener",
    page_icon="📊",
    layout="wide",
)

# -----------------------------
# Constants
# -----------------------------
DATA_FILE = "universe.csv"
PITCH_TICKERS = {"SFM"}  # 60-second pitch company must not be recommended

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_data(path):
    d = pd.read_csv(path)
    # Normalize column names
    d.columns = [c.strip() for c in d.columns]

    numeric_cols = [
        "market_cap", "GPA", "BM", "STA", "SNOA", "Altman_Z", "F_SCORE",
        "f_roa", "f_cfo", "f_droa", "f_accrual", "f_dlever",
        "f_dliquid", "f_eq_offer", "f_dmargin", "f_dturn",
        "ROA", "CFOA", "delta_ROA", "leverage", "delta_leverage",
        "current_ratio", "delta_liquidity", "gross_margin",
        "delta_margin", "asset_turnover", "delta_turnover",
    ]
    for c in numeric_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    if "ticker" in d.columns:
        d["ticker"] = d["ticker"].astype(str).str.upper().str.strip()

    # Valuation quality control:
    # Negative B/M is not meaningful for a long-only value ranking.
    # Extremely large absolute B/M values in this dataset are data-basis
    # distortions, so leave them missing instead of allowing them to dominate.
    if "BM" in d.columns:
        d["BM_raw"] = d["BM"]
        d.loc[(d["BM"] <= 0) | (d["BM"].abs() > 20), "BM"] = np.nan

    if "company_name" not in d.columns and "name" in d.columns:
        d["company_name"] = d["name"]

    if "sector" not in d.columns:
        d["sector"] = "Unknown"

    # Remove the pitch company from recommendations/universe as required.
    if "ticker" in d.columns:
        d = d[~d["ticker"].isin(PITCH_TICKERS)].copy()

    return d


def pct_rank_high_good(series):
    """Percentile score where 100 = strongest/highest."""
    return series.rank(pct=True, method="average") * 100


def pct_rank_high_bad(series):
    """Percentile score where 100 = riskiest/highest."""
    return series.rank(pct=True, method="average") * 100


def fmt_pct(x):
    if pd.isna(x):
        return "—"
    return f"{x:.1%}"


def fmt_num(x, decimals=2):
    if pd.isna(x):
        return "—"
    return f"{x:.{decimals}f}"


def compact_money(x):
    if pd.isna(x):
        return "—"
    x = float(x)
    if abs(x) >= 1e12:
        return f"${x/1e12:.2f}T"
    if abs(x) >= 1e9:
        return f"${x/1e9:.2f}B"
    if abs(x) >= 1e6:
        return f"${x/1e6:.1f}M"
    return f"${x:,.0f}"


def add_relative_metrics(d):
    out = d.copy()
    for metric in ["STA", "SNOA", "Altman_Z", "GPA", "BM", "F_SCORE"]:
        if metric in out.columns:
            means = out.groupby("sector")[metric].transform("mean")
            out[f"{metric}_sector_mean"] = means
            out[f"{metric}_demeaned"] = out[metric] - means
    return out


def build_sell_screen(
    d,
    sta_threshold_pct,
    snoa_threshold_pct,
    altman_cutoff,
    min_red_flags,
    use_sector_relative_accruals,
):
    out = d.copy()

    sta_col = "STA_demeaned" if use_sector_relative_accruals else "STA"
    snoa_col = "SNOA_demeaned" if use_sector_relative_accruals else "SNOA"

    # Higher STA/SNOA = greater accrual concern.
    out["STA_risk_pct"] = pct_rank_high_bad(out[sta_col])
    out["SNOA_risk_pct"] = pct_rank_high_bad(out[snoa_col])

    out["flag_STA"] = np.where(
        out[sta_col].notna(),
        out["STA_risk_pct"] >= sta_threshold_pct,
        np.nan,
    )
    out["flag_SNOA"] = np.where(
        out[snoa_col].notna(),
        out["SNOA_risk_pct"] >= snoa_threshold_pct,
        np.nan,
    )
    out["flag_Altman"] = np.where(
        out["Altman_Z"].notna(),
        out["Altman_Z"] < altman_cutoff,
        np.nan,
    )

    flags = out[["flag_STA", "flag_SNOA", "flag_Altman"]]
    out["sell_red_flags"] = flags.fillna(False).sum(axis=1).astype(int)
    out["sell_metrics_available"] = flags.notna().sum(axis=1).astype(int)

    # Require enough observed metrics to evaluate the requested red-flag count.
    out["SELL_candidate"] = (
        (out["sell_red_flags"] >= min_red_flags)
        & (out["sell_metrics_available"] >= min_red_flags)
    )

    return out


def build_buy_screen(
    d,
    min_fscore,
    top_buy_pct,
    w_gpa,
    w_bm,
    w_fscore,
    use_sector_relative_buy,
):
    out = d.copy()

    gpa_col = "GPA_demeaned" if use_sector_relative_buy else "GPA"
    bm_col = "BM_demeaned" if use_sector_relative_buy else "BM"
    fs_col = "F_SCORE_demeaned" if use_sector_relative_buy else "F_SCORE"

    out["GPA_strength_pct"] = pct_rank_high_good(out[gpa_col])
    out["BM_strength_pct"] = pct_rank_high_good(out[bm_col])
    out["F_SCORE_strength_pct"] = pct_rank_high_good(out[fs_col])

    total_weight = w_gpa + w_bm + w_fscore
    if total_weight == 0:
        total_weight = 1.0

    out["BUY_score"] = (
        w_gpa * out["GPA_strength_pct"]
        + w_bm * out["BM_strength_pct"]
        + w_fscore * out["F_SCORE_strength_pct"]
    ) / total_weight

    # Require all three BUY-side metrics to avoid turning missing values into a false signal.
    complete = out[["GPA", "BM", "F_SCORE"]].notna().all(axis=1)
    eligible = complete & (out["F_SCORE"] >= min_fscore)

    cutoff = np.nan
    if eligible.any():
        cutoff = out.loc[eligible, "BUY_score"].quantile(1 - top_buy_pct / 100)

    out["BUY_candidate"] = eligible & (out["BUY_score"] >= cutoff)
    out["BUY_rank"] = np.nan
    if out["BUY_candidate"].any():
        ranks = out.loc[out["BUY_candidate"], "BUY_score"].rank(
            ascending=False, method="min"
        )
        out.loc[ranks.index, "BUY_rank"] = ranks

    return out, cutoff


def buy_case(row):
    parts = []
    if pd.notna(row.get("GPA_strength_pct")):
        parts.append(f"GPA is in the {row['GPA_strength_pct']:.0f}th percentile")
    if pd.notna(row.get("BM_strength_pct")):
        parts.append(f"B/M is in the {row['BM_strength_pct']:.0f}th percentile")
    if pd.notna(row.get("F_SCORE")):
        parts.append(f"F_SCORE is {int(row['F_SCORE'])}/9")
    if pd.notna(row.get("BUY_score")):
        parts.append(f"combined BUY score is {row['BUY_score']:.1f}/100")

    if not parts:
        return "Insufficient data."

    interpretation = []
    gpa_pct = row.get("GPA_strength_pct")
    bm_pct = row.get("BM_strength_pct")
    fs = row.get("F_SCORE")

    if pd.notna(gpa_pct):
        if gpa_pct >= 75:
            interpretation.append("strong profitability")
        elif gpa_pct >= 50:
            interpretation.append("above-median profitability")
        else:
            interpretation.append("weaker profitability")

    if pd.notna(bm_pct):
        if bm_pct >= 75:
            interpretation.append("an attractive relative valuation")
        elif bm_pct >= 50:
            interpretation.append("a reasonable relative valuation")
        else:
            interpretation.append("a less attractive relative valuation")

    if pd.notna(fs):
        if fs >= 8:
            interpretation.append("very strong financial strength")
        elif fs >= 7:
            interpretation.append("solid financial strength")
        else:
            interpretation.append("more mixed financial strength")

    if interpretation:
        return "; ".join(parts) + ". This combination suggests " + ", ".join(interpretation) + " under the current BUY rules."
    return "; ".join(parts) + "."


def sell_case(row):
    reasons = []
    if row.get("flag_STA") is True or row.get("flag_STA") == 1:
        reasons.append(f"high STA risk ({row.get('STA_risk_pct', np.nan):.0f}th percentile)")
    if row.get("flag_SNOA") is True or row.get("flag_SNOA") == 1:
        reasons.append(f"high SNOA risk ({row.get('SNOA_risk_pct', np.nan):.0f}th percentile)")
    if row.get("flag_Altman") is True or row.get("flag_Altman") == 1:
        reasons.append(f"Altman Z of {row.get('Altman_Z', np.nan):.2f} is below the selected cutoff")

    if not reasons:
        return "No selected SELL red flags."

    count = int(row.get("sell_red_flags", len(reasons)))
    if count >= 3:
        conclusion = "Together, the accrual and distress signals indicate a broad deterioration-risk profile under the current SELL rules."
    elif count == 2:
        conclusion = "The combination of two independent warning signals makes the company a stronger risk candidate than a single-metric flag."
    else:
        conclusion = "This is a single warning signal and should be interpreted more cautiously."

    return "Triggers " + ", ".join(reasons) + ". " + conclusion


# -----------------------------
# Load
# -----------------------------
try:
    df = load_data(DATA_FILE)
except FileNotFoundError:
    st.error(
        "universe.csv was not found. Run build_universe.ipynb in Google Colab, "
        "download universe.csv, and place it beside app.py in your GitHub repository."
    )
    st.stop()

required = {"ticker", "company_name", "sector", "market_cap", "GPA", "BM", "STA", "SNOA", "Altman_Z", "F_SCORE"}
missing_required = sorted(required.difference(df.columns))
if missing_required:
    st.error("The CSV is missing required columns: " + ", ".join(missing_required))
    st.stop()

# -----------------------------
# Sidebar: universe
# -----------------------------
st.sidebar.title("Screen controls")
st.sidebar.caption("Every control below recomputes the analysis immediately.")

all_sectors = sorted([x for x in df["sector"].dropna().unique().tolist() if str(x).strip()])
selected_sectors = st.sidebar.multiselect(
    "Sectors to include",
    options=all_sectors,
    default=all_sectors,
)

mcap_floor_m = st.sidebar.number_input(
    "Minimum market cap ($ millions)",
    min_value=0,
    max_value=100000,
    value=300,
    step=100,
)

st.sidebar.divider()
st.sidebar.subheader("SELL screen")

use_sector_relative_accruals = st.sidebar.radio(
    "Accrual ranking basis",
    ["Raw", "Sector-relative"],
    horizontal=True,
) == "Sector-relative"

sta_threshold_pct = st.sidebar.slider(
    "Flag STA at or above percentile",
    min_value=50,
    max_value=99,
    value=90,
    step=1,
)

snoa_threshold_pct = st.sidebar.slider(
    "Flag SNOA at or above percentile",
    min_value=50,
    max_value=99,
    value=90,
    step=1,
)

altman_cutoff = st.sidebar.slider(
    "Flag Altman Z below",
    min_value=-1.0,
    max_value=5.0,
    value=1.81,
    step=0.05,
)

min_red_flags = st.sidebar.slider(
    "Minimum SELL red flags",
    min_value=1,
    max_value=3,
    value=2,
    step=1,
)

st.sidebar.divider()
st.sidebar.subheader("BUY screen")

use_sector_relative_buy = st.sidebar.radio(
    "BUY ranking basis",
    ["Raw", "Sector-relative"],
    horizontal=True,
) == "Sector-relative"

min_fscore = st.sidebar.slider(
    "Minimum F_SCORE",
    min_value=0,
    max_value=9,
    value=7,
    step=1,
)

top_buy_pct = st.sidebar.slider(
    "Top BUY score percentile to show (%)",
    min_value=1,
    max_value=50,
    value=10,
    step=1,
)

st.sidebar.caption("BUY score weights")
w_gpa = st.sidebar.slider("GPA weight", 0, 5, 2, 1)
w_bm = st.sidebar.slider("Book-to-Market weight", 0, 5, 2, 1)
w_fscore = st.sidebar.slider("F_SCORE weight", 0, 5, 3, 1)

exclude_sell_from_buy = st.sidebar.checkbox(
    "Exclude SELL-flagged firms from BUY list",
    value=True,
)

# -----------------------------
# Filter universe
# -----------------------------
screen = df.copy()
screen = screen[screen["sector"].isin(selected_sectors)]
screen = screen[screen["market_cap"].fillna(-np.inf) >= mcap_floor_m * 1_000_000].copy()
screen = add_relative_metrics(screen)

sell_df = build_sell_screen(
    screen,
    sta_threshold_pct=sta_threshold_pct,
    snoa_threshold_pct=snoa_threshold_pct,
    altman_cutoff=altman_cutoff,
    min_red_flags=min_red_flags,
    use_sector_relative_accruals=use_sector_relative_accruals,
)

analysis_df, buy_cutoff = build_buy_screen(
    sell_df,
    min_fscore=min_fscore,
    top_buy_pct=top_buy_pct,
    w_gpa=w_gpa,
    w_bm=w_bm,
    w_fscore=w_fscore,
    use_sector_relative_buy=use_sector_relative_buy,
)

if exclude_sell_from_buy:
    analysis_df["BUY_candidate"] = analysis_df["BUY_candidate"] & ~analysis_df["SELL_candidate"]
    analysis_df["BUY_rank"] = np.nan
    if analysis_df["BUY_candidate"].any():
        ranks = analysis_df.loc[analysis_df["BUY_candidate"], "BUY_score"].rank(
            ascending=False, method="min"
        )
        analysis_df.loc[ranks.index, "BUY_rank"] = ranks

buy = analysis_df[analysis_df["BUY_candidate"]].sort_values(["BUY_rank", "ticker"]).copy()
sell = analysis_df[analysis_df["SELL_candidate"]].sort_values(
    ["sell_red_flags", "STA_risk_pct", "SNOA_risk_pct"],
    ascending=[False, False, False],
).copy()

# -----------------------------
# Header
# -----------------------------
st.title("ACCT 3343 · Quantitative BUY / SELL Screener")
st.info(
    "**Course-based analysis.** SELL candidates are identified from accounting/accrual "
    "and distress warning signals (STA, SNOA, Altman Z). BUY candidates use a different "
    "analysis: profitability, value, and financial strength (GPA, Book-to-Market, F_SCORE)."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Companies screened", f"{len(analysis_df):,}")
c2.metric("SELL candidates", f"{len(sell):,}")
c3.metric("BUY candidates", f"{len(buy):,}")
complete_buy = analysis_df[["GPA", "BM", "F_SCORE"]].notna().all(axis=1).sum()
c4.metric("Complete BUY data", f"{complete_buy:,}")

if "data_generated_at" in analysis_df.columns and analysis_df["data_generated_at"].notna().any():
    st.caption(f"Dataset generated: {analysis_df['data_generated_at'].dropna().iloc[0]}")

tabs = st.tabs(
    ["Overview", "SELL screen", "BUY screen", "Company drill-down", "Methodology & data quality"]
)

# -----------------------------
# Overview
# -----------------------------
with tabs[0]:
    st.subheader("Screen overview")

    o1, o2 = st.columns(2)
    with o1:
        st.markdown("#### Top BUY candidates")
        if buy.empty:
            st.warning("No BUY candidates under the selected settings.")
        else:
            show_buy = buy.head(10).copy()
            cols = ["BUY_rank", "ticker", "company_name", "sector", "GPA", "BM", "F_SCORE", "BUY_score"]
            st.dataframe(
                show_buy[cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "GPA": st.column_config.NumberColumn(format="%.3f"),
                    "BM": st.column_config.NumberColumn(format="%.3f"),
                    "BUY_score": st.column_config.NumberColumn(format="%.1f"),
                    "F_SCORE": st.column_config.NumberColumn(format="%d"),
                },
            )

    with o2:
        st.markdown("#### Strongest SELL warnings")
        if sell.empty:
            st.success("No SELL candidates under the selected settings.")
        else:
            show_sell = sell.head(10).copy()
            cols = ["ticker", "company_name", "sector", "sell_red_flags", "STA", "SNOA", "Altman_Z"]
            st.dataframe(
                show_sell[cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "STA": st.column_config.NumberColumn(format="%.3f"),
                    "SNOA": st.column_config.NumberColumn(format="%.3f"),
                    "Altman_Z": st.column_config.NumberColumn(format="%.2f"),
                },
            )

    st.markdown("#### Interactive map: value vs. profitability")
    plot_df = analysis_df.dropna(subset=["GPA", "BM"]).copy()
    if not plot_df.empty:
        plot_df["Status"] = np.select(
            [
                plot_df["BUY_candidate"] & plot_df["SELL_candidate"],
                plot_df["BUY_candidate"],
                plot_df["SELL_candidate"],
            ],
            ["Overlap", "BUY", "SELL"],
            default="Other",
        )
        fig = px.scatter(
            plot_df,
            x="BM",
            y="GPA",
            size="market_cap",
            color="Status",
            hover_name="ticker",
            hover_data=["company_name", "sector", "F_SCORE", "Altman_Z"],
            labels={"BM": "Book-to-Market", "GPA": "Gross Profitability"},
        )
        # Visual zoom only: firms outside this displayed range remain in all calculations.
        fig.update_xaxes(range=[0, 2])
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Chart is visually zoomed to Book-to-Market values from 0 to 2 for readability. "
            "Companies outside this displayed range remain included in the screening calculations."
        )

    st.markdown("#### Quantitative investment cases")
    bc1, bc2 = st.columns(2)
    with bc1:
        st.markdown("**BUY cases**")
        if buy.empty:
            st.write("No candidates under current settings.")
        else:
            for _, row in buy.head(3).iterrows():
                st.markdown(f"**{row['ticker']} — {row['company_name']}**  \n{buy_case(row)}")

    with bc2:
        st.markdown("**SELL cases**")
        if sell.empty:
            st.write("No candidates under current settings.")
        else:
            for _, row in sell.head(3).iterrows():
                st.markdown(f"**{row['ticker']} — {row['company_name']}**  \n{sell_case(row)}")

# -----------------------------
# SELL
# -----------------------------
with tabs[1]:
    st.header("SELL screen · Accrual quality + financial distress")
    st.write(
        "The SELL analysis uses three warning signals. High STA and high SNOA indicate "
        "greater accrual concern; Altman Z below the selected cutoff indicates financial-distress concern. "
        "A firm becomes a SELL candidate when it reaches the selected minimum number of red flags."
    )

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Universe entering", f"{len(analysis_df):,}")
    s2.metric("STA flags", f"{int(analysis_df['flag_STA'].fillna(False).sum()):,}")
    s3.metric("SNOA flags", f"{int(analysis_df['flag_SNOA'].fillna(False).sum()):,}")
    s4.metric("Altman flags", f"{int(analysis_df['flag_Altman'].fillna(False).sum()):,}")

    sell_display = analysis_df.sort_values(
        ["SELL_candidate", "sell_red_flags", "STA_risk_pct"],
        ascending=[False, False, False]
    ).copy()

    sell_display["Market Cap"] = sell_display["market_cap"].apply(compact_money)
    sell_display["STA"] = sell_display["STA"] * 100
    sell_display["SNOA"] = sell_display["SNOA"] * 100

    sell_display = sell_display[
        [
            "ticker", "company_name", "sector", "Market Cap",
            "STA", "STA_risk_pct", "SNOA", "SNOA_risk_pct",
            "Altman_Z", "flag_STA", "flag_SNOA", "flag_Altman",
            "sell_red_flags", "SELL_candidate"
        ]
    ]

    st.dataframe(
        sell_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ticker": "Ticker",
            "company_name": "Company",
            "sector": "Sector",
            "STA": st.column_config.NumberColumn("STA (%)", format="%.1f"),
            "STA_risk_pct": st.column_config.NumberColumn("STA risk percentile", format="%.1f"),
            "SNOA": st.column_config.NumberColumn("SNOA (%)", format="%.1f"),
            "SNOA_risk_pct": st.column_config.NumberColumn("SNOA risk percentile", format="%.1f"),
            "Altman_Z": st.column_config.NumberColumn("Altman Z", format="%.2f"),
            "flag_STA": "STA flag",
            "flag_SNOA": "SNOA flag",
            "flag_Altman": "Altman flag",
            "sell_red_flags": "Red flags",
            "SELL_candidate": "SELL candidate",
        },
    )

    if not sell.empty:
        st.markdown("#### Why the top SELL names are flagged")
        for _, row in sell.head(5).iterrows():
            st.markdown(f"**{row['ticker']} — {row['company_name']}:** {sell_case(row)}")

# -----------------------------
# BUY
# -----------------------------
with tabs[2]:
    st.header("BUY screen · Profitability + value + financial strength")
    st.write(
        "The BUY analysis is deliberately different from the SELL analysis. "
        "It combines GPA, Book-to-Market, and Piotroski F_SCORE. "
        "Each measure is converted to a percentile score, then combined using the sidebar weights."
    )

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Eligible F_SCORE", f"{int((analysis_df['F_SCORE'] >= min_fscore).fillna(False).sum()):,}")
    b2.metric("BUY score cutoff", "—" if pd.isna(buy_cutoff) else f"{buy_cutoff:.1f}")
    b3.metric("BUY candidates", f"{len(buy):,}")
    b4.metric("Top F_SCORE", "—" if analysis_df["F_SCORE"].dropna().empty else f"{analysis_df['F_SCORE'].max():.0f}/9")

    buy_display = analysis_df.sort_values(
        ["BUY_candidate", "BUY_score"],
        ascending=[False, False]
    ).copy()

    buy_display["Market Cap"] = buy_display["market_cap"].apply(compact_money)
    buy_display["GPA"] = buy_display["GPA"] * 100

    buy_display = buy_display[
        [
            "BUY_rank", "ticker", "company_name", "sector", "Market Cap",
            "GPA", "GPA_strength_pct", "BM", "BM_strength_pct",
            "F_SCORE", "F_SCORE_strength_pct", "BUY_score", "BUY_candidate"
        ]
    ]

    st.dataframe(
        buy_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "BUY_rank": "BUY rank",
            "ticker": "Ticker",
            "company_name": "Company",
            "sector": "Sector",
            "GPA": st.column_config.NumberColumn("GPA (%)", format="%.1f"),
            "GPA_strength_pct": st.column_config.NumberColumn("GPA percentile", format="%.1f"),
            "BM": st.column_config.NumberColumn("B/M", format="%.3f"),
            "BM_strength_pct": st.column_config.NumberColumn("B/M percentile", format="%.1f"),
            "F_SCORE": st.column_config.NumberColumn("F_SCORE", format="%.0f"),
            "F_SCORE_strength_pct": st.column_config.NumberColumn("F_SCORE percentile", format="%.1f"),
            "BUY_score": st.column_config.NumberColumn("BUY score", format="%.1f"),
            "BUY_candidate": "BUY candidate",
        },
    )

    if not buy.empty:
        st.markdown("#### Why the top BUY names rank highly")
        for _, row in buy.head(5).iterrows():
            st.markdown(f"**{row['ticker']} — {row['company_name']}:** {buy_case(row)}")

# -----------------------------
# Drill-down
# -----------------------------
with tabs[3]:
    st.header("Company drill-down")
    if analysis_df.empty:
        st.warning("No companies remain under the universe filters.")
    else:
        choices = analysis_df.sort_values("ticker")["ticker"].tolist()
        ticker = st.selectbox("Select a company", choices)
        row = analysis_df.loc[analysis_df["ticker"] == ticker].iloc[0]

        st.subheader(f"{row['ticker']} · {row['company_name']}")
        st.caption(f"{row['sector']} · Market cap {compact_money(row['market_cap'])}")

        d1, d2, d3, d4, d5, d6 = st.columns(6)
        d1.metric("GPA", fmt_pct(row["GPA"]))
        d2.metric("B/M", fmt_num(row["BM"], 3))
        d3.metric("F_SCORE", "—" if pd.isna(row["F_SCORE"]) else f"{int(row['F_SCORE'])}/9")
        d4.metric("STA", fmt_pct(row["STA"]))
        d5.metric("SNOA", fmt_pct(row["SNOA"]))
        d6.metric("Altman Z", fmt_num(row["Altman_Z"], 2))

        status = []
        if row["BUY_candidate"]:
            status.append("BUY candidate")
        if row["SELL_candidate"]:
            status.append("SELL candidate")
        st.write("**Current screen status:** " + (", ".join(status) if status else "Not selected as BUY or SELL under current thresholds."))

        st.markdown("#### F_SCORE components")
        component_cols = [
            ("f_roa", "Positive ROA"),
            ("f_cfo", "Positive CFO"),
            ("f_droa", "Improving ROA"),
            ("f_accrual", "CFO > ROA"),
            ("f_dlever", "Lower leverage"),
            ("f_dliquid", "Higher liquidity"),
            ("f_eq_offer", "No equity issuance"),
            ("f_dmargin", "Higher gross margin"),
            ("f_dturn", "Higher asset turnover"),
        ]
        rows = []
        for col, label in component_cols:
            val = row[col] if col in analysis_df.columns else np.nan
            rows.append({"F_SCORE test": label, "Point": val})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("#### Quantitative interpretation")
        st.write("**BUY side:** " + buy_case(row))
        st.write("**SELL side:** " + sell_case(row))

# -----------------------------
# Methodology & data quality
# -----------------------------
with tabs[4]:
    st.header("Methodology & data quality")

    st.markdown(
        """
### Universe
- U.S.-listed companies from Yahoo Finance's screener.
- Financial Services are excluded in the universe-builder because accrual and Altman measures are not well suited to banks.
- **SFM is excluded** because it was used for the 60-second pitch and cannot be one of this app's recommendations.
- The app reads a prepared CSV; it does not call Yahoo Finance on every Streamlit rerun.

### SELL methodology
- **STA** = (Net Income − Cash Flow from Operations) / Total Assets.
- **SNOA** = (Operating Assets − Operating Liabilities) / Total Assets.
- High STA and SNOA are treated as accrual warning signals.
- **Altman Z** uses the course formula; values below the user-selected cutoff are flagged.
- A SELL candidate needs the selected number of red flags. BUY metrics are not used to create SELL flags.

### BUY methodology
- **GPA** = Gross Profit / Total Assets.
- **Book-to-Market** = Book Equity / Market Equity. Negative B/M and extreme absolute B/M values above 20 are excluded from the BUY ranking as valuation data-quality cases rather than treated as cheap stocks.
- **F_SCORE** is the 9-point Piotroski-style financial-strength score taught in class:
  positive ROA, positive CFO, improving ROA, CFO > ROA, falling leverage,
  improving liquidity, no equity issuance, improving gross margin, and improving asset turnover.
- GPA, B/M, and F_SCORE are percentile-ranked; the user controls their weights.
- BUY candidates must meet the minimum F_SCORE and fall in the selected top percentage of the combined BUY score.

### Missing-data rule
Missing values remain missing. The app does **not** use `fillna(0)` to manufacture financial observations.
A company must have all three BUY metrics to enter the BUY ranking. SELL flags are only evaluated where the relevant metric exists.

### F_SCORE equity issuance proxy
Yahoo Finance coverage of explicit equity issuance is inconsistent. The universe builder first looks for a cash-flow statement equity-issuance line.
If that is unavailable, it uses the change in ordinary shares outstanding as a transparent proxy. The source used is stored in `eq_offer_source`.
        """
    )

    st.markdown("#### Coverage for current filters")
    st.caption("Counts below reflect only the companies currently included by the sidebar universe filters.")
    metrics = ["GPA", "BM", "STA", "SNOA", "Altman_Z", "F_SCORE"]
    miss = pd.DataFrame(
        {
            "Metric": metrics,
            "Missing": [int(analysis_df[m].isna().sum()) for m in metrics],
            "Available": [int(analysis_df[m].notna().sum()) for m in metrics],
        }
    )
    st.dataframe(miss, use_container_width=True, hide_index=True)

    if "eq_offer_source" in analysis_df.columns:
        st.markdown("#### Equity-issuance data source")
        source_counts = (
            analysis_df["eq_offer_source"]
            .fillna("missing")
            .value_counts()
            .rename_axis("Source")
            .reset_index(name="Companies")
        )
        st.dataframe(source_counts, use_container_width=True, hide_index=True)

    st.caption(
        "This is a course screening tool, not personalized investment advice. "
        "Recommendations change when the user changes the selected quantitative rules."
    )
