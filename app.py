import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from scipy.stats import t as t_dist   # for Grubbs' test
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pdfkit

# Page configuration: set title and wide layout
st.set_page_config(
    page_title="Weyland-Yutani Mining Dashboard",
    layout="wide",
)

# Default Google Sheets CSV URL
CSV_URL_DEFAULT = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vRx7FuaguRcCHCwQOJyPv1oDCHM7u7oq5yDmI-iV0IoPOa2uroqOG8qZtD3ZvlB1CpqsOMw9Ri9mkS5/"
    "pub?gid=809861880&single=true&output=csv"
)

# Helper functions


def load_data(csv_url: str) -> pd.DataFrame:
    """Load CSV from Google Sheets, parse dates, drop empty columns."""
    df = pd.read_csv(csv_url)

    # Drop unnamed empty columns
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date").reset_index(drop=True)

    return df


def compute_stats(df: pd.DataFrame, mine_cols: list[str]) -> pd.DataFrame:
    """
    For each mine and for Total:
    - Mean
    - Std
    - Median
    - IQR
    """
    rows = []

    # Per-mine statistics
    for col in mine_cols:
        s = df[col].dropna()
        rows.append(
            {
                "Mine": col,
                "Mean": s.mean(),
                "Std": s.std(ddof=1),
                "Median": s.median(),
                "IQR": s.quantile(0.75) - s.quantile(0.25),
            }
        )

    # Total output statistics
    total_series = df[mine_cols].sum(axis=1)
    rows.append(
        {
            "Mine": "Total",
            "Mean": total_series.mean(),
            "Std": total_series.std(ddof=1),
            "Median": total_series.median(),
            "IQR": total_series.quantile(0.75) - total_series.quantile(0.25),
        }
    )

    return pd.DataFrame(rows)


def detect_iqr_anomalies(
    df: pd.DataFrame, mine_cols: list[str], k: float = 1.5
) -> pd.DataFrame:
    """IQR-rule anomalies for each mine."""
    all_rows = []

    for col in mine_cols:
        s = df[col].dropna()

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - k * iqr
        upper = q3 + k * iqr

        mask = (df[col] < lower) | (df[col] > upper)

        if mask.any():
            outliers = df.loc[mask, ["Date", "Day_idx", "Weekday"]].copy()
            outliers["Mine"] = col
            outliers["Output"] = df.loc[mask, col].values
            outliers["Lower_bound"] = lower
            outliers["Upper_bound"] = upper
            all_rows.append(outliers)

    if all_rows:
        return pd.concat(all_rows, ignore_index=True)

    return pd.DataFrame(
        columns=[
            "Date",
            "Day_idx",
            "Weekday",
            "Mine",
            "Output",
            "Lower_bound",
            "Upper_bound",
        ]
    )


def detect_zscore_anomalies(
    df: pd.DataFrame, mine_cols: list[str], z_thresh: float = 3.0
) -> pd.DataFrame:
    """Z-score anomalies for each mine."""
    all_rows = []

    for col in mine_cols:
        s = df[col].dropna()

        mean = s.mean()
        std = s.std(ddof=1)
        if std == 0 or pd.isna(std):
            continue

        z = (df[col] - mean) / std
        mask = z.abs() > z_thresh

        if mask.any():
            outliers = df.loc[mask, ["Date", "Day_idx", "Weekday"]].copy()
            outliers["Mine"] = col
            outliers["Output"] = df.loc[mask, col].values
            outliers["Z"] = z[mask].values
            all_rows.append(outliers)

    if all_rows:
        return pd.concat(all_rows, ignore_index=True)

    return pd.DataFrame(columns=["Date", "Day_idx", "Weekday", "Mine", "Output", "Z"])


def detect_ma_percent_anomalies(
    df: pd.DataFrame,
    mine_cols: list[str],
    window: int = 7,
    pct_thresh: float = 30.0,
) -> pd.DataFrame:
    """Moving-average percentage deviation anomalies."""
    all_rows = []

    for col in mine_cols:
        s = df[col]

        ma = s.rolling(window=window, min_periods=window).mean()
        valid = ma.notna() & (ma != 0)
        deviation_pct = (s - ma).abs() / ma * 100

        mask = valid & (deviation_pct > pct_thresh)

        if mask.any():
            outliers = df.loc[mask, ["Date", "Day_idx", "Weekday"]].copy()
            outliers["Mine"] = col
            outliers["Output"] = s[mask].values
            outliers["MA"] = ma[mask].values
            outliers["Deviation_pct"] = deviation_pct[mask].values
            all_rows.append(outliers)

    if all_rows:
        return pd.concat(all_rows, ignore_index=True)

    return pd.DataFrame(
        columns=[
            "Date",
            "Day_idx",
            "Weekday",
            "Mine",
            "Output",
            "MA",
            "Deviation_pct",
        ]
    )


def detect_grubbs_anomalies(
    df: pd.DataFrame,
    mine_cols: list[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Grubbs' test anomalies for each mine."""
    all_rows = []

    for col in mine_cols:
        s = df[col].dropna()
        N = len(s)
        if N < 3:
            continue

        mean = s.mean()
        std = s.std(ddof=1)
        if std == 0 or pd.isna(std):
            continue

        G = (s - mean).abs() / std

        t_crit = t_dist.ppf(1 - alpha / (2 * N), N - 2)
        G_crit = ((N - 1) / np.sqrt(N)) * np.sqrt(
            t_crit**2 / (N - 2 + t_crit**2)
        )

        mask = G > G_crit

        if mask.any():
            outliers = df.loc[s.index[mask], ["Date", "Day_idx", "Weekday"]].copy()
            outliers["Mine"] = col
            outliers["Output"] = s[mask].values
            outliers["G"] = G[mask].values
            outliers["G_crit"] = G_crit
            all_rows.append(outliers)

    if all_rows:
        return pd.concat(all_rows, ignore_index=True)

    return pd.DataFrame(
        columns=["Date", "Day_idx", "Weekday", "Mine", "Output", "G", "G_crit"]
    )


def compute_trendlines(df: pd.DataFrame, mine_cols: list[str], degree: int):
    """
    Polynomial trendlines for each mine:
    x = Day_idx, y = output.
    Returns DataFrame with Date, Mine, Trend.
    """
    if degree <= 0:
        return None

    rows = []
    x_all = df["Day_idx"].values.astype(float)

    for col in mine_cols:
        y_all = df[col].values.astype(float)

        mask = ~np.isnan(x_all) & ~np.isnan(y_all)
        x = x_all[mask]
        y = y_all[mask]

        if len(x) <= degree:
            continue

        coeffs = np.polyfit(x, y, degree)
        y_pred = np.polyval(coeffs, x)

        tmp = pd.DataFrame(
            {
                "Date": df.loc[mask, "Date"].values,
                "Mine": col,
                "Trend": y_pred,
            }
        )
        rows.append(tmp)

    if not rows:
        return None

    return pd.concat(rows, ignore_index=True)


def build_anomaly_summary(
    df: pd.DataFrame,
    stats_df: pd.DataFrame,
    anomalies_iqr: pd.DataFrame,
    anomalies_z: pd.DataFrame,
    anomalies_ma: pd.DataFrame,
    anomalies_grubbs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Unified anomaly table:

    Columns:
      - Date
      - Mine
      - Value
      - Spike_or_drop (vs mine median)
      - Methods (comma-separated tests)
    """
    frames = []

    if not anomalies_iqr.empty:
        tmp = anomalies_iqr[["Date", "Mine", "Output"]].copy()
        tmp["Method"] = "IQR"
        frames.append(tmp)

    if not anomalies_z.empty:
        tmp = anomalies_z[["Date", "Mine", "Output"]].copy()
        tmp["Method"] = "Z-score"
        frames.append(tmp)

    if not anomalies_ma.empty:
        tmp = anomalies_ma[["Date", "Mine", "Output"]].copy()
        tmp["Method"] = "Moving average"
        frames.append(tmp)

    if not anomalies_grubbs.empty:
        tmp = anomalies_grubbs[["Date", "Mine", "Output"]].copy()
        tmp["Method"] = "Grubbs"
        frames.append(tmp)

    if not frames:
        return pd.DataFrame(
            columns=["Date", "Mine", "Value", "Spike_or_drop", "Methods"]
        )

    combined = pd.concat(frames, ignore_index=True)

    grouped = (
        combined.groupby(["Date", "Mine", "Output"])["Method"]
        .apply(lambda s: ", ".join(sorted(s.unique())))
        .reset_index()
        .rename(columns={"Output": "Value", "Method": "Methods"})
    )

    # Spike / Drop classification
    median_map = stats_df.set_index("Mine")["Median"].to_dict()

    def classify_spike_drop(row):
        med = median_map.get(row["Mine"])
        if pd.isna(med):
            return ""
        if row["Value"] > med:
            return "Spike"
        elif row["Value"] < med:
            return "Drop"
        else:
            return "Neutral"

    grouped["Spike_or_drop"] = grouped.apply(classify_spike_drop, axis=1)

    grouped = grouped[["Date", "Mine", "Value", "Spike_or_drop", "Methods"]]
    grouped = grouped.sort_values(["Date", "Mine"]).reset_index(drop=True)

    return grouped


def build_overall_chart_data_uri(df: pd.DataFrame, mine_cols: list[str]) -> str:
    """
    Draw a static line chart (all mines over time) with matplotlib,
    return it as base64 data URI string to embed into HTML.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    for col in mine_cols:
        ax.plot(df["Date"], df[col], label=col)

    ax.set_title("Daily output per mine")
    ax.set_xlabel("Date")
    ax.set_ylabel("Output")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    img_bytes = buf.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    return f"data:image/png;base64,{img_b64}"


def build_html_report(
    df: pd.DataFrame,
    stats_df: pd.DataFrame,
    combined_anomalies: pd.DataFrame,
    mine_cols: list[str],
) -> str:
    """
    Build a clean HTML report:
    - Overview
    - Overall chart
    - Per-mine statistics
    - Combined anomalies
    """
    start_date = df["Date"].min()
    end_date = df["Date"].max()
    n_days = df["Date"].nunique()
    n_mines = len(mine_cols)

    df_total = df.copy()
    df_total["Total_output"] = df_total[mine_cols].sum(axis=1)
    total_output_sum = df_total["Total_output"].sum()
    total_output_mean = df_total["Total_output"].mean()

    chart_uri = build_overall_chart_data_uri(df, mine_cols)

    stats_html = stats_df.to_html(index=False, float_format=lambda x: f"{x:,.2f}")
    anomalies_html = (
        combined_anomalies.to_html(index=False, float_format=lambda x: f"{x:,.2f}")
        if not combined_anomalies.empty
        else "<p>No anomalies detected.</p>"
    )

    html = f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Weyland-Yutani Mines – Daily Output Report</title>
      <style>
        @page {{
          margin: 20mm;
        }}
        body {{
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
          margin: 0;
          color: #222;
          line-height: 1.4;
          font-size: 12px;
        }}
        h1, h2, h3 {{
          color: #222;
          margin: 0 0 6px 0;
        }}
        h1 {{
          font-size: 20px;
          border-bottom: 2px solid #444;
          padding-bottom: 4px;
          margin-bottom: 10px;
        }}
        h2 {{
          font-size: 16px;
          margin-top: 16px;
        }}
        h3 {{
          font-size: 14px;
          margin-top: 10px;
        }}
        .section {{
          margin-top: 14px;
        }}
        table {{
          border-collapse: collapse;
          width: 100%;
          margin-top: 6px;
          font-size: 11px;
        }}
        th, td {{
          border: 1px solid #ddd;
          padding: 4px 6px;
          text-align: right;
        }}
        th {{
          background-color: #f4f4f4;
          font-weight: 600;
        }}
        td:first-child, th:first-child {{
          text-align: left;
        }}
        tr:nth-child(even) td {{
          background-color: #fafafa;
        }}
        .kpi-row {{
          margin-top: 4px;
        }}
        .kpi-box {{
          display: inline-block;
          margin-right: 10px;
          margin-top: 4px;
          padding: 6px 10px;
          border-radius: 4px;
          background-color: #f7f7f7;
          border: 1px solid #ddd;
        }}
        .chart-container {{
          text-align: center;
          margin-top: 10px;
        }}
        .chart-container img {{
          max-width: 100%;
          height: auto;
          border: 1px solid #ddd;
          border-radius: 4px;
        }}
        .small-muted {{
          font-size: 10px;
          color: #666;
        }}
        .page-break {{
          page-break-before: always;
        }}
      </style>
    </head>
    <body>

      <!-- Title -->
      <h1>Weyland-Yutani Mines – Daily Output &amp; Anomaly Report</h1>
      <p class="small-muted">
        Date range: {start_date.date()} – {end_date.date()}<br/>
        Generated by: Data Engineering Dashboard
      </p>

      <!-- 1. Overview -->
      <div class="section">
        <h2>1. Overview</h2>
        <div class="kpi-row">
          <div class="kpi-box">
            <strong>Number of mines</strong><br/>{n_mines}
          </div>
          <div class="kpi-box">
            <strong>Number of days</strong><br/>{n_days}
          </div>
          <div class="kpi-box">
            <strong>Total output (all mines)</strong><br/>{total_output_sum:,.2f}
          </div>
          <div class="kpi-box">
            <strong>Average daily total output</strong><br/>{total_output_mean:,.2f}
          </div>
        </div>
      </div>

      <!-- 2. Overall production chart -->
      <div class="section">
        <h2>2. Overall production chart</h2>
        <p class="small-muted">
          Daily output per mine over time (lines generated from the current simulator dataset).
        </p>
        <div class="chart-container">
          <img src="{chart_uri}" alt="Daily output chart" />
        </div>
      </div>

      <!-- Page break before tables -->
      <div class="page-break"></div>

      <!-- 3. Per-mine statistics -->
      <div class="section">
        <h2>3. Per-mine statistics</h2>
        <p class="small-muted">
          Mean, standard deviation, median, and interquartile range (IQR) for each mine and for the total.
        </p>
        {stats_html}
      </div>

      <!-- 4. Anomaly events -->
      <div class="section">
        <h2>4. Anomaly events</h2>
        <p class="small-muted">
          Detected anomalies with spike/drop classification and the list of tests that flagged each point.
        </p>
        {anomalies_html}
      </div>

    </body>
    </html>
    """

    return html


def html_to_pdf_bytes(html: str) -> bytes:
    """
    Convert HTML string to PDF bytes using pdfkit.
    Requires wkhtmltopdf installed on the system.
    """
    wkhtml_path = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
    config = pdfkit.configuration(wkhtmltopdf=wkhtml_path)

    pdf_bytes = pdfkit.from_string(html, output_path=False, configuration=config)
    return pdf_bytes


# Streamlit app

st.title("Weyland-Yutani Mining Ops Dashboard")
st.caption(
    "Interactive analysis of simulated daily output across mines: "
    "summary statistics, anomaly detection, charts, and exportable PDF reports."
)
st.markdown("---")

st.sidebar.title("Controls")
csv_url = st.sidebar.text_input("Google Sheets CSV URL", CSV_URL_DEFAULT)

# Anomaly tests to run
st.sidebar.markdown("### Anomaly tests to run")
run_iqr = st.sidebar.checkbox("IQR rule", value=True)
run_z = st.sidebar.checkbox("Z-score rule", value=True)
run_ma = st.sidebar.checkbox("Moving average % rule", value=True)
run_grubbs = st.sidebar.checkbox("Grubbs' test", value=True)

st.sidebar.markdown("---")

# Sensitivity controls
k_iqr = st.sidebar.slider("IQR multiplier k", 1.0, 5.0, 1.5, 0.1)
z_thresh = st.sidebar.slider("Z-score threshold", 1.0, 5.0, 3.0, 0.1)
ma_window = st.sidebar.slider("MA window (days)", 3, 30, 7, 1)
ma_pct = st.sidebar.slider("MA deviation threshold (%)", 5, 200, 30, 5)
alpha_grubbs = st.sidebar.slider(
    "Grubbs' alpha (significance)", 0.001, 0.10, 0.05, 0.005
)

st.sidebar.markdown("---")

# Chart options
st.sidebar.markdown("### Chart options")
chart_type = st.sidebar.selectbox(
    "Chart type",
    ["Line", "Bar", "Stacked area"],
)
trend_degree = st.sidebar.selectbox(
    "Trendline degree",
    [0, 1, 2, 3, 4],
    format_func=lambda d: "No trendline" if d == 0 else f"Degree {d}",
    index=1,
)

anomaly_for_chart = st.sidebar.selectbox(
    "Highlight anomalies on chart",
    ["None", "IQR", "Z-score", "Moving average", "Grubbs"],
)

if csv_url:
    df = load_data(csv_url)

    # Exclude 'Event Multiplier' from mine columns
    mine_cols = [
        c
        for c in df.columns
        if c not in ["Date", "Day_idx", "Weekday", "Event Multiplier"]
    ]

    # Compute statistics
    stats_df = compute_stats(df, mine_cols)

    # Convert dataframe to long format for charts
    df_long = df.melt(
        id_vars=["Date", "Day_idx", "Weekday"],
        value_vars=mine_cols,
        var_name="Mine",
        value_name="Output",
    )

    # Detect anomalies (for tables, chart, and combined list)
    anomalies_iqr = (
        detect_iqr_anomalies(df, mine_cols, k=k_iqr)
        if run_iqr or anomaly_for_chart == "IQR"
        else pd.DataFrame()
    )
    anomalies_z = (
        detect_zscore_anomalies(df, mine_cols, z_thresh=z_thresh)
        if run_z or anomaly_for_chart == "Z-score"
        else pd.DataFrame()
    )
    anomalies_ma = (
        detect_ma_percent_anomalies(
            df, mine_cols, window=ma_window, pct_thresh=float(ma_pct)
        )
        if run_ma or anomaly_for_chart == "Moving average"
        else pd.DataFrame()
    )
    anomalies_grubbs = (
        detect_grubbs_anomalies(df, mine_cols, alpha=alpha_grubbs)
        if run_grubbs or anomaly_for_chart == "Grubbs"
        else pd.DataFrame()
    )

    combined_anomalies = build_anomaly_summary(
        df,
        stats_df,
        anomalies_iqr,
        anomalies_z,
        anomalies_ma,
        anomalies_grubbs,
    )

    # Layout: top columns
    col_left, col_right = st.columns([2, 1])

    # Charts
    with col_left:
        st.subheader("Mines output over time")

        tooltip = ["Date:T", "Mine:N", "Output:Q"]

        if chart_type == "Stacked area":
            main_chart = (
                alt.Chart(df_long)
                .mark_area()
                .encode(
                    x="Date:T",
                    y=alt.Y("Output:Q", stack="zero"),
                    color="Mine:N",
                    tooltip=tooltip,
                )
            )
        else:
            base = (
                alt.Chart(df_long)
                .encode(
                    x="Date:T",
                    y="Output:Q",
                    color="Mine:N",
                    tooltip=tooltip,
                )
            )
            if chart_type == "Line":
                main_chart = base.mark_line()
            else:
                main_chart = base.mark_bar()

        trend_df = compute_trendlines(df, mine_cols, degree=trend_degree)
        if trend_df is not None and trend_degree > 0:
            trend_chart = (
                alt.Chart(trend_df)
                .mark_line(strokeDash=[4, 4])
                .encode(
                    x="Date:T",
                    y="Trend:Q",
                    color="Mine:N",
                    tooltip=["Date:T", "Mine:N", "Trend:Q"],
                )
            )
            chart = main_chart + trend_chart
        else:
            chart = main_chart

        anomalies_for_chart = None
        if anomaly_for_chart == "IQR":
            anomalies_for_chart = anomalies_iqr
        elif anomaly_for_chart == "Z-score":
            anomalies_for_chart = anomalies_z
        elif anomaly_for_chart == "Moving average":
            anomalies_for_chart = anomalies_ma
        elif anomaly_for_chart == "Grubbs":
            anomalies_for_chart = anomalies_grubbs

        if anomalies_for_chart is not None and not anomalies_for_chart.empty:
            anomaly_points = anomalies_for_chart[["Date", "Mine", "Output"]].copy()

            anomaly_chart = (
                alt.Chart(anomaly_points)
                .mark_circle(size=80, color="red")
                .encode(
                    x="Date:T",
                    y="Output:Q",
                    tooltip=["Date:T", "Mine:N", "Output:Q"],
                )
            )

            chart = chart + anomaly_chart

        st.altair_chart(chart.interactive(), use_container_width=True)

    # Summary statistics (right column)
    with col_right:
        st.subheader("Summary statistics")
        st.dataframe(
            stats_df.style.format(
                {
                    "Mean": "{:,.1f}",
                    "Std": "{:,.1f}",
                    "Median": "{:,.1f}",
                    "IQR": "{:,.1f}",
                }
            )
        )

    st.markdown("---")

    # Combined anomaly list
    st.subheader("Combined anomaly list (all methods)")
    if combined_anomalies.empty:
        st.write("No anomalies detected by any method.")
    else:
        st.dataframe(combined_anomalies)

    # Individual anomaly tables
    if run_iqr:
        st.subheader(f"IQR-based anomalies (k = {k_iqr})")
        st.write("Rule: value outside [Q1 - k·IQR, Q3 + k·IQR].")
        if anomalies_iqr.empty:
            st.write("No IQR outliers detected.")
        else:
            st.dataframe(anomalies_iqr)

    if run_z:
        st.subheader(f"Z-score-based anomalies (|z| > {z_thresh})")
        st.write("Rule: z = (x - mean) / std.")
        if anomalies_z.empty:
            st.write("No z-score outliers detected.")
        else:
            st.dataframe(anomalies_z)

    if run_ma:
        st.subheader(
            f"Moving-average anomalies (window={ma_window}, deviation > {ma_pct}%)"
        )
        st.write("Rule: |x - MA| / MA * 100 > threshold.")
        if anomalies_ma.empty:
            st.write("No moving-average outliers detected.")
        else:
            st.dataframe(anomalies_ma)

    if run_grubbs:
        st.subheader(f"Grubbs' test anomalies (alpha = {alpha_grubbs})")
        st.write("Rule: G = |x - mean| / std; G > G_crit(alpha, N).")
        if anomalies_grubbs.empty:
            st.write("No Grubbs outliers detected.")
        else:
            st.dataframe(anomalies_grubbs)

    # PDF report
    st.markdown("---")
    st.subheader("Download report")
    if st.button("Generate PDF report"):
        html = build_html_report(df, stats_df, combined_anomalies, mine_cols)
        try:
            pdf_bytes = html_to_pdf_bytes(html)
            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name="weyland_yutani_report.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"PDF generation failed: {e}")

else:
    st.info("Please paste the CSV URL.")
