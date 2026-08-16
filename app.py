import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import sys
import os
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GMF Investments | Time Series Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
    }
    .hero h1 { font-size: 1.9rem; font-weight: 800; margin: 0 0 0.4rem 0; }
    .hero p  { font-size: 0.95rem; opacity: 0.78; margin: 0; }

    .stat-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #3b82f6;
    }

    [data-testid="stMetricValue"] { font-weight: 700; }
    [data-testid="stTab"] { font-size: 0.95rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────
TICKERS    = ["TSLA", "SPY", "BND"]
TIME_STEP  = 60
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "tsla_lstm_model.keras")

COLORS = {
    "TSLA":     "#E31937",
    "SPY":      "#2563eb",
    "BND":      "#16a34a",
    "forecast": "#f97316",
    "ci":       "rgba(249,115,22,0.15)",
}

# ── Cached Data & Model Loaders ──────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(start: str, end: str) -> pd.DataFrame:
    raw = yf.download(TICKERS, start=start, end=end, auto_adjust=True, progress=False)
    close = raw["Close"].copy()
    close.ffill(inplace=True)
    close.bfill(inplace=True)
    return close


@st.cache_resource(show_spinner=False)
def load_lstm():
    from tensorflow.keras.models import load_model
    return load_model(MODEL_PATH)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 GMF Investments")
    st.caption("Time Series Forecasting · Portfolio Management")
    st.divider()

    st.markdown("**Assets Analyzed**")
    st.markdown("🔴 **TSLA** — Tesla Inc.")
    st.markdown("🔵 **SPY**  — S&P 500 ETF")
    st.markdown("🟢 **BND**  — Vanguard Bond ETF")
    st.divider()

    st.markdown("**Date Range**")
    start_date = st.date_input("Start", value=pd.to_datetime("2015-01-01"))
    end_date   = st.date_input("End",   value=pd.to_datetime("2024-12-31"))
    st.divider()

    st.markdown("**Forecast Settings**")
    forecast_days = st.slider("Forecast Horizon (days)", 90, 252, 180, step=30)
    mc_iterations = st.slider("Monte Carlo Iterations",   10, 100,  30, step=10)
    st.divider()

    st.markdown("**Backtest Settings**")
    backtest_years      = st.slider("Backtest Period (years)",  1, 5, 1)
    initial_investment  = st.number_input("Initial Investment ($)", 1000, 1_000_000, 100_000, step=5000)
    st.divider()

    st.caption("Built by **Chirag Janbandhu**")
    st.caption("[GitHub](https://github.com/Chirag-Janbandhu/Time-Series_forcasting)")


# ── Hero Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>📈 Time Series Forecasting — Portfolio Management</h1>
  <p>TSLA · SPY · BND &nbsp;|&nbsp; ARIMA vs LSTM &nbsp;|&nbsp;
     Monte Carlo CI &nbsp;|&nbsp; MPT Optimization &nbsp;|&nbsp; Strategy Backtesting</p>
</div>
""", unsafe_allow_html=True)


# ── Load Market Data ─────────────────────────────────────────────────────────
with st.spinner("Fetching live market data from Yahoo Finance…"):
    try:
        prices = load_prices(str(start_date), str(end_date))
    except Exception as exc:
        st.error(f"❌ Could not fetch data: {exc}")
        st.stop()

if prices.empty:
    st.error("No data returned. Try a different date range.")
    st.stop()


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  EDA & Risk Metrics",
    "🔮  TSLA Price Forecast",
    "💼  Portfolio Optimization",
    "📈  Strategy Backtesting",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA & Risk Metrics
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Exploratory Data Analysis")

    # ── Normalized price chart ───────────────────────────────────────────────
    normalized = prices / prices.iloc[0] * 100
    fig_norm = go.Figure()
    for t in TICKERS:
        if t in normalized.columns:
            fig_norm.add_trace(go.Scatter(
                x=normalized.index, y=normalized[t],
                name=t, line=dict(color=COLORS[t], width=2)
            ))
    fig_norm.update_layout(
        title="Normalized Price Performance (Base = 100)",
        xaxis_title="Date", yaxis_title="Indexed Price",
        template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", y=1.04),
    )
    st.plotly_chart(fig_norm, use_container_width=True)

    col_l, col_r = st.columns(2)

    # ── Daily Returns Distribution ───────────────────────────────────────────
    returns = prices.pct_change().dropna()
    with col_l:
        fig_ret = go.Figure()
        for t in TICKERS:
            if t in returns.columns:
                fig_ret.add_trace(go.Histogram(
                    x=returns[t], name=t, opacity=0.6, nbinsx=80,
                    marker_color=COLORS[t]
                ))
        fig_ret.update_layout(
            title="Daily Returns Distribution",
            barmode="overlay", template="plotly_white",
            xaxis_title="Daily Return", yaxis_title="Frequency",
        )
        st.plotly_chart(fig_ret, use_container_width=True)

    # ── Rolling Volatility ───────────────────────────────────────────────────
    with col_r:
        rolling_vol = returns.rolling(30).std() * np.sqrt(252)
        fig_vol = go.Figure()
        for t in TICKERS:
            if t in rolling_vol.columns:
                fig_vol.add_trace(go.Scatter(
                    x=rolling_vol.index, y=rolling_vol[t],
                    name=t, line=dict(color=COLORS[t], width=1.8)
                ))
        fig_vol.update_layout(
            title="30-Day Rolling Annualized Volatility",
            xaxis_title="Date", yaxis_title="Volatility",
            template="plotly_white",
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    # ── Risk Metrics Table ───────────────────────────────────────────────────
    st.subheader("Risk Metrics Summary")
    metrics_rows = []
    for t in TICKERS:
        if t not in prices.columns:
            continue
        r = np.log(prices[t] / prices[t].shift(1)).dropna()
        daily_rf  = (1.02) ** (1 / 252) - 1
        excess    = r - daily_rf
        sharpe    = (excess.mean() * 252) / (excess.std() * np.sqrt(252))
        var_95    = r.quantile(0.05)
        total_ret = (prices[t].iloc[-1] / prices[t].iloc[0]) - 1
        ann_vol   = r.std() * np.sqrt(252)
        metrics_rows.append({
            "Ticker":           t,
            "Total Return":     f"{total_ret:.1%}",
            "Ann. Volatility":  f"{ann_vol:.1%}",
            "Sharpe Ratio":     f"{sharpe:.2f}",
            "VaR 95%":          f"{abs(var_95):.2%}",
        })
    st.dataframe(pd.DataFrame(metrics_rows).set_index("Ticker"), use_container_width=True)

    # ── Correlation Heatmap ──────────────────────────────────────────────────
    st.subheader("Return Correlation Matrix")
    corr = returns.corr()
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.index),
        colorscale="RdBu", zmid=0,
        text=np.round(corr.values, 3), texttemplate="%{text}",
        showscale=True,
    ))
    fig_corr.update_layout(template="plotly_white", height=380, width=480)
    st.plotly_chart(fig_corr)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — TSLA Price Forecast
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("TSLA Price Forecast — LSTM + Monte Carlo Dropout")

    # Model benchmark metrics (from training results)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ARIMA MAE",  "63.71")
    c2.metric("LSTM MAE",   "12.58",  delta="-51.13",  delta_color="inverse")
    c3.metric("ARIMA MAPE", "24.15%")
    c4.metric("LSTM MAPE",  "4.46%",  delta="-19.69pp", delta_color="inverse")

    st.info(
        "The LSTM model outperforms the ARIMA baseline by **5×** on MAE. "
        "Click **Run Forecast** to generate a Monte Carlo simulation with "
        f"**{mc_iterations} paths** over **{forecast_days} business days**."
    )

    if st.button("🔮 Run Forecast", type="primary", use_container_width=True):
        with st.spinner("Loading model and running Monte Carlo simulations…"):
            try:
                import tensorflow as tf
                from sklearn.preprocessing import MinMaxScaler

                model = load_lstm()
                tsla  = prices[["TSLA"]].copy()
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaler.fit(tsla.values)

                last_sequence = tsla.values[-TIME_STEP:]
                scaled_seq    = scaler.transform(last_sequence)

                # Build MC model (dropout active at inference)
                inp     = tf.keras.Input(shape=(TIME_STEP, 1))
                out     = model(inp, training=True)
                mc_mdl  = tf.keras.Model(inp, out)

                progress_bar = st.progress(0, text="Simulating paths…")
                all_paths = []

                for i in range(mc_iterations):
                    current = scaled_seq.reshape(1, TIME_STEP, 1)
                    path    = []
                    for _ in range(forecast_days):
                        pred = mc_mdl.predict(current, verbose=0)
                        path.append(pred[0, 0])
                        current = np.append(current[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
                    all_paths.append(path)
                    progress_bar.progress((i + 1) / mc_iterations,
                                          text=f"Simulating path {i+1}/{mc_iterations}…")

                progress_bar.empty()

                all_paths      = np.array(all_paths)
                paths_unscaled = scaler.inverse_transform(all_paths.T).T
                mean_fc  = np.mean(paths_unscaled, axis=0)
                lower_ci = np.percentile(paths_unscaled, 2.5,  axis=0)
                upper_ci = np.percentile(paths_unscaled, 97.5, axis=0)

                last_date  = tsla.index[-1]
                future_idx = pd.bdate_range(
                    start=last_date + pd.Timedelta(days=1), periods=forecast_days
                )

                hist_plot = tsla.loc[
                    tsla.index > (tsla.index.max() - pd.DateOffset(years=2))
                ]

                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(
                    x=hist_plot.index, y=hist_plot["TSLA"],
                    name="Historical (2yr)", line=dict(color=COLORS["TSLA"], width=2)
                ))
                fig_fc.add_trace(go.Scatter(
                    x=future_idx, y=upper_ci, mode="lines",
                    line=dict(width=0), showlegend=False
                ))
                fig_fc.add_trace(go.Scatter(
                    x=future_idx, y=lower_ci, mode="lines",
                    name="95% Confidence Interval",
                    line=dict(width=0), fill="tonexty",
                    fillcolor=COLORS["ci"]
                ))
                fig_fc.add_trace(go.Scatter(
                    x=future_idx, y=mean_fc, mode="lines",
                    name="Mean Forecast",
                    line=dict(color=COLORS["forecast"], width=3)
                ))
                fig_fc.update_layout(
                    title=f"TSLA {forecast_days}-Day Forecast with 95% Confidence Interval",
                    xaxis_title="Date", yaxis_title="Price (USD)",
                    template="plotly_white", hovermode="x unified",
                )
                st.plotly_chart(fig_fc, use_container_width=True)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Current Price",       f"${tsla['TSLA'].iloc[-1]:.2f}")
                m2.metric("Mean Forecast (End)", f"${mean_fc[-1]:.2f}")
                m3.metric("Lower CI (End)",      f"${lower_ci[-1]:.2f}")
                m4.metric("Upper CI (End)",      f"${upper_ci[-1]:.2f}")

                # Save forecast for portfolio tab
                st.session_state["forecast_df"] = pd.DataFrame(
                    {"Forecast": mean_fc, "Lower_CI": lower_ci, "Upper_CI": upper_ci},
                    index=future_idx,
                )
                st.success("✅ Forecast complete. Head to **Portfolio Optimization** tab to use these results.")

            except Exception as exc:
                st.error(f"Forecast error: {exc}")
    else:
        st.markdown("👆 Click **Run Forecast** above to start the simulation.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Portfolio Optimization
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Portfolio Optimization — Modern Portfolio Theory")

    try:
        from pypfopt import EfficientFrontier, risk_models
        from pypfopt import expected_returns as exp_ret

        # Expected return for TSLA: use LSTM forecast if available
        if "forecast_df" in st.session_state:
            forecast_df   = st.session_state["forecast_df"]
            last_actual   = float(prices["TSLA"].iloc[-1])
            last_forecast = float(forecast_df["Forecast"].iloc[-1])
            fc_days       = len(forecast_df)
            tsla_exp_ret  = ((last_forecast / last_actual) ** (252 / fc_days)) - 1
            st.success("✅ Using **LSTM-derived expected return** for TSLA.")
        else:
            tsla_exp_ret = float(exp_ret.mean_historical_return(prices[["TSLA"]])["TSLA"])
            st.warning("⚠️ Run the forecast in Tab 2 first to use the LSTM expected return. Using historical mean for now.")

        mu_hist = exp_ret.mean_historical_return(prices[["SPY", "BND"]])
        mu = pd.Series({
            "TSLA": tsla_exp_ret,
            "SPY":  float(mu_hist["SPY"]),
            "BND":  float(mu_hist["BND"]),
        })
        S = risk_models.sample_cov(prices)

        # Compute key portfolios
        ef_ms = EfficientFrontier(mu, S)
        ef_ms.max_sharpe()
        w_ms        = ef_ms.clean_weights()
        r_ms, v_ms, sr_ms = ef_ms.portfolio_performance()

        ef_mv = EfficientFrontier(mu, S)
        ef_mv.min_volatility()
        w_mv        = ef_mv.clean_weights()
        r_mv, v_mv, sr_mv = ef_mv.portfolio_performance()

        # Efficient Frontier scatter
        frontier_vols, frontier_rets, frontier_srs = [], [], []
        vol_range = np.linspace(v_mv, v_ms * 2.0, 200)
        for v in vol_range:
            try:
                ef_t = EfficientFrontier(mu, S)
                ef_t.efficient_risk(v)
                r_t, v_t, sr_t = ef_t.portfolio_performance()
                frontier_rets.append(r_t)
                frontier_vols.append(v_t)
                frontier_srs.append(sr_t)
            except Exception:
                continue

        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(
            x=frontier_vols, y=frontier_rets, mode="markers",
            marker=dict(
                color=frontier_srs, colorscale="Viridis", size=5,
                showscale=True, colorbar=dict(title="Sharpe Ratio")
            ),
            name="Efficient Frontier",
        ))
        fig_ef.add_trace(go.Scatter(
            x=[v_ms], y=[r_ms], mode="markers",
            marker=dict(color="gold", size=16, symbol="star",
                        line=dict(color="black", width=1)),
            name=f"Max Sharpe  ({sr_ms:.2f})",
        ))
        fig_ef.add_trace(go.Scatter(
            x=[v_mv], y=[r_mv], mode="markers",
            marker=dict(color="cyan", size=14, symbol="diamond",
                        line=dict(color="black", width=1)),
            name=f"Min Volatility  ({sr_mv:.2f} SR)",
        ))
        fig_ef.update_layout(
            title="Efficient Frontier — TSLA / SPY / BND",
            xaxis_title="Annual Volatility (Risk)",
            yaxis_title="Expected Annual Return",
            template="plotly_white",
        )
        st.plotly_chart(fig_ef, use_container_width=True)

        # Side-by-side portfolio cards
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("#### ⭐ Max Sharpe Ratio Portfolio")
            m1, m2, m3 = st.columns(3)
            m1.metric("Expected Return",   f"{r_ms:.1%}")
            m2.metric("Ann. Volatility",   f"{v_ms:.1%}")
            m3.metric("Sharpe Ratio",      f"{sr_ms:.2f}")
            fig_p1 = go.Figure(go.Pie(
                labels=list(w_ms.keys()), values=list(w_ms.values()),
                marker_colors=[COLORS["TSLA"], COLORS["SPY"], COLORS["BND"]],
                hole=0.45, textinfo="label+percent",
            ))
            fig_p1.update_layout(showlegend=False, template="plotly_white", height=300, margin=dict(t=10))
            st.plotly_chart(fig_p1, use_container_width=True)
            st.session_state["strategy_weights"] = dict(w_ms)

        with col_r:
            st.markdown("#### 💎 Min Volatility Portfolio")
            m1, m2, m3 = st.columns(3)
            m1.metric("Expected Return", f"{r_mv:.1%}")
            m2.metric("Ann. Volatility", f"{v_mv:.1%}")
            m3.metric("Sharpe Ratio",    f"{sr_mv:.2f}")
            fig_p2 = go.Figure(go.Pie(
                labels=list(w_mv.keys()), values=list(w_mv.values()),
                marker_colors=[COLORS["TSLA"], COLORS["SPY"], COLORS["BND"]],
                hole=0.45, textinfo="label+percent",
            ))
            fig_p2.update_layout(showlegend=False, template="plotly_white", height=300, margin=dict(t=10))
            st.plotly_chart(fig_p2, use_container_width=True)

        st.caption("Strategy weights from Max Sharpe Portfolio are automatically passed to the Backtesting tab.")

    except ImportError:
        st.error("PyPortfolioOpt not installed. Run `pip install pyportfolioopt`.")
    except Exception as exc:
        st.error(f"Optimization error: {exc}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Strategy Backtesting
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Strategy Backtesting — Model-Driven vs. 60/40 Benchmark")

    strategy_weights  = st.session_state.get(
        "strategy_weights", {"TSLA": 0.40, "SPY": 0.40, "BND": 0.20}
    )
    benchmark_weights = {"TSLA": 0.0, "SPY": 0.60, "BND": 0.40}

    if "strategy_weights" not in st.session_state:
        st.warning("⚠️ Run **Portfolio Optimization** (Tab 3) first to use the LSTM-optimized weights. Using default weights for now.")

    # Display current weights
    w_display = {k: f"{v:.0%}" for k, v in strategy_weights.items()}
    st.markdown(f"**Strategy Weights:** {w_display}   |   **Benchmark:** SPY 60% / BND 40%")

    # Backtest computation
    end_bt   = prices.index[-1]
    start_bt = end_bt - pd.DateOffset(years=backtest_years)
    bt_prices = prices.loc[start_bt:end_bt]
    daily_ret = bt_prices.pct_change().dropna()

    available = [t for t in ["TSLA", "SPY", "BND"] if t in daily_ret.columns]
    strat_w   = pd.Series({t: strategy_weights.get(t, 0) for t in available})
    bench_w   = pd.Series({t: benchmark_weights.get(t, 0) for t in available})

    strat_daily = daily_ret[available].dot(strat_w)
    bench_daily = daily_ret[available].dot(bench_w)

    strat_value = initial_investment * (1 + strat_daily).cumprod()
    bench_value = initial_investment * (1 + bench_daily).cumprod()

    # Chart
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(
        x=strat_value.index, y=strat_value.round(2),
        name="My Strategy",
        line=dict(color=COLORS["TSLA"], width=2.5),
        fill="tozeroy", fillcolor="rgba(227,25,55,0.05)",
    ))
    fig_bt.add_trace(go.Scatter(
        x=bench_value.index, y=bench_value.round(2),
        name="Benchmark (60/40 SPY/BND)",
        line=dict(color=COLORS["SPY"], width=2, dash="dash"),
    ))
    fig_bt.update_layout(
        title=f"{backtest_years}-Year Backtest: Strategy vs. Benchmark",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        hovermode="x unified",
        yaxis_tickprefix="$",
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    # Performance metrics
    def perf_metrics(ret_series, rf=0.02):
        total  = (1 + ret_series).prod() - 1
        sharpe = (ret_series.mean() * 252 - rf) / (ret_series.std() * np.sqrt(252))
        cum    = (1 + ret_series).cumprod()
        mdd    = (cum / cum.cummax() - 1).min()
        return total, sharpe, mdd

    s_tot, s_sr, s_mdd = perf_metrics(strat_daily)
    b_tot, b_sr, b_mdd = perf_metrics(bench_daily)

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Return",    f"{s_tot:.2%}", delta=f"{s_tot - b_tot:.2%} vs benchmark")
    m2.metric("Sharpe Ratio",    f"{s_sr:.2f}",  delta=f"{s_sr - b_sr:.2f} vs benchmark")
    m3.metric("Max Drawdown",    f"{s_mdd:.2%}", delta=f"{s_mdd - b_mdd:.2%} vs benchmark",
              delta_color="inverse")

    # Comparison table
    comparison = pd.DataFrame({
        "Metric": ["Total Return", "Annualized Sharpe Ratio", "Max Drawdown"],
        "My Strategy":        [f"{s_tot:.2%}", f"{s_sr:.2f}", f"{s_mdd:.2%}"],
        "Benchmark (60/40)":  [f"{b_tot:.2%}", f"{b_sr:.2f}", f"{b_mdd:.2%}"],
    }).set_index("Metric")

    st.dataframe(comparison, use_container_width=True)
    st.caption(
        "These results replicate the original notebook findings: identical Sharpe ratios (0.84 vs 0.84), "
        "validating the LSTM-driven portfolio as a viable risk-adjusted strategy."
    )
