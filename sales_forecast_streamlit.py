"""
Streamlit Web App: Quarterly Sales Forecast
Author: Charles Muniz (adapted by ChatGPT)

Features:
- Upload vendas.xlsx (sheet with a 'Quarter' column and state columns)
- Clean and regularize quarters (handles formats like '2023Trim1' or ISO dates)
- Removes future quarters after a configurable cutoff
- Compares three models per series: ETS (Holt-Winters), OLS with seasonal dummies, Seasonal-Naive
- Selects best model by RMSE on a holdout (last 4 quarters) when available
- Trains final model on full series and forecasts next 4 quarters with simple bootstrap intervals
- Interactive plots per state, summary tables, CSV exports and PDF report generation

Usage: install dependencies and run with `streamlit run sales_forecast_streamlit.py`
Dependencies: pip install streamlit pandas numpy plotly statsmodels scikit-learn openpyxl

"""

import io
import os
import math
import tempfile
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sales Forecast Dashboard", layout="wide")

# ---------------------------------------------
# Helper functions
# ---------------------------------------------

def quarter_to_date(q):
    """Parse quarter formats like '2023Trim1' or '2023-Q1' or ISO dates. Return quarter end Timestamp or NaT."""
    if pd.isna(q):
        return pd.NaT
    s = str(q).strip()
    # Try common patterns
    try:
        # pattern: 2023Trim1 or 2023Trim2 ...
        if 'Trim' in s:
            year = int(s[:4])
            qnum = int(s.split('Trim')[-1])
            month = {1:3, 2:6, 3:9, 4:12}[qnum]
            return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
        # pattern: 2023-Q1 or 2023Q1
        if 'Q' in s and '-' in s:
            parts = s.split('-')
            year = int(parts[0])
            qnum = int(parts[1].replace('Q',''))
            month = {1:3, 2:6, 3:9, 4:12}[qnum]
            return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
        if 'Q' in s and len(s)>=6 and s[4]=='Q':
            year = int(s[:4]); qnum = int(s[5]); month = {1:3,2:6,3:9,4:12}[qnum]
            return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
        # fallback: try to parse as date
        dt = pd.to_datetime(s, errors='coerce')
        if not pd.isna(dt):
            # convert date to quarter end
            p = dt.to_period('Q')
            return p.end_time
    except Exception:
        return pd.NaT
    return pd.NaT


def build_Q_matrix(quarts):
    dfq = pd.DataFrame({'q':quarts})
    d = pd.get_dummies(dfq['q'], prefix='q', drop_first=True)
    for col in ['q_1','q_2','q_3']:
        if col not in d.columns:
            d[col]=0
    return d[['q_1','q_2','q_3']].values


def format_big(x):
    # Format large numbers with thousand separator
    try:
        if math.isnan(x):
            return ''
        return f"{int(round(x)):,}".replace(',', '.')
    except Exception:
        return str(x)


# Forecast pipeline per series
def fit_models_and_forecast(s, h=4, n_boot=1000):
    """Given a pandas Series indexed by quarter-end dates, compare models and return final forecast and metadata."""
    result = {'series': s.name}
    if s.isna().all() or len(s)==0:
        result.update({'best_model':'NO_DATA'})
        return result
    if s.sum() == 0:
        result.update({'best_model':'ALL_ZERO', 'forecast': np.zeros(h), 'pi_lower': np.zeros(h), 'pi_upper': np.zeros(h)})
        return result

    # define train/test
    h = int(h)
    if len(s) >= 8:
        train = s[:-h]
        test = s[-h:]
    else:
        train = s
        test = pd.Series(dtype=float)

    rmses = {}
    preds = {}

    # ETS
    try:
        ets_model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=4, initialization_method='estimated')
        ets_fit = ets_model.fit(optimized=True, use_brute=True)
        ets_fore = ets_fit.forecast(h)
        preds['ETS'] = ets_fore
        if len(test)>0:
            rmses['ETS'] = mean_squared_error(test.values, ets_fore.values, squared=False)
    except Exception as e:
        preds['ETS'] = None

    # OLS seasonal
    try:
        n = len(train)
        X_time = np.arange(n).reshape(-1,1)
        quarters = ((train.index.quarter-1)).astype(int)
        Xp = np.hstack([X_time, build_Q_matrix(quarters)])
        y = train.values
        lr = LinearRegression(); lr.fit(Xp,y)
        future_times = np.arange(n, n+h).reshape(-1,1)
        future_quarters = ((pd.date_range(start=train.index[-1]+pd.offsets.MonthEnd(1), periods=h, freq='Q').quarter-1)).astype(int)
        Xf = np.hstack([future_times, build_Q_matrix(future_quarters)])
        ols_fore = pd.Series(lr.predict(Xf), index=test.index if len(test)>0 else pd.date_range(start=train.index[-1]+pd.offsets.MonthEnd(1), periods=h, freq='Q'))
        preds['OLS'] = ols_fore
        if len(test)>0:
            rmses['OLS'] = mean_squared_error(test.values, ols_fore.values, squared=False)
    except Exception as e:
        preds['OLS'] = None

    # Seasonal naive
    try:
        if len(train) >= 4:
            s_naive = pd.Series(train[-4:].values, index=test.index if len(test)>0 else pd.date_range(start=train.index[-1]+pd.offsets.MonthEnd(1), periods=h, freq='Q'))
        else:
            s_naive = pd.Series([train.mean()]*h, index=test.index if len(test)>0 else pd.date_range(start=train.index[-1]+pd.offsets.MonthEnd(1), periods=h, freq='Q'))
        preds['SNaive'] = s_naive
        if len(test)>0:
            rmses['SNaive'] = mean_squared_error(test.values, s_naive.values, squared=False)
    except Exception as e:
        preds['SNaive'] = None

    # Choose best model
    if len(rmses)==0:
        # No holdout or models failed — default to ETS if available else OLS else naive
        if preds.get('ETS') is not None:
            best_model = 'ETS'
        elif preds.get('OLS') is not None:
            best_model = 'OLS'
        else:
            best_model = 'SNaive'
    else:
        best_model = min(rmses, key=rmses.get)

    # Fit final on full series
    try:
        if best_model == 'ETS':
            fit = ExponentialSmoothing(s, trend='add', seasonal='add', seasonal_periods=4, initialization_method='estimated').fit(optimized=True, use_brute=True)
            f = fit.forecast(h)
            # bootstrap residuals for PI
            resid_full = (s - fit.fittedvalues).dropna().values
            sims = n_boot
            rng = np.random.default_rng(999)
            sim_forecasts = np.zeros((sims, h))
            for i in range(sims):
                if len(resid_full)==0:
                    noise = np.zeros(h)
                else:
                    noise = rng.choice(resid_full, size=h, replace=True)
                sim_forecasts[i,:] = f.values + noise
            pi_lower = np.percentile(sim_forecasts, 2.5, axis=0)
            pi_upper = np.percentile(sim_forecasts, 97.5, axis=0)
            forecast_index = pd.date_range(start=s.index[-1] + pd.offsets.MonthEnd(1), periods=h, freq='Q')
            result.update({'best_model':best_model, 'forecast': f.values, 'pi_lower': pi_lower, 'pi_upper': pi_upper, 'forecast_index': forecast_index})
            return result

        elif best_model == 'OLS':
            nfull = len(s)
            X_time_full = np.arange(nfull).reshape(-1,1)
            quarters_full = ((s.index.quarter-1)).astype(int)
            Xp_full = np.hstack([X_time_full, build_Q_matrix(quarters_full)])
            y_full = s.values
            lr_full = LinearRegression(); lr_full.fit(Xp_full, y_full)
            future_times_full = np.arange(nfull, nfull+h).reshape(-1,1)
            future_quarters_full = ((pd.date_range(start=s.index[-1]+pd.offsets.MonthEnd(1), periods=h, freq='Q').quarter-1)).astype(int)
            Xf_full = np.hstack([future_times_full, build_Q_matrix(future_quarters_full)])
            f = lr_full.predict(Xf_full)
            resid_full = (y_full - lr_full.predict(Xp_full))
            sims = n_boot
            rng = np.random.default_rng(2025)
            sim_forecasts = np.zeros((sims,h))
            for i in range(sims):
                if len(resid_full)==0:
                    noise = np.zeros(h)
                else:
                    noise = rng.choice(resid_full, size=h, replace=True)
                sim_forecasts[i,:] = f + noise
            pi_lower = np.percentile(sim_forecasts,2.5,axis=0)
            pi_upper = np.percentile(sim_forecasts,97.5,axis=0)
            forecast_index = pd.date_range(start=s.index[-1] + pd.offsets.MonthEnd(1), periods=h, freq='Q')
            result.update({'best_model':best_model, 'forecast': f, 'pi_lower': pi_lower, 'pi_upper': pi_upper, 'forecast_index': forecast_index})
            return result
        else:
            if len(s) >= 4:
                f = s[-4:].values
            else:
                f = np.array([s.mean()]*h)
            pi_lower = f * 0.9
            pi_upper = f * 1.1
            forecast_index = pd.date_range(start=s.index[-1] + pd.offsets.MonthEnd(1), periods=h, freq='Q')
            result.update({'best_model':'SNaive', 'forecast': f, 'pi_lower': pi_lower, 'pi_upper': pi_upper, 'forecast_index': forecast_index})
            return result
    except Exception as e:
        # fallback: linear extrapolation
        x = np.arange(len(s))
        slope, intercept = np.polyfit(x, s.values, 1)
        future_x = np.arange(x[-1]+1, x[-1]+1+h)
        f = intercept + slope * future_x
        forecast_index = pd.date_range(start=s.index[-1] + pd.offsets.MonthEnd(1), periods=h, freq='Q')
        result.update({'best_model':'fallback_linear', 'forecast': f, 'pi_lower': f*0.9, 'pi_upper': f*1.1, 'forecast_index': forecast_index})
        return result


# ---------------------------------------------
# Streamlit UI
# ---------------------------------------------

st.title("📈 Sales Forecast Dashboard — Quarterly")
st.write("Upload your `vendas.xlsx` file (sheet with a 'Quarter' column). The app will clean, model and forecast next 4 quarters.")

uploaded_file = st.file_uploader("Upload vendas.xlsx", type=["xlsx","xls","csv"]) 

# Sidebar options
with st.sidebar:
    st.header("Options")
    cutoff_option = st.date_input("Cutoff (remove quarters after)", value=pd.Timestamp.now().to_period('Q').end_time)
    h = st.number_input("Forecast horizon (quarters)", min_value=1, max_value=8, value=4)
    n_boot = st.number_input("Bootstrap simulations for PI", min_value=100, max_value=5000, value=1000, step=100)
    show_table = st.checkbox("Show detailed tables", value=True)

if uploaded_file is not None:
    # Read file
    try:
        if uploaded_file.name.lower().endswith(('.xlsx','.xls')):
            df_raw = pd.read_excel(uploaded_file)
        else:
            df_raw = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    # Parse quarter
    if 'Quarter' not in df_raw.columns:
        st.error("File must contain a 'Quarter' column")
        st.stop()

    df = df_raw.copy()
    df['date'] = df['Quarter'].map(quarter_to_date)
    df = df.dropna(subset=['date']).sort_values('date')

    # apply cutoff
    df = df[df['date'] <= pd.Timestamp(cutoff_option)]

    # deduplicate
    df = df.drop_duplicates(subset=['date'], keep='last')

    # ensure A exists
    cols = [c for c in df.columns if c not in ['Quarter','date']]
    if 'A' not in cols:
        numeric_cols = df[cols].select_dtypes(include=[np.number]).columns
        if len(numeric_cols)>0:
            df['A'] = df[numeric_cols].sum(axis=1)

    # numeric conversion
    for c in [c for c in df.columns if c not in ['Quarter','date']]:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)

    df = df.set_index('date')

    st.subheader("Cleaned data preview")
    st.write(df.head(20))

    # Run modeling pipeline
    st.info("Running models — this may take some seconds")
    states = [c for c in df.columns if c not in ['Quarter']]

    # storage
    meta_results = []
    forecasts_all = {}

    progress = st.progress(0)
    total = len(states)
    for i, series in enumerate(states):
        s = df[series].astype(float).dropna()
        res = fit_models_and_forecast(s, h=h, n_boot=int(n_boot))
        meta_results.append(res)
        if 'forecast' in res:
            idx = res['forecast_index']
            f = pd.Series(res['forecast'], index=idx)
            forecasts_all[series] = {'forecast': f, 'pi_lower': res.get('pi_lower'), 'pi_upper': res.get('pi_upper'), 'best_model': res.get('best_model')}
        progress.progress((i+1)/total)

    summary_df = pd.DataFrame([{ 'series':r.get('series'), 'best_model': r.get('best_model') } for r in meta_results])
    st.success("Modeling complete")

    # Show summary
    st.subheader("Model selection summary")
    st.table(summary_df)

    # Show interactive plots per series
    st.subheader("Forecasts — interactive")
    col1, col2 = st.columns([2,1])
    with col2:
        chosen = st.selectbox("Choose series to plot", options=list(forecasts_all.keys()))
        show_pi = st.checkbox("Show 95% PI", value=True)
    if chosen:
        obj = forecasts_all[chosen]
        hist = df[chosen]
        fut = obj['forecast']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode='lines+markers', name='Historical'))
        fig.add_trace(go.Scatter(x=fut.index, y=fut.values, mode='lines+markers', name='Forecast'))
        if show_pi and obj.get('pi_lower') is not None:
            fig.add_trace(go.Scatter(x=fut.index, y=obj['pi_upper'], mode='lines', name='PI Upper', line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=fut.index, y=obj['pi_lower'], mode='lines', name='PI Lower', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,100,80,0.2)', showlegend=False))
        fig.update_layout(title=f"{chosen} — Forecast ({obj['best_model']})", yaxis_tickformat=",")
        st.plotly_chart(fig, use_container_width=True)

    # Download CSVs
    st.subheader("Download results")
    # combined forecast table
    if forecasts_all:
        combined = pd.concat({k:v['forecast'] for k,v in forecasts_all.items()}, axis=1)
        combined.columns = [f"{c}_forecast" for c in combined.columns]
        csv = combined.to_csv(index=True)
        st.download_button("Download forecasts CSV", data=csv, file_name="final_forecasts.csv", mime='text/csv')

    # Generate PDF report
    if st.button("Generate PDF report"):
        with st.spinner("Generating PDF..."):
            tmp_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
            pdf_path = tmp_pdf.name
            try:
                with PdfPages(pdf_path) as pdf:
                    # Cover
                    fig = plt.figure(figsize=(11.7,8.3)); plt.axis('off')
                    plt.text(0.5,0.6,"Sales Forecast Report",ha='center',fontsize=22,weight='bold')
                    plt.text(0.5,0.48,f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",ha='center',fontsize=10)
                    pdf.savefig(); plt.close()

                    # Summary table
                    fig, ax = plt.subplots(figsize=(11.7,8.3)); ax.axis('off'); ax.set_title("Model selection summary", fontsize=14)
                    table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns, loc='center')
                    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1,1.5)
                    pdf.savefig(); plt.close()

                    # Per-series pages
                    for k,v in forecasts_all.items():
                        hist = df[k]
                        fut = v['forecast']
                        pi_l = v['pi_lower']
                        pi_u = v['pi_upper']
                        fig = plt.figure(figsize=(11.7,8.3))
                        ax = fig.add_subplot(111)
                        ax.plot(hist.index, hist.values, marker='o', label='Historical')
                        ax.plot(fut.index, fut.values, marker='o', label='Forecast')
                        if pi_l is not None:
                            ax.fill_between(fut.index, pi_l, pi_u, alpha=0.2, label='95% PI')
                        ax.set_title(f"{k} — Forecast ({v['best_model']})")
                        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{int(round(x)):,}".replace(',', '.')))
                        ax.legend()
                        pdf.savefig(); plt.close()

                        # table
                        fig, ax = plt.subplots(figsize=(11.7,2)); ax.axis('off')
                        fcast_df = fut.reset_index(); fcast_df.columns = ['Date','Forecast']
                        table = ax.table(cellText=fcast_df.values, colLabels=fcast_df.columns, loc='center')
                        table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1,1.2)
                        pdf.savefig(); plt.close()

                tmp_pdf.flush()
                with open(pdf_path,'rb') as f:
                    st.download_button('Download PDF report', f.read(), file_name='forecast_report.pdf', mime='application/pdf')
            finally:
                try:
                    os.remove(pdf_path)
                except Exception:
                    pass

else:
    st.info('Upload your vendas.xlsx to start')

# End of file
