import os
import yfinance as yf
import pandas as pd

def download_yf_data(tickers, start, end, interval="1d", output_dir="data", auto_adjust=False):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading {tickers} from {start} to {end} (interval={interval})...")
    data = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        group_by='ticker',
        threads=True,
        auto_adjust=auto_adjust,
        progress=True
    )

    all_data_list = []

    if isinstance(data.columns, pd.MultiIndex):
        for tk in tickers:
            if tk in data.columns.levels[0]:
                df_tk = data[tk].copy()
                df_tk["Ticker"] = tk  # <-- Add ticker column
                all_data_list.append(df_tk)

                csv_path = os.path.join(output_dir, f"{tk}_{start}_to_{end}.csv")
                df_tk.to_csv(csv_path)
                print(f"Saved: {csv_path}")

        # Combined flat CSV with ticker col
        combined_df = pd.concat(all_data_list)
        combined_csv = os.path.join(output_dir, f"combined_{start}_to_{end}.csv")
        combined_df.to_csv(combined_csv)
        print(f"Saved combined CSV: {combined_csv}")

    else:
        # Single ticker
        df_single = data.copy()
        df_single["Ticker"] = tickers[0] if tickers else "UNKNOWN"
        df_single.to_csv(os.path.join(output_dir, f"{tickers[0]}_{start}_to_{end}.csv"))
        all_data_list.append(df_single)

    # Save Excel with ticker column
    excel_path = os.path.join(output_dir, f"yf_{start}_to_{end}.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        for df in all_data_list:
            tk = df["Ticker"].iloc[0]
            df.to_excel(writer, sheet_name=tk[:31])
    print(f"Saved Excel workbook: {excel_path}")

    # Return combined DataFrame
    return pd.concat(all_data_list)
