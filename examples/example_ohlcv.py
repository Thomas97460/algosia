#!/usr/bin/env python3
"""
Exemple d'utilisation du client pour télécharger l'OHLCV.
Montre :
  - récupération en DataFrame pandas
  - export CSV
  - export JSON
"""

from algosia.algosia_client import AlgosiaClient

# Refresh token longue durée (utilisateur final / service)
REFRESH_TOKEN = "TON_REFRESH_TOKEN_ICI"

EXCHANGE = "binance"
PAIR     = "BTCUSDT"

START_TS = "2025-09-10T09:00:00Z"
END_TS   = "2025-09-10T09:01:00Z"


def main():
    client = AlgosiaClient(
        refresh_token=REFRESH_TOKEN,
        base_url="https://requestor.app.algosia.ai",
        verbose=True,
    )

    # 1) Récupération en DataFrame pandas
    df = client.get_ohlcv(
        exchange=EXCHANGE,
        pair=PAIR,
        start_ts=START_TS,
        end_ts=END_TS,
        max_points=15000,
        output="pandas",
    )
    print("\n--- OHLCV / pandas ---")
    print(df.head())

    # 2) Récupération + export CSV
    csv_path = "out/ohlcv_sample.csv"
    client.get_ohlcv(
        exchange=EXCHANGE,
        pair=PAIR,
        start_ts=START_TS,
        end_ts=END_TS,
        max_points=15000,
        output="csv",
        dest_path=csv_path,
    )
    print(f"\nCSV écrit dans {csv_path}")

    # 3) Récupération + export JSON
    json_obj = client.get_ohlcv(
        exchange=EXCHANGE,
        pair=PAIR,
        start_ts=START_TS,
        end_ts=END_TS,
        max_points=15000,
        output="json",
        dest_path="out/ohlcv_sample.json",
    )
    print("\n--- OHLCV / json (aperçu) ---")
    print(json_obj[:2])  # aperçu


if __name__ == "__main__":
    main()
