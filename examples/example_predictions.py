#!/usr/bin/env python3
"""
Exemple d'utilisation du client pour télécharger les prédictions.
Montre :
  - récupération en DataFrame pandas
  - export CSV
  - export JSON

La route côté client est :
  /predictions/<exchange>/<pair>/<model_name>?start_timestamp=...&end_timestamp=...

Réponse attendue : [[timestamp, probability], ...]
"""

from algosia.algosia_client import AlgosiaClient

REFRESH_TOKEN = "TON_REFRESH_TOKEN_ICI"

EXCHANGE   = "binance"
PAIR       = "BTCUSDT"
MODEL_NAME = "long_LSTM_1s_BTCUSDT_2024_09-12"

START_TS = "2025-09-10T09:00:00Z"
END_TS   = "2025-09-10T09:01:00Z"


def main():
    client = AlgosiaClient(
        refresh_token=REFRESH_TOKEN,
        base_url="https://requestor.app.algosia.ai",
        verbose=True,
    )

    # 1) Récupération en pandas
    df = client.get_predictions(
        exchange=EXCHANGE,
        pair=PAIR,
        model_name=MODEL_NAME,
        start_ts=START_TS,
        end_ts=END_TS,
        max_points=15000,
        output="pandas",
    )
    print("\n--- PREDICTIONS / pandas ---")
    print(df.head())

    # 2) Export CSV
    csv_path = "out/preds_sample.csv"
    client.get_predictions(
        exchange=EXCHANGE,
        pair=PAIR,
        model_name=MODEL_NAME,
        start_ts=START_TS,
        end_ts=END_TS,
        max_points=15000,
        output="csv",
        dest_path=csv_path,
    )
    print(f"\nCSV écrit dans {csv_path}")

    # 3) Export JSON
    json_obj = client.get_predictions(
        exchange=EXCHANGE,
        pair=PAIR,
        model_name=MODEL_NAME,
        start_ts=START_TS,
        end_ts=END_TS,
        max_points=15000,
        output="json",
        dest_path="out/preds_sample.json",
    )
    print("\n--- PREDICTIONS / json (aperçu) ---")
    print(json_obj[:2])  # aperçu


if __name__ == "__main__":
    main()
