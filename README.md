# algosia

Client Python pour récupérer des données OHLCV et prédictions de modèles depuis l'API Algosia.

## Authentification

Vous avez besoin d'un **refresh token** longue durée fourni par Algosia.

## Usage

### Initialisation du client

```python
from algosia.algosia_client import AlgosiaClient

client = AlgosiaClient(
    refresh_token="VOTRE_REFRESH_TOKEN",
    verbose=1  # 0: silencieux, 1: barre de progression, 2: logs détaillés
)
```

### Récupération OHLCV (bougies 1 seconde)

```python
# Format pandas DataFrame
df = client.get_ohlcv(
    exchange="binance",
    pair="BTCUSDT",
    start_ts="2025-01-01T00:00:00Z",
    end_ts="2025-01-02T00:00:00Z",
    max_points=15000,
    output="pandas"
)

# Export CSV
client.get_ohlcv(
    exchange="binance",
    pair="BTCUSDT",
    start_ts="2025-01-01T00:00:00Z",
    end_ts="2025-01-02T00:00:00Z",
    output="csv",
    dest_path="data/ohlcv.csv"
)

# Export JSON
data = client.get_ohlcv(
    exchange="binance",
    pair="BTCUSDT",
    start_ts="2025-01-01T00:00:00Z",
    end_ts="2025-01-02T00:00:00Z",
    output="json",
    dest_path="data/ohlcv.json"
)
```

### Récupération des prédictions

```python
# Format pandas DataFrame
df = client.get_predictions(
    exchange="binance",
    pair="BTCUSDT",
    model_name="long_LSTM_1s_BTCUSDT_2024_09-12",
    start_ts="2025-01-01T00:00:00Z",
    end_ts="2025-01-02T00:00:00Z",
    output="pandas"
)

# Export CSV ou JSON
client.get_predictions(
    exchange="binance",
    pair="BTCUSDT",
    model_name="long_LSTM_1s_BTCUSDT_2024_09-12",
    start_ts="2025-01-01T00:00:00Z",
    end_ts="2025-01-02T00:00:00Z",
    output="csv",
    dest_path="data/predictions.csv"
)
```

## Exchanges et paires supportés

**Binance**: BTCUSDT, ETHUSDT, XRPUSDT, BNBUSDT, SOLUSDT  
**Kraken**: BTCUSD, ETHUSD, SOLUSD, XRPUSD, XDGUSD

Les modèles disponibles sont listés dans `algosia/env.py`.

## Formats de timestamps

Accepte: epoch Unix, datetime Python, ou ISO 8601 string (ex: `"2025-01-01T00:00:00Z"`).

## Exemple complet

Voir `examples/example_ohlcv.py` pour un exemple d'utilisation détaillé.