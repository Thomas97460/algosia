import base64, json, time, requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from tqdm.auto import tqdm
from .env import EXCHANGES_CONFIG


API_KEY_DEFAULT = "AIzaSyA4AOIsbt5TQje_sqI1eE6Wg1yp50TqVq0"


class C:
    R  = "\033[0m"; B  = "\033[1m"; K  = "\033[90m"
    W  = "\033[97m"; RD = "\033[91m"; GN = "\033[92m"
    YL = "\033[93m"; BL = "\033[94m"; MG = "\033[95m"; CN = "\033[96m"


class AlgosiaClient:
    def __init__(self,
                 refresh_token,
                 api_key: str = API_KEY_DEFAULT,
                 base_url: str = "https://requestor.app.algosia.ai",
                 verbose: int = 1):
        """Initialise le client avec refresh_token longue durée, api_key (fixe),
        base_url de l'API et mode verbeux.

        verbose:
          0 → silencieux
          1 → barre de progression (tqdm) sur les downloads
          2 → logs détaillés (mode actuel par défaut).
        Met en cache l'id_token Firebase et son expiration décodée."""
        self.api_key = api_key
        self.refresh_token = refresh_token
        self.base_url = base_url.rstrip("/")
        if isinstance(verbose, bool):
            level = 2 if verbose else 0
        else:
            level = int(verbose)
        self.verbose_level = max(0, min(2, level))
        self.verbose = self.verbose_level > 0  # compat héritée
        self._id_token = None
        self._token_exp = 0.0

    def _now(self):
        """Retourne un timestamp UTC lisible pour les logs."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%fZ")

    def _log(self, kind, msg):
        """Affiche un message selon le niveau de verbosité."""
        if self.verbose_level == 0 and kind != "bad":
            return
        if self.verbose_level == 1 and kind in ("info", "ok"):
            return
        prefix, col = {
            "ok": ("✔", C.GN),
            "info": ("•", C.CN),
            "warn": ("⚠", C.YL),
            "bad": ("✘", C.RD),
        }.get(kind, ("•", C.W))
        text = f"{col}{prefix} [{self._now()}] {msg}{C.R}"
        if self.verbose_level == 1 and tqdm is not None:
            tqdm.write(text)
        else:
            print(text, flush=True)

    def _jwt_valid(self):
        """True si on a déjà un id_token non expiré (avec 15s de marge)."""
        if not self._id_token or not self._token_exp:
            return False
        return time.time() + 15 < self._token_exp

    def _decode_exp(self, jwt_token):
        """Extrait 'exp' du JWT (sans vérif de signature)."""
        try:
            payload_b64 = jwt_token.split(".")[1]
            pad = "=" * (-len(payload_b64) % 4)
            payload = base64.urlsafe_b64decode(payload_b64 + pad)
            return float(json.loads(payload).get("exp", 0))
        except Exception:
            return 0.0

    def _refresh_id_token(self):
        """Récupère un id_token neuf via securetoken.googleapis.com
        en utilisant le refresh_token longue durée."""
        url = f"https://securetoken.googleapis.com/v1/token?key={self.api_key}"
        data = {"grant_type": "refresh_token", "refresh_token": self.refresh_token}
        self._log("info", "Auth: demande d’un id_token via refresh_token")
        r = requests.post(
            url,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=20,
        )
        self._log("info", f"Auth: réponse {r.status_code} {r.reason}")
        r.raise_for_status()
        self._id_token = r.json()["id_token"]
        self._token_exp = self._decode_exp(self._id_token)

    def _ensure_token(self):
        """Retourne un id_token valide en mémoire (refresh si besoin)."""
        if not self._jwt_valid():
            self._refresh_id_token()
        return self._id_token

    def _isoz(self, ts):
        """Normalise un timestamp (epoch / datetime / str ISO) en
        'YYYY-mm-ddTHH:MM:SS.sssZ' UTC."""
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        if isinstance(ts, datetime):
            return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        s = str(ts)
        if s.endswith("Z"):
            return s
        if "+" in s:
            return datetime.fromisoformat(s).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        return datetime.fromisoformat(s + "+00:00").astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    def _to_utc_dt(self, ts):
        """Convertit epoch / datetime / str ISO en datetime timezone-aware UTC."""
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        if isinstance(ts, datetime):
            return ts.astimezone(timezone.utc)
        s = str(ts)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).astimezone(timezone.utc)

    def _save_csv(self, df: pd.DataFrame, dest_csv: str | None):
        """Écrit un DataFrame en CSV si dest_csv est fourni."""
        if not dest_csv:
            return
        p = Path(dest_csv).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        self._log("ok", f"CSV écrit → {p}")

    def _save_json(self, obj, dest_json: str | None):
        """Écrit un objet JSON si dest_json est fourni."""
        if not dest_json:
            return
        p = Path(dest_json).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        self._log("ok", f"JSON écrit → {p}")

    def _get(self,
             path,
             params=None,
             max_retries=3,
             cooldown=1.5,
             backoff_5xx=15.0,
             req_index: int | None = None,
             req_total: int | None = None):
        """GET authentifié avec retries (429/5xx) et refresh auto sur 401."""
        for attempt in range(1, max_retries + 1):
            token = self._ensure_token()
            url = f"{self.base_url}/{path.lstrip('/')}"
            hdrs = {"Authorization": f"Bearer {token}"}
            try:
                detail_parts = []
                if params is not None:
                    detail_parts.append(str(params))
                if req_index is not None:
                    if req_total is not None:
                        detail_parts.append(f"[req {req_index}/{req_total}]")
                    else:
                        detail_parts.append(f"[req {req_index}]")
                if max_retries > 1 and attempt > 1:
                    detail_parts.append(f"(retry {attempt}/{max_retries})")
                detail = " ".join(detail_parts) if detail_parts else ""
                self._log("info", f"GET {url} {detail}".rstrip())
                r = requests.get(url, headers=hdrs, params=params or {}, timeout=240)
                self._log("info", f"⇐ {r.status_code} {r.reason} ({len(r.content or b'')} bytes)")
                if r.status_code == 401 and attempt < max_retries:
                    self._log("warn", "401, refresh token")
                    self._refresh_id_token()
                    time.sleep(cooldown)
                    continue
                if r.status_code in (500,501,502,503,504,429) and attempt < max_retries:
                    wait = backoff_5xx if r.status_code != 429 else cooldown
                    self._log("warn", f"{r.status_code}, retry in {wait}s")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r
            except requests.RequestException as e:
                if attempt >= max_retries:
                    self._log("bad", f"ABORT {e}")
                    raise
                self._log("warn", f"Exception {e}, retry in {cooldown}s")
                time.sleep(cooldown)
        raise RuntimeError("GET a échoué après retries")

    @staticmethod
    def _parse_tabular_payload(content, content_type):
        """Parse un payload tabulaire renvoyé par l'API (JSON ou CSV).
        Retourne une liste de lignes homogènes : [timestamp, float...].
        Cas attendu:
        - JSON: [[ts, v1, v2,...], [ts, v1, v2,...], ...]
        - CSV:  ligne par échantillon, timestamp en col[0], nombres après."""
        rows = []
        ct = (content_type or "").lower()
        is_json_like = "json" in ct or (content and content[:1] in (b"[", b"{"))

        if is_json_like:
            data = json.loads(content)
            if not data:
                return rows
            if isinstance(data, list) and data and isinstance(data[0], list):
                for arr in data:
                    ts = arr[0]
                    nums = [float(x) for x in arr[1:]]
                    rows.append([ts, *nums])
                return rows
            if isinstance(data, list):
                for item in data:
                    rows.append(item if isinstance(item, list) else [item])
                return rows
            rows.append([data])
            return rows

        text = content.decode("utf-8", "replace").splitlines()
        import csv as _csv
        rd = _csv.reader(text)
        first = next(rd, None)
        has_header = first and any("time" in str(x).lower() for x in first)
        if not has_header and first:
            ts = first[0]
            nums = [float(x) for x in first[1:]]
            rows.append([ts, *nums])
        for line in rd:
            ts = line[0]
            nums = [float(x) for x in line[1:]]
            rows.append([ts, *nums])
        return rows

    def _range_fetch_loop(self, route, start_ts, end_ts, max_points):
        """Boucle de fenêtrage temporel (par pas de max_points secondes).
        Renvoie la concaténation de toutes les lignes parsées."""
        sdt = self._to_utc_dt(start_ts)
        edt = self._to_utc_dt(end_ts)
        cur = sdt
        out_rows = []
        if edt < sdt:
            raise ValueError("end_ts doit être >= start_ts")
        span_seconds = int((edt - sdt).total_seconds()) + 1
        req_total = max(1, (span_seconds + max_points - 1) // max_points)
        if self.verbose_level == 1 and tqdm is None:
            raise RuntimeError("tqdm est requis pour verbose=1. Installez 'tqdm'.")
        progress = None
        if self.verbose_level == 1:
            progress = tqdm(total=req_total, desc=f"{route}", unit="req", leave=False)
        req_index = 0
        try:
            while cur <= edt:
                stop = min(edt, cur + timedelta(seconds=max_points - 1))
                params = {
                    "start_timestamp": self._isoz(cur),
                    "end_timestamp": self._isoz(stop),
                }
                req_index += 1
                r = self._get(
                    route,
                    params=params,
                    req_index=req_index,
                    req_total=req_total,
                )
                if progress is not None:
                    progress.update(1)
                chunk = self._parse_tabular_payload(
                    r.content,
                    r.headers.get("Content-Type", ""),
                )
                if not chunk:
                    self._log("warn", f"Tranche vide {cur.isoformat()} → {stop.isoformat()}")
                    cur = stop + timedelta(seconds=1)
                    continue
                out_rows.extend(chunk)
                last_ts = self._to_utc_dt(chunk[-1][0])
                self._log("ok", f"+{len(chunk)} lignes jusqu'à {last_ts.isoformat()}")
                cur = stop + timedelta(seconds=1)
        finally:
            if progress is not None:
                progress.close()
        return out_rows

    def _rows_to_df(self, rows, columns):
        """Transforme une liste de lignes en DataFrame trié par timestamp.
        columns[0] est le timestamp."""
        df = pd.DataFrame(rows, columns=columns)
        ts_col = columns[0]
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
        df = df.sort_values(ts_col).reset_index(drop=True)
        return df

    def _finalize_output(self, df, output, dest_path):
        """Prépare le retour demandé:
        - 'pandas' : retourne le DataFrame
        - 'csv'    : écrit CSV dest_path, retourne dest_path
        - 'json'   : écrit JSON dest_path (records), retourne la liste de dicts."""
        if output == "pandas":
            return df
        if output == "csv":
            self._save_csv(df, dest_path)
            return dest_path
        if output == "json":
            df_json = df.copy()
            for col in df_json.columns:
                if is_datetime64_any_dtype(df_json[col]):
                    series = df_json[col]
                    if getattr(series.dt, "tz", None) is None:
                        series = series.dt.tz_localize(timezone.utc)
                    else:
                        series = series.dt.tz_convert(timezone.utc)
                    df_json[col] = series.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            data_out = df_json.to_dict(orient="records")
            self._save_json(data_out, dest_path)
            return data_out
        raise ValueError("output doit être 'pandas', 'csv' ou 'json'")

    def _validate_exchange_pair(self, exchange, pair):
        """Vérifie que l'exchange et la paire sont autorisés et retourne les identifiants normalisés."""
        exchange_key = str(exchange).lower()
        if exchange_key not in EXCHANGES_CONFIG:
            allowed = ", ".join(sorted(EXCHANGES_CONFIG))
            raise ValueError(f"exchange '{exchange}' non supporté. Exchanges autorisés: {allowed}")
        pair_key = str(pair).upper()
        symbols = EXCHANGES_CONFIG[exchange_key]["symbols"]
        if pair_key not in symbols:
            allowed_pairs = ", ".join(sorted(symbols))
            raise ValueError(
                f"paire '{pair}' non supportée pour l'exchange '{exchange}'. "
                f"Paires autorisées: {allowed_pairs}"
            )
        return exchange_key, pair_key, symbols[pair_key]

    def _validate_model(self, exchange_key, pair_key, model_name):
        """Vérifie que le modèle est autorisé pour l'exchange et la paire donnés."""
        models = EXCHANGES_CONFIG[exchange_key]["symbols"][pair_key]["models"]
        if model_name not in models:
            allowed_models = ", ".join(sorted(models))
            raise ValueError(
                f"modèle '{model_name}' non supporté pour {exchange_key}/{pair_key}. "
                f"Modèles autorisés: {allowed_models}"
            )
        return model_name

    def get_ohlcv(self,
                  exchange,
                  pair,
                  start_ts,
                  end_ts,
                  max_points=15000,
                  output="pandas",
                  dest_path: str | None = None):
        """Récupère l'OHLCV (bougies 1s) entre start_ts et end_ts.
        Les données sont fenêtrées par blocs de max_points secondes. Le retour
        API est un tableau style [[timestamp, open, high, low, close, volume],...].
        Colonnes finales:
        ['open_timestamp','open','high','low','close','volume'].
        output: 'pandas' | 'csv' | 'json'.
        dest_path: chemin de sortie si csv/json."""
        exchange_key, pair_key, _ = self._validate_exchange_pair(exchange, pair)
        rows = self._range_fetch_loop(
            f"ohlcv/{exchange_key}/{pair_key}",
            start_ts,
            end_ts,
            max_points,
        )
        df = self._rows_to_df(
            rows,
            ["open_timestamp", "open", "high", "low", "close", "volume"],
        )
        return self._finalize_output(df, output, dest_path)

    def get_predictions(self,
                        exchange,
                        pair,
                        model_name,
                        start_ts,
                        end_ts,
                        max_points=15000,
                        output="pandas",
                        dest_path: str | None = None):
        """Récupère les prédictions d'un modèle entre start_ts et end_ts.
        Même logique que l'OHLCV: on découpe par max_points secondes puis on
        concatène. Le retour API est supposé du type
        [[timestamp, probability], ...].
        Colonnes finales:
        ['timestamp','probability'].
        output: 'pandas' | 'csv' | 'json'.
        dest_path: chemin de sortie si csv/json."""
        exchange_key, pair_key, _ = self._validate_exchange_pair(exchange, pair)
        model_name_str = str(model_name)
        self._validate_model(exchange_key, pair_key, model_name_str)
        rows = self._range_fetch_loop(
            f"predictions/{exchange_key}/{pair_key}/{model_name_str}",
            start_ts,
            end_ts,
            max_points,
        )
        df = self._rows_to_df(
            rows,
            ["timestamp", "probability"],
        )
        return self._finalize_output(df, output, dest_path)
