# V2 · Funding-aware backtest + carry signal (V2a → V2b)

## Objetivo
Separar claramente:
- **V2a:** introducir el coste/ingreso real de funding en el PnL manteniendo la señal V1 (momento Ridge).
- **V2b:** añadir carry como componente de señal/overlay sobre el PnL ya funding-aware.

## Datos y ETL
- Fuente: Binance `/fapi/v1/fundingRate` (8h). Se agregan 3 periodos diarios.
- Formato: `funding_rate_8h` y `funding_rate_1d` en **fracción** (0.0005 = 0.05%).
- Convención de signo (Binance): `funding_rate > 0` ⇒ **long paga, short cobra**.
- PIT: sólo datos ≤ fecha; sin forward-fill.
- Parquet `funding_1d.parquet`: `date, symbol, exchange, funding_rate_8h, funding_rate_1d, collected_at`.
- CLI ETL funding:  
  `python scripts/etl/binance_futures.py funding --start 2019-11-01 --end 2025-07-01 --output-file data/processed/binance/funding_1d.parquet`

## PnL con funding (backtester)
- PnL_total = PnL_precio + PnL_funding.
- Por símbolo/día (respetando signo Binance):  
  **PnL_fundingᵢ,t = sign(positionᵢ,t) · |notionalᵢ,t| · ( − funding_rate_1dᵢ,t )**  
  - Long & funding_rate>0 → pagas (PnL_funding<0).  
  - Short & funding_rate>0 → cobras (PnL_funding>0).
- Comisiones: igual que V1; funding siempre se aplica (se use o no como feature).

## Features V2 (para V2b)
- Reutiliza features V1 (mom/liquidez).
- Añade pocas métricas de funding/carry:
  - `funding_1d`, `funding_7d_mean`, `funding_30d_mean`
  - `carry_score = funding_30d_mean * 365` (anualizado en fracción)
  - opcional `carry_rank_cs` (percentil cross-sectional)
- **No look-ahead**: todas las métricas de funding para fecha t usan datos ≤ t−1.
- CLI:  
  `python scripts/etl/binance_futures.py features-v2 --ohlcv-file data/processed/binance/ohlcv_1d.parquet --funding-file data/processed/binance/funding_1d.parquet --universe-dir data/processed/universe/v0b --output-file data/processed/features/v2/features.parquet --start 2019-11-01 --end 2025-07-01 --rebalance-freq 1W`

## Estrategias
- **V2a (baseline realista)**  
  - Señal: Ridge V1 (momento).  
  - Portfolio: LS neutral, inverse-vol, gross=1.0, sin caps; overlay de vol target opcional.  
  - Cambia sólo el PnL (suma funding).

- **V2b (carry en la señal)**  
  - Opción A (recomendada primero): mix lineal  
    - z_mom y z_carry se calculan **cross-sectional por fecha** (z-score dentro del universo en t).  
    - `signal = w_mom * z_mom + w_carry * z_carry`, grid corto (0.8/0.2, 0.7/0.3, 0.6/0.4).  
  - Opción B: Ridge con features_v2 (mom + carry + liquidez); monitor coeficientes para que carry no domine.  
  - Portfolio: igual que V2a; funding siempre aplicado.

## Validación
- Subperíodos: 2020–22 y 2023–24.
- Métricas: Sharpe, Vol, MaxDD, Turnover, IC/RankIC global y por componente (mom solo, carry solo, mix).
- Ablations: V2a (mom+funding coste), carry-only, mix.
- Chequeo leakage: agregaciones y medias sólo con datos anteriores; sin ffill.

## Riesgos y mitigación
- Rate limits 429: backoff/sleep.
- Gaps funding: no ffill; log de símbolos/fechas faltantes.
- Dimensión: limitar a 3–4 features de funding para evitar overfitting inicial.

## Comandos esperados
- Funding ETL:  
  `python scripts/etl/binance_futures.py funding --start 2019-11-01 --end 2025-07-01 --output-file data/processed/binance/funding_1d.parquet`
- Features V2:  
  `python scripts/etl/binance_futures.py features-v2 --ohlcv-file data/processed/binance/ohlcv_1d.parquet --funding-file data/processed/binance/funding_1d.parquet --universe-dir data/processed/universe/v0b --output-file data/processed/features/v2/features.parquet --start 2019-11-01 --end 2025-07-01 --rebalance-freq 1W`
- Backtest V2a:  
  `python scripts/run_backtest.py --strategy v2a_momentum_funding`
- Backtest V2b (mix lineal):  
  `python scripts/run_backtest.py --strategy v2b_carry_mix`
