# V1 subperíodos: weekly vs 2W (LS, gross=1.0)

Source runs: Ridge LS sin overlay, inverse-vol, sin caps. Datos 2019-11 a 2025-07 (PIT universes v0b, features_v1).

## Métricas financieras

| Frecuencia | Tramo | Sharpe | Vol anual | MaxDD |
|-----------|-------|--------|-----------|-------|
| 1W (baseline) | 2020-2022 | 0.86 | 0.19 | -0.23 |
| 1W (baseline) | 2023-2024 | 0.585 | 0.21 | -0.27 |
| 2W (experimento) | 2020-2022 | 0.25 | 0.15 | -0.31 |
| 2W (experimento) | 2023-2024 | 1.97 | 0.33 | -0.35 |

## Señal (IC / RankIC cross-sectional)

| Frecuencia | Tramo | RankIC medio | t-stat |
|-----------|-------|--------------|--------|
| 1W | 2020-2022 | 0.084 | 2.87 |
| 1W | 2023-2024 | 0.0466 | 2.15 |
| 2W | 2020-2022 | (no calc.) | — |
| 2W | 2023-2024 | (no calc.) | — |

Notas:
- La señal mantiene edge positivo en ambos tramos; el 2W explota en 2023-24 pero se degrada fuerte en 2020-22 → depende de régimen.
- No se aplicaron caps ni overlays; costos básicos incluidos.

## Turnover (indicativo)

- 1W: mayor turnover, pero aceptable con comisiones asumidas en el backtest.
- 2W: menor turnover; podría justificar usar 2W en regímenes de alta señal/menor ruido.

## Conclusiones rápidas

- Baseline oficial: **1W LS gross=1.0** (robusta en ambos tramos).
- 2W queda como **config “agresiva por régimen”**: excelente en 2023-24, floja en 2020-22; candidata a scheduler en V3+.
- Próximo paso V2: integrar funding/carry; V3+: scheduler de rebalance por régimen (vol BTC/HMM) para alternar 1W/2W sin overfitting.
