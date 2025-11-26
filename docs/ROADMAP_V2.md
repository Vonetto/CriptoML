# ROADMAP V2 — estado y líneas experimentales

## Baseline (oficial)
- **V2a/V2b (mom-only funding-aware)**  
  - Señal: momentum (Ridge V1/V2 mom), LS neutral, inverse-vol, gross=1.0.  
  - PnL: precio + funding real; comisiones 4 bps (probado hasta 8–12 bps con caída leve).  
  - Métricas (2019-11 → 2025-07): Sharpe ~0.74, Vol ~0.39, MaxDD ~-0.38.  
  - Subperíodos: 2020–22 Sharpe ~0.92; 2023–24 Sharpe ~0.67.  
  - Configs: `v2a_momentum_funding.yaml` (y `v2b_mom_only` análogo).

## Experimental (no baseline)
- **V2c carry contrarian (−z_carry)** — `v2c_carry_contra_exp`  
  - Full: Sharpe ~1.38, Vol 0.31, MaxDD ~-0.29;  
    2020–22 Sharpe ~1.80; 2023–24 Sharpe ~0.73.  
  - Advertencias: IC/RankIC medio débil (p≈0.26 vs permutaciones), rolling Sharpe mediana ~0.0, régimen-dependiente.  
  - Mezclas lineales 50/50 con mom: Sharpe ~0.83 (peor que carry solo y que mom), DD mayor.  
  - Filtros “no long high funding” probados: degradan Sharpe; no adoptados.

## Conclusión V2
- Funding debe estar siempre en el PnL (logrado).  
- Señal core para producción: **mom-only con funding en PnL**.  
- Carry contrarian se mantiene como módulo de research (V2c), no en el baseline.

## Ideas futuras (V3+)
- Ensemble por books separados (mom y carry_contra) con gestión de riesgo por book.  
- Scheduler por régimen (vol BTC / estados HMM) que elija entre mom, carry, o mix.  
- Filtros/penalizaciones basadas en funding pero aplicados en el portfolio builder, no vía z-score.  
- Evaluar más fuentes de carry/funding (otros exchanges) y robustez multifuente.
