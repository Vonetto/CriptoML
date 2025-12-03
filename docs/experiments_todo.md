# Experiments To-Do (branch: experiments)

Mantén este archivo como checklist vivo de ideas/experimentos que podemos probar sin miedo al overfitting, pero siempre contrastando contra el baseline **momentum funding-aware + slippage_alpha=0.001**.

## Data quality / Liquidez
- [ ] Implementar detector de “prints inside spread sin resting order” (wash/fake volume) y combinarlo con `vol_per_trade`/`min_trades_filter`; regenerar universos V0b y medir impacto.
- [ ] Añadir filtro de outliers de volumen por z-score (p99/p995) en `build_universe_v0b` y comparar universos.

## Costos / Slippage
- [ ] Calibrar `slippage_alpha` por símbolo usando spreads intradía (order book) y probar estrés (0.001 vs 0.005 vs 0.01) sobre baseline.
- [ ] Probar modelo de impacto simplificado: cost ∝ sigma * sqrt(volume_fraction).
- [ ] Comparar trades simulados vs supuestos de slippage por bucket de volatilidad (inspirado en discusiones de r/algotrading): ajustar alpha si es sistemáticamente optimista/pesimista.
- [ ] Slippage por buckets de volatilidad (nuevo): usar pct_bins y multiplicadores; medir Sharpe/MDD vs slippage fijo.
- [ ] Slippage basado en spread real: en curso recolección (cron cada 10 min → data/raw/binance_futures/spreads.parquet). Luego agregar mediana/p90 diaria por símbolo y re-correr baseline/prop con alpha ∝ spread_pct.
- [x] Impacto simplificado cost = k * sigma_30d * sqrt(turnover):
  - Probado k=0.5/1.0/2.0 (sin slippage_alpha): destruye la estrategia (Sharpe -2.5 / -5.3 / -9.4, MDD ≈ -1). No viable con esos parámetros.
  - Pendiente solo si se reintenta con k << 0.5 y/o calibración con spreads reales.

## Overlays de riesgo
- [x] Confidence scaling suave solo en longs (|mom21|/vol, 0.9–1.1, clip 2.5, clamp DD 0.9–1.05; shorts sin cambio).
  - Config: `v2a_momentum_prop_tight_21d_cap05_beta_conf_longonly`.
  - Resultados full (2019-11→2025-07): Sharpe 2.32, AR 0.80, Vol 0.27, MDD -0.090 (mejor DD vs baseline -0.107).
  - Stress costos (8bps + slip 0.002): Sharpe 2.26, MDD -0.091.
  - OOS 2023-01→2025-07: Sharpe 2.32, MDD -0.090.
  - Bootstrap bloques 6w (300 muestras): conf gana a baseline en ~54% Sharpe y ~53% MDD (medianas SR 2.31 vs 2.28; MDD -0.124 vs -0.129).
  - Estado: **promover como baseline experimental** (combina prop_tight + cap5 + beta60 + conf_longonly).
- [ ] Volatility gating simple: usar ATR/σ_30d de BTC para escalar gross (sin IC/metas) y comparar contra baseline.
- [ ] Scheduler 1W/2W con proporción objetivo de estados (p33/p66) y permutación de estados para validar que supera random.
- [x] Overlay de “prop constraints”: simular daily-loss-limit / equity-stop (funded account) y medir impacto en DD/Sharpe vs baseline (idea de r/quantfinance).
  - Hecho (variante funding LS momentum, slippage_alpha=0.001):
    - Configs: `v2a_momentum_prop_tight` (loss 3%, cooloff 2, equity_stop 30%), `v2a_momentum_prop_mid` (loss 4%, cooloff 1, equity_stop 35%).
    - Resultados vs baseline (2019-11→2025-07):
      - Baseline (sin prop): Sharpe 0.55, AR 0.21, Vol 0.38, MDD -0.39.
      - Prop_tight: Sharpe 2.03, AR 0.53, Vol 0.26, MDD -0.17 (subperíodos 2020-22 SR 2.04; 2023-25 SR 2.19).
      - Prop_mid: Sharpe 1.88, AR 0.59, Vol 0.31, MDD -0.17.
    - Robustez:
      - Block bootstrap (bloques 6 rebalance, 300 muestras): observación ~mediana de la distribución (pctl_sr≈0.48 para tight, 0.47 mid; pctl_mdd ~0.47/0.69). Mejora no depende del orden.
    - Pendiente: decidir si se adopta como overlay opcional; mantener baseline sin overlay.

## Señales
- [ ] Carry contrarian: repetir permutaciones/holdouts con slippage_alpha=0.001 y reportar subperíodos; decidir si queda como V2c experimental estable.
  - Hecho: carry_contra (z_carry negativo, slippage_alpha=0.001)
    - Base: Sharpe 1.50, AR 0.465, Vol 0.311, MDD -0.289; 2020-22 SR 2.18, 2023-25 SR 0.74; bootstrap pctl_sr ~0.50, pctl_mdd ~0.53.
    - Ajustes selection_top_pct 0.15 y carry_filter off: sin efecto.
    - Con overlay prop_tight (loss 3%, cooloff 2): Sharpe 1.83, Vol 0.248, MDD -0.202; 2020-22 SR 2.04, 2023-25 SR 1.72, MDD -0.105. Mejora clara en tramo débil, pero sigue experimental (inferior a momentum+prop_tight_21d).
- [ ] Momentum tweaks: probar ventana 21d y 45d; evaluar IC/RankIC y Sharpe vs baseline.
- [ ] Blends ligeros (ej. 70/30 ret_30d/ret_7d) ya fallaron; documentar y no repetir salvo nueva evidencia.
- [ ] Limitar grids de ventanas/params (evitar data snooping): pocas opciones predefinidas + OOS/perm tests al estilo de la crítica al “grid EMA” vista en r/algotrading.
  - Hecho: ventanas probadas 10d/14d/21d/45d (LS funding-aware, slippage_alpha=0.001).
    - 10d: Sharpe -0.18, MDD -0.59 → descartado.
    - 14d: Sharpe 0.48, MDD -0.54 → peor que baseline.
    - 45d: Sharpe 0.07, MDD -0.51 → descartado.
    - 21d: Sharpe 0.81, MDD -0.35; subperíodos 2020-22 SR 0.73, 2023-25 SR 1.02. Mejora vs baseline (30d SR 0.55, MDD -0.39).
    - Combinado con overlay prop:
      - `v2a_momentum_prop_tight_21d`: SR 3.29/1.90/2.65 (2020-21 / 2022-23 / 2024-25H1), MDD min -0.163. Muy superior al baseline y consistente.
      - `v2a_momentum_prop_mid_21d`: SR 3.39/1.04/2.38, MDD min -0.132. También fuerte, pero peor en 2022-23 frente a prop_tight.
    - Candidata: 21d; descartar 10/14/45.
    - Blends mom21/mom7 probados (sin prop): 80/20 y 60/40 suben Sharpe frente a 21d pelado (SR ~0.95 y 1.02; MDD ~-0.28/-0.26). Con overlay prop_tight, todos los blends bajan Sharpe/AR vs prop_tight_21d; la mejora marginal de MDD (-0.143) no compensa. Se descarta blend con prop; mantener señal prop_tight_21d pura.
    - Overlays de volatility gating (step) sobre 21d probados:
       - target_vol 0.30/0.40, max lev 1.2–1.5 → Sharpe ~0.88–0.90, MDD ~-0.29/-0.34. Peor que 21d simple y mucho peor que prop_tight. Descartado.

### Aumentar alpha (nuevos experimentos propuestos)
- [ ] Portfolio tweaks (sin cambiar señal):
- [x] Capping de peso por activo en esquema inverse-vol (p.ej. 5–10%).
- [x] Top/Bottom por percentil dinámico (evitar forzar posiciones cuando el universo efectivo es pequeño).
- [x] Beta-neutralization vs BTC (restar beta estimada en las weights) y comparar Sharpe/MDD/alpha.
- **Estado cap:** cap 5% es el mejor: Sharpe 2.10, MDD -0.121 (full), reduce turnover (~-11%). Cap 7.5/10% también mejoran DD pero Sharpe inferior. Dyn 25/25 empeora → descartar. Stress de costes (8bps+slip 0.002) cap5 mantiene Sharpe 2.07, MDD -0.123.
- **Estado beta-neutral (oficial experimental previo):** cap5 + beta hedge vs BTC (lookback 60d) Sharpe 2.31, MDD -0.107; subperíodos 2020-22 y 2023-25 mejores que baseline y cap; bootstrap: SR mejor 72% vs baseline, 70% vs cap. Lookback 40d/90d: 40d peor; 90d Sharpe 2.27, MDD -0.140.
- **Estado actual (baseline experimental propuesto):** beta60 + cap5 + prop_tight + confidence_longonly (arriba) → Sharpe 2.32, MDD -0.090; robusto a costes/OOS/bootstrap.
- [x] Max-diversification weighting (cov 30d shrinked) en lugar de inverse-vol:
  - Config base: `v2a_momentum_prop_tight_21d_cap05_beta_maxdiv_cov45` (cov 45d).
  - Full: Sharpe 2.38, AR 0.83, Vol 0.271, MDD -0.087.
  - Ajustes probados:
    - Cov 20d → Sharpe 2.27, MDD -0.115 (peor).
    - Caps 5/6/7.5% → Sharpe ~2.37–2.38, MDD -0.087 (cambia poco).
    - Gross 1.2 (cap5) → Sharpe 2.50, AR 1.10, MDD -0.107 (más retorno, más DD).
    - Gross 1.2 + confidence_longonly (cap5) → Sharpe 2.53, AR 1.22, MDD -0.101 (agresivo).
  - Robustez:
    - Defensivo (cov45, cap5, g=1.0): stress Sharpe 2.32 / MDD -0.089; OOS 2023–25 Sharpe 2.63 / MDD -0.087.
    - Agresivo (cov45, cap5, g=1.2 + conf_longonly): stress Sharpe 2.46 / MDD -0.103; OOS 2023–25 Sharpe 2.75 / MDD -0.096.
    - Bootstrap (agresivo vs defensivo): SR mediana 2.49 vs 2.36; agresivo > defensivo en 66% Sharpe; MDD mediana -0.127 vs -0.118 (agresivo peor en DD en 62% de muestras).
  - Elección: agresivo = baseline principal; defensivo = variante low-risk.
- [x] Crowd-penalty (penalizar señal por correlación media 30d, power=1):
  - Config base: `v2_aggressive_crowdpen` (gross 1.2, cap5, conf long-only, max-div).
  - Full: Sharpe 2.66, AR 1.22, Vol 0.320, MDD -0.107 (mejor Sharpe que agresivo original, DD similar).
  - Variantes probadas: power 0.5 (Sharpe 2.60, MDD -0.104), power 1.5 (Sharpe 2.51, MDD -0.174), lb20 (igual), cap4 (no corrido).
  - Estado: overlay opcional prometedor; PCA-neutral descartado (MDD ~-0.21).
- [ ] Señal Ridge extendida (features extra de momentum/liquidez):
  - Añadir avg_volume_quote_7d/30d, persistencia de volumen (días > umbral), skew/kurt de retornos, (más adelante) spread/funding si disponible.
  - Re-entrenar Ridge con validación walk-forward estricta (sin grid grande) y comparar IC/Sharpe frente a señal actual.
- [ ] Meta-labeling ligero / filtro de trades:
  - [x] Clasificador simple (logit) para predecir éxito de la señal momentum en el siguiente rebalance; probar filtro por probabilidad > p.
  - [x] Validar con permutación/holdout para evitar data-snooping.
  - **Estado actual (logit, probs en `meta_prob.parquet`, thresholds 0.40/0.55/0.60):**
    - Con prop_tight_21d_cap05_beta: Sharpe cae a 1.73–2.15 vs 2.31 base, MDD igual (-0.1066), AR cae a 0.42–0.67. N períodos menor (286 vs 293) por filtrado.
    - Conclusión: meta-filter no aporta; queda como opcional/experimental, no promover.
- [ ] Costos realistas (cuando spreads estén listos):
  - Recalibrar slippage_alpha con spreads mediana/p90; probar buckets por volatilidad.
  - Turnover-aware weighting (penalizar señales de baja convicción para reducir churn).
- [ ] Timing / overlays de exposición (retorno corto plazo):
  - [x] Cash overlay por régimen BTC … **Probado: todas las variantes 0.0/0.3/0.5/0.7 empeoran Sharpe/AR; descartado.**
  - [x] Ventana caliente/fría … **Probado (varios grids): mejoras marginales; permutación pctl ~88%, no concluyente; descartado.**
  - [x] Breadth filter … **Probado variantes (umbrales, binario, long-only): todas degradan Sharpe/AR; descartado.**

  
  - **Nuevos (atrevidos):**
    - [ ] Book satélite de reversión corta: mom_5d invertida, gross 0.3–0.5, cap 3%, beta hedge, prop_tight; eval full y OOS 2023–25; posible blend con core.
    - [ ] Book satélite carry+trend entrenable: features mom_21/63, carry_30d, vol, skew/kurt; ElasticNet/Ridge WF trimestral; gross 1.0, cap 5%, beta hedge; evaluar como libro separado y blend 50/50.
    - [ ] Portfolio “risk parity + liquidity”: weights ∝ 1/vol ajustados por liquidity score (vol_30d*num_trades_30d); constraints gross=1, cap=7%, beta hedge, target vol 20%; comparar MDD/Sharpe vs core.

## Crowd / Copy (exploratorio)
- [ ] Evaluar viabilidad de crowd-follow vs crowd-fade: necesitar fuente de rankings de “traders populares” o wallets a copiar; si no hay datos, documentar bloqueo.
- [ ] Si se consigue fuente: construir señal básica (top-N providers) y medir fees/slippage para ver si es monetizable; vigilar martingalas escondidas (aprendido de r/copytrading).

## Meta-labeling / IC gating (experimentales)
- [ ] Dejar meta-label y IC gating marcados como “experimental”; sólo repetir si cambiamos umbrales o señales.

## Regímenes macro/BTC
- [ ] Hacer train/test por corte temporal para cualquier scheduler basado en BTC vol/dd/funding; incluir benchmark “siempre 2W”.
- [ ] Permutación de scheduler manteniendo % de días stress para validar significancia.

## Evaluación y reporting
- [ ] Para cada experimento: guardar Sharpe/Vol/DD + RankIC/IC + rolling Sharpe 1Y + turnover.
- [ ] Documentar resultados fallidos para no re-testear sin motivo.
- [x]  Validaciones robustez para señal 21d + prop_tight:
  - [x] Block bootstrap con overlay prop_tight_21d (bloques 6): pctl_sr~0.53, pctl_mdd~0.30.
  - [x] Stress de costos (comm 8bps, slippage_alpha 0.002): SR sigue alto (tramos 3.21/1.78/2.56), MDD ~-0.17.
  - [x] Placebo overlay (loss_limit 100%): cae a SR ~0.9, confirma que el stop/cooldown es el driver.
  - [x] Turnover mediana ~0.99; rolling SR 52w media 2.56 (min -0.015, max 8.76).
  - [x] Walk-forward simple: train hasta 2022-12 SR 2.06, OOS 2023-25 SR 2.43, MDD -0.163.
 
