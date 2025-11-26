# Adaptive Neutral-Carry Alpha (CriptoML)

Este repositorio aloja un pipeline de research cuantitativo para construir, backtestear y validar estrategias cross-sectional en el mercado de criptomonedas. El foco está en señales de momentum/carry/on-chain y en una evolución iterativa (V0–V7) que agrega complejidad sin perder trazabilidad.

## Roadmap de versiones
- **V0–V1**: Infraestructura de datos diaria, sin sesgo de supervivencia, motores básicos de backtest y señales técnicas de precio/volumen.
- **V2–V3**: Integración de carry (funding) y factores on-chain.
- **V4–V5**: Portafolios market-neutral con control de beta y neutralización de características.
- **V6–V7**: Adaptación a regímenes de mercado (reglas + HMM) y validación institucional con CPCV y Deflated Sharpe.

Cada versión se documenta mediante configs (`configs/strategy`) y se "congela" con un tag de git (`v0`, `v1`, ...). Los resultados de backtests viven en `experiments/` para auditar el workflow completo.

## API Keys / Secrets
Guarda credenciales sensibles fuera del repositorio. Para CoinMarketCap (tier gratuito funciona para el pre-filtro) copia `.env.example` a `.env` y exporta la variable `CMC_KEY` (acepta también `CMC_PRO_API_KEY`):

```bash
cp .env.example .env
echo "CMC_KEY=tu_token" >> .env
```

Los pipelines que usan `CoinMarketCapClient` leerán la clave desde esa variable de entorno automáticamente (también puedes pasarla vía `--cmc-key` en los scripts ETL).

## Estructura del repo
```
.
├── configs/            # YAMLs de estrategia, datos y motor de backtest
├── data/               # Datos locales (raw/processed/external) - ignorados en git
├── experiments/        # Resultados de backtests (logs, curvas, métricas)
├── notebooks/          # Exploración y EDA
├── scripts/            # Entry points (run_backtest, ingesta, etc.)
└── src/crypto_alpha/   # Código reusable (data, features, modelos, backtest, utils)
```

## V0 – Baseline Momentum Daily
- **Universo**: top 40 por volumen USD, excluyendo stablecoins/wrapped, rebalance mensual.
- **Datos**: OHLCV diarios (Binance o fuente equivalente) desde 2017.
- **Señal**: momentum absoluto a 30 días (retorno acumulado).
- **Portafolio**: rebalance semanal, top 20% rankeado por la señal, pesos `inverse_vol` (σ 30d). 
- **Costos**: comisión fija 5 bps, sin slippage.
- **Backtest**: capital inicial USD 10k.

El archivo `configs/strategy/v0_baseline.yaml` captura estos parámetros. Ejecutar `python scripts/run_backtest.py --strategy v0_baseline` (una vez implementada la lógica) reproducirá el experimento y guardará resultados en `experiments/v0_baseline_*`.

## V1 – Ridge cross-sectional (perps, PIT)
- **Datos/Universo**: OHLCV diarios de perps Binance, universos PIT mensuales en `data/processed/universe/v0b/` (top 30–40 por liquidez; sin sesgo de supervivencia).  
- **Features/target**: `scripts/etl/binance_futures.py features-v1 …` genera `features.parquet` con retornos 1/3/7/30d, vol 7/30d, SMA gaps, RSI‑14, ratios de volumen, proxies de liquidez y `forward_return_5d` / `target_excess_return_5d`.  
- **Modelo**: Ridge rolling (~2 años ventana), `scripts/train_v1.py --features-file ... --output-file ... --save-coefs ...` → predicciones PIT en `data/processed/predictions/v1/ridge_predictions.parquet` y coeficientes en `ridge_coefs.parquet`.  
- **Evaluación señal**: `scripts/evaluate_predictions.py` calcula IC/RankIC y compara vs momentum (ret_7d/30d). En la muestra actual, RankIC ≈ 0.076 (t‑stat > 4), indicando edge cross-sectional.  
- **Estrategias**:
  - **Long-only**: `configs/strategy/v1_ridge.yaml` (overlay proporcional configurable). Aun con overlay, Sharpe bajo; sirve como smoke test.  
  - **Long-short**: `configs/strategy/v1_ridge_ls_conservative.yaml` (gross=1.0) y `..._aggro.yaml` (gross=1.5). Sin overlay. Resultado base (gross=1.0): Sharpe ~0.51, vol ~0.21, MaxDD ~‑0.26 en 2020‑2025.  
- **Backtests**: `python scripts/run_backtest.py --strategy v1_ridge_ls_conservative` (o `..._aggro`).  
- **Plot**: `scripts/plot_equity.py --experiment ... --btc-parquet data/processed/binance/ohlcv_1d.parquet` para comparar V0/V1/BTC.  
- **Análisis de coeficientes**: `scripts/analyze_ridge_coefs.py` genera `experiments/v1_eval/coefs/` con stats y boxplot interactivo.

### Baseline V1 (LS) elegido
- **Baseline oficial**: semanal, gross=1.0, inverse-vol, sin caps ni overlay.  
  - 2020-22: Sharpe 0.86, vol 0.11, MaxDD -0.07  
  - 2023-24: Sharpe 0.58, vol 0.27, MaxDD -0.18  
  - Señal mantiene RankIC/IC positivos en ambos tramos.
- **Config agresiva / experimento**: quincenal (2W), gross=1.0, inverse-vol.  
  - 2020-22: Sharpe 0.25, vol 0.14, MaxDD -0.18 (peor que weekly)  
  - 2023-24: Sharpe 1.97, vol 0.37, MaxDD -0.18 (excelente en régimen reciente)  
  - Útil como candidato a overlay por régimen, no como baseline global.

## Herramientas clave
- `run_grid.py`: barrido de parámetros de overlay (target_vol, min_scale, dd_proportionality, cooldown, etc.) usando overrides en caliente.
- `evaluate_predictions.py`: IC/RankIC y comparación top‑K contra señales baseline.
- `run_backtest.py --override key=value`: inyecta cambios sin editar YAML (útil para experimentar con riesgo, leverage, pesos).

## Estado actual (nov 2025)
- Señal V1 validada estadísticamente (RankIC positivo).  
- Monetización long-only limitada: overlay no logra Sharpe atractivo.  
- Portafolio long-short (gross 1.0–1.5) con Sharpe ~0.5 y DD moderado: base recomendada para avanzar a V2.  
- Próximo foco: refinar overlay ligero si se requiere, añadir control de beta/funding y seguir roadmap hacia carry (V2).
