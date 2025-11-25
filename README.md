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

## Próximos pasos sugeridos
1. Implementar loaders reales en `src/crypto_alpha/data/` (descarga/parquet/limpieza).
2. Completar el motor de backtest en `src/crypto_alpha/backtest/engine.py` y el script `scripts/run_backtest.py`.
3. Añadir notebooks de sanity checks (`notebooks/01_v0_momentum...`).
4. Etiquetar la versión cuando el pipeline de V0 sea reproducible.
