# Roadmap Técnico – Adaptive Neutral-Carry Alpha

**Nota:** Este roadmap es vivo y puede cambiar cuando aparezcan hallazgos nuevos. Sirve como brújula para priorizar trabajo y documentar qué supone cada versión (V0–V7).

## 0. Filosofía General

La estrategia crece en iteraciones cortas. Cada versión es una compuerta: no se avanza hasta comprobar que los datos están limpios, el backtest es razonable y la hipótesis de alpha tiene evidencia out-of-sample. La idea es validar primero el laboratorio (datos + motor), luego las señales, más tarde la construcción de portafolios y finalmente la adaptación a regímenes y validación institucional.

---

## Fase I – Infraestructura y Señales Técnicas (V0a, V0b, V1)

### V0a – Baseline rápido (spot)
- **Objetivo:** Sanidad del pipeline con datos spot diarios y momentum 30d.
- **Datos:** OHLCV spot diarios (Binance, yfinance, etc.).
- **Universo:** Top 30–40 por market cap + volumen promedio 30d, excluye stablecoins/wrapped, reconstitución mensual.
- **Señal:** `mom30_{i,t} = log(P_{i,t}) - log(P_{i,t-30})`.
- **Estrategia:** Rebalance semanal, long-only top 20% (equal weight o inverse vol).
- **Rol:** Confirmar ingestión, motor y métricas básicas (Sharpe, MaxDD, equity curve).

### V0b – Universo realista en perps (point-in-time)
- **Objetivo:** Usar el universo operativo real (perps) reconstruido point-in-time.
- **Datos:** OHLCV diarios de futuros perpetuos (Binance/Bybit). Open interest y volumen para filtros.
- **Universo:** Top 30–40 contratos por open interest/volumen con filtros de liquidez; excluir stablecoins y activos ilíquidos. Reconstitución mensual.
- **Señal y reglas:** Igual que V0a (momentum 30d, rebalance semanal, long-only) pero sobre el universo de perps.
- **Rol:** Cerrar la fase de infraestructura eliminando sesgos de supervivencia.

### V1 – Modelo técnico cross-sectional (excess return)
- **Objetivo:** Pasar de una señal a un modelo que prediga retorno en exceso.
- **Target:** `y_i = r_i(5d) - median_{j in U_t} r_j(5d)` (retornos a 5d menos la mediana cross-sectional).
- **Features:** Retornos `1d/3d/7d/30d`, volatilidades `7d/30d`, ratios de volumen, distancias a SMAs 10/30/60, RSI 14, etc., todos como z-score por fecha.
- **Modelo:** Ridge Regression con ventana rolling (~2 años train → próximo bloque test).
- **PnL:** Introducir costo de funding aproximado para reflejar perps.
- **Estrategia:** Rebalance semanal, long-only top 20% según predicción, pesos inverse vol. Evaluar IC/RankIC y estabilidad vs V0b.

---

## Fase II – Factores adicionales (V2, V3)

### V2 – Carry (funding) como factor estructural
- **Datos:** Funding rate por activo (cada 8h). Guardar acumulados.
- **Features:** Funding 1d/3d/7d, anualizado, z-scores cross-sectional. Funding real pasa a ser parte del PnL.
- **Estrategias:** Carry puro (long donde el funding es negativo) y modelo técnico+carry con features extendidos.
- **Meta:** Demostrar que el carry añade alpha neto.

### V3 – Factor on-chain
- **Datos:** NVT, MVRV, direcciones activas, volumen on-chain, etc. con lag T+1.
- **Features:** Cambios porcentuales 7d/30d, ratios `mcap/active_addresses`, `mcap/tx_volume`, etc.
- **Uso:** Factor value on-chain aislado y luego integrar en el modelo con momentum+carry.
- **Meta:** Que on-chain mejore Sharpe o drawdowns sin introducir demasiado ruido.

---

## Fase III – Portafolios Neutralizados (V4, V5)

### V4 – Long-short y control de beta
- **Señal:** Combinación `score = w_mom*z_mom + w_carry*z_carry + w_value*z_value` o salida del modelo multifactores.
- **Portafolio:** Long top X%, short bottom X%, pesos inverse vol u optimización convexa (CVXPY) con restricciones de beta neta ≈ 0, leverage máximo y límites por activo.
- **Beta:** Estimar beta vs BTC/índice con ventana rolling (~60d) y controlar exposición.

### V5 – Neutralización de features
- **Objetivo:** Quitar componentes ligados a beta, market cap, sectores, etc. de las señales.
- **Método:** Proyectar señales en el complemento ortogonal de la matriz de riesgos `R`. Formalmente `F_neut = F - R (R^+ F)`.
- **Resultado:** Señales limpias de factores de riesgo obvios antes de armar portafolios.

---

## Fase IV – Regímenes y Validación (V6, V7)

### V6 – Adaptación heurística a regímenes
- **Reglas:** Filtro de volatilidad BTC (si vol30 > p90, bajar exposición), control de drawdown propio (si MaxDD supera umbral, reducir tamaño o pausar).
- **Meta:** Modulación dinámica del riesgo sin cambiar la generación de señales.

### V7 – Full power (HMM + impacto + CPCV/DSR)
- **HMM:** Modelos de regímenes sobre retornos/vol de BTC; adaptar pesos y apalancamiento según estado.
- **Frecuencia:** Migrar a 1H si es viable, incluir impacto y fricción con una ley tipo raíz cuadrada (`impact ≈ Y * sigma * sqrt(Q/V)`).
- **Validación:** Combinatorial Purged Cross-Validation para splits robustos y Deflated Sharpe Ratio para corregir por data-mining.

---

## Especificación Técnica V0 (detalle)

### 1. Universo y datos
- **Frecuencia:** Diaria (1D) a cierre UTC.
- **Datos faltantes:** Se deja vacío (nunca se rellena). Si no hay barra, el activo no participa ese día.
- **V0a:** OHLCV spot; universo mensual top 30–40 por market cap + volumen 30d, excluye stablecoins/wrapped. Sirve como smoke test.
- **V0b:** OHLCV de perps; universo point-in-time mensual top 30–40 por open interest/volumen 30d con filtros de liquidez. Activos muertos deben aparecer cuando existían.

### 2. Transformaciones y señal
- **Log-precio:** `p_{i,t} = log(Close_{i,t})`.
- **Retorno diario:** `r_{i,t} = p_{i,t} - p_{i,t-1}` (sirve para volatilidad).
- **Momentum 30d:** `mom30_{i,t} = p_{i,t} - p_{i,t-30}` = `log(Close_{i,t} / Close_{i,t-30})`. Requiere 30 datos válidos.
- **Elegibilidad:** El activo debe pertenecer a `U_t`, tener historial suficiente y sin gaps en la ventana.

### 3. Reglas de trading
- **Rebalance:** Semanal (definir calendario, ej. lunes usando cierre del domingo).
- **Ranking:** Ordenar activos del universo válido por `mom30` descendente.
- **Selección:** Top `q%` (p. ej. 20%). Si hay 40 válidos → top 8.
- **Pesos:**
  - Equal weight: `w = 1/N`.
  - Inverse vol: usar `sigma_{i,t}` como raíz de la media de `r^2` 30d, asignar `w_i ∝ 1/sigma_i` y normalizar.
- **Restricciones:** Long-only, suma de pesos = 1, sin apalancamiento ni shorts.

### 4. Backtest y PnL
- **Retornos entre rebalanceos:** `R_price = Close_{t+1} / Close_t - 1` acumulado entre fechas de rebalance. Retorno de portafolio = suma de `w_i * R_price_i`.
- **Costos:** comisión fija `c_fee ≈ 0.0005` (0.05%) aplicada al turnover `sum |w_new - w_old|`. En V0 no se modela funding real ni impacto, sólo esta fricción.
- **Actualización de capital:** `V_{t+1} = V_t * (1 + R_net)` con `R_net = R_gross - cost_fee`.

### 5. Métricas
- Sharpe y volatilidad anualizada.
- Rentabilidad anualizada y Max Drawdown.
- % semanas positivas.
- Benchmarks: buy & hold BTC y buy & hold equal-weight del universo (rebalance mensual o trimestral).
- Opcional: IC/RankIC entre `mom30` y retorno futuro 5d/7d.

### 6. Trampas comunes
- **Lookahead:** Nunca usar datos posteriores al rebalance para calcular señales o universos.
- **Sesgo de supervivencia:** En V0b el universo debe reconstruirse point-in-time (no usar lista actual para el pasado).
- **Listados recientes:** Exigir mínimo 30–60 días antes de dejar entrar un activo.
- **Datos incompletos:** Si hay gaps dentro de la ventana de 30 días, el activo queda fuera en esa fecha.

---

## Uso interno
- Este archivo vive en `docs/roadmap.md` y es la referencia viva del plan. Actualizarlo cuando se cierre una versión (tag + config) o cambien prioridades.
