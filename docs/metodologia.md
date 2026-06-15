# Metodología: modelos de predicción de resultados de fútbol

> Reporte técnico que consolida la revisión de literatura sobre predicción de
> resultados de selecciones nacionales (foco: Copa del Mundo FIFA) y fundamenta
> el diseño del modelo para el Mundial 2026.
>
> Las afirmaciones de este documento provienen de una revisión con verificación
> adversarial de fuentes (21 afirmaciones confirmadas, 4 refutadas). Las
> refutadas y las salvedades están al final, en [§9](#9-salvedades-y-afirmaciones-refutadas).

## Índice

1. [Panorama: tres familias de métodos](#1-panorama-tres-familias-de-métodos)
2. [Modelos de goles basados en Poisson](#2-modelos-de-goles-basados-en-poisson)
3. [Sistemas de rating de fuerza](#3-sistemas-de-rating-de-fuerza)
4. [El enfoque ganador: Random Forest híbrido](#4-el-enfoque-ganador-random-forest-híbrido)
5. [Simulación del torneo (Monte Carlo)](#5-simulación-del-torneo-monte-carlo)
6. [Evaluación: RPS y benchmark de apuestas](#6-evaluación-rps-y-benchmark-de-apuestas)
7. [Datos de entrenamiento](#7-datos-de-entrenamiento)
8. [Implicaciones para nuestro diseño (Mundial 2026)](#8-implicaciones-para-nuestro-diseño-mundial-2026)
9. [Salvedades y afirmaciones refutadas](#9-salvedades-y-afirmaciones-refutadas)
10. [Referencias](#10-referencias)

---

## 1. Panorama: tres familias de métodos

La literatura converge en tres familias que **se combinan**, no que compiten:

| Familia | Qué modela | Representantes |
|---|---|---|
| **Modelos de goles** | La distribución del marcador `(X, Y)` | Maher (1982), Dixon-Coles (1997), Karlis-Ntzoufras (2003), Baio-Blangiardo (2010) |
| **Ratings de fuerza** | Un escalar de "qué tan bueno" es cada equipo | World Football Elo, ranking FIFA, SPI |
| **ML híbrido** | El resultado/goles a partir de *features* | Groll & Ley et al. (2018–2021): Random Forest + habilidades-Poisson |

El **resultado más robusto y replicado** de toda la revisión: el mejor predictor
no es ninguna familia aislada, sino un **híbrido** donde los parámetros de
habilidad estimados por un modelo de goles tipo Poisson se usan como *feature*
de un Random Forest. Ver [§4](#4-el-enfoque-ganador-random-forest-híbrido).

---

## 2. Modelos de goles basados en Poisson

La idea fundacional: los goles que marca un equipo en un partido siguen una
distribución de **Poisson**, cuya tasa `λ` depende de la fuerza ofensiva del
equipo, la defensiva del rival y la ventaja de local.

### 2.1 Poisson independiente (Maher, 1982)

Sea `X` los goles del local e `Y` los del visitante. El modelo base asume que
son **Poisson independientes**:

$$
X \sim \text{Poisson}(\lambda), \qquad Y \sim \text{Poisson}(\mu)
$$

$$
P(X = x) = \frac{e^{-\lambda}\,\lambda^{x}}{x!}, \qquad x = 0, 1, 2, \dots
$$

con tasas parametrizadas de forma **log-lineal** por equipo:

$$
\log \lambda = \mu_0 + h + \alpha_i - \beta_j
$$
$$
\log \mu = \mu_0 + \alpha_j - \beta_i
$$

donde:
- `α_i` = **fuerza ofensiva** (attack) del equipo `i`,
- `β_j` = **fuerza defensiva** (defence) del equipo `j` (más alta ⇒ concede menos),
- `h` = **ventaja de localía** (un escalar > 0),
- `μ_0` = intercepto global (nivel medio de goles).

Por independencia, la probabilidad de un **marcador exacto** es el producto:

$$
P(X = x,\, Y = y) = \frac{e^{-\lambda}\lambda^{x}}{x!}\cdot\frac{e^{-\mu}\mu^{y}}{y!}
$$

**Identificabilidad:** se impone una restricción (p. ej. `Σ_i α_i = 0`) porque
sumar una constante a todos los ataques y restarla a las defensas deja `λ`
invariante. Maher concluyó que este modelo "describe los marcadores con
razonable precisión", con pequeñas discrepancias sistemáticas — sobre todo que
**subestima los empates**.

### 2.2 Poisson bivariado (Karlis & Ntzoufras, 2003)

El defecto del modelo independiente es que ignora la **correlación** entre los
goles de ambos equipos (Maher la estimó en ≈ 0.2). El Poisson bivariado la
introduce con una construcción de **shock compartido**:

$$
X = W_1 + W_3, \qquad Y = W_2 + W_3, \qquad W_k \sim \text{Poisson}(\lambda_k) \text{ indep.}
$$

El término común `W_3` induce covarianza positiva:

$$
\mathrm{Cov}(X, Y) = \lambda_3 \ge 0
$$

La función de masa conjunta es:

$$
P(X=x, Y=y) = e^{-(\lambda_1+\lambda_2+\lambda_3)} \frac{\lambda_1^{x}}{x!}\frac{\lambda_2^{y}}{y!}
\sum_{k=0}^{\min(x,y)} \binom{x}{k}\binom{y}{k} k!\left(\frac{\lambda_3}{\lambda_1\lambda_2}\right)^{k}
$$

(cuando `λ_3 = 0` se recupera exactamente el Poisson independiente).

**Variante diagonal-inflada:** para corregir la subestimación de empates,
Karlis & Ntzoufras añaden masa extra a la **diagonal** (marcadores `x = y`):

$$
P(X=x, Y=y) =
\begin{cases}
(1-p)\,\text{BP}(x, y; \lambda_1,\lambda_2,\lambda_3) + p\,D(x; \theta) & x = y \\
(1-p)\,\text{BP}(x, y; \lambda_1,\lambda_2,\lambda_3) & x \neq y
\end{cases}
$$

donde `BP` es la bivariada de arriba, `D` una distribución sobre la diagonal y
`p` el peso de inflado. Esto mejora el ajuste de `0-0`, `1-1`, etc. y permite
**sobredispersión**.

### 2.3 Dixon-Coles (1997) — el estándar práctico

Dixon-Coles es la solución más usada al problema de los empates y los marcadores
bajos, con **dos aportes**:

**(a) Corrección de dependencia en marcadores bajos.** Multiplica el producto de
Poisson por un factor `τ` que solo toca los cuatro marcadores `{0,0}, {0,1},
{1,0}, {1,1}`:

$$
P(X=x, Y=y) = \tau_{\lambda,\mu}(x,y)\cdot \frac{e^{-\lambda}\lambda^{x}}{x!}\cdot \frac{e^{-\mu}\mu^{y}}{y!}
$$

$$
\tau_{\lambda,\mu}(x,y) =
\begin{cases}
1 - \lambda\mu\rho & (x,y) = (0,0) \\
1 + \lambda\rho & (x,y) = (0,1) \\
1 + \mu\rho & (x,y) = (1,0) \\
1 - \rho & (x,y) = (1,1) \\
1 & \text{en otro caso}
\end{cases}
$$

El parámetro `ρ` captura la dependencia: con `ρ > 0` se infla la probabilidad de
empates bajos. (Esta es exactamente la función `dc_tau` ya implementada en
[`models/dixon_coles.py`](../src/worldcup/models/dixon_coles.py).)

**(b) Ponderación temporal.** Los partidos antiguos importan menos. En vez de la
verosimilitud ordinaria, se maximiza una **pseudo-verosimilitud ponderada**:

$$
\mathcal{L}(\theta) = \sum_{m=1}^{M} \phi(t_m)\,\ell_m(\theta), \qquad
\phi(t_m) = e^{-\xi\,(t_{\text{hoy}} - t_m)}
$$

donde `ℓ_m` es la log-verosimilitud del partido `m`, `t_m` su fecha y `ξ ≥ 0` la
**tasa de decaimiento**. `ξ = 0` ⇒ todos los partidos pesan igual; `ξ` grande ⇒
solo cuenta lo reciente. Se elige `ξ` por validación (en [config.yaml](../config/config.yaml)
está como `models.dixon_coles.xi`).

### 2.4 Jerárquico bayesiano (Baio & Blangiardo, 2010)

Misma estructura log-lineal, pero tratada bayesianamente. Los goles de cada
equipo se modelan como dos Poisson:

$$
\log \theta_{g1} = \text{home} + \text{att}_{h(g)} + \text{def}_{a(g)}
$$
$$
\log \theta_{g2} = \text{att}_{a(g)} + \text{def}_{h(g)}
$$

con parámetros de ataque/defensa por equipo y un parámetro `home` constante. La
ventaja bayesiana es el **shrinkage**: las habilidades de equipos con pocos
partidos se "encogen" hacia la media global, reduciendo sobreajuste.

> ⚠️ El detalle de implementación (efectos intercambiables vía MCMC en OpenBUGS)
> **no quedó verificado** — usar la estructura del modelo, no asumir el método de
> ajuste exacto. Ver [§9](#9-salvedades-y-afirmaciones-refutadas).

---

## 3. Sistemas de rating de fuerza

Resumen la fuerza de un equipo en **un escalar** dinámico, que luego puede
alimentar la capa de goles o el ML.

### 3.1 World Football Elo Ratings

Adaptación del Elo de ajedrez. Tras cada partido, el rating se actualiza:

$$
R_n = R_o + K\,(W - W_e)
$$

donde:
- `R_o`, `R_n` = rating antes y después,
- `W` = resultado real ∈ {1 victoria, 0.5 empate, 0 derrota},
- `W_e` = **resultado esperado**, dado por la curva logística:

$$
W_e = \frac{1}{1 + 10^{-dr/400}}, \qquad dr = (R_{\text{local}} - R_{\text{visitante}}) + 100
$$

El `+100` es la **ventaja de localía** en puntos Elo (≈ 100). Y `K` es el peso
del ajuste, con **dos refinamientos clave** respecto al Elo de ajedrez:

**(a) `K` escalonado por importancia del partido:**

| Tipo de partido | `K` |
|---|---|
| Final de Copa del Mundo | 60 |
| Torneo continental / clasificatorio de Mundial | 50 |
| Copa continental menor | 40 |
| Clasificatorios menores | 30 |
| Amistoso | 20 |

**(b) Ajuste por diferencia de goles.** Se escala `K` según el margen:

$$
K' = K \cdot G, \qquad
G = \begin{cases}
1 & \text{margen} \le 1 \\
1.5 & \text{margen} = 2 \\
1.75 & \text{margen} = 3 \\
1.75 + \dfrac{N - 3}{8} & \text{margen} = N \ge 4
\end{cases}
$$

Esto valida y **refina** el Elo que ya implementamos en
[`ratings/elo.py`](../src/worldcup/ratings/elo.py): nuestra versión ya tiene
localía y multiplicador por margen; el escalonado de `K` por importancia es la
mejora pendiente más directa.

### 3.2 Ranking FIFA y SPI

- **Ranking FIFA**: puntos oficiales por confederación y resultados. Útil como
  *feature*, pero —ver [§4](#4-el-enfoque-ganador-random-forest-híbrido)— resulta
  **menos informativo** que las habilidades-Poisson.
- **SPI (FiveThirtyEight)**: combinaba ratings ofensivo/defensivo basados en
  resultados (75%) y en plantel (25%). *Nota: discontinuado en 2023–2025; sirve
  como referencia metodológica, no como fuente viva.*

---

## 4. El enfoque ganador: Random Forest híbrido

Esta es la contribución central de la línea **Groll, Ley, Schauberger & Van
Eetvelde (2019)** y trabajos derivados — y el hallazgo más fuerte de la revisión.

### 4.1 La idea

En lugar de usar un modelo de goles *o* un ML, se **encadenan**:

```
                    paso 1                          paso 2
  partidos históricos ──► habilidades-Poisson ──► feature ──► Random Forest ──► goles / 1X2
  (ponderados)            (att/def por equipo)               (+ rank FIFA,
                                                              odds, GDP, ...)
```

### 4.2 Paso 1 — Habilidades por ranking Poisson ponderado

Se ajusta un modelo de goles (el mejor fue el **Poisson bivariado**, §2.2) por
**máxima verosimilitud ponderada** sobre *todos* los partidos internacionales de
los ~8 años previos, con un peso doble por partido:

$$
w_m = \underbrace{e^{-\xi\,(t_{\text{hoy}} - t_m)}}_{\text{decaimiento temporal}} \times \underbrace{I_m}_{\text{importancia}}
$$

- **Decaimiento temporal:** la mejor **semivida** fue ≈ **3 años** (seleccionada
  por RPS promedio). Semivida `h` ⇒ `ξ = ln(2)/h`.
- **Importancia del partido** `I_m`:

  | Tipo | `I_m` |
  |---|---|
  | Amistoso | 1.0 |
  | Clasificatorio | 2.5 |
  | Torneo continental | 3.0 |
  | Copa del Mundo | 4.0 |

El resultado por equipo es un **parámetro de habilidad** (esencialmente su
ataque menos defensa estimados), que condensa su fuerza ofensiva/defensiva.

### 4.3 Paso 2 — Random Forest sobre los goles

El Random Forest **aprende el número de goles** que marca un equipo en un
partido (cada partido aporta 2 observaciones: local y visitante). Sus *features*:

- **Habilidad-Poisson** del equipo y del rival ← *la variable más importante*,
- ranking/puntos FIFA,
- cuotas o **consenso de casas de apuestas**,
- covariables de contexto: PIB del país, valor de mercado del plantel, edad
  media, nº de jugadores en Champions League, confederación del rival, etc.

> Un Random Forest promedia muchos árboles de decisión entrenados sobre
> remuestreos bootstrap y subconjuntos aleatorios de *features*; reduce varianza
> y captura interacciones no lineales sin que haya que especificarlas.

### 4.4 El hallazgo clave (con números)

En el análisis de **importancia de variables**:

$$
\text{Habilidad-Poisson} \;\gg\; \text{Ranking FIFA} \;>\; \text{Cuotas (Oddset)} \;>\; \text{resto}
$$

> *"the abilities are by far the most important predictor in the random forest and
> carry clearly more information than all other predictors"* — Groll et al. (2018).

Y en evaluación **out-of-sample** (leave-one-World-Cup-out sobre Mundiales
2002–2014, ~250 partidos), el RF híbrido **igualó/superó levemente a las
cuotas**:

| Modelo | RPS (menor = mejor) |
|---|---|
| **Random Forest híbrido** | **0.187** |
| Cuotas de casas de apuestas | 0.188 |

### 4.5 Extensión "doblemente híbrida"

Trabajos posteriores (Mundial Femenino 2019, EURO 2020) integran **dos métodos
de ranking** como *features* del mismo RF:
1. habilidad-Poisson (de resultados históricos),
2. **consenso de casas de apuestas** (convertido a habilidad).

La habilidad-Poisson sigue dominando; el consenso de apuestas resulta algo más
informativo que el ranking FIFA. EURO 2020 añade además ratings **plus-minus**
de jugadores.

---

## 5. Simulación del torneo (Monte Carlo)

Tener un buen modelo de *partido* no basta: el torneo tiene estructura (grupos +
eliminatorias). Se proyecta por **Monte Carlo**:

1. Para cada partido, el modelo da una distribución del marcador (o de 1-X-2).
2. Se **muestrea** un resultado de esa distribución.
3. Se resuelve la fase de grupos (puntos, desempates) y el cuadro eliminatorio
   (en empates de eliminatoria → prórroga/penales).
4. Se repite **N veces** (típicamente decenas de miles).
5. Las **frecuencias** dan las probabilidades: `P(gana grupo)`, `P(llega a
   semis)`, `P(campeón)`, etc.

$$
\hat{P}(\text{equipo } i \text{ es campeón}) = \frac{1}{N}\sum_{s=1}^{N} \mathbb{1}\{\text{campeón}_s = i\}
$$

> ⚠️ La *metodología* Monte Carlo está confirmada; cifras concretas de
> simulaciones/probabilidades que circulan en blogs **no** lo están (ver §9).

---

## 6. Evaluación: RPS y benchmark de apuestas

### 6.1 Ranked Probability Score (métrica principal)

El resultado de un partido es **ordinal**: victoria local ≻ empate ≻ victoria
visitante. El RPS es la métrica estándar porque **respeta ese orden** —penaliza
más confundir local↔visitante que local↔empate—. Para `r` categorías ordenadas
con probabilidades predichas `p_j` y resultado real `e_j` (one-hot):

$$
\text{RPS} = \frac{1}{r-1}\sum_{i=1}^{r-1}\left(\sum_{j=1}^{i}(p_j - e_j)\right)^2
$$

Para `r = 3` (1-X-2), con orden `(local, empate, visitante)`. **Menor es mejor.**
Es la métrica que usamos en [`evaluation/metrics.py`](../src/worldcup/evaluation/metrics.py)
y la que Groll et al. usan tanto para evaluar como para **seleccionar el modelo**.

### 6.2 Métricas complementarias

- **Log-loss / verosimilitud multinomial:**
  $$
  \text{LogLoss} = -\sum_{j=1}^{r} e_j \log p_j
  $$
  Penaliza fuerte la confianza mal puesta, pero **ignora el orden**.

- **Brier score:**
  $$
  \text{Brier} = \sum_{j=1}^{r} (p_j - e_j)^2
  $$
  Generaliza el error cuadrático; el RPS es esencialmente su versión
  *acumulada/ordinal*.

### 6.3 El benchmark: las casas de apuestas

Las **cuotas** (implied probabilities, tras quitar el margen) son el *benchmark*
natural y **muy difícil de superar**: incorporan información de mercado. Que el
RF híbrido empate con ellas (RPS 0.187 vs 0.188) se considera un resultado
fuerte — aunque con la salvedad de muestra pequeña (§9).

---

## 7. Datos de entrenamiento

| Dato | Uso | Tenemos |
|---|---|---|
| **Resultados internacionales históricos** | habilidades-Poisson, Elo, modelo de goles | ✅ 49.417 partidos (1872–2026) |
| **Ranking / puntos FIFA** | *feature* del RF | ✅ snapshot hasta 2024-06 |
| **Importancia del partido** | peso `I_m` (amistoso vs Mundial) | 🟡 derivable de `tournament` |
| **Decaimiento temporal** | peso `φ(t)` (semivida ≈ 3 años) | ✅ fechas disponibles |
| **Cuotas / consenso de apuestas** | *feature* potente + benchmark | ❌ pendiente |
| **Covariables de equipo/jugadores** | PIB, valor de mercado, edad, plus-minus | ❌ pendiente |
| **xG (expected goals)** | *feature* avanzada (cobertura limitada en selecciones) | ❌ pendiente / pregunta abierta |

**Ponderación recomendada (de la literatura):** estimar habilidades sobre ~8
años de partidos con peso `w_m = e^{-ξ(t_hoy − t_m)} · I_m`, semivida ≈ 3 años.

---

## 8. Implicaciones para nuestro diseño (Mundial 2026)

La arquitectura de 3 capas que montamos **sigue siendo correcta**. El reporte
ajusta la **capa de modelo**:

| Componente | Plan inicial | Plan según evidencia |
|---|---|---|
| Modelo principal | Dixon-Coles puro | **Habilidad-Poisson → Random Forest híbrido** |
| Rol de Dixon-Coles | núcleo único | **motor de habilidades** + generador de marcadores para la simulación |
| ML | XGBoost (benchmark) | **Random Forest** (es lo que tiene evidencia); XGBoost = exploración |
| Elo | localía + margen | + **`K` escalonado por importancia** |
| Evaluación | RPS / log-loss / Brier | ✅ sin cambios; añadir **cuotas como benchmark** |
| Simulación | Monte Carlo | ✅ sin cambios; ajustar al **formato 48 equipos** y 3 anfitriones |

**Ruta de implementación sugerida (incremental):**

1. `DixonColesModel.fit()` — MLE ponderada. Primer ladrillo en *cualquier* ruta;
   produce las habilidades.
2. Ponderación `w_m` (decaimiento temporal + importancia del partido).
3. Simulador Monte Carlo (grupos + bracket 2026) → ya da predicciones usables.
4. Modelo de goles → habilidades como *feature* + Random Forest híbrido.
5. Conseguir cuotas de apuestas (feature + benchmark) y covariables de plantel.

Cada paso deja un modelo **evaluable con RPS** contra Mundiales pasados.

---

## 9. Salvedades y afirmaciones refutadas

**Salvedades de las afirmaciones confirmadas:**

- La superioridad sobre las cuotas (RPS 0.187 vs 0.188) viene de **un solo grupo
  de autores** (Groll/Ley), sobre **~250 partidos**, con margen estrecho. No es
  un benchmark validado independientemente por terceros.
- La evidencia de ML confirmada es de **Random Forests**; **no** se verificó
  literatura sólida sobre XGBoost ni redes neuronales en selecciones, ni sobre
  **xG** como *feature*. Quedan como [preguntas abiertas](#preguntas-abiertas).
- `eloratings.net` es JS-renderizado; sus fórmulas se corroboraron vía caché de
  búsqueda y Wikipedia, no por fetch directo.

**Afirmaciones REFUTADAS en verificación (NO usar):**

1. Detalle de implementación MCMC/OpenBUGS de Baio & Blangiardo (voto 1-2). La
   *estructura* del modelo sí está confirmada; el *método de ajuste* no.
2. *"Elo supera al ranking FIFA en eliminatorias, AUC 0.775 vs 0.695"* (voto
   **0-3**; fuente ResearchGate de baja calidad).
3. El dataset de *"115 partidos de eliminatorias 1994–2022"* de esa misma fuente
   (voto 1-2).
4. *"100.000 simulaciones, USA campeona al 28.1%"* (Mundial Femenino 2019; voto
   1-2). La metodología Monte Carlo sí; el número exacto no.

### Preguntas abiertas

- ¿Rinden XGBoost / redes neuronales igual o mejor que el RF híbrido en
  selecciones? Sin benchmark verificado.
- ¿Aporta el **xG** o datos granulares de jugadores en partidos de selección,
  dado el menor volumen de datos de tracking que en ligas de clubes?
- **Formato 2026** (48 equipos, 12 grupos, nueva fase eliminatoria): cómo ajustar
  la simulación y la **ventaja local de los 3 anfitriones** (EE.UU., México,
  Canadá).
- ¿Cuánta ventaja real ofrecen los modelos sobre las cuotas en validación
  independiente, y cómo integrar el consenso de apuestas **sin sobreajustar** a él?

---

## 10. Referencias

**Modelos de goles**
- Maher, M. J. (1982). *Modelling association football scores.* Statistica
  Neerlandica 36(3):109–118. https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9574.1982.tb00782.x
- Dixon, M. & Coles, S. (1997). *Modelling Association Football Scores and
  Inefficiencies in the Football Betting Market.* JRSS-C 46(2):265–280.
- Karlis, D. & Ntzoufras, I. (2003). *Analysis of sports data by using bivariate
  Poisson models.* JRSS-D 52(3):381–393. http://www2.stat-athens.aueb.gr/~jbn/papers2/08_Karlis_Ntzoufras_2003_RSSD.pdf
- Baio, G. & Blangiardo, M. (2010). *Bayesian hierarchical model for the
  prediction of football results.* J. Applied Statistics 37(2):253–264.
  https://www.semanticscholar.org/paper/Bayesian-hierarchical-model-for-the-prediction-of-Baio-Blangiardo/1a974c7f4e90d9f56498d2711ebccb9fcc2e09d6

**Ratings**
- World Football Elo Ratings — metodología: https://www.eloratings.net/about
- Wikipedia, *World Football Elo Ratings*: https://en.wikipedia.org/wiki/World_Football_Elo_Ratings

**ML híbrido**
- Groll, A., Ley, C., Schauberger, G. & Van Eetvelde, H. (2019). *A hybrid random
  forest to predict soccer matches in international tournaments.* JQAS
  15(4):271–287. https://arxiv.org/pdf/1806.03208
- Groll et al. (2019). *Prediction of the FIFA Women's World Cup 2019* (doubly
  hybrid). https://arxiv.org/pdf/1906.01131
- Groll et al. (2021). *Hybrid ML approach for EURO 2020.* https://arxiv.org/pdf/2106.05799

**Evaluación**
- Constantinou, A. & Fenton, N. *Solving the problem of inadequate scoring rules…*
  https://www.eecs.qmul.ac.uk/~norman/papers/evaluating_predictive_accuracy_football.pdf

**Práctico**
- dashee87, *Predicting Football Results With Dixon-Coles and Time-Weighting*:
  https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/

---

*Documento generado a partir de una revisión de literatura con verificación
adversarial de fuentes. Última actualización: 2026-06-15.*
