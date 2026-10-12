# Artículo Valoración de Opciones mediante un Árbol Binomial Adaptativo

### Un enfoque basado en máxima verosimilitud, volatilidad condicional y distribuciones de colas pesadas

**Autor:** José María López Espinosa  
**Lenguaje:** Python  
**Área:** Finanzas cuantitativas · Derivados financieros · Econometría financiera  
**Año:** 2026

---

## 1. Descripción del proyecto

Este repositorio contiene la implementación en Python del modelo desarrollado en el artículo **«Valoración de Opciones mediante un Árbol Binomial Adaptativo»**.

La investigación estudia una extensión del modelo binomial clásico de Cox-Ross-Rubinstein (CRR), incorporando información procedente de la dinámica estadística de los rendimientos financieros mediante un modelo AR(2)-GARCH(1,1) con innovaciones t-Student.

El objetivo principal es analizar si una estructura temporal de volatilidad estimada a partir de datos históricos permite mejorar la valoración de opciones financieras frente al modelo CRR convencional.

Para ello, se desarrolla un sistema que combina:

- Estimación de parámetros por máxima verosimilitud.
- Modelización de la media condicional mediante AR(2).
- Estimación de volatilidad condicional mediante GARCH(1,1).
- Innovaciones t-Student estandarizadas.
- Construcción de árboles binomiales recombinantes.
- Calibración de modelos con cotizaciones reales de opciones.
- Comparación empírica frente al modelo CRR.
- Evaluación temporal fuera de muestra.
- Cálculo de sensibilidades y análisis de cobertura.
- Visualización y exportación de los resultados.

La implementación utiliza datos reales obtenidos de Yahoo Finance y del Departamento del Tesoro de Estados Unidos.

## 2. Fundamentos matemáticos

### 2.1. Modelo AR(2)-GARCH(1,1)-t

Los rendimientos logarítmicos se definen como:

$$
r_t=\ln\left(\frac{S_t}{S_{t-1}}\right)
$$

La dinámica condicional del rendimiento se representa mediante:

$$
r_t=\mu_t+\varepsilon_t
$$

donde la media condicional sigue un proceso autorregresivo de segundo orden:

$$
\mu_t=\mu+\phi_1r_{t-1}+\phi_2r_{t-2}
$$

La varianza condicional viene determinada por:

$$
h_t=\omega+\alpha\varepsilon_{t-1}^2+\beta h_{t-1}
$$

Se consideran innovaciones con distribución t-Student estandarizada:

$$
\varepsilon_t=\sqrt{h_t}z_t
$$

$$
z_t=\sqrt{\frac{\nu-2}{\nu}}T_t,
\qquad T_t\sim t_\nu
$$

con media nula, varianza unitaria y $\nu>2$.

El vector de parámetros es:

$$
\theta=(\mu,\phi_1,\phi_2,\omega,\alpha,\beta,\nu)
$$

### 2.2. Estimación por máxima verosimilitud

Los parámetros se obtienen mediante la maximización de la función de log-verosimilitud condicional:

$$
\hat{\theta}=
\underset{\theta}{\operatorname{argmax}}
\sum_{t=1}^{T}
\ln f(r_t\mid\mathcal{F}_{t-1};\theta)
$$

La implementación emplea el algoritmo SLSQP de SciPy, con restricciones sobre los parámetros y tres puntos iniciales de optimización.

El modelo incorpora restricciones de estacionariedad para el proceso autorregresivo y de persistencia para GARCH.

### 2.3. Árbol binomial adaptativo

A partir de las estimaciones del modelo GARCH se construye una estructura de varianza condicional prevista.

Sea $A(t)$ la varianza acumulada estimada hasta el horizonte $t$.

Para un vencimiento $T$ y $N$ pasos, se define:

$$
a=\sqrt{\frac{A(T)}{N}}
$$

Los factores multiplicativos del árbol son:

$$
u=e^a,\qquad d=e^{-a}
$$

A diferencia de una discretización temporal uniforme, el modelo construye una malla adaptada a los incrementos de varianza acumulada.

En cada intervalo temporal $\Delta t_i$, la probabilidad neutral al riesgo se calcula como:

$$
p_i=
\frac{e^{(r-q)\Delta t_i}-d}{u-d}
$$

donde $r$ representa el tipo de interés continuamente compuesto y $q$ la tasa aproximada de dividendos.

El valor de una opción americana se determina mediante inducción hacia atrás, comparando en cada nodo el valor de continuación y el valor de ejercicio inmediato:

$$
V_i=\max\left\{
\text{valor intrínseco},
e^{-r\Delta t_i}
[p_iV_{i+1}^{u}+(1-p_i)V_{i+1}^{d}]
\right\}
$$

Para reducir los efectos de discretización se promedian los precios obtenidos con árboles de 500 y 501 pasos.

**Importante:** La calibración de la curva de varianza constituye un ajuste reducido a las cotizaciones de mercado. No representa por sí sola una derivación completa del cambio de medida de probabilidad para un proceso GARCH.

## 3. Modelos comparados

El programa considera cuatro especificaciones para la evaluación de los rendimientos:

| Modelo | Media | Volatilidad | Innovaciones |
|---|---|---|---|
| AR2-GARCH-t | AR(2) | GARCH(1,1) | Student-t |
| Media-GARCH-t | Constante | GARCH(1,1) | Student-t |
| AR2-GARCH-normal | AR(2) | GARCH(1,1) | Normal |
| Student constante | Constante | Constante | Student-t |

Adicionalmente, para la valoración de opciones se consideran cuatro configuraciones:

| Configuración | Descripción |
|---|---|
| `Adapt_hist` | Árbol adaptativo con varianza prevista sin calibración a opciones |
| `CRR_hist` | Árbol CRR con volatilidad histórica anualizada |
| `Adapt_cal` | Árbol adaptativo calibrado a cotizaciones de opciones |
| `CRR_cal` | Árbol CRR calibrado a cotizaciones de opciones |

La calibración se realiza mediante minimización del error cuadrático medio frente a los precios medios bid-ask.

## 4. Instalación y requisitos

### Requisitos

- Python 3.10 o posterior recomendado.
- Conexión a Internet.
- Acceso a Yahoo Finance y al Tesoro estadounidense.
- Entorno gráfico compatible con Matplotlib.
- Datos disponibles de acciones o ETF estadounidenses con opciones estandarizadas.

### Instalación de dependencias

Desde una terminal, ejecutar:

```bash
python -m pip install numpy pandas scipy matplotlib yfinance pandas_market_calendars requests
```

Las bibliotecas utilizadas son:

| Biblioteca | Función |
|---|---|
| NumPy | Cálculo numérico y operaciones vectorizadas |
| Pandas | Manipulación de series temporales y tablas |
| SciPy | Optimización, funciones estadísticas y cálculo numérico |
| Matplotlib | Representación gráfica |
| yfinance | Descarga de cotizaciones e históricos |
| pandas_market_calendars | Calendario bursátil de NYSE |
| Requests | Descarga de información del Tesoro estadounidense |

## 5. Ejecución del programa

Descargar el archivo Python y ejecutar:

```bash
python "CODIGO DEFINITIVO.py"
```

También puede renombrarse el archivo como `terminal_binomial.py`, utilizando entonces:

```bash
python terminal_binomial.py
```

### Parámetros de entrada

La aplicación funciona mediante una interfaz de consola.

Al comenzar, solicita:

```text
ARBOL ADAPTATIVO / CRR

Ticker [SPY]:
años de historico [5]:
Actualizar cada cuantos segundos (0 = al cerrar los graficos) [0]:
```

**Ejemplo de configuración:**

- Ticker: `SPY`
- Histórico: 5 años
- Actualización: 0 segundos

Los valores predeterminados permiten ejecutar el análisis sobre SPY utilizando cinco años de observaciones históricas.

Cuando se selecciona un intervalo positivo de actualización, el programa descarga nuevamente la información y recalibra los modelos después de cada ejecución y del intervalo de espera.

Con intervalo cero, el usuario puede cerrar las figuras y decidir desde la consola si desea efectuar una nueva captura.

El programa también permite interrumpir la ejecución mediante `Ctrl+C`.

## 6. Datos y metodología empírica

### 6.1. Obtención de información financiera

El programa obtiene automáticamente:

1. Precios históricos ajustados del subyacente.
2. Cotizaciones recientes del activo con frecuencia de un minuto.
3. Dividendos distribuidos durante los últimos 365 días.
4. Cadenas de opciones call y put.
5. Tipos de interés de letras del Tesoro estadounidense.
6. Calendario oficial de sesiones de negociación de NYSE.

Los vencimientos de las opciones se seleccionan por proximidad a horizontes de 30, 60 y 90 días.

Los tipos de interés utilizados se obtienen mediante conversión de las tasas de descuento del Tesoro e interpolación entre los vencimientos disponibles.

### 6.2. Selección de contratos

Para reducir el efecto de cotizaciones potencialmente poco representativas, se aplican distintos criterios de selección:

- Bid estrictamente positivo y ask no inferior al bid.
- Spread relativo bid-ask no superior al 20%.
- Interés abierto mínimo de 10 contratos.
- Última operación registrada con antigüedad máxima de siete días.
- Precio de ejercicio situado a menos del 5% del precio del subyacente.
- Contratos estandarizados denominados en dólares estadounidenses.
- Comprobaciones de cotas de valoración y consistencia entre calls y puts.

Por vencimiento se seleccionan hasta 12 precios de ejercicio próximos al precio del subyacente, siempre que sobrevivan suficientes parejas call-put.

### 6.3. Evaluación fuera de muestra

El procedimiento incorpora dos análisis diferenciados.

**Evaluación temporal de rendimientos**

La muestra histórica se divide cronológicamente:

- 80% para estimación.
- 20% para evaluación.

Los parámetros se estiman en el bloque inicial y se evalúan sobre las observaciones posteriores, sin reestimación en cada fecha del bloque de prueba.

Se calculan las métricas Negative Log-Likelihood (NLL) y QLIKE, acompañadas de intervalos de confianza mediante bootstrap por bloques con 2.000 remuestreos.

**Evaluación transversal de opciones**

Los precios de ejercicio seleccionados se dividen alternativamente entre contratos de calibración y contratos de evaluación.

Los parámetros calibrados se evalúan posteriormente sobre contratos que no participaron en la optimización.

Esta división permite examinar el ajuste transversal, aunque no constituye una validación temporal futura de los precios de las opciones.

## 7. Métricas de valoración

Para comparar el comportamiento de los cuatro modelos de valoración se utilizan:

**Error absoluto medio (MAE)**

$$
MAE=\frac{1}{n}\sum_{i=1}^{n}|P_i^{modelo}-P_i^{mercado}|
$$

**Raíz del error cuadrático medio (RMSE)**

$$
RMSE=
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(P_i^{modelo}-P_i^{mercado})^2
}
$$

**Sesgo medio**

$$
Bias=
\frac{1}{n}
\sum_{i=1}^{n}
(P_i^{modelo}-P_i^{mercado})
$$

También se calcula el porcentaje de precios teóricos situados dentro del intervalo bid-ask observado.

Los resultados se presentan tanto de forma agregada como por vencimiento y conjunto de calibración o evaluación.

## 8. Sensibilidades y cobertura

El sistema calcula numéricamente las principales sensibilidades de las opciones:

| Sensibilidad | Interpretación |
|---|---|
| Delta | Sensibilidad al precio del subyacente |
| Gamma | Variación de delta ante cambios del subyacente |
| Vega | Sensibilidad ante cambios en la volatilidad |
| Theta | Sensibilidad al paso del tiempo |
| Rho | Sensibilidad a los tipos de interés |

Las sensibilidades se estiman mediante diferencias finitas para los modelos adaptativo y CRR y para una referencia obtenida mediante inversión numérica de la volatilidad implícita en un árbol CRR americano.

En la implementación se incluyen las columnas `vega_1pp`, `theta_dia` y `rho_1pp`, según las convenciones numéricas establecidas en el código.

### Cobertura mediante delta

Se construye asimismo una medida de cobertura basada en el punto medio entre dos estimaciones de delta:

$$
\eta=\frac{\Delta_{Adapt}+\Delta_{IV}}{2}
$$

y una medida de discrepancia:

$$
z=\frac{|\Delta_{Adapt}-\Delta_{IV}|}{2}
$$

Para un contrato sobre 100 acciones:

$$
N_{acciones}=100\eta
$$

El programa exporta tanto la cantidad de acciones resultante como la medida de discrepancia asociada.

Estas magnitudes son propuestas de cobertura bajo los modelos considerados, no garantías de eliminación del riesgo.

## 9. Resultados y visualizaciones

Cada ejecución genera automáticamente un directorio de resultados:

```text
resultados_binomial/
└── TICKER_FECHA_HORA/
    ├── captura.json
    ├── informe.txt
    ├── historico.csv
    ├── spot_1min.csv
    ├── dividendos_365_dias.csv
    ├── tipos.csv
    ├── opciones_originales.csv
    ├── parametros_completos.csv
    ├── parametros_train.csv
    ├── evaluacion_temporal.csv
    ├── intervalos_temporales.csv
    ├── varianza_prevista.csv
    ├── calibracion.csv
    ├── precios_opciones.csv
    ├── opciones_resultados.csv
    ├── errores_opciones.csv
    ├── cobertura.csv
    ├── convergencia_sin_recalibrar.csv
    ├── 01_historico.png
    ├── 02_comparacion.png
    ├── 03_griegas.png
    └── 04_escenarios.png
```

La carpeta incluye otros archivos CSV auxiliares, entre ellos distribuciones y escenarios de valoración bajo la medida neutral al riesgo, pérdidas temporales y comparaciones de sensibilidades.

### Figuras generadas

**Figura 1 — Histórico y estimación**

Contiene precios históricos ajustados, rendimientos diarios, volatilidad condicional anualizada y distribución de residuos estandarizados.

**Figura 2 — Comparación de modelos**

Representa los precios teóricos frente a las cotizaciones de mercado, errores de valoración, métricas en contratos reservados y contraste entre un árbol europeo y Black-Scholes con varianza equivalente.

**Figura 3 — Sensibilidades y cobertura**

Presenta Delta, Gamma, Vega, Theta y Rho para las distintas referencias de valoración, además de las cantidades de acciones propuestas para cobertura.

**Figura 4 — Escenarios del árbol**

Incluye trayectorias de cuantiles, medianas y medias bajo la medida neutral al riesgo, distribuciones terminales y curvas de varianza acumulada de los modelos calibrados.

Los escenarios neutrales al riesgo no deben interpretarse como predicciones de los precios futuros bajo la distribución histórica.

## 10. Análisis de convergencia

El programa evalúa la sensibilidad de los precios teóricos al número de pasos del árbol mediante:

$$
N\in\{100,250,500,1000\}
$$

Para cada configuración se calculan los errores de valoración en los contratos reservados y las diferencias de precios respecto a la referencia de 500 pasos.

Las pruebas de convergencia utilizan los parámetros previamente calibrados, sin volver a optimizarlos para cada tamaño del árbol.

## 11. Reproducibilidad y limitaciones

Este proyecto está orientado a la investigación y experimentación académica.

Deben tenerse en cuenta las siguientes consideraciones:

- **Datos de mercado:** Yahoo Finance puede proporcionar cotizaciones retrasadas, incompletas o sujetas a revisiones.
- **Sincronización:** Las cadenas de opciones y el precio del subyacente no necesariamente representan una captura perfectamente simultánea.
- **Última operación:** La fecha de la última transacción de una opción no determina cuándo se actualizaron sus cotizaciones bid y ask.
- **Dividendos:** El rendimiento por dividendos es una aproximación basada en distribuciones históricas de los últimos 365 días.
- **Calibración:** El ajuste transversal no equivale a una prueba temporal fuera de muestra de precios de opciones.
- **Optimización:** Algunos activos o periodos pueden generar problemas de convergencia o insuficiencia de contratos válidos.
- **Coste computacional:** El uso de árboles de cientos de pasos, calibraciones y cálculos de sensibilidades puede exigir tiempos de ejecución elevados.
- **Alcance:** El programa está diseñado para acciones y ETF estadounidenses denominados en USD con opciones estandarizadas.
- **Medida neutral al riesgo:** La transformación de la estructura GARCH histórica mediante un factor de escala no implica una especificación completa de la dinámica GARCH bajo una medida martingala equivalente.

El programa no sustituye los datos faltantes por observaciones simuladas o parámetros inventados. Si la información requerida no está disponible o una etapa esencial de la estimación falla, la ejecución correspondiente se interrumpe.

Se conservan archivos de las capturas realizadas, parámetros estimados y metadatos para facilitar la trazabilidad de los resultados.

Para una reproducción más estricta de los experimentos es recomendable conservar también las versiones exactas de Python y de las dependencias utilizadas.

## 12. Alcance de la investigación

La implementación permite investigar hasta qué punto una estructura dinámica de volatilidad obtenida mediante modelos econométricos aporta información útil para la valoración de opciones.

La comparación empírica se centra en el ajuste a cotizaciones, la evaluación de la dinámica de rendimientos, la estabilidad numérica y las diferencias de sensibilidades entre modelos.

El programa no implementa una estrategia de negociación ni un backtest de rentabilidad de carteras con operaciones ejecutadas.

Por tanto, no se presentan sus resultados como evidencia directa de rentabilidades obtenibles mediante trading.

## 13. Referencia académica

**López Espinosa, José María (2026).**  
*Valoración de Opciones mediante un Árbol Binomial Adaptativo: Un enfoque basado en máxima verosimilitud, volatilidad condicional y distribuciones de colas pesadas.*

Este repositorio contiene la implementación computacional asociada a la investigación.

## 14. Autor

**José María López Espinosa**

Estudiante de Economía — Universidad de Sevilla.

Áreas de interés:

- Finanzas cuantitativas.
- Modelización econométrica.
- Derivados financieros.
- Procesos estocásticos.
- Valoración de activos financieros.

---

**Aviso:** Este proyecto tiene fines académicos y de investigación. Los resultados y estimaciones no constituyen recomendaciones personalizadas de inversión ni garantías sobre el comportamiento futuro de los mercados.
