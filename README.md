# Análisis Predictivo de Precios de Propiedades Residenciales en India

**Proyecto Final — Ciencia de Datos | UABC**  
**Autor:** Juan Garfias  
**Dataset:** House Prices India — 187,531 propiedades · 81 ciudades · 21 variables

---

## Resultados

| Modelo | R² | MAE | RMSE |
|---|---|---|---|
| **Random Forest** | **0.92** | ₹736,441 | ₹1,808,673 |
| Ridge Regression (baseline) | 0.58 | ₹2,993,116 | ₹4,022,467 |

> El modelo explica el **92% de la varianza** en precios de propiedades nunca vistas.

---

## Visualizaciones

### Distribución del Precio — Multimodalidad natural del mercado

![Distribución del precio](reports/figures/01_distribucion_precio.png)

El precio **no sigue una distribución normal**: tiene 4 picos en ₹5M, ₹7M, ₹15M y ₹22M. Esto no es ruido — refleja la segmentación natural del mercado (económico, familiar, ejecutivo y lujo). El log-transform mejora la simetría pero no elimina la multimodalidad.

---

### Correlaciones — ¿Qué variable se relaciona más con el precio?

![Heatmap de correlación](reports/figures/02_heatmap_correlacion.png)

Las 3 variables con mayor correlación lineal (Pearson) con el precio son **Bathroom (r=0.59)**, carpet_area_num (r=0.58) y bhk_count (r=0.56). La ubicación (`city_avg_price`) tiene r=0.44 lineal, pero en el modelo no-lineal es la **segunda variable más importante (24%)** — la relación precio-ubicación es no lineal.

---

### Precio por Ciudad — El mercado premium está concentrado

![Precio mediano por ciudad](reports/figures/05_precio_por_ciudad.png)

**New Delhi (₹15.8M), Gurgaon (₹14.5M) y Mumbai (₹14M)** lideran el mercado. Hay una caída abrupta después del puesto 4: el mercado premium está geográficamente muy concentrado.

---

### Precio por Orientación — Hallazgo contraintuitivo

![Precio por orientación](reports/figures/05b_precio_por_orientacion.png)

Las casas orientadas al **Sur (₹13.6M) son las más caras** — no las del Norte (₹7.2M). En el mercado indio, la orientación Sur está asociada al **Vastu Shastra** (filosofía arquitectónica) y mayor luminosidad en invierno. Las orientadas al Sur-Este son las más baratas (₹6M), una diferencia del 127%.

---

### Feature Importance — ¿Qué decide el precio?

![Feature importance](reports/figures/07_feature_importance.png)

| Rank | Variable | Importancia |
|---|---|---|
| 1 | `carpet_area_num` — Área en pies cuadrados | 42% |
| 2 | `city_avg_price` — Precio mediano de la ciudad | 24% |
| 3–5 | `total_floors`, `Bathroom`, `floor_ratio` | ~6% cada una |

El área es el factor dominante, pero la ubicación (ciudad) tiene el doble de peso que cualquier característica del inmueble.

---

### Curva de Aprendizaje — Diagnóstico del modelo

![Curva de aprendizaje](reports/figures/08_curva_aprendizaje.png)

Train R²≈0.985 vs Validation R²≈0.915: brecha de ~7 puntos indica **overfitting moderado**. El modelo generaliza bien (R²=0.92 en test), pero podría mejorar limitando `max_depth` o incrementando `min_samples_leaf`.

---

### Residuos del Random Forest

![Gráfico de residuos](reports/figures/06_residuos.png)

El gráfico muestra **heterocedasticidad**: el error crece con el precio. Las propiedades más caras tienen características más únicas y difíciles de generalizar — comportamiento esperado en mercados de lujo.

---

### Segmentación de Mercado — 4 perfiles con KMeans

![Método del codo](reports/figures/09_elbow_method.png)

El método del codo no muestra un quiebre marcado (los datos no forman clusters perfectos), por lo que se eligió **k=4 por interpretabilidad de negocio**.

| Segmento | BHK | Área | Precio/sqft | Perfil |
|---|---|---|---|---|
| Cluster 0 | 3 | 1,300 sqft | ₹8,087 | Inversión Económica |
| Cluster 1 | 2 | 800 sqft | ₹6,500 | Perfil Familiar |
| Cluster 2 | 2 | 688 sqft | ₹5,814 | Perfil Ejecutivo |
| Cluster 3 | 2 | 105 sqft | ₹53,488 | **Lujo Premium** |

> El Cluster 3 (Lujo Premium) se distingue por un `price_per_sqft` extremadamente alto (₹53,488 vs ~₹6,500 del resto) — ubicaciones exclusivas donde el precio por metro cuadrado es el diferenciador, no el tamaño.

---

### Visualizaciones Interactivas

Los archivos HTML en `reports/figures/` se abren en el navegador:

| Archivo | Contenido |
|---|---|
| [`04_scatter_interactivo.html`](reports/figures/04_scatter_interactivo.html) | Scatter área vs precio, coloreado por BHK (Plotly) |
| [`10_clusters_interactivo.html`](reports/figures/10_clusters_interactivo.html) | PCA 2D de los 4 clusters de mercado (Plotly) |

---

## Estructura del Proyecto

```
house_price_project/
│
├── data/
│   ├── raw/                        ← Coloca aquí house_prices.csv (ver instrucciones)
│   └── processed/                  ← Se genera al ejecutar el notebook 01
│
├── notebooks/
│   ├── 01_EDA_limpieza.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling_regresion.ipynb
│   └── 04_modeling_clustering.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   └── models.py
│
├── app/
│   └── app.py                      ← Interfaz web Streamlit
│
├── reports/
│   ├── figures/                    ← 11 gráficas (PNG + HTML interactivos)
│   └── reporte_hallazgos.md
│
├── requirements.txt
└── README.md
```

---

## Instalación

```bash
git clone https://github.com/RicardoGarfias/house_price_project.git
cd house_price_project

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

### Obtener el dataset

El archivo `house_prices.csv` no está en el repositorio (102 MB). Descárgalo y colócalo en:

```
data/raw/house_prices.csv
```

---

## Orden de Ejecución

| Paso | Notebook | Genera |
|------|----------|--------|
| 1 | `01_EDA_limpieza.ipynb` | `data/processed/house_prices_clean.csv` |
| 2 | `02_feature_engineering.ipynb` | Variables nuevas + 11 gráficas en `reports/figures/` |
| 3 | `03_modeling_regresion.ipynb` | `app/model_rf.pkl` (363 MB, excluido del repo) |
| 4 | `04_modeling_clustering.ipynb` | `app/model_kmeans.pkl`, `app/scaler_kmeans.pkl` |

```bash
cd app
streamlit run app.py
# Abre http://localhost:8501
```

---

## Ingeniería de Atributos

8 variables derivadas creadas en el notebook 02:

| Variable | Descripción |
|---|---|
| `price_per_sqft` | Precio por pie cuadrado |
| `floor_ratio` | Posición relativa del piso (0=planta baja, 1=último) |
| `is_luxury` | Palabras clave de lujo en título/descripción |
| `furnishing_score` | Ordinal: 0=sin amueblar · 1=semi · 2=amueblado |
| `has_garden_view` | Vista a jardín o parque |
| `has_parking` | Estacionamiento incluido |
| `city_avg_price` | Precio mediano de la ciudad (evita 81 dummies) |
| `has_society` | Pertenece a un condominio |

---

## Hallazgos Principales

1. **El precio es multimodal**, con picos en ₹5M, ₹7M, ₹15M y ₹22M — reflejo directo de los 4 segmentos de mercado.
2. **Bathroom tiene mayor correlación lineal** con el precio (r=0.59) que la ubicación (r=0.44), pero en el modelo no-lineal la ubicación es la segunda variable más importante (24%). La relación precio-ubicación es no lineal.
3. **New Delhi, Gurgaon y Mumbai** concentran el mercado premium, siendo 2× más caras que el promedio.
4. **Las casas orientadas al Sur son las más caras** (₹13.6M), no las del Norte (₹7.2M) — fenómeno relacionado con el Vastu Shastra en la arquitectura india.
5. **Overfitting moderado** (Train R²=0.985 vs Test R²=0.92); ajustar `max_depth` reduciría la brecha sin sacrificar precisión.

---

## Dependencias

```
pandas>=2.0 · numpy>=1.24 · matplotlib>=3.7 · seaborn>=0.12
plotly>=5.14 · scikit-learn>=1.3 · joblib>=1.3 · streamlit>=1.28
```
