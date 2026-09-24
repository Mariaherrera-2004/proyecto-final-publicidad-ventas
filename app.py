# -*- coding: utf-8 -*-
"""
Dashboard - Proyecto Final: Gasto en publicidad digital y ventas en e-commerce
Programación para Ciencia de Datos II - U Compensar

Autora: María Fernanda Herrera Contreras
Docente: William Eduardo Clavijo Bohorquez

Cómo ejecutar localmente:
    pip install -r requirements.txt
    python app.py
    -> abrir http://127.0.0.1:8050 en el navegador

Este dashboard reúne, en una sola herramienta interactiva, los hallazgos de las
tres etapas del proyecto (contextualización, profundización y transferencia):
    1. Análisis exploratorio de datos (EDA)
    2. Contraste de hipótesis (prueba t de Welch / Mann-Whitney)
    3. Regresión lineal (simple, múltiple y Ridge con regularización)
    4. Regresión logística (clasificación de "mes de venta alta")
    5. Simulador interactivo para el equipo financiero
"""

import numpy as np
import pandas as pd
from scipy import stats

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import (
    r2_score, mean_squared_error, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
)
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 1. CARGA Y PREPARACIÓN DE DATOS
# ---------------------------------------------------------------------------

df = pd.read_csv("data/ventas_ecommerce.csv", parse_dates=["mes"])
df = df.sort_values("mes").reset_index(drop=True)

FEATURES = ["gasto_publicidad_millones", "visitas_web_miles", "tasa_conversion_pct"]
TARGET = "ventas_millones"

mediana_ventas = df[TARGET].median()
mediana_gasto = df["gasto_publicidad_millones"].median()
df["venta_alta"] = (df[TARGET] > mediana_ventas).astype(int)

# Outliers detectados en la etapa exploratoria (|z| > 2)
z = (df[TARGET] - df[TARGET].mean()) / df[TARGET].std(ddof=1)
outlier_idx = df.index[np.abs(z) > 2]

PALETA = {
    "primario": "#4C72B0",
    "secundario": "#DD8452",
    "acento": "#C44E52",
    "exito": "#55A868",
    "fondo": "#F7F9FC",
    "texto": "#1F2A44",
}

# ---------------------------------------------------------------------------
# 2. MODELOS BASE (se recalculan también dentro de los callbacks para
#    reflejar los controles interactivos, pero aquí se dejan versiones
#    "de referencia" usadas en las pestañas de contexto / conclusiones)
# ---------------------------------------------------------------------------

X_full = df[FEATURES].to_numpy()
y_full = df[TARGET].to_numpy()

X_simple = df[["gasto_publicidad_millones"]].to_numpy()
modelo_simple = LinearRegression().fit(X_simple, y_full)
r_pearson = np.corrcoef(df["gasto_publicidad_millones"], df[TARGET])[0, 1]
r2_simple = r2_score(y_full, modelo_simple.predict(X_simple))

# Regresión logística de referencia (todas las observaciones, C=1, lbfgs)
Xl = df[["gasto_publicidad_millones"]].to_numpy()
yl = df["venta_alta"].to_numpy()
modelo_logit_ref = LogisticRegression(C=1, solver="lbfgs").fit(Xl, yl)

# Regresión lineal múltiple de referencia (todas las observaciones, sin
# separar train/test) — se usa en el simulador para dar una predicción con
# toda la información disponible.
modelo_multiple_ref = LinearRegression().fit(X_full, y_full)

# ---------------------------------------------------------------------------
# 3. FUNCIONES AUXILIARES
# ---------------------------------------------------------------------------

def kpi_card(titulo, valor, subtitulo, color):
    return dbc.Card(
        dbc.CardBody([
            html.P(titulo, className="kpi-titulo"),
            html.H3(valor, className="kpi-valor", style={"color": color}),
            html.P(subtitulo, className="kpi-subtitulo"),
        ]),
        className="kpi-card shadow-sm",
    )


def figura_base(fig, titulo=None):
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Segoe UI, Roboto, Helvetica, Arial", size=13,
                  color=PALETA["texto"]),
        margin=dict(l=40, r=30, t=50 if titulo else 20, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    if titulo:
        fig.update_layout(title=dict(text=titulo, x=0.02, font=dict(size=16)))
    return fig


# ---------------------------------------------------------------------------
# 4. INICIALIZACIÓN DE LA APP
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    title="Dashboard | Publicidad y Ventas E-commerce",
)
server = app.server  # necesario para despliegue (Binder / gunicorn)

# ---------------------------------------------------------------------------
# 5. LAYOUT — ENCABEZADO
# ---------------------------------------------------------------------------

encabezado = dbc.Navbar(
    dbc.Container([
        html.Div([
            html.H4("📊 Publicidad digital y ventas — E-commerce",
                     className="text-white mb-0"),
            html.Small("Proyecto Final · Programación para Ciencia de Datos II · "
                       "María Fernanda Herrera Contreras",
                       className="text-white-50"),
        ])
    ], fluid=True),
    color=PALETA["texto"],
    dark=True,
    className="mb-3",
)

# ---------------------------------------------------------------------------
# 5.1 PESTAÑA — CONTEXTO / RESUMEN EJECUTIVO
# ---------------------------------------------------------------------------

tab_contexto = dbc.Container([
    dbc.Row([
        dbc.Col(kpi_card("Meses analizados", f"{len(df)}", "enero 2021 – diciembre 2025",
                          PALETA["primario"]), md=3),
        dbc.Col(kpi_card("Correlación gasto–ventas", f"r = {r_pearson:.2f}",
                          "asociación positiva moderada-alta", PALETA["secundario"]), md=3),
        dbc.Col(kpi_card("Poder explicativo (simple)", f"R² = {r2_simple:.2f}",
                          "regresión lineal, 1 variable", PALETA["exito"]), md=3),
        dbc.Col(kpi_card("Meses atípicos", f"{len(outlier_idx)}",
                          "campaña viral y falla operativa", PALETA["acento"]), md=3),
    ], className="g-3 mb-4"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Planteamiento del problema", className="mb-3"),
            html.P("Una tienda de comercio electrónico de tamaño mediano invierte cada "
                   "mes en publicidad digital (Meta, TikTok Ads, email marketing), pero "
                   "el equipo de mercadeo no tiene claridad sobre qué tan efectiva es esa "
                   "inversión para generar ventas reales, y el área financiera necesita "
                   "saber si vale la pena seguir aumentando el presupuesto."),
            html.P("Este dashboard reúne las tres etapas del proyecto — exploración de "
                   "datos, contraste de hipótesis y modelos predictivos (regresión lineal "
                   "y logística) — para responder, con evidencia estadística, si el gasto "
                   "en publicidad realmente impulsa las ventas y qué tan confiable es esa "
                   "relación."),
            html.Hr(),
            html.H6("Variables del conjunto de datos"),
            html.Ul([
                html.Li([html.B("gasto_publicidad_millones: "),
                         "inversión mensual en publicidad digital (millones COP)."]),
                html.Li([html.B("ventas_millones: "), "ventas totales del mes (millones COP)."]),
                html.Li([html.B("visitas_web_miles: "), "visitas al sitio web (miles)."]),
                html.Li([html.B("tasa_conversion_pct: "),
                         "porcentaje de visitantes que terminan comprando."]),
            ]),
        ])), md=7),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Cómo navegar el dashboard", className="mb-3"),
            html.Ol([
                html.Li("Exploración: distribución, evolución temporal y outliers."),
                html.Li("Contraste de hipótesis: ¿el gasto alto genera más ventas?"),
                html.Li("Regresión: modelo lineal simple, múltiple y con "
                        "regularización (Ridge)."),
                html.Li("Clasificación: ¿se puede predecir si un mes será de "
                        "\"venta alta\"?"),
                html.Li("Simulador: estima ventas y probabilidad de venta alta "
                        "para un presupuesto hipotético."),
            ]),
            html.P("Todos los gráficos son interactivos: use los filtros, "
                   "deslizadores y menús desplegables de cada pestaña para "
                   "explorar los datos.", className="text-muted small mb-0"),
        ])), md=5),
    ], className="g-3"),
], fluid=True, className="py-3")

# ---------------------------------------------------------------------------
# 5.2 PESTAÑA — EXPLORACIÓN DE DATOS
# ---------------------------------------------------------------------------

tab_exploracion = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Label("Rango de meses a analizar", className="fw-bold"),
            dcc.RangeSlider(
                id="rango-fechas",
                min=0, max=len(df) - 1, step=1,
                value=[0, len(df) - 1],
                marks={i: d.strftime("%Y") for i, d in enumerate(df["mes"])
                       if d.month == 1},
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ], md=8),
        dbc.Col([
            html.Label("Resaltar valores atípicos", className="fw-bold"),
            dcc.Checklist(
                id="chk-outliers",
                options=[{"label": " Resaltar meses atípicos (|z| > 2)", "value": "on"}],
                value=["on"],
                className="mt-2",
            ),
        ], md=4),
    ], className="mb-3 g-3"),

    dbc.Row([
        dbc.Col(dcc.Graph(id="graf-histograma"), md=6),
        dbc.Col(dcc.Graph(id="graf-evolucion"), md=6),
    ], className="g-3"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(id="resumen-eda")), md=12),
    ], className="g-3 mt-1"),
], fluid=True, className="py-3")

# ---------------------------------------------------------------------------
# 5.3 PESTAÑA — CONTRASTE DE HIPÓTESIS
# ---------------------------------------------------------------------------

tab_hipotesis = dbc.Container([
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("H₀ y H₁"),
            html.P("H₀: μ_alto ≤ μ_bajo  (el gasto no incrementa las ventas)"),
            html.P("H₁: μ_alto > μ_bajo  (el gasto sí incrementa las ventas)"),
            html.P("Prueba t de Welch (una cola), α = 0.05. Los meses se dividen "
                   f"según la mediana del gasto publicitario ({mediana_gasto:.2f} "
                   "millones COP)."),
            html.Div([
                html.Label("Incluir los 2 meses atípicos en la prueba", className="fw-bold"),
                dcc.RadioItems(
                    id="radio-outliers-hip",
                    options=[
                        {"label": " Incluir outliers (dataset completo)", "value": "con"},
                        {"label": " Excluir outliers (prueba de robustez)", "value": "sin"},
                    ],
                    value="con",
                    className="mt-2",
                ),
            ], className="mt-3"),
        ])), md=4),
        dbc.Col(dcc.Graph(id="graf-boxplot"), md=8),
    ], className="g-3 mb-3"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(id="resultado-hipotesis")), md=12),
    ], className="g-3"),
], fluid=True, className="py-3")

# ---------------------------------------------------------------------------
# 5.4 PESTAÑA — REGRESIÓN (SIMPLE / MÚLTIPLE / RIDGE)
# ---------------------------------------------------------------------------

tab_regresion = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Label("Modelo de regresión", className="fw-bold"),
            dcc.Dropdown(
                id="dd-modelo",
                options=[
                    {"label": "Regresión lineal simple (solo gasto en publicidad)",
                     "value": "simple"},
                    {"label": "Regresión lineal múltiple (gasto + visitas + conversión)",
                     "value": "multiple"},
                    {"label": "Regresión Ridge (múltiple + regularización L2)",
                     "value": "ridge"},
                ],
                value="multiple", clearable=False,
            ),
        ], md=6),
        dbc.Col([
            html.Label("Fuerza de regularización (alpha) — solo aplica a Ridge",
                       className="fw-bold"),
            dcc.Slider(id="slider-alpha", min=0.1, max=50, step=0.1, value=5,
                       marks={0.1: "0.1", 5: "5", 10: "10", 20: "20", 50: "50"},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], md=6),
    ], className="mb-3 g-3"),

    dbc.Row([
        dbc.Col(dcc.Graph(id="graf-dispersion-modelo"), md=7),
        dbc.Col(dbc.Card(dbc.CardBody(id="metricas-modelo")), md=5),
    ], className="g-3"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(id="explicacion-modelo")), md=12),
    ], className="g-3 mt-1"),
], fluid=True, className="py-3")

# ---------------------------------------------------------------------------
# 5.5 PESTAÑA — CLASIFICACIÓN (REGRESIÓN LOGÍSTICA)
# ---------------------------------------------------------------------------

tab_logistica = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Label("Umbral de decisión (probabilidad ≥ umbral → \"venta alta\")",
                       className="fw-bold"),
            dcc.Slider(id="slider-umbral", min=0.1, max=0.9, step=0.05, value=0.6,
                       marks={i / 10: str(i / 10) for i in range(1, 10)},
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.P("Un falso negativo (no anticipar un mes de venta alta) es más "
                   "costoso para logística e inventario que un falso positivo, por "
                   "lo que se prioriza la sensibilidad (recall).",
                   className="text-muted small mt-2"),
        ], md=12),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(dcc.Graph(id="graf-sigmoide"), md=7),
        dbc.Col(dcc.Graph(id="graf-matriz-confusion"), md=5),
    ], className="g-3"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(id="metricas-logistica")), md=12),
    ], className="g-3 mt-1"),
], fluid=True, className="py-3")

# ---------------------------------------------------------------------------
# 5.6 PESTAÑA — SIMULADOR
# ---------------------------------------------------------------------------

tab_simulador = dbc.Container([
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Simulador para el equipo financiero", className="mb-2"),
            html.P("Ajuste un presupuesto hipotético de publicidad y vea la venta "
                   "esperada y la probabilidad de que sea un mes de \"venta alta\", "
                   "según los modelos entrenados con los 60 meses históricos.",
                   className="text-muted"),
            html.Label("Gasto hipotético en publicidad (millones COP)", className="fw-bold"),
            dcc.Slider(
                id="slider-simulador",
                min=float(df["gasto_publicidad_millones"].min()),
                max=float(df["gasto_publicidad_millones"].max()) + 5,
                step=0.5,
                value=float(df["gasto_publicidad_millones"].median()),
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Div(id="aviso-extrapolacion", className="text-warning small mt-2"),
        ])), md=12),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(dcc.Graph(id="graf-gauge-ventas"), md=6),
        dbc.Col(dcc.Graph(id="graf-gauge-prob"), md=6),
    ], className="g-3"),
], fluid=True, className="py-3")

# ---------------------------------------------------------------------------
# 5.7 PESTAÑA — CONCLUSIONES
# ---------------------------------------------------------------------------

tab_conclusiones = dbc.Container([
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Principales hallazgos", className="mb-3"),
            html.Ul([
                html.Li("Existe una relación positiva y estadísticamente muy "
                        "significativa entre el gasto en publicidad y las ventas "
                        "(prueba t de Welch, p ≈ 1.18 × 10⁻⁸; Mann-Whitney U, "
                        "p ≈ 2.29 × 10⁻⁹), con un tamaño del efecto grande "
                        "(d de Cohen ≈ 1.68)."),
                html.Li("La regresión lineal simple explica ~51.5 % de la "
                        "variabilidad de las ventas (R² = 0.515); la regresión "
                        "múltiple, evaluada con validación cruzada (5 folds), "
                        "alcanza en promedio R² ≈ 0.45, y sube a R² ≈ 0.48 al "
                        "introducir regularización Ridge (α ≈ 5), que además "
                        "corrige el signo contraintuitivo de visitas_web_miles "
                        "causado por la multicolinealidad (VIF ≈ 7.5) con el "
                        "gasto publicitario."),
                html.Li("Un único split 70/30 es muy sensible a dónde caen los dos "
                        "meses atípicos (campaña viral y falla operativa); la "
                        "validación cruzada da una estimación más estable del "
                        "desempeño real del modelo."),
                html.Li("La regresión logística, usando solo el gasto en "
                        "publicidad, predice si un mes será de \"venta alta\" "
                        "con 94.4 % de exactitud y sensibilidad perfecta "
                        "(umbral = 0.6), priorizando no dejar pasar meses de alta "
                        "demanda."),
            ]),
            html.H5("Recomendación para el equipo financiero", className="mt-3 mb-2"),
            html.P("La evidencia respalda mantener o incrementar moderadamente el "
                   "presupuesto de publicidad digital: cada millón adicional "
                   "invertido se asocia, en promedio, con un aumento cercano a "
                   "3.4–3.9 millones COP en ventas mensuales. Sin embargo, la "
                   "relación no es puramente lineal ni exclusivamente causal: se "
                   "recomienda (1) recolectar variables de estacionalidad y "
                   "competencia, (2) monitorear el retorno marginal para detectar "
                   "rendimientos decrecientes, y (3) usar el modelo con "
                   "regularización y validación cruzada — no un único split — "
                   "como base para decisiones presupuestales recurrentes."),
        ])), md=12),
    ]),
], fluid=True, className="py-3")

# ---------------------------------------------------------------------------
# 6. LAYOUT PRINCIPAL
# ---------------------------------------------------------------------------

app.layout = html.Div([
    encabezado,
    dbc.Container([
        dbc.Tabs([
            dbc.Tab(tab_contexto, label="🏠 Contexto"),
            dbc.Tab(tab_exploracion, label="🔍 Exploración"),
            dbc.Tab(tab_hipotesis, label="⚖️ Contraste de hipótesis"),
            dbc.Tab(tab_regresion, label="📈 Regresión"),
            dbc.Tab(tab_logistica, label="🎯 Clasificación"),
            dbc.Tab(tab_simulador, label="🧮 Simulador"),
            dbc.Tab(tab_conclusiones, label="✅ Conclusiones"),
        ]),
        html.Footer(
            html.P("Fundación Universitaria Compensar · Ingeniería en Ciencia de "
                   "Datos · Programación para Ciencia de Datos II · 2026",
                   className="text-center text-muted small mt-4 mb-3"),
        ),
    ], fluid=True),
])

# ===========================================================================
# 7. CALLBACKS
# ===========================================================================

# --- 7.1 Exploración de datos -----------------------------------------------

@app.callback(
    Output("graf-histograma", "figure"),
    Output("graf-evolucion", "figure"),
    Output("resumen-eda", "children"),
    Input("rango-fechas", "value"),
    Input("chk-outliers", "value"),
)
def actualizar_eda(rango, chk):
    i0, i1 = rango
    d = df.iloc[i0:i1 + 1].copy()
    resaltar = "on" in (chk or [])

    fig_hist = px.histogram(d, x=TARGET, nbins=12, color_discrete_sequence=[PALETA["primario"]])
    fig_hist.update_traces(marker_line_color="white", marker_line_width=1)
    fig_hist = figura_base(fig_hist, "Distribución de las ventas mensuales")
    fig_hist.update_xaxes(title="Ventas (millones COP)")
    fig_hist.update_yaxes(title="Frecuencia (n.º de meses)")

    fig_evo = go.Figure()
    fig_evo.add_trace(go.Scatter(x=d["mes"], y=d[TARGET], mode="lines+markers",
                                  line=dict(color=PALETA["secundario"]),
                                  name="Ventas mensuales"))
    if resaltar:
        atipicos = d.loc[d.index.intersection(outlier_idx)]
        if len(atipicos):
            fig_evo.add_trace(go.Scatter(
                x=atipicos["mes"], y=atipicos[TARGET], mode="markers",
                marker=dict(color=PALETA["acento"], size=13, symbol="star"),
                name="Mes atípico (|z| > 2)"))
    fig_evo = figura_base(fig_evo, "Evolución mensual de las ventas")
    fig_evo.update_xaxes(title="Mes")
    fig_evo.update_yaxes(title="Ventas (millones COP)")

    resumen = dbc.Row([
        dbc.Col([
            html.H6("Gasto en publicidad (millones COP)"),
            html.P(f"Media: {d['gasto_publicidad_millones'].mean():.2f} · "
                   f"Mediana: {d['gasto_publicidad_millones'].median():.2f} · "
                   f"Desv. estándar: {d['gasto_publicidad_millones'].std(ddof=1):.2f}"),
        ], md=6),
        dbc.Col([
            html.H6("Ventas (millones COP)"),
            html.P(f"Media: {d[TARGET].mean():.2f} · "
                   f"Mediana: {d[TARGET].median():.2f} · "
                   f"Desv. estándar: {d[TARGET].std(ddof=1):.2f}"),
        ], md=6),
    ])
    return fig_hist, fig_evo, resumen


# --- 7.2 Contraste de hipótesis --------------------------------------------

@app.callback(
    Output("graf-boxplot", "figure"),
    Output("resultado-hipotesis", "children"),
    Input("radio-outliers-hip", "value"),
)
def actualizar_hipotesis(modo):
    d = df.drop(index=outlier_idx) if modo == "sin" else df

    grupo_alto = d.loc[d["gasto_publicidad_millones"] > mediana_gasto, TARGET]
    grupo_bajo = d.loc[d["gasto_publicidad_millones"] <= mediana_gasto, TARGET]

    t_stat, p_val = stats.ttest_ind(grupo_alto, grupo_bajo, equal_var=False,
                                     alternative="greater")
    u_stat, p_mw = stats.mannwhitneyu(grupo_alto, grupo_bajo, alternative="greater")

    n1, n2 = len(grupo_alto), len(grupo_bajo)
    s1, s2 = grupo_alto.std(ddof=1), grupo_bajo.std(ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    cohen_d = (grupo_alto.mean() - grupo_bajo.mean()) / sp

    fig = go.Figure()
    fig.add_trace(go.Box(y=grupo_bajo, name="Bajo gasto", marker_color=PALETA["primario"]))
    fig.add_trace(go.Box(y=grupo_alto, name="Alto gasto", marker_color=PALETA["secundario"]))
    fig = figura_base(fig, "Ventas mensuales según nivel de gasto en publicidad")
    fig.update_yaxes(title="Ventas (millones COP)")

    decision = "Se rechaza H₀" if p_val < 0.05 else "No se rechaza H₀"
    color_decision = PALETA["exito"] if p_val < 0.05 else PALETA["acento"]

    resultado = dbc.Row([
        dbc.Col([
            html.H6(f"n = {n1 + n2} meses ({n1} alto gasto / {n2} bajo gasto)"),
            html.P(f"Media alto gasto: {grupo_alto.mean():.2f} millones COP  |  "
                   f"Media bajo gasto: {grupo_bajo.mean():.2f} millones COP"),
            html.P(f"Diferencia de medias: {grupo_alto.mean() - grupo_bajo.mean():.2f} "
                   "millones COP"),
        ], md=4),
        dbc.Col([
            html.P(f"Prueba t de Welch: t = {t_stat:.3f}, p (una cola) = {p_val:.2e}"),
            html.P(f"Mann-Whitney U: U = {u_stat:.1f}, p (una cola) = {p_mw:.2e}"),
            html.P(f"Tamaño del efecto (d de Cohen): {cohen_d:.3f}"),
        ], md=4),
        dbc.Col([
            html.H5(decision, style={"color": color_decision}),
            html.P("Con α = 0.05, existe evidencia de que los meses de alto gasto "
                   "publicitario presentan, en promedio, ventas mayores que los "
                   "meses de bajo gasto." if p_val < 0.05 else
                   "No hay evidencia suficiente para afirmar que el alto gasto "
                   "publicitario genera mayores ventas."),
        ], md=4),
    ])
    return fig, resultado


# --- 7.3 Regresión -----------------------------------------------------------

@app.callback(
    Output("graf-dispersion-modelo", "figure"),
    Output("metricas-modelo", "children"),
    Output("explicacion-modelo", "children"),
    Input("dd-modelo", "value"),
    Input("slider-alpha", "value"),
)
def actualizar_regresion(tipo_modelo, alpha):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    if tipo_modelo == "simple":
        X = df[["gasto_publicidad_millones"]].to_numpy()
        cv_scores = cross_val_score(LinearRegression(), X, y_full, cv=kf, scoring="r2")
        modelo = LinearRegression().fit(X, y_full)
        y_pred = modelo.predict(X)
        nombre = "Regresión lineal simple"
        texto_modelo = (f"ventas = {modelo.intercept_:.2f} + "
                        f"{modelo.coef_[0]:.3f} × gasto_publicidad")
        explicacion = ("Modelo de referencia de la etapa de contextualización: usa "
                       "únicamente el gasto en publicidad como predictor. Es el más "
                       "simple de interpretar, pero deja sin explicar ~48 % de la "
                       "variabilidad de las ventas.")

    elif tipo_modelo == "multiple":
        X = X_full
        cv_scores = cross_val_score(LinearRegression(), X, y_full, cv=kf, scoring="r2")
        modelo = LinearRegression().fit(X, y_full)
        y_pred = modelo.predict(X)
        nombre = "Regresión lineal múltiple"
        coefs = dict(zip(FEATURES, modelo.coef_))
        texto_modelo = (f"ventas = {modelo.intercept_:.2f} + "
                        f"{coefs['gasto_publicidad_millones']:.2f}×gasto + "
                        f"{coefs['visitas_web_miles']:.2f}×visitas + "
                        f"{coefs['tasa_conversion_pct']:.2f}×conversión")
        explicacion = ("Incorpora visitas al sitio y tasa de conversión. Sin "
                       "regularizar, la fuerte correlación entre gasto y visitas "
                       "(VIF ≈ 7.5) produce coeficientes inestables — por eso el "
                       "coeficiente de 'visitas' puede aparecer negativo, algo "
                       "contraintuitivo que la regularización corrige.")

    else:  # ridge
        scaler = StandardScaler().fit(X_full)
        Xs = scaler.transform(X_full)
        cv_scores = cross_val_score(Ridge(alpha=alpha), Xs, y_full, cv=kf, scoring="r2")
        modelo = Ridge(alpha=alpha).fit(Xs, y_full)
        y_pred = modelo.predict(Xs)
        nombre = f"Regresión Ridge (α = {alpha:g})"
        coefs = dict(zip(FEATURES, modelo.coef_))
        texto_modelo = (f"ventas = {modelo.intercept_:.2f} + "
                        f"{coefs['gasto_publicidad_millones']:.2f}×gasto_z + "
                        f"{coefs['visitas_web_miles']:.2f}×visitas_z + "
                        f"{coefs['tasa_conversion_pct']:.2f}×conversión_z  "
                        "(variables estandarizadas)")
        explicacion = ("La regularización L2 (Ridge) penaliza coeficientes grandes "
                       "y reparte el efecto entre variables correlacionadas de "
                       "forma más estable, controlando la multicolinealidad entre "
                       "gasto y visitas sin necesidad de eliminar ninguna variable. "
                       "Ajuste el deslizador de alpha para ver el efecto: valores "
                       "muy altos sobre-regularizan (subajuste); valores muy bajos "
                       "se acercan a la regresión múltiple sin regularizar.")

    r2_insample = r2_score(y_full, y_pred)
    mse_cv = mean_squared_error(y_full, y_pred)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_full, y=y_pred, mode="markers",
                              marker=dict(color=PALETA["primario"], size=9, opacity=0.75),
                              name="Meses"))
    lims = [min(y_full.min(), y_pred.min()) - 10, max(y_full.max(), y_pred.max()) + 10]
    fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                              line=dict(color=PALETA["acento"], dash="dash"),
                              name="Predicción perfecta (y = x)"))
    fig = figura_base(fig, f"{nombre}: ventas reales vs. predichas")
    fig.update_xaxes(title="Ventas reales (millones COP)")
    fig.update_yaxes(title="Ventas predichas (millones COP)")

    metricas = html.Div([
        html.H5(nombre),
        html.P(texto_modelo, className="small text-muted"),
        html.Table([
            html.Tr([html.Td("R² (todas las observaciones)"),
                     html.Td(f"{r2_insample:.3f}", className="fw-bold")]),
            html.Tr([html.Td("R² promedio (validación cruzada, 5 folds)"),
                     html.Td(f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}",
                             className="fw-bold")]),
            html.Tr([html.Td("MSE (todas las observaciones)"),
                     html.Td(f"{mse_cv:.2f}", className="fw-bold")]),
        ], className="table table-sm mt-2"),
    ])

    return fig, metricas, explicacion


# --- 7.4 Clasificación (regresión logística) --------------------------------

@app.callback(
    Output("graf-sigmoide", "figure"),
    Output("graf-matriz-confusion", "figure"),
    Output("metricas-logistica", "children"),
    Input("slider-umbral", "value"),
)
def actualizar_logistica(umbral):
    Xtr, Xte, ytr, yte = train_test_split(Xl, yl, test_size=0.3, random_state=42,
                                           stratify=yl)
    modelo = LogisticRegression(C=1, solver="lbfgs").fit(Xtr, ytr)
    proba_te = modelo.predict_proba(Xte)[:, 1]
    y_pred = (proba_te >= umbral).astype(int)

    acc = accuracy_score(yte, y_pred)
    prec = precision_score(yte, y_pred, zero_division=0)
    rec = recall_score(yte, y_pred, zero_division=0)
    f1 = f1_score(yte, y_pred, zero_division=0)
    cm = confusion_matrix(yte, y_pred)

    xs = np.linspace(df["gasto_publicidad_millones"].min() - 1,
                      df["gasto_publicidad_millones"].max() + 1, 200).reshape(-1, 1)
    curva = modelo.predict_proba(xs)[:, 1]

    fig_sig = go.Figure()
    fig_sig.add_trace(go.Scatter(x=xs.ravel(), y=curva, mode="lines",
                                  line=dict(color=PALETA["acento"], width=3),
                                  name="Probabilidad estimada"))
    fig_sig.add_trace(go.Scatter(x=Xtr.ravel(), y=ytr, mode="markers",
                                  marker=dict(color=PALETA["primario"], opacity=0.6),
                                  name="Entrenamiento"))
    fig_sig.add_trace(go.Scatter(x=Xte.ravel(), y=yte, mode="markers",
                                  marker=dict(color=PALETA["exito"], symbol="square"),
                                  name="Prueba"))
    fig_sig.add_hline(y=umbral, line_dash="dash", line_color="gray",
                       annotation_text=f"Umbral = {umbral}")
    fig_sig = figura_base(fig_sig, "Probabilidad de mes de venta alta según el gasto")
    fig_sig.update_xaxes(title="Gasto en publicidad (millones COP)")
    fig_sig.update_yaxes(title="Probabilidad de venta alta")

    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        labels=dict(x="Predicho", y="Real", color="n"),
                        x=["Venta normal/baja", "Venta alta"],
                        y=["Venta normal/baja", "Venta alta"])
    fig_cm = figura_base(fig_cm, f"Matriz de confusión (umbral = {umbral})")
    fig_cm.update_coloraxes(showscale=False)

    metricas = dbc.Row([
        dbc.Col(kpi_card("Exactitud", f"{acc:.1%}", "aciertos totales", PALETA["primario"]), md=3),
        dbc.Col(kpi_card("Sensibilidad (recall)", f"{rec:.1%}",
                          "meses de venta alta detectados", PALETA["exito"]), md=3),
        dbc.Col(kpi_card("Precisión", f"{prec:.1%}",
                          "aciertos entre los predichos como altos", PALETA["secundario"]), md=3),
        dbc.Col(kpi_card("F1-score", f"{f1:.3f}", "balance precisión/sensibilidad",
                          PALETA["acento"]), md=3),
    ], className="g-3")

    return fig_sig, fig_cm, metricas


# --- 7.5 Simulador -----------------------------------------------------------

@app.callback(
    Output("graf-gauge-ventas", "figure"),
    Output("graf-gauge-prob", "figure"),
    Output("aviso-extrapolacion", "children"),
    Input("slider-simulador", "value"),
)
def actualizar_simulador(gasto_hipotetico):
    visitas_prom = df["visitas_web_miles"].mean()
    conversion_prom = df["tasa_conversion_pct"].mean()
    X_sim = np.array([[gasto_hipotetico, visitas_prom, conversion_prom]])
    ventas_pred = modelo_multiple_ref.predict(X_sim)[0]

    prob_alta = modelo_logit_ref.predict_proba([[gasto_hipotetico]])[0, 1]

    fig_ventas = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(ventas_pred),
        title={"text": "Ventas mensuales esperadas (millones COP)"},
        gauge={
            "axis": {"range": [0, max(260, ventas_pred + 20)]},
            "bar": {"color": PALETA["primario"]},
            "steps": [
                {"range": [0, mediana_ventas], "color": "#E8EEF7"},
                {"range": [mediana_ventas, 260], "color": "#D6E3F5"},
            ],
            "threshold": {"line": {"color": PALETA["acento"], "width": 4},
                          "value": mediana_ventas},
        },
    ))
    fig_ventas = figura_base(fig_ventas)

    fig_prob = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(prob_alta) * 100,
        number={"suffix": "%"},
        title={"text": "Probabilidad de mes de \"venta alta\""},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": PALETA["exito"] if prob_alta >= 0.6 else PALETA["secundario"]},
            "threshold": {"line": {"color": PALETA["acento"], "width": 4}, "value": 60},
        },
    ))
    fig_prob = figura_base(fig_prob)

    aviso = ""
    if not (df["gasto_publicidad_millones"].min() <= gasto_hipotetico
            <= df["gasto_publicidad_millones"].max()):
        aviso = ("⚠️ Este valor está fuera del rango histórico observado "
                 f"({df['gasto_publicidad_millones'].min():.1f}–"
                 f"{df['gasto_publicidad_millones'].max():.1f} millones COP); "
                 "la predicción es una extrapolación del modelo.")

    return fig_ventas, fig_prob, aviso


# ---------------------------------------------------------------------------
# 8. EJECUCIÓN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
