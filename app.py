# -*- coding: utf-8 -*-
"""
Dashboard - Proyecto Final: Gasto en publicidad digital y ventas en e-commerce
Programación para Ciencia de Datos II - U Compensar

Autora: María Fernanda Herrera Contreras
Docente: William Eduardo Clavijo Bohorquez
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

# Paleta de colores moderna y profesional
PALETA = {
    "primario": "#6366F1",     # Indigo Electrónico
    "secundario": "#0EA5E9",   # Azul Sky / Cyan
    "acento": "#F43F5E",       # Rose / Rojo Coral
    "exito": "#10B981",        # Esmeralda
    "warning": "#F59E0B",      # Ámbar
    "fondo": "#F8FAFC",        # Slate 50
    "texto": "#0F172A",        # Slate 900
    "muted": "#64748B",        # Slate 500
}

# ---------------------------------------------------------------------------
# 2. MODELOS BASE DE REFERENCIA
# ---------------------------------------------------------------------------

X_full = df[FEATURES].to_numpy()
y_full = df[TARGET].to_numpy()

X_simple = df[["gasto_publicidad_millones"]].to_numpy()
modelo_simple = LinearRegression().fit(X_simple, y_full)
r_pearson = np.corrcoef(df["gasto_publicidad_millones"], df[TARGET])[0, 1]
r2_simple = r2_score(y_full, modelo_simple.predict(X_simple))

# Regresión logística de referencia
Xl = df[["gasto_publicidad_millones"]].to_numpy()
yl = df["venta_alta"].to_numpy()
modelo_logit_ref = LogisticRegression(C=1, solver="lbfgs").fit(Xl, yl)

# Regresión lineal múltiple de referencia
modelo_multiple_ref = LinearRegression().fit(X_full, y_full)

# ---------------------------------------------------------------------------
# 3. FUNCIONES AUXILIARES DE DISEÑO
# ---------------------------------------------------------------------------

def kpi_card(titulo, valor, subtitulo, color):
    return dbc.Card(
        dbc.CardBody([
            html.Div(titulo, className="kpi-titulo"),
            html.Div(valor, className="kpi-valor", style={"color": color}),
            html.Div(subtitulo, className="kpi-subtitulo"),
        ]),
        className="kpi-card shadow-sm",
        style={"--kpi-color": color}
    )


def figura_base(fig, titulo=None):
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Plus Jakarta Sans, -apple-system, sans-serif", size=12, color=PALETA["texto"]),
        margin=dict(l=50, r=30, t=65 if titulo else 25, b=45),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.01, font=dict(size=11, color=PALETA["muted"])),
        hoverlabel=dict(bgcolor=PALETA["texto"], font_size=12, font_family="Plus Jakarta Sans", font_color="white"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9", zerolinecolor="#E2E8F0")
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", zerolinecolor="#E2E8F0")
    if titulo:
        fig.update_layout(title=dict(text=f"<b>{titulo}</b>", x=0.01, y=0.98, font=dict(size=14, color=PALETA["texto"])))
    return fig

# ---------------------------------------------------------------------------
# 4. INICIALIZACIÓN DE LA APP CON RECURSOS ESTÉTICOS
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap",
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
    ],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    title="Analytics | Publicidad & Ventas E-commerce",
)
server = app.server

# ---------------------------------------------------------------------------
# 5. LAYOUT — ENCABEZADO MODERNO
# ---------------------------------------------------------------------------

encabezado = dbc.Navbar(
    dbc.Container([
        html.Div([
            html.Div([
                html.I(className="fa-solid fa-chart-line text-indigo me-2 fs-4", style={"color": PALETA["primario"]}),
                html.H4("Analytics & Impacto Publicitario — E-commerce", className="text-white d-inline-block fw-bold align-middle mb-0"),
                html.Span("PROYECTO FINAL", className="brand-badge ms-3 align-middle")
            ], className="d-flex align-items-center mb-1"),
            html.Small("Programación para Ciencia de Datos II · Autora: María Fernanda Herrera Contreras · Docente: William Eduardo Clavijo Bohorquez",
                       className="text-white-50 font-monospace")
        ])
    ], fluid=True),
    className="app-header mb-4",
    dark=True,
)

# ---------------------------------------------------------------------------
# 5.1 PESTAÑA — CONTEXTO
# ---------------------------------------------------------------------------

tab_contexto = dbc.Container([
    dbc.Row([
        dbc.Col(kpi_card("Meses analizados", f"{len(df)}", "Enero 2021 – Diciembre 2025", PALETA["primario"]), md=3),
        dbc.Col(kpi_card("Correlación Gasto–Ventas", f"r = {r_pearson:.2f}", "Asociación positiva fuerte", PALETA["secundario"]), md=3),
        dbc.Col(kpi_card("Poder explicativo", f"R² = {r2_simple:.2f}", "Regresión lineal simple", PALETA["exito"]), md=3),
        dbc.Col(kpi_card("Meses atípicos", f"{len(outlier_idx)}", "Campaña viral & Falla operativa", PALETA["acento"]), md=3),
    ], className="g-3 mb-4"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div([
                html.I(className="fa-solid fa-bullseye me-2 text-indigo", style={"color": PALETA["primario"]}),
                html.Span("Planteamiento del problema", className="card-title-modern")
            ], className="d-flex align-items-center mb-3"),
            html.P("Una tienda de comercio electrónico invierte mensualmente en publicidad digital (Meta Ads, TikTok Ads, Email Marketing). El equipo directivo requiere determinar la efectividad real de dicha inversión para generar ingresos sostenibles.", className="text-secondary"),
            html.P("Este dashboard integra tres fases metodológicas: Análisis Exploratorio de Datos (EDA), Contraste de Hipótesis y Modelado Predictivo (Regresión Lineal Múltiple/Ridge y Clasificación Logística) para orientar decisiones financieras basadas en evidencia estadística.", className="text-secondary"),
            html.Hr(className="my-3 text-muted"),
            html.H6("Variables clave del modelo", className="fw-bold mb-2 text-dark"),
            html.Ul([
                html.Li([html.B("gasto_publicidad_millones: ", className="text-dark"), "Inversión mensual en pauta digital (millones COP)."]),
                html.Li([html.B("ventas_millones: ", className="text-dark"), "Ingresos totales por ventas en el mes (millones COP)."]),
                html.Li([html.B("visitas_web_miles: ", className="text-dark"), "Tráfico total recibido en el sitio web (miles)."]),
                html.Li([html.B("tasa_conversion_pct: ", className="text-dark"), "Porcentaje de visitantes que completan una compra."]),
            ], className="text-secondary mb-0 ps-3"),
        ])), md=7),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div([
                html.I(className="fa-solid fa-compass me-2 text-indigo", style={"color": PALETA["primario"]}),
                html.Span("Guía de navegación", className="card-title-modern")
            ], className="d-flex align-items-center mb-3"),
            html.Ol([
                html.Li([html.B("Exploración: "), "Distribución de ventas, evolución temporal y detección de valores atípicos."]),
                html.Li([html.B("Contraste de hipótesis: "), "Evaluación estadística del impacto de alto vs. bajo gasto."]),
                html.Li([html.B("Regresión: "), "Modelos lineales, multivariados y regularización Ridge para mitigar colinealidad."]),
                html.Li([html.B("Clasificación: "), "Regresión logística para estimar la probabilidad de un mes de venta alta."]),
                html.Li([html.B("Simulador: "), "Proyección en tiempo real de ingresos según el presupuesto publicitario asignado."]),
            ], className="text-secondary ps-3 mb-3"),
            html.Div([
                html.I(className="fa-solid fa-lightbulb me-2 text-warning"),
                html.Small("Todos los gráficos son interactivos. Modifique los controles y deslizadores para evaluar escenarios.", className="text-muted")
            ], className="p-2 rounded bg-light border d-flex align-items-center"),
        ])), md=5),
    ], className="g-3"),
], fluid=True, className="py-2")

# ---------------------------------------------------------------------------
# 5.2 PESTAÑA — EXPLORACIÓN DE DATOS
# ---------------------------------------------------------------------------

tab_exploracion = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Label("Rango de meses a analizar:", className="fw-semibold mb-0 me-2"),
                html.Span(id="badge-rango-fechas", className="badge px-3 py-1 rounded-pill shadow-sm", style={"backgroundColor": "#6366F1", "color": "#FFFFFF", "fontSize": "0.85rem", "fontWeight": "600"})
            ], className="d-flex align-items-center mb-2"),
            dcc.RangeSlider(
                id="rango-fechas",
                min=0, max=len(df) - 1, step=1,
                value=[0, len(df) - 1],
                marks={
                    0: "Ene 2021",
                    12: "Ene 2022",
                    24: "Ene 2023",
                    36: "Ene 2024",
                    48: "Ene 2025",
                    59: "Dic 2025"
                },
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ], md=8),
        dbc.Col([
            html.Label("Filtro de valores atípicos", className="fw-semibold mb-2"),
            dcc.Checklist(
                id="chk-outliers",
                options=[{"label": " Resaltar meses atípicos (|z| > 2)", "value": "on"}],
                value=["on"],
                className="mt-1 text-secondary",
            ),
        ], md=4),
    ], className="mb-4 g-3 p-3 bg-white rounded-3 border"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="graf-histograma"))), md=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="graf-evolucion"))), md=6),
    ], className="g-3"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(id="resumen-eda")), md=12),
    ], className="g-3 mt-1"),
], fluid=True, className="py-2")

# ---------------------------------------------------------------------------
# 5.3 PESTAÑA — CONTRASTE DE HIPÓTESIS
# ---------------------------------------------------------------------------

tab_hipotesis = dbc.Container([
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Formulación de Hipótesis", className="card-title-modern mb-3"),
            html.Div([
                html.P([html.Span("H₀: ", className="fw-bold text-danger"), "μ_alto ≤ μ_bajo (El gasto no incrementa las ventas significativas)"]),
                html.P([html.Span("H₁: ", className="fw-bold text-success"), "μ_alto > μ_bajo (El alto gasto incrementa sustancialmente las ventas)"]),
            ], className="p-3 bg-light rounded border mb-3"),
            html.P(f"Prueba t de Welch (una cola) y Mann-Whitney U, con α = 0.05. Punto de cohorte por mediana de gasto ({mediana_gasto:.2f} M COP).", className="text-muted small"),
            html.Div([
                html.Label("Tratamiento de valores atípicos", className="fw-semibold mb-2"),
                dcc.RadioItems(
                    id="radio-outliers-hip",
                    options=[
                        {"label": " Incluir outliers (Dataset completo)", "value": "con"},
                        {"label": " Excluir outliers (Prueba de robustez)", "value": "sin"},
                    ],
                    value="con",
                    className="text-secondary",
                ),
            ], className="mt-3 pt-3 border-top"),
        ])), md=4),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="graf-boxplot"))), md=8),
    ], className="g-3 mb-3"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(id="resultado-hipotesis")), md=12),
    ], className="g-3"),
], fluid=True, className="py-2")

# ---------------------------------------------------------------------------
# 5.4 PESTAÑA — REGRESIÓN
# ---------------------------------------------------------------------------

tab_regresion = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Label("Selección de Modelo de Regresión", className="fw-semibold mb-2"),
            dcc.Dropdown(
                id="dd-modelo",
                options=[
                    {"label": "Regresión Lineal Simple (Solo Gasto Publicitario)", "value": "simple"},
                    {"label": "Regresión Lineal Múltiple (Gasto + Visitas + Conversión)", "value": "multiple"},
                    {"label": "Regresión Ridge (Múltiple + Regularización L2)", "value": "ridge"},
                ],
                value="multiple", clearable=False,
            ),
        ], md=12),
    ], className="mb-4 g-3 p-3 bg-white rounded-3 border"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="graf-dispersion-modelo"))), md=7),
        dbc.Col(dbc.Card(dbc.CardBody(id="metricas-modelo")), md=5),
    ], className="g-3"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(id="explicacion-modelo")), md=12),
    ], className="g-3 mt-1"),
], fluid=True, className="py-2")

# ---------------------------------------------------------------------------
# 5.5 PESTAÑA — CLASIFICACIÓN
# ---------------------------------------------------------------------------

tab_logistica = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Label("Umbral de Decisión Probabilístico (Probabilidad ≥ Umbral → Venta Alta)", className="fw-semibold mb-2"),
            dcc.Slider(id="slider-umbral", min=0.1, max=0.9, step=0.05, value=0.6,
                       marks={i / 10: str(i / 10) for i in range(1, 10)},
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.P("Priorizamos la sensibilidad (Recall) para evitar falsos negativos en temporadas de alta demanda operacional.", className="text-muted small mt-2 mb-0"),
        ], md=12),
    ], className="mb-4 p-3 bg-white rounded-3 border"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="graf-sigmoide"))), md=7),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="graf-matriz-confusion"))), md=5),
    ], className="g-3"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(id="metricas-logistica")), md=12),
    ], className="g-3 mt-1"),
], fluid=True, className="py-2")

# ---------------------------------------------------------------------------
# 5.6 PESTAÑA — SIMULADOR INTERACTIVO
# ---------------------------------------------------------------------------

tab_simulador = dbc.Container([
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Simulador Presupuestal Financiero", className="card-title-modern mb-2"),
            html.P("Ajuste la inversión mensual estimada para proyectar las ventas esperadas y la probabilidad de registrar un mes de venta alta.", className="text-secondary small mb-3"),
            html.Label("Inversión mensual en publicidad (Millones COP)", className="fw-semibold mb-2"),
            dcc.Slider(
                id="slider-simulador",
                min=float(df["gasto_publicidad_millones"].min()),
                max=float(df["gasto_publicidad_millones"].max()) + 5,
                step=0.5,
                value=float(df["gasto_publicidad_millones"].median()),
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Div(id="aviso-extrapolacion", className="mt-3"),
        ])), md=12),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="graf-gauge-ventas"))), md=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="graf-gauge-prob"))), md=6),
    ], className="g-3"),
], fluid=True, className="py-2")

# ---------------------------------------------------------------------------
# 5.7 PESTAÑA — CONCLUSIONES Y RECOMENDACIONES
# ---------------------------------------------------------------------------

tab_conclusiones = dbc.Container([
    dbc.Row([
        # Módulo 1: Hallazgos Estadísticos
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div([
                html.I(className="fa-solid fa-chart-pie me-2 fs-5", style={"color": PALETA["primario"]}),
                html.Span("Principales Hallazgos Estadísticos", className="card-title-modern")
            ], className="d-flex align-items-center mb-3 pb-2 border-bottom"),
            html.Ul([
                html.Li([
                    html.B("Impacto en Ventas: "),
                    "Existe una relación altamente significativa entre inversión publicitaria e ingresos (Welch p ≈ 1.18×10⁻⁸; Mann-Whitney p ≈ 2.29×10⁻⁹) con un tamaño de efecto grande (Cohen's d = 1.68)."
                ], className="mb-3 text-secondary small"),
                html.Li([
                    html.B("Modelos de Regresión: "),
                    "El modelo lineal simple explica el 51.5% de la variabilidad. El modelo múltiple evaluado con Validación Cruzada (5-fold) alcanza R² ≈ 0.45 y sube a R² ≈ 0.48 con regularización Ridge (α = 5), corrigiendo la multicolinealidad estructural (VIF ≈ 7.5)."
                ], className="mb-3 text-secondary small"),
                html.Li([
                    html.B("Clasificación Logística: "),
                    "Con un umbral óptimo (0.60), la regresión logística predice los meses de alta venta con 94.4% de exactitud y 100% de sensibilidad."
                ], className="mb-0 text-secondary small"),
            ], className="ps-3 mb-0"),
        ]), className="h-100"), md=6),

        # Módulo 2: Recomendaciones Estratégicas
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div([
                html.I(className="fa-solid fa-lightbulb me-2 fs-5", style={"color": PALETA["exito"]}),
                html.Span("Recomendaciones para Decisiones", className="card-title-modern")
            ], className="d-flex align-items-center mb-3 pb-2 border-bottom"),
            html.Div([
                html.H6("Retorno Estimado de Inversión", className="fw-bold text-dark mb-2"),
                html.P("La evidencia cuantitativa valida la rentabilidad del canal digital: cada millón COP adicional invertido en publicidad proyecta un incremento estimado entre 3.4 y 3.9 millones COP en ventas brutas.", className="text-secondary small mb-0"),
            ], className="mb-3 p-3 bg-light rounded border"),
            html.Div([
                html.H6("Estándar de Modelación Corporativa", className="fw-bold text-dark mb-2"),
                html.P("Se aconseja utilizar la regularización Ridge con validación cruzada como norma corporativa para la asignación presupuestal, asegurando proyecciones estables ante la volatilidad del mercado.", className="text-secondary small mb-0"),
            ], className="p-3 bg-light rounded border"),
        ]), className="h-100"), md=6),
    ], className="g-3 d-flex align-items-stretch"),
], fluid=True, className="py-2")

# ---------------------------------------------------------------------------
# 6. LAYOUT PRINCIPAL CON TABS VECTORIALES
# ---------------------------------------------------------------------------

app.layout = html.Div([
    encabezado,
    dbc.Container([
        dbc.Tabs([
            dbc.Tab(tab_contexto, label="Contexto", tab_id="tab-contexto", label_style={"cursor": "pointer"}),
            dbc.Tab(tab_exploracion, label="Exploración", tab_id="tab-exploracion", label_style={"cursor": "pointer"}),
            dbc.Tab(tab_hipotesis, label="Contraste de Hipótesis", tab_id="tab-hipotesis", label_style={"cursor": "pointer"}),
            dbc.Tab(tab_regresion, label="Regresión", tab_id="tab-regresion", label_style={"cursor": "pointer"}),
            dbc.Tab(tab_logistica, label="Clasificación", tab_id="tab-logistica", label_style={"cursor": "pointer"}),
            dbc.Tab(tab_simulador, label="Simulador", tab_id="tab-simulador", label_style={"cursor": "pointer"}),
            dbc.Tab(tab_conclusiones, label="Conclusiones", tab_id="tab-conclusiones", label_style={"cursor": "pointer"}),
        ], className="custom-tabs"),
        html.Footer(
            html.Div([
                html.P("Fundación Universitaria Compensar · Facultad de Ingeniería · Ciencia de Datos II · 2026", className="text-center text-muted small mb-0")
            ], className="py-3"),
        ),
    ], fluid=True),
])

# ===========================================================================
# 7. CALLBACKS DE LA APLICACIÓN
# ===========================================================================

# --- 7.1 Exploración --------------------------------------------------------

@app.callback(
    Output("graf-histograma", "figure"),
    Output("graf-evolucion", "figure"),
    Output("resumen-eda", "children"),
    Output("badge-rango-fechas", "children"),
    Input("rango-fechas", "value"),
    Input("chk-outliers", "value"),
)
def actualizar_eda(rango, chk):
    i0, i1 = rango
    d = df.iloc[i0:i1 + 1].copy()
    resaltar = "on" in (chk or [])

    fecha_ini = d["mes"].min().strftime("%b %Y").capitalize()
    fecha_fin = d["mes"].max().strftime("%b %Y").capitalize()
    texto_badge = f"{fecha_ini} – {fecha_fin} ({len(d)} meses)"

    fig_hist = px.histogram(d, x=TARGET, nbins=12, color_discrete_sequence=[PALETA["primario"]])
    fig_hist.update_traces(marker_line_color="white", marker_line_width=1.5, opacity=0.9)
    fig_hist = figura_base(fig_hist, "Distribución de Ventas Mensuales")
    fig_hist.update_xaxes(title="Ventas (millones COP)")
    fig_hist.update_yaxes(title="Frecuencia (Meses)")

    fig_evo = go.Figure()
    fig_evo.add_trace(go.Scatter(x=d["mes"], y=d[TARGET], mode="lines+markers",
                                  line=dict(color=PALETA["primario"], width=2.5),
                                  marker=dict(size=6, color=PALETA["primario"]),
                                  name="Ventas mensuales"))
    if resaltar:
        atipicos = d.loc[d.index.intersection(outlier_idx)]
        if len(atipicos):
            fig_evo.add_trace(go.Scatter(
                x=atipicos["mes"], y=atipicos[TARGET], mode="markers",
                marker=dict(color=PALETA["acento"], size=12, symbol="diamond"),
                name="Mes atípico (|z| > 2)"))
    fig_evo = figura_base(fig_evo, "Evolución Temporal de Ventas")
    fig_evo.update_xaxes(title="Fecha")
    fig_evo.update_yaxes(title="Ventas (millones COP)")

    resumen = dbc.Row([
        dbc.Col([
            html.H6("Inversión Publicitaria (Millones COP)", className="fw-bold text-dark mb-2"),
            html.Div([
                html.Span(f"Media: {d['gasto_publicidad_millones'].mean():.2f} M", className="badge bg-light text-dark border me-2 p-2"),
                html.Span(f"Mediana: {d['gasto_publicidad_millones'].median():.2f} M", className="badge bg-light text-dark border me-2 p-2"),
                html.Span(f"Desv. Est: {d['gasto_publicidad_millones'].std(ddof=1):.2f} M", className="badge bg-light text-dark border me-2 p-2"),
            ]),
        ], md=6),
        dbc.Col([
            html.H6("Ventas Totales (Millones COP)", className="fw-bold text-dark mb-2"),
            html.Div([
                html.Span(f"Media: {d[TARGET].mean():.2f} M", className="badge bg-light text-dark border me-2 p-2"),
                html.Span(f"Mediana: {d[TARGET].median():.2f} M", className="badge bg-light text-dark border me-2 p-2"),
                html.Span(f"Desv. Est: {d[TARGET].std(ddof=1):.2f} M", className="badge bg-light text-dark border me-2 p-2"),
            ]),
        ], md=6),
    ])
    return fig_hist, fig_evo, resumen, texto_badge

# --- 7.2 Hipótesis -----------------------------------------------------------

@app.callback(
    Output("graf-boxplot", "figure"),
    Output("resultado-hipotesis", "children"),
    Input("radio-outliers-hip", "value"),
)
def actualizar_hipotesis(modo):
    d = df.drop(index=outlier_idx) if modo == "sin" else df

    grupo_alto = d.loc[d["gasto_publicidad_millones"] > mediana_gasto, TARGET]
    grupo_bajo = d.loc[d["gasto_publicidad_millones"] <= mediana_gasto, TARGET]

    t_stat, p_val = stats.ttest_ind(grupo_alto, grupo_bajo, equal_var=False, alternative="greater")
    u_stat, p_mw = stats.mannwhitneyu(grupo_alto, grupo_bajo, alternative="greater")

    n1, n2 = len(grupo_alto), len(grupo_bajo)
    s1, s2 = grupo_alto.std(ddof=1), grupo_bajo.std(ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    cohen_d = (grupo_alto.mean() - grupo_bajo.mean()) / sp

    fig = go.Figure()
    fig.add_trace(go.Box(y=grupo_bajo, name="Bajo Gasto", marker_color=PALETA["secundario"], boxmean=True))
    fig.add_trace(go.Box(y=grupo_alto, name="Alto Gasto", marker_color=PALETA["primario"], boxmean=True))
    fig = figura_base(fig, "Comparativa de Ventas según Nivel de Inversión")
    fig.update_yaxes(title="Ventas (millones COP)")

    decision = "Se rechaza H₀" if p_val < 0.05 else "No se rechaza H₀"
    color_decision = PALETA["exito"] if p_val < 0.05 else PALETA["acento"]

    resultado = dbc.Row([
        dbc.Col([
            html.H6(f"Muestra: n = {n1 + n2} meses", className="fw-bold mb-2"),
            html.P(f"Promedio Alto Gasto: {grupo_alto.mean():.2f} M COP", className="small text-muted mb-1"),
            html.P(f"Promedio Bajo Gasto: {grupo_bajo.mean():.2f} M COP", className="small text-muted mb-1"),
            html.P(f"Diferencia neta: {grupo_alto.mean() - grupo_bajo.mean():.2f} M COP", className="fw-semibold text-dark mb-0"),
        ], md=4),
        dbc.Col([
            html.H6("Pruebas Estadísticas", className="fw-bold mb-2"),
            html.P(f"Welch t: {t_stat:.3f} | p-val: {p_val:.2e}", className="small mb-1"),
            html.P(f"Mann-Whitney U: {u_stat:.1f} | p-val: {p_mw:.2e}", className="small mb-1"),
            html.P(f"Efecto Cohen's d: {cohen_d:.3f} (Alto)", className="fw-semibold text-indigo mb-0"),
        ], md=4),
        dbc.Col([
            html.Div(decision, className="fw-extrabold fs-5 mb-2", style={"color": color_decision}),
            html.P("Existe suficiente evidencia estadística para concluir que la alta inversión publicitaria incrementa de forma significativa los ingresos mensuales.", className="small text-muted mb-0" if p_val < 0.05 else "No hay evidencia suficiente."),
        ], md=4),
    ])
    return fig, resultado

# --- 7.3 Regresión -----------------------------------------------------------

@app.callback(
    Output("graf-dispersion-modelo", "figure"),
    Output("metricas-modelo", "children"),
    Output("explicacion-modelo", "children"),
    Input("dd-modelo", "value"),
)
def actualizar_regresion(tipo_modelo):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    if tipo_modelo == "simple":
        X = df[["gasto_publicidad_millones"]].to_numpy()
        cv_scores = cross_val_score(LinearRegression(), X, y_full, cv=kf, scoring="r2")
        modelo = LinearRegression().fit(X, y_full)
        y_pred = modelo.predict(X)
        nombre = "Regresión Lineal Simple"
        texto_modelo = f"Ventas = {modelo.intercept_:.2f} + {modelo.coef_[0]:.3f} × Gasto"
        explicacion = "Modelo bivariado de referencia. Establece la tendencia base entre pauta e ingresos pero omite factores adicionales."

    elif tipo_modelo == "multiple":
        X = X_full
        cv_scores = cross_val_score(LinearRegression(), X, y_full, cv=kf, scoring="r2")
        modelo = LinearRegression().fit(X, y_full)
        y_pred = modelo.predict(X)
        nombre = "Regresión Lineal Múltiple"
        coefs = dict(zip(FEATURES, modelo.coef_))
        texto_modelo = f"Ventas = {modelo.intercept_:.2f} + {coefs['gasto_publicidad_millones']:.2f}×Gasto + {coefs['visitas_web_miles']:.2f}×Visitas + {coefs['tasa_conversion_pct']:.2f}×Conv"
        explicacion = "Modelo multivariado. Muestra cierta inestabilidad en coeficientes debido a alta colinealidad estructural (VIF ≈ 7.5)."

    else:
        alpha = 5.0
        scaler = StandardScaler().fit(X_full)
        Xs = scaler.transform(X_full)
        cv_scores = cross_val_score(Ridge(alpha=alpha), Xs, y_full, cv=kf, scoring="r2")
        modelo = Ridge(alpha=alpha).fit(Xs, y_full)
        y_pred = modelo.predict(Xs)
        nombre = f"Regresión Ridge (L2, α = {alpha:g})"
        coefs = dict(zip(FEATURES, modelo.coef_))
        texto_modelo = f"Ventas_z = {modelo.intercept_:.2f} + {coefs['gasto_publicidad_millones']:.2f}×Gasto_z + {coefs['visitas_web_miles']:.2f}×Visitas_z + {coefs['tasa_conversion_pct']:.2f}×Conv_z"
        explicacion = "La regularización L2 (Ridge, α = 5) penaliza coeficientes grandes y atenúa la multicolinealidad estructural entre gasto publicitario y visitas al sitio web, mejorando la estabilidad del modelo."

    r2_insample = r2_score(y_full, y_pred)
    mse_cv = mean_squared_error(y_full, y_pred)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_full, y=y_pred, mode="markers",
                              marker=dict(color=PALETA["primario"], size=9, opacity=0.8),
                              name="Observaciones"))
    lims = [min(y_full.min(), y_pred.min()) - 10, max(y_full.max(), y_pred.max()) + 10]
    fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                              line=dict(color=PALETA["acento"], dash="dash", width=1.5),
                              name="Ideal (y = x)"))
    fig = figura_base(fig, f"{nombre}: Ventas Reales vs. Predichas")
    fig.update_xaxes(title="Ventas Reales (millones COP)")
    fig.update_yaxes(title="Ventas Predichas (millones COP)")

    filas_coefs = []
    if tipo_modelo == "ridge":
        coefs = dict(zip(FEATURES, modelo.coef_))
        filas_coefs = [
            html.Tr([html.Td("β Gasto (Estandarizado)"), html.Td(f"{coefs['gasto_publicidad_millones']:.3f}", className="fw-bold text-end text-indigo")]),
            html.Tr([html.Td("β Visitas (Estandarizado)"), html.Td(f"{coefs['visitas_web_miles']:.3f}", className="fw-bold text-end text-indigo")]),
            html.Tr([html.Td("β Conversión (Estandarizado)"), html.Td(f"{coefs['tasa_conversion_pct']:.3f}", className="fw-bold text-end text-indigo")]),
        ]

    metricas = html.Div([
        html.H6(nombre, className="fw-bold mb-1"),
        html.P(texto_modelo, className="small text-muted mb-3 font-monospace"),
        html.Table([
            html.Tr([html.Td("R² (Dataset Completo)"), html.Td(f"{r2_insample:.3f}", className="fw-bold text-end")]),
            html.Tr([html.Td("R² Promedio (CV 5-fold)"), html.Td(f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}", className="fw-bold text-indigo text-end")]),
            html.Tr([html.Td("MSE (Error Cuadrático Medio)"), html.Td(f"{mse_cv:.2f}", className="fw-bold text-end")]),
            *filas_coefs
        ], className="table-modern"),
    ])

    return fig, metricas, explicacion

# --- 7.4 Clasificación -------------------------------------------------------

@app.callback(
    Output("graf-sigmoide", "figure"),
    Output("graf-matriz-confusion", "figure"),
    Output("metricas-logistica", "children"),
    Input("slider-umbral", "value"),
)
def actualizar_logistica(umbral):
    Xtr, Xte, ytr, yte = train_test_split(Xl, yl, test_size=0.3, random_state=42, stratify=yl)
    modelo = LogisticRegression(C=1, solver="lbfgs").fit(Xtr, ytr)
    proba_te = modelo.predict_proba(Xte)[:, 1]
    y_pred = (proba_te >= umbral).astype(int)

    acc = accuracy_score(yte, y_pred)
    prec = precision_score(yte, y_pred, zero_division=0)
    rec = recall_score(yte, y_pred, zero_division=0)
    f1 = f1_score(yte, y_pred, zero_division=0)
    cm = confusion_matrix(yte, y_pred)

    xs = np.linspace(df["gasto_publicidad_millones"].min() - 1, df["gasto_publicidad_millones"].max() + 1, 200).reshape(-1, 1)
    curva = modelo.predict_proba(xs)[:, 1]

    fig_sig = go.Figure()
    fig_sig.add_trace(go.Scatter(x=xs.ravel(), y=curva, mode="lines",
                                  line=dict(color=PALETA["primario"], width=3),
                                  name="Curva Sigmoide"))
    fig_sig.add_trace(go.Scatter(x=Xtr.ravel(), y=ytr, mode="markers",
                                  marker=dict(color=PALETA["muted"], opacity=0.5, size=7),
                                  name="Entrenamiento"))
    fig_sig.add_trace(go.Scatter(x=Xte.ravel(), y=yte, mode="markers",
                                  marker=dict(color=PALETA["exito"], symbol="square", size=8),
                                  name="Prueba"))
    fig_sig.add_hline(y=umbral, line_dash="dash", line_color=PALETA["acento"],
                       annotation_text=f"Umbral = {umbral}")
    fig_sig = figura_base(fig_sig, "Curva de Probabilidad Logística")
    fig_sig.update_xaxes(title="Gasto Publicitario (millones COP)")
    fig_sig.update_yaxes(title="Probabilidad de Venta Alta")

    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Purples",
                        labels=dict(x="Predicho", y="Real", color="Frecuencia"),
                        x=["Venta Normal/Baja", "Venta Alta"],
                        y=["Venta Normal/Baja", "Venta Alta"])
    fig_cm = figura_base(fig_cm, f"Matriz de Confusión (Umbral: {umbral})")
    fig_cm.update_coloraxes(showscale=False)

    metricas = dbc.Row([
        dbc.Col(kpi_card("Exactitud (Accuracy)", f"{acc:.1%}", "Aciertos globales", PALETA["primario"]), md=3),
        dbc.Col(kpi_card("Sensibilidad (Recall)", f"{rec:.1%}", "Detección de meses altos", PALETA["exito"]), md=3),
        dbc.Col(kpi_card("Precisión", f"{prec:.1%}", "Acierto sobre predicción alta", PALETA["secundario"]), md=3),
        dbc.Col(kpi_card("F1-Score", f"{f1:.3f}", "Balance de clasificación", PALETA["acento"]), md=3),
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
        number={"suffix": " M", "font": {"family": "Plus Jakarta Sans", "weight": 700}},
        title={"text": "<b>Ventas Estimadas (Millones COP)</b>", "font": {"size": 13, "family": "Plus Jakarta Sans"}},
        gauge={
            "axis": {"range": [0, max(260, ventas_pred + 20)]},
            "bar": {"color": PALETA["primario"]},
            "steps": [
                {"range": [0, mediana_ventas], "color": "#F1F5F9"},
                {"range": [mediana_ventas, 260], "color": "#EEF2FF"},
            ],
            "threshold": {"line": {"color": PALETA["acento"], "width": 3}, "value": mediana_ventas},
        },
    ))
    fig_ventas = figura_base(fig_ventas)

    fig_prob = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(prob_alta) * 100,
        number={"suffix": "%", "font": {"family": "Plus Jakarta Sans", "weight": 700}},
        title={"text": "<b>Probabilidad de Mes de Venta Alta</b>", "font": {"size": 13, "family": "Plus Jakarta Sans"}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": PALETA["exito"] if prob_alta >= 0.6 else PALETA["warning"]},
            "threshold": {"line": {"color": PALETA["acento"], "width": 3}, "value": 60},
        },
    ))
    fig_prob = figura_base(fig_prob)

    aviso = None
    if not (df["gasto_publicidad_millones"].min() <= gasto_hipotetico <= df["gasto_publicidad_millones"].max()):
        aviso = html.Div([
            html.I(className="fa-solid fa-triangle-exclamation me-2 text-warning fs-5"),
            html.Span(f"Nota: El valor ingresado ({gasto_hipotetico:.1f} M) se encuentra fuera del rango histórico observado ({df['gasto_publicidad_millones'].min():.1f} - {df['gasto_publicidad_millones'].max():.1f} M COP). La proyección corresponde a una extrapolación.", className="small text-secondary")
        ], className="p-2 px-3 rounded bg-warning-subtle border border-warning d-flex align-items-center")

    return fig_ventas, fig_prob, aviso

# ---------------------------------------------------------------------------
# 8. EJECUCIÓN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
