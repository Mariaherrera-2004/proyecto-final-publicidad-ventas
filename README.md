# Dashboard — Publicidad digital y ventas en e-commerce

Proyecto Final (Etapa de transferencia) — Programación para Ciencia de Datos II
Fundación Universitaria Compensar — Ingeniería en Ciencia de Datos

**Autora:** María Fernanda Herrera Contreras
**Docente:** William Eduardo Clavijo Bohorquez

## Contenido de la carpeta

```
proyecto/
├── app.py                   # Aplicación Dash (dashboard interactivo)
├── dashboard_binder.ipynb   # Notebook para lanzar el dashboard en Binder
├── requirements.txt         # Dependencias de Python
├── runtime.txt              # Versión de Python para Binder
├── data/
│   └── ventas_ecommerce.csv # Dataset (60 meses)
├── assets/
│   └── estilos.css          # Estilos del dashboard (Dash carga esta carpeta automáticamente)
└── README.md
```

## 1. Ejecutar el dashboard localmente (Paso a paso)

Para ejecutar este proyecto en tu computadora, abre la terminal (PowerShell o CMD en Windows) y sigue estos pasos:

### Paso 1: Ubicarse en la carpeta del proyecto
Asegúrate de estar dentro de la carpeta donde están los archivos `app.py` y `requirements.txt`.
- Si descargaste o clonaste el repositorio de GitHub, entra a la carpeta con:
  ```powershell
  cd proyecto-final-publicidad-ventas
  ```
  *(Nota: Si ya abriste la terminal directamente dentro de la carpeta del proyecto, no necesitas hacer `cd`).*

### Paso 2: Crear el entorno virtual de Python
Crea un entorno aislado para instalar las librerías necesarias sin afectar tu sistema:
```powershell
python -m venv venv
```

### Paso 3: Activar el entorno virtual
- **En Windows (PowerShell / CMD - Por defecto):**
  ```powershell
  .\venv\Scripts\activate
  ```
  *(Verás que al inicio de tu línea de comandos aparece `(venv)` indicando que está activo).*
- **En macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```

### Paso 4: Instalar las dependencias
Instala todas las librerías necesarias (`dash`, `pandas`, `scikit-learn`, `plotly`, etc.):
```powershell
pip install -r requirements.txt
```

### Paso 5: Iniciar el dashboard
Ejecuta la aplicación principal:
```powershell
python app.py
```

### Paso 6: Abrir en el navegador
Abre tu navegador web e ingresa a la siguiente dirección:
👉 **http://127.0.0.1:8050**

---

## 2. Subir el proyecto a GitHub

1. Cree un repositorio nuevo en GitHub (por ejemplo `proyecto-final-ecommerce-dashboard`).
2. Desde esta carpeta:

```bash
git init
git add .
git commit -m "Dashboard proyecto final: publicidad y ventas e-commerce"
git branch -M main
git remote add origin https://github.com/<su-usuario>/<su-repositorio>.git
git push -u origin main
```

> Verifique que la carpeta `data/` con `ventas_ecommerce.csv` quede incluida en el repositorio (no debe ir en un `.gitignore`).

## 3. Publicar y ver el dashboard en Binder

1. Vaya a **https://mybinder.org**
2. Pegue la URL de su repositorio de GitHub.
3. Deje la rama en `main`.
4. En "Path to a notebook file (optional)" seleccione **File** y escriba `dashboard_binder.ipynb`, o simplemente deje la ruta en blanco y abra el notebook manualmente una vez cargue JupyterLab.
5. Haga clic en **launch**. La primera vez puede tardar varios minutos mientras Binder construye el entorno (instala `requirements.txt`).
6. Cuando abra JupyterLab, abra `dashboard_binder.ipynb` y ejecute **Run → Run All Cells**. Haga clic en el enlace "Abrir el dashboard interactivo" que aparece al final.
7. Copie la URL que le da mybinder.org (algo como `https://mybinder.org/v2/gh/<usuario>/<repo>/main`) — esa es la que va en el archivo de enlaces del entregable final.

> Nota: Binder no mantiene el servidor activo indefinidamente (se apaga tras un periodo de inactividad); cada persona que abra el enlace debe ejecutar el notebook para "despertar" el dashboard.

## 4. Descripción del dashboard

El dashboard tiene 7 pestañas:

1. **Contexto** — planteamiento del problema y KPIs generales.
2. **Exploración** — histograma y evolución temporal de ventas, con filtro de rango de fechas y resaltado de outliers.
3. **Contraste de hipótesis** — prueba t de Welch / Mann-Whitney U, con opción de incluir o excluir los meses atípicos.
4. **Regresión** — comparación interactiva entre regresión lineal simple, múltiple y Ridge (con deslizador de regularización), evaluadas con validación cruzada.
5. **Clasificación** — regresión logística para predecir "mes de venta alta", con deslizador de umbral de decisión y matriz de confusión en tiempo real.
6. **Simulador** — permite ingresar un presupuesto hipotético de publicidad y ver la venta esperada y la probabilidad de un mes de venta alta.
7. **Conclusiones** — hallazgos y recomendación para el equipo financiero.

## 5. Fuente de los datos

`ventas_ecommerce.csv`: dataset construido para este análisis (60 meses, enero 2021 – diciembre 2025) con variables `gasto_publicidad_millones`, `ventas_millones`, `visitas_web_miles` y `tasa_conversion_pct`. Incluye dos meses atípicos insertados intencionalmente (una campaña viral y una falla operativa) usados en las etapas de contextualización y profundización del curso.
