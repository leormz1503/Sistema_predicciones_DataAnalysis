import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium, folium_static
from folium.plugins import HeatMap, MarkerCluster
from datetime import datetime

# Importar clases y funciones del archivo original
try:
    from Codigos_Final import (
        Configuracion, 
        cargar_datos, 
        limpiar_datos_basicos, 
        filtrar_delitos_grupo,
        AnalizadorDescriptivo,
        PronosticadorDelitos
    )
except ImportError:
    st.error("No se pudo importar Códigos_Final.py. Asegúrate de que el archivo existe en el directorio.")
    st.stop()

# Configuración de la página
st.set_page_config(
    page_title="Tablero de Análisis Delictivo CDMX",
    page_icon="🛡️",
    layout="wide"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #31333F; /* Color de texto oscuro para contraste */
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #ff4b4b;
        color: #000000; /* Color negro para la pestaña activa */
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# FUNCIONES DE CARGA Y PROCESAMIENTO
# ==========================================

@st.cache_data
def load_data(filepath):
    """Carga y limpia los datos usando las funciones de Códigos_Final"""
    df = cargar_datos(filepath)
    if df is not None:
        return limpiar_datos_basicos(df)
    return None

@st.cache_resource
def get_model_results(df, delito_type):
    """Entrena el modelo y devuelve los resultados"""
    config = Configuracion()
    pronosticador = PronosticadorDelitos()
    
    if delito_type == "Privación de la libertad":
        delitos_list = config.DELITOS_PRIVACION_LIBERTAD
        grupo_nombre = "Privación de Libertad"
    else:
        delitos_list = config.DELITOS_ARMAS
        grupo_nombre = "Delitos con Armas"
        
    df_filtrado = filtrar_delitos_grupo(df, delitos_list, grupo_nombre)
    
    if len(df_filtrado) == 0:
        return None, None
        
    datos_serie = pronosticador.preparar_datos_serie_temporal(df_filtrado, grupo_nombre)
    
    if datos_serie is not None:
        resultados = pronosticador.entrenar_modelo_xgboost(datos_serie, grupo_nombre)
        return resultados, df_filtrado
    
    return None, df_filtrado

# ==========================================
# FUNCIONES DE VISUALIZACIÓN (ADAPTADAS)
# ==========================================

def plot_analisis_descriptivo(datos, titulo):
    """Versión adaptada de analisis_distribucion_geografica para Streamlit"""
    if datos is None or len(datos) == 0:
        st.warning("No hay datos para mostrar.")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"ANÁLISIS DESCRIPTIVO - {titulo}", fontsize=16, fontweight="bold")
    
    # 1. Top alcaldias
    ax1 = axes[0, 0]
    top_alcaldias = datos["alcaldia_hecho"].value_counts().head(10)
    sns.barplot(x=top_alcaldias.index, y=top_alcaldias.values, ax=ax1, palette="viridis")
    ax1.set_title("TOP 10 ALCALDÍAS", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Alcaldías")
    ax1.set_ylabel("Número de Delitos")
    ax1.tick_params(axis="x", rotation=45)
    
    # 2. Distribucion por tipo de delito
    ax2 = axes[0, 1]
    top_delitos = datos["delito"].value_counts().head(8)
    
    # Función para ocultar porcentajes pequeños (< 4%) que suelen encimarse
    def autopct_fun(pct):
        return f'{pct:.1f}%' if pct > 4 else ''

    # Usamos leyenda externa para evitar textos encimados en el gráfico de pastel
    wedges, texts, autotexts = ax2.pie(
        top_delitos.values, 
        autopct=autopct_fun, 
        startangle=90,
        textprops={'fontsize': 9},
        pctdistance=0.85
    )
    ax2.legend(
        wedges, 
        top_delitos.index,
        title="Tipos de Delito",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize='small'
    )
    ax2.set_title("DISTRIBUCIÓN POR TIPO DE DELITO", fontsize=12, fontweight="bold")
    
    # 3. Distribucion por año
    ax3 = axes[1, 0]
    distribucion_anio = datos["anio"].value_counts().sort_index()
    ax3.plot(distribucion_anio.index, distribucion_anio.values, marker="o", linewidth=2, color="#2E86C1")
    ax3.set_title("EVOLUCIÓN TEMPORAL", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Año")
    ax3.set_ylabel("Número de Delitos")
    ax3.grid(True, alpha=0.3)
    
    # 4. Distribucion por hora
    ax4 = axes[1, 1]
    distribucion_hora = datos["hora"].value_counts().sort_index()
    # Usamos matplotlib estándar para controlar mejor los ticks del eje X (cada 2 horas)
    ax4.bar(distribucion_hora.index, distribucion_hora.values, color=sns.color_palette("magma", len(distribucion_hora)))
    ax4.set_title("DISTRIBUCIÓN POR HORA", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Hora del Día")
    ax4.set_ylabel("Número de Delitos")
    ax4.set_xticks(range(0, 24, 2)) # Mostrar cada 2 horas para evitar solapamiento
    
    plt.tight_layout()
    return fig

def plot_pronostico(resultados, titulo):
    """Visualiza el pronóstico futuro"""
    if resultados is None or resultados["pronostico_futuro"] is None:
        return None
        
    pronostico = resultados["pronostico_futuro"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Datos históricos recientes (últimos 12 meses para contexto)
    y_test = resultados["y_test"]
    if len(y_test) > 12:
        y_test = y_test[-12:]
    
    # Plot históricos
    ax.plot(y_test.index, y_test.values, label="Histórico (Prueba)", marker="o", color="gray", alpha=0.6)
    
    # Plot pronóstico
    ax.plot(pronostico["fecha"], pronostico["pronostico"], label="Pronóstico", marker="o", color="#FF5733", linewidth=2, linestyle="--")
    
    # Intervalos de confianza (simulados visualmente para el ejemplo)
    ax.fill_between(pronostico["fecha"], 
                   pronostico["pronostico"] * 0.9, 
                   pronostico["pronostico"] * 1.1, 
                   color="#FF5733", alpha=0.2, label="Intervalo de Confianza (Est.)")

    ax.set_title(f"PRONÓSTICO A 6 MESES - {titulo}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Número de Casos")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

# ==========================================
# INTERFAZ PRINCIPAL
# ==========================================

st.title("🛡️ Sistema de Análisis y Predicción de Delitos CDMX")
st.markdown("### Proyecto Final - Data Analysis and AI Tools")

# Cargar datos
RUTA_ARCHIVO = "carpetasFGJ_acumulado_2025_01 (1).csv"
with st.spinner("Cargando base de datos..."):
    df_main = load_data(RUTA_ARCHIVO)

if df_main is None:
    st.error("No se pudieron cargar los datos. Verifica la ruta del archivo.")
    st.stop()

# Sidebar - Menú Principal
st.sidebar.header("Configuración del Análisis")

# Selección de Tipo de Delito
tipo_delito = st.sidebar.radio(
    "Selecciona el Tipo de Delito:",
    ["Privación de la libertad", "Delitos con Armas"]
)

# Entrenar/Obtener modelo para el delito seleccionado
with st.spinner(f"Procesando datos para {tipo_delito}..."):
    resultados_modelo, df_filtrado = get_model_results(df_main, tipo_delito)

if df_filtrado is None or len(df_filtrado) == 0:
    st.warning(f"No se encontraron registros para {tipo_delito}.")
    st.stop()

# Tabs para las diferentes secciones
tab1, tab2, tab3 = st.tabs(["📊 Análisis Descriptivo", "🔮 Predicciones y Métricas", "🗺️ Mapa de Riesgo"])

# --- TAB 1: ANÁLISIS DESCRIPTIVO ---
with tab1:
    st.subheader(f"Análisis Descriptivo: {tipo_delito}")
    
    # Métricas clave
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de Casos (2017-2025)", f"{len(df_filtrado):,}")
    col2.metric("Promedio Mensual", f"{len(df_filtrado)/((2025-2017)*12):.1f}")
    col3.metric("Alcaldía con más casos", df_filtrado["alcaldia_hecho"].mode()[0])
    col4.metric("Hora más frecuente", f"{df_filtrado["hora"].mode()[0]}:00")
    
    # Gráficas
    fig_desc = plot_analisis_descriptivo(df_filtrado, tipo_delito.upper())
    st.pyplot(fig_desc)

# --- TAB 2: PREDICCIONES Y MÉTRICAS ---
with tab2:
    st.subheader(f"Modelo Predictivo (XGBoost): {tipo_delito}")
    
    if resultados_modelo:
        # Métricas del modelo
        st.markdown("#### Métricas de Desempeño del Modelo")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("MAE (Error Absoluto Medio)", f"{resultados_modelo["metricas"]["mae"]:.4f}")
        m_col2.metric("RMSE (Raíz del Error Cuadrático Medio)", f"{resultados_modelo["metricas"]["rmse"]:.4f}")
        m_col3.metric("R² (Coeficiente de Determinación)", f"{resultados_modelo["metricas"]["r2"]:.4f}")
        
        # Gráfica de Pronóstico
        st.markdown("#### Pronóstico a 6 Meses")
        fig_pron = plot_pronostico(resultados_modelo, tipo_delito.upper())
        st.pyplot(fig_pron)
        
        # Tabla de Pronósticos
        st.markdown("#### Datos del Pronóstico")
        pronostico_df = resultados_modelo["pronostico_futuro"].copy()
        pronostico_df["fecha"] = pronostico_df["fecha"].dt.strftime("%Y-%m-%d")
        st.dataframe(pronostico_df.style.format({"pronostico": "{:.2f}"}))
        
    else:
        st.warning("No hay suficientes datos para generar un modelo predictivo confiable.")

# --- TAB 3: MAPA DE RIESGO ---
with tab3:
    st.subheader("Mapa Interactivo de Riesgo y Rutas")
    
    # Filtro de Alcaldía para el mapa
    alcaldias_disponibles = sorted(df_filtrado["alcaldia_hecho"].unique())
    alcaldia_seleccionada = st.selectbox("Selecciona una Alcaldía para visualizar:", alcaldias_disponibles)
    
    # Filtrar datos por alcaldía
    df_mapa = df_filtrado[df_filtrado["alcaldia_hecho"] == alcaldia_seleccionada].copy()
    
    # Validar coordenadas
    if "latitud" in df_mapa.columns and "longitud" in df_mapa.columns:
        df_mapa = df_mapa.dropna(subset=["latitud", "longitud"])
        # Filtrar coordenadas erróneas (fuera de CDMX aprox)
        df_mapa = df_mapa[
            (df_mapa["latitud"].between(19.0, 19.6)) & 
            (df_mapa["longitud"].between(-99.4, -98.8))
        ]
        
        if len(df_mapa) > 0:
            st.markdown(f"Mostrando **{len(df_mapa)}** incidentes en **{alcaldia_seleccionada}**.")
            
            # Centro del mapa
            lat_center = df_mapa["latitud"].mean()
            lon_center = df_mapa["longitud"].mean()
            
            m = folium.Map(location=[lat_center, lon_center], zoom_start=13, tiles="CartoDB positron")
            
            # 1. Mapa de Calor (Heatmap)
            heat_data = [[row["latitud"], row["longitud"]] for index, row in df_mapa.iterrows()]
            HeatMap(heat_data, radius=15, blur=10).add_to(m)
            
            # 2. Puntos de Alto Riesgo (Cluster)
            marker_cluster = MarkerCluster().add_to(m)
            # Limitamos a 500 puntos para rendimiento si son muchos
            for idx, row in df_mapa.head(500).iterrows():
                folium.Marker(
                    location=[row["latitud"], row["longitud"]],
                    popup=f"{row["delito"]}<br>{row["fecha_hecho"]}",
                    icon=folium.Icon(color="red", icon="info-sign")
                ).add_to(marker_cluster)
            
            # 3. "Rutas de Riesgo" (Simulación visual conectando puntos densos)
            # Como no tenemos rutas reales, conectamos los puntos secuencialmente en el tiempo 
            # para visualizar patrones de movimiento o secuencia de eventos recientes (últimos 10 casos)
            df_recent = df_mapa.sort_values("fecha_hecho", ascending=False).head(10)
            if len(df_recent) > 1:
                route_points = [[row["latitud"], row["longitud"]] for idx, row in df_recent.iterrows()]
                folium.PolyLine(
                    route_points, 
                    weight=2, 
                    color="blue", 
                    opacity=0.6, 
                    dash_array="5, 10",
                tooltip="Secuencia de últimos eventos"
            ).add_to(m)
            st.info("🔵 La línea azul discontinua conecta los 10 eventos más recientes para identificar patrones secuenciales.")

            folium_static(m, width=1000, height=600)
            
        else:
            st.warning("No se encontraron coordenadas válidas para esta alcaldía en el rango de la CDMX.")
    else:
        st.error("El dataset no contiene columnas de latitud y longitud necesarias para el mapa.")

st.markdown("---")
st.caption("Desarrollado por Leonardo Ramirez y Santiago Ulloa")
