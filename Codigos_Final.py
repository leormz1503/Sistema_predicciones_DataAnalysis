"""
SISTEMA DE ANALISIS DELICTIVO - GRUPOS ESPECIFICOS
Analisis descriptivo y pronosticos para delitos de Armas y Privacion de Libertad
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACION Y DEFINICION DE GRUPOS DE DELITOS
# =============================================================================

class Configuracion:
    """Configuracion centralizada del sistema"""
    
    # Grupo 1: Privacion de la libertad / Secuestro
    DELITOS_PRIVACION_LIBERTAD = [
        "SECUESTRO",
        "SECUESTRO EXPRESS (PARA COMETER ROBO O EXTORSIÓN)",
        "PLAGIO O SECUESTRO", 
        "DESAPARICION FORZADA DE PERSONAS",
        "PRIVACION DE LA LIBERTAD PERSONAL",
        "PRIVACION DE LA LIBERTAD PERSONAL (REALIZAR ACTO SEXUAL)",
        "PRIVACIÓN DE LA LIBERTAD PERSONAL (SI LIBERA DENTRO OF 24 HORAS)",
        "RETENCIÓN O SUSTRACCIÓN DE MENORES INCAPACES",
        "SUSTRACCIÓN DE MENORES",
        "RETENCIÓN DE MENORES",
        "TRAFICO DE INFANTES",
        "ROBO DE INFANTE",
        "PRIV. ILEGAL DE LA LIB. Y ROBO DE VEHICULO"
    ]
    
    # Grupo 2: Armas
    DELITOS_ARMAS = [
        "PORTACION ARMA/PROHIB.",
        "PORTACION DE ARMA DE FUEGO",
        "LEY FEDERAL DE ARMAS DE FUEGO Y EXPLOSIVOS",
        "HOMICIDIO POR ARMA DE FUEGO",
        "HOMICIDIO CULPOSO POR ARMA DE FUEGO", 
        "LESIONES INTENCIONALES POR ARMA DE FUEGO",
        "LESIONES INTENCIONALES POR ARMA BLANCA",
        "HOMICIDIO POR ARMA BLANCA",
        "FEMINICIDIO POR DISPARO DE ARMA DE FUEGO",
        "FEMINICIDIO POR ARMA BLANCA",
        "DISPAROS DE ARMA DE FUEGO",
        "TENTATIVA DE HOMICIDIO",
        "PORTACIÓN, FABRICACIÓN E IMPORTACIÓN DE OBJETOS APTOS PARA AGREDIR"
    ]

# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def cargar_datos(ruta_archivo):
    """Carga y valida el dataset"""
    try:
        df = pd.read_csv(ruta_archivo)
        print(f"Dataset cargado: {len(df):,} registros")
        print(f"Columnas disponibles: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"Error cargando archivo: {e}")
        return None

def limpiar_datos_basicos(df):
    """Limpia datos basicos del dataset"""
    df_limpio = df.dropna(subset=['delito', 'alcaldia_hecho']).copy()
    
    # Convertir fechas
    df_limpio['fecha_hecho'] = pd.to_datetime(df_limpio['fecha_hecho'], errors='coerce')
    df_limpio['hora_hecho'] = pd.to_datetime(df_limpio['hora_hecho'], format='%H:%M:%S', errors='coerce')
    
    # Filtrar desde 2017
    df_limpio = df_limpio[df_limpio['fecha_hecho'].dt.year >= 2017]
    
    # Extraer componentes temporales
    df_limpio['anio'] = df_limpio['fecha_hecho'].dt.year
    df_limpio['mes'] = df_limpio['fecha_hecho'].dt.month
    df_limpio['hora'] = df_limpio['hora_hecho'].dt.hour
    df_limpio['dia_semana'] = df_limpio['fecha_hecho'].dt.dayofweek
    
    return df_limpio

def filtrar_delitos_grupo(df, lista_delitos, nombre_grupo):
    """Filtra delitos por grupo con manejo de mayusculas/minusculas"""
    delitos_upper = [delito.upper() for delito in lista_delitos]
    df_filtrado = df[df['delito'].str.upper().isin(delitos_upper)].copy()
    print(f"Delitos de {nombre_grupo}: {len(df_filtrado):,} registros")
    return df_filtrado

# =============================================================================
# ANALISIS DESCRIPTIVO
# =============================================================================

class AnalizadorDescriptivo:
    """Maneja analisis descriptivo de los datos"""
    
    def generar_reporte_estadistico(self, datos, nombre_grupo):
        """Genera reporte estadistico descriptivo"""
        print("=" * 80)
        print(f"REPORTE ESTADISTICO - {nombre_grupo}")
        print("=" * 80)
        
        if datos is None or len(datos) == 0:
            print("No hay datos disponibles para este grupo")
            return
        
        print(f"Total de registros: {len(datos):,}")
        print(f"Periodo: {datos['fecha_hecho'].min().year} - {datos['fecha_hecho'].max().year}")
        print(f"Alcaldias diferentes: {datos['alcaldia_hecho'].nunique()}")
        print(f"Tipos de delito diferentes: {datos['delito'].nunique()}")
        
        print(f"\nTOP 5 ALCALDIAS:")
        top_alcaldias = datos['alcaldia_hecho'].value_counts().head(5)
        for i, (alcaldia, conteo) in enumerate(top_alcaldias.items(), 1):
            porcentaje = (conteo / len(datos)) * 100
            print(f"  {i}. {alcaldia}: {conteo:,} casos ({porcentaje:.1f}%)")
        
        print(f"\nTOP 5 TIPOS DE DELITO:")
        top_delitos = datos['delito'].value_counts().head(5)
        for i, (delito, conteo) in enumerate(top_delitos.items(), 1):
            porcentaje = (conteo / len(datos)) * 100
            print(f"  {i}. {delito}: {conteo:,} casos ({porcentaje:.1f}%)")
        
        print(f"\nDISTRIBUCIÓN POR AÑO:")
        distribucion_anio = datos['anio'].value_counts().sort_index()
        for anio, conteo in distribucion_anio.items():
            print(f"  {anio}: {conteo:,} casos")
        
        print("=" * 80)
    
    def analisis_distribucion_geografica(self, datos, titulo):
        """Analiza distribucion geografica de delitos"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'ANALISIS DESCRIPTIVO - {titulo}', fontsize=16, fontweight='bold')
        
        # 1. Top alcaldias
        ax1 = axes[0, 0]
        top_alcaldias = datos['alcaldia_hecho'].value_counts().head(10)
        ax1.bar(range(len(top_alcaldias)), top_alcaldias.values, color='skyblue', alpha=0.8)
        ax1.set_title('TOP 10 ALCALDIAS', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Alcaldias')
        ax1.set_ylabel('Número de Delitos')
        ax1.set_xticks(range(len(top_alcaldias)))
        ax1.set_xticklabels(top_alcaldias.index, rotation=45, ha='right')
        
        # 2. Distribucion por tipo de delito
        ax2 = axes[0, 1]
        top_delitos = datos['delito'].value_counts().head(8)
        ax2.pie(top_delitos.values, labels=top_delitos.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title('DISTRIBUCIÓN POR TIPO OF DELITO', fontsize=12, fontweight='bold')
        
        # 3. Distribucion por año
        ax3 = axes[1, 0]
        distribucion_anio = datos['anio'].value_counts().sort_index()
        ax3.plot(distribucion_anio.index, distribucion_anio.values, marker='o', linewidth=2)
        ax3.set_title('EVOLUCIÓN TEMPORAL', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Año')
        ax3.set_ylabel('Número de Delitos')
        ax3.grid(True, alpha=0.3)
        
        # 4. Distribucion por hora
        ax4 = axes[1, 1]
        distribucion_hora = datos['hora'].value_counts().sort_index()
        ax4.bar(distribucion_hora.index, distribucion_hora.values, alpha=0.7, color='orange')
        ax4.set_title('DISTRIBUCIÓN POR HORA', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Hora del Día')
        ax4.set_ylabel('Número de Delitos')
        ax4.set_xticks(range(0, 24, 2))
        
        plt.tight_layout()
        plt.show()
        
        return top_alcaldias
    
    def analisis_comparativo_grupos(self, datos_privacion, datos_armas):
        """Analisis comparativo entre los dos grupos de delitos"""
        if datos_privacion is None or datos_armas is None:
            print("No hay datos suficientes para análisis comparativo")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ANALISIS COMPARATIVO - GRUPOS DELICTIVOS', fontsize=16, fontweight='bold')
        
        # 1. Volumen total comparativo
        ax1 = axes[0, 0]
        totales = [len(datos_privacion), len(datos_armas)]
        grupos = ['Privación Libertad', 'Delitos Armas']
        colores = ['#FF6B6B', '#4ECDC4']
        
        barras = ax1.bar(grupos, totales, color=colores, alpha=0.8)
        ax1.set_title('COMPARATIVA: VOLUMEN TOTAL', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Número de Delitos')
        
        for barra, valor in zip(barras, totales):
            ax1.text(barra.get_x() + barra.get_width()/2, barra.get_height() + max(totales)*0.01, 
                    f'{valor:,}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Top alcaldias comparativo
        ax2 = axes[0, 1]
        top_privacion = datos_privacion['alcaldia_hecho'].value_counts().head(5)
        top_armas = datos_armas['alcaldia_hecho'].value_counts().head(5)
        
        x_pos = np.arange(max(len(top_privacion), len(top_armas)))
        ancho = 0.35
        
        if len(top_privacion) > 0:
            ax2.bar(x_pos[:len(top_privacion)] - ancho/2, top_privacion.values, ancho, 
                   label='Privación Libertad', alpha=0.7, color='#FF6B6B')
        
        if len(top_armas) > 0:
            ax2.bar(x_pos[:len(top_armas)] + ancho/2, top_armas.values, ancho, 
                   label='Delitos Armas', alpha=0.7, color='#4ECDC4')
        
        ax2.set_title('TOP ALCALDIAS COMPARATIVO', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Alcaldias')
        ax2.set_ylabel('Número de Delitos')
        ax2.legend()
        ax2.set_xticks(x_pos[:max(len(top_privacion), len(top_armas))])
        ax2.set_xticklabels(top_privacion.index if len(top_privacion) >= len(top_armas) else top_armas.index, 
                           rotation=45)
        
        # 3. Distribucion por hora comparativa
        ax3 = axes[1, 0]
        horas_privacion = datos_privacion['hora'].value_counts().sort_index()
        horas_armas = datos_armas['hora'].value_counts().sort_index()
        
        ax3.plot(horas_privacion.index, horas_privacion.values, label='Privación Libertad', 
                color='#FF6B6B', linewidth=2)
        ax3.plot(horas_armas.index, horas_armas.values, label='Delitos Armas', 
                color='#4ECDC4', linewidth=2)
        ax3.set_title('DISTRIBUCIÓN POR HORA COMPARATIVA', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Hora del Día')
        ax3.set_ylabel('Número de Delitos')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(range(0, 24, 2))
        
        # 4. Evolucion temporal comparativa
        ax4 = axes[1, 1]
        evol_privacion = datos_privacion['anio'].value_counts().sort_index()
        evol_armas = datos_armas['anio'].value_counts().sort_index()
        
        ax4.plot(evol_privacion.index, evol_privacion.values, marker='o', 
                label='Privación Libertad', color='#FF6B6B', linewidth=2)
        ax4.plot(evol_armas.index, evol_armas.values, marker='s', 
                label='Delitos Armas', color='#4ECDC4', linewidth=2)
        ax4.set_title('EVOLUCIÓN TEMPORAL COMPARATIVA', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Año')
        ax4.set_ylabel('Número de Delitos')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# =============================================================================
# MODELOS PREDICTIVOS Y PRONOSTICOS - VERSION MEJORADA
# =============================================================================

class PronosticadorDelitos:
    """Maneja pronosticos de series temporales - VERSION MEJORADA"""
    
    def __init__(self):
        self.modelos = {}
    
    def preparar_datos_serie_temporal(self, datos, grupo_nombre):
        """Prepara datos para serie temporal con FEATURES AVANZADAS"""
        try:
            # Agrupar por mes
            datos['fecha_hecho'] = pd.to_datetime(datos['fecha_hecho'])
            serie_mensual = datos.set_index('fecha_hecho').resample('M').size()
            serie_mensual.name = 'conteo'
            
            if len(serie_mensual) < 24:
                print(f"Datos insuficientes para {grupo_nombre}: {len(serie_mensual)} meses")
                return None
            
            # Crear DataFrame para modelo
            df_serie = pd.DataFrame({'conteo': serie_mensual})
            df_serie = df_serie.dropna()
            
            # Características temporales básicas
            df_serie['mes'] = df_serie.index.month
            df_serie['trimestre'] = df_serie.index.quarter
            df_serie['anio'] = df_serie.index.year
            df_serie['mes_sin'] = np.sin(2 * np.pi * df_serie.index.month / 12)
            df_serie['mes_cos'] = np.cos(2 * np.pi * df_serie.index.month / 12)
            
            # Lags (Retardos)
            for lag in [1, 2, 3, 6, 12]:
                df_serie[f'lag_{lag}'] = df_serie['conteo'].shift(lag)
            
            # INGENIERÍA DE CARACTERÍSTICAS AVANZADAS (DEL PRIMER CÓDIGO)
            
            # Medias Móviles (Tendencia)
            df_serie['media_3m'] = df_serie['conteo'].shift(1).rolling(window=3).mean()
            df_serie['media_6m'] = df_serie['conteo'].shift(1).rolling(window=6).mean()
            df_serie['media_12m'] = df_serie['conteo'].shift(1).rolling(window=12).mean()
            
            # Desviación Estándar (Volatilidad/Riesgo)
            df_serie['std_6m'] = df_serie['conteo'].shift(1).rolling(window=6).std()
            df_serie['std_12m'] = df_serie['conteo'].shift(1).rolling(window=12).std()

            # Cambio de Tendencia (Derivada)
            df_serie['diff_12m'] = df_serie['conteo'].diff(periods=12)

            df_serie = df_serie.dropna()
            print(f"Datos preparados para {grupo_nombre}: {len(df_serie)} meses")
            return df_serie
            
        except Exception as e:
            print(f"Error preparando datos para {grupo_nombre}: {e}")
            return None
    
    def entrenar_modelo_xgboost(self, datos_serie, grupo_nombre, meses_pronostico=6):
        """Entrena y optimiza modelo XGBoost con GridSearch y TimeSeriesSplit"""
        if datos_serie is None or len(datos_serie) < 24:
            print(f"Datos insuficientes para optimización en {grupo_nombre}")
            return None
        
        try:
            # Separar características y objetivo
            X = datos_serie.drop('conteo', axis=1)
            y = datos_serie['conteo']
            
            # 1. Estrategia de Validación Cruzada Temporal
            tscv = TimeSeriesSplit(n_splits=5)
            
            # 2. Hiperparámetros a optimizar (DEL PRIMER CÓDIGO)
            param_grid = {
                'n_estimators': [100, 250, 500],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.7, 0.9],
                'colsample_bytree': [0.7, 0.9]
            }
            
            # 3. GridSearchCV para optimización
            xgb = XGBRegressor(random_state=42)
            gscv = GridSearchCV(
                estimator=xgb,
                param_grid=param_grid,
                scoring='neg_mean_squared_error',
                cv=tscv,
                verbose=0,
                n_jobs=-1
            )
            
            # 4. Entrenar con Grid Search
            print(f"Optimizando hiperparámetros para {grupo_nombre}...")
            gscv.fit(X, y)
            
            # Obtener el mejor modelo
            modelo_optimizado = gscv.best_estimator_
            print(f"Mejores parámetros: {gscv.best_params_}")
            
            # 5. División Final Train/Test para evaluación
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Re-entrenar el mejor modelo con conjunto de entrenamiento
            modelo_optimizado.fit(X_train, y_train)
            
            # 6. Evaluar modelo optimizado
            preds = modelo_optimizado.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            
            print(f"\nMetricas {grupo_nombre} (Modelo Optimizado):")
            print(f"  MAE: {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²: {r2:.4f}")
            
            # Generar pronóstico futuro
            pronostico_futuro = self.generar_pronostico_futuro(modelo_optimizado, datos_serie, meses_pronostico)
            
            resultados = {
                'modelo': modelo_optimizado,
                'metricas': {'mae': mae, 'rmse': rmse, 'r2': r2},
                'X_test': X_test,
                'y_test': y_test,
                'predicciones': preds,
                'pronostico_futuro': pronostico_futuro
            }
            
            return resultados
            
        except Exception as e:
            print(f"Error entrenando modelo para {grupo_nombre}: {e}")
            # Fallback: usar modelo simple si falla la optimización
            return self.entrenar_modelo_simple(datos_serie, grupo_nombre, meses_pronostico)
    
    def entrenar_modelo_simple(self, datos_serie, grupo_nombre, meses_pronostico=6):
        """Modelo simple como fallback si falla la optimización"""
        try:
            X = datos_serie.drop('conteo', axis=1)
            y = datos_serie['conteo']
            
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            modelo = XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            
            modelo.fit(X_train, y_train)
            preds = modelo.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            
            print(f"\nMetricas {grupo_nombre} (Modelo Simple):")
            print(f"  MAE: {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²: {r2:.4f}")
            
            pronostico_futuro = self.generar_pronostico_futuro(modelo, datos_serie, meses_pronostico)
            
            return {
                'modelo': modelo,
                'metricas': {'mae': mae, 'rmse': rmse, 'r2': r2},
                'X_test': X_test,
                'y_test': y_test,
                'predicciones': preds,
                'pronostico_futuro': pronostico_futuro
            }
            
        except Exception as e:
            print(f"Error con modelo simple para {grupo_nombre}: {e}")
            return None
    
    def generar_pronostico_futuro(self, modelo, datos_serie, meses_pronostico):
        """Genera pronóstico para meses futuros"""
        try:
            pronosticos = []
            datos_actuales = datos_serie.copy()
            ultima_fecha = datos_actuales.index[-1]
            
            for i in range(meses_pronostico):
                proxima_fecha = ultima_fecha + pd.DateOffset(months=i+1)
                
                # Preparar características
                caracteristicas = {
                    'mes': proxima_fecha.month,
                    'trimestre': proxima_fecha.quarter,
                    'anio': proxima_fecha.year,
                    'mes_sin': np.sin(2 * np.pi * proxima_fecha.month / 12),
                    'mes_cos': np.cos(2 * np.pi * proxima_fecha.month / 12)
                }
                
                # Lags
                for lag in [1, 2, 3, 6, 12]:
                    lag_fecha = proxima_fecha - pd.DateOffset(months=lag)
                    if lag_fecha in datos_actuales.index:
                        caracteristicas[f'lag_{lag}'] = datos_actuales.loc[lag_fecha, 'conteo']
                    else:
                        caracteristicas[f'lag_{lag}'] = datos_actuales['conteo'].iloc[-1]
                
                # Convertir a DataFrame
                caracteristicas_df = pd.DataFrame([caracteristicas])
                columnas_entrenamiento = datos_actuales.drop('conteo', axis=1).columns
                
                # Asegurar columnas
                for col in columnas_entrenamiento:
                    if col not in caracteristicas_df.columns:
                        caracteristicas_df[col] = 0
                
                caracteristicas_df = caracteristicas_df[columnas_entrenamiento]
                
                # Predecir
                prediccion = modelo.predict(caracteristicas_df)[0]
                prediccion = max(0, prediccion)
                
                pronosticos.append({'fecha': proxima_fecha, 'pronostico': prediccion})
            
            return pd.DataFrame(pronosticos)
                
        except Exception as e:
            print(f"Error generando pronóstico futuro: {e}")
            return None
    
    def visualizar_pronosticos(self, resultados_privacion, resultados_armas):
        """Visualiza comparativa de pronósticos"""
        if not resultados_privacion or not resultados_armas:
            print("Datos insuficientes para visualización de pronósticos")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('PRONÓSTICOS - COMPARATIVA GRUPOS DELICTIVOS', fontsize=16, fontweight='bold')
        
        # Pronósticos Privación de Libertad
        ax1 = axes[0]
        if resultados_privacion['pronostico_futuro'] is not None:
            pronostico_privacion = resultados_privacion['pronostico_futuro']
            ax1.bar(range(len(pronostico_privacion)), pronostico_privacion['pronostico'], 
                   color='#FF6B6B', alpha=0.8)
            ax1.set_title('PRIVACIÓN DE LIBERTAD - PRONÓSTICO 6 MESES', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Meses Futuros')
            ax1.set_ylabel('Casos Pronosticados')
            ax1.set_xticks(range(len(pronostico_privacion)))
            ax1.set_xticklabels([f'M+{i+1}' for i in range(len(pronostico_privacion))])
        
        # Pronósticos Delitos con Armas
        ax2 = axes[1]
        if resultados_armas['pronostico_futuro'] is not None:
            pronostico_armas = resultados_armas['pronostico_futuro']
            ax2.bar(range(len(pronostico_armas)), pronostico_armas['pronostico'], 
                   color='#4ECDC4', alpha=0.8)
            ax2.set_title('DELITOS CON ARMAS - PRONÓSTICO 6 MESES', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Meses Futuros')
            ax2.set_ylabel('Casos Pronosticados')
            ax2.set_xticks(range(len(pronostico_armas)))
            ax2.set_xticklabels([f'M+{i+1}' for i in range(len(pronostico_armas))])
        
        plt.tight_layout()
        plt.show()

# =============================================================================
# FUNCION PRINCIPAL
# =============================================================================

def analizar_grupos_delitos(ruta_archivo):
    """Funcion principal para analizar los grupos de delitos especificos"""
    
    print("SISTEMA DE ANÁLISIS PARA GRUPOS DELICTIVOS ESPECIFICOS")
    print("=" * 70)
    
    # Cargar y limpiar datos
    df = cargar_datos(ruta_archivo)
    if df is None:
        return {}
    
    df_limpio = limpiar_datos_basicos(df)
    print(f"Registros válidos desde 2017: {len(df_limpio):,}")
    
    # Inicializar componentes
    config = Configuracion()
    analizador = AnalizadorDescriptivo()
    pronosticador = PronosticadorDelitos()
    
    resultados = {}
    
    # ANALISIS GRUPO 1: PRIVACION DE LIBERTAD
    print("\n" + "=" * 60)
    print("ANÁLISIS GRUPO 1: PRIVACIÓN DE LIBERTAD")
    print("=" * 60)
    
    datos_privacion = filtrar_delitos_grupo(df_limpio, config.DELITOS_PRIVACION_LIBERTAD, "Privación de Libertad")
    
    if len(datos_privacion) > 0:
        analizador.generar_reporte_estadistico(datos_privacion, "PRIVACIÓN DE LIBERTAD")
        distribucion_privacion = analizador.analisis_distribucion_geografica(datos_privacion, "PRIVACIÓN DE LIBERTAD")
        
        # Preparar datos para pronósticos
        datos_serie_privacion = pronosticador.preparar_datos_serie_temporal(datos_privacion, "Privación de Libertad")
        if datos_serie_privacion is not None:
            resultados_privacion = pronosticador.entrenar_modelo_xgboost(datos_serie_privacion, "Privación de Libertad")
            resultados['privacion'] = resultados_privacion
    else:
        print("No hay datos suficientes para Privación de Libertad")
        datos_privacion = None
    
    # ANALISIS GRUPO 2: DELITOS CON ARMAS
    print("\n" + "=" * 60)
    print("ANÁLISIS GRUPO 2: DELITOS CON ARMAS")
    print("=" * 60)
    
    datos_armas = filtrar_delitos_grupo(df_limpio, config.DELITOS_ARMAS, "Delitos con Armas")
    
    if len(datos_armas) > 0:
        analizador.generar_reporte_estadistico(datos_armas, "DELITOS CON ARMAS")
        distribucion_armas = analizador.analisis_distribucion_geografica(datos_armas, "DELITOS CON ARMAS")
        
        # Preparar datos para pronósticos
        datos_serie_armas = pronosticador.preparar_datos_serie_temporal(datos_armas, "Delitos con Armas")
        if datos_serie_armas is not None:
            resultados_armas = pronosticador.entrenar_modelo_xgboost(datos_serie_armas, "Delitos con Armas")
            resultados['armas'] = resultados_armas
    else:
        print("No hay datos suficientes para Delitos con Armas")
        datos_armas = None
    
    # ANALISIS COMPARATIVO
    print("\n" + "=" * 60)
    print("ANÁLISIS COMPARATIVO ENTRE GRUPOS")
    print("=" * 60)
    
    if datos_privacion is not None and datos_armas is not None:
        analizador.analisis_comparativo_grupos(datos_privacion, datos_armas)
    
    # PRONÓSTICOS COMPARATIVOS
    print("\n" + "=" * 60)
    print("SISTEMA DE PRONÓSTICOS")
    print("=" * 60)
    
    if 'privacion' in resultados and 'armas' in resultados:
        pronosticador.visualizar_pronosticos(resultados['privacion'], resultados['armas'])
        
        # Mostrar resumen de pronósticos
        print("\nRESUMEN DE PRONÓSTICOS:")
        if resultados['privacion']['pronostico_futuro'] is not None:
            print("\nPrivación de Libertad - Próximos 6 meses:")
            for _, fila in resultados['privacion']['pronostico_futuro'].iterrows():
                print(f"  {fila['fecha'].strftime('%Y-%m')}: {fila['pronostico']:.1f} casos")
        
        if resultados['armas']['pronostico_futuro'] is not None:
            print("\nDelitos con Armas - Próximos 6 meses:")
            for _, fila in resultados['armas']['pronostico_futuro'].iterrows():
                print(f"  {fila['fecha'].strftime('%Y-%m')}: {fila['pronostico']:.1f} casos")
    
    print("\n" + "=" * 70)
    print("ANÁLISIS COMPLETADO")
    print("=" * 70)
    
    return resultados

# =============================================================================
# EJECUCION
# =============================================================================

if __name__ == "__main__":
    # Configuración de visualización
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Ruta del archivo - AJUSTA ESTA RUTA
    ruta_archivo = "carpetasFGJ_acumulado_2025_01 (1).csv"  # Ajusta la ruta según tu sistema
    
    # Ejecutar análisis
    try:
        resultados = analizar_grupos_delitos(ruta_archivo)
        print("Proceso finalizado exitosamente")
        
    except Exception as e:
        print(f"Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()