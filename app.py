# ==============================
# IMPORTS
# ==============================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import hmean

# ==============================
# CONFIGURACIÓN DE STREAMLIT
# ==============================
st.set_page_config(
    page_title="Monitor IDEW - CAPRD",
    page_icon="📊",
    layout="wide"
)

# ==============================
# CONFIGURACIÓN GRÁFICA MATPLOTLIB
# ==============================
plt.style.use('ggplot')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelcolor': '#2C3E50',
    'axes.edgecolor': '#2C3E50',
    'axes.titlesize': 14,
    'figure.titlesize': 18
})

# ==============================
# TABLA DE CALIFICACIONES
# ==============================
TABLA_CALIFICACIONES = [
    (95, float('inf'), 'A', 'Excelente'),
    (90, 95, 'B', 'Bueno'),
    (80, 90, 'C', 'Justo'),
    (60, 80, 'D', 'Pobre'),
    (0, 60, 'F', 'Reprobado')
]

# ==============================
# FUNCIONES AUXILIARES
# ==============================
def obtener_calificacion(idew):
    for min_, max_, letra, desempeno in TABLA_CALIFICACIONES:
        if min_ <= idew < max_:
            return f"{letra} - {desempeno}"
    return "N/A"

@st.cache_data
def cargar_datos(ruta):
    """Carga los datos desde la ruta especificada y valida columnas. Usa caché para velocidad."""
    df = pd.read_excel(ruta).set_index('Fecha')
    required = ['INFLAC', 'DESEMP', 'RFISC', 'CREC']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes en {ruta}: {missing}")
    return df

@st.cache_data
def calcular_theta_sdh(df):
    variables = ['INFLAC', 'DESEMP', 'RFISC', 'CREC']
    sd_i = df[variables].std(ddof=0)
    if (sd_i <= 0).any():
        raise ValueError("Alguna desviación estándar es 0; no puede calcularse la media armónica.")
    sd_h = hmean(sd_i)
    theta = [sd_h/sd for sd in sd_i]
    return theta, sd_h

def calcular_idew(df, theta, año):
    fila = df.loc[año]
    idew = (1 - theta[0]*abs(fila['INFLAC'])
            - theta[1]*fila['DESEMP']
            + theta[2]*fila['RFISC']
            + theta[3]*fila['CREC']) * 100
    return idew, fila

@st.cache_data
def calcular_idew_vectorizado(df, theta):
    v = df[['INFLAC','DESEMP','RFISC','CREC']].copy()
    idew = (1 - theta[0]*v['INFLAC'].abs()
              - theta[1]*v['DESEMP']
              + theta[2]*v['RFISC']
              + theta[3]*v['CREC']) * 100
    idew.name = 'IDEW'
    return idew

# ==============================
# FUNCIONES GRÁFICAS
# ==============================
def crear_termometro(ax, valor, año, calificacion):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.add_patch(plt.Rectangle((10, 0.3), 80, 0.4,
                               facecolor='#ECF0F1', edgecolor='#2C3E50', lw=1.5))
    for division in range(20, 100, 20):
        x_pos = 10 + division*0.8
        ax.plot([x_pos, x_pos], [0.25, 0.35], color='#2C3E50', lw=1)
    ax.add_patch(plt.Rectangle((10, 0.3), max(0, min(100, valor))*0.8, 0.4,
                               facecolor='greenyellow', edgecolor='none',
                               hatch='xxxx', linewidth=0.75, alpha=0.8))
    ax.set_title(f"IDEW {año}\n{valor:.1f}%\n{calificacion}",
                 fontweight='bold', pad=4, linespacing=1.2, y=1.08)

def crear_componentes(ax, datos, año):
    variables = {
        'Inflación': datos['INFLAC']*100,
        'Desempleo': datos['DESEMP']*100,
        'Resultado Fiscal': datos['RFISC']*100,
        'Crecimiento': datos['CREC']*100
    }
    colores = ['#E74C3C', '#F39C12', '#3498DB', '#27AE60']
    etiquetas = [f"{k}\n({v:.1f}%)" for k, v in variables.items()]
    ax.barh(etiquetas, list(variables.values()),
            color=colores, height=0.6, edgecolor='white')
    ax.axvline(0, color='#2C3E50', linestyle='--', alpha=0.7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('Porcentaje (%)', fontsize=10)

def graficar_linea_idew(serie_idew, pais, inicio=2000, fin=2030):
    idx = pd.Index(range(inicio, fin+1), name='Fecha')
    s = serie_idew.reindex(idx)
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(s.index, s.values, marker='o', linewidth=2)
    ax.set_title(f"{pais}: Índice de Desempeño Económico (IDEW) {inicio}-{fin}")
    ax.set_xlabel("Año")
    ax.set_ylabel("IDEW (%)")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(inicio, fin)
    ymin = np.nanmin(s.values) if np.isfinite(np.nanmin(s.values)) else 0
    ymax = np.nanmax(s.values) if np.isfinite(np.nanmax(s.values)) else 100
    if np.isfinite(ymin) and np.isfinite(ymax):
        margen = max(2, 0.05*(ymax - ymin))
        ax.set_ylim(ymin - margen, ymax + margen)
    plt.tight_layout()
    return fig # Modificado para devolver la figura en lugar de mostrarla directamente

# ==============================
# LÓGICA PRINCIPAL DE STREAMLIT
# ==============================
def main():
    # Diccionario maestro de configuración con rutas relativas a la carpeta "data"
    paises = {
        "Costa Rica": "data/IDECR_06122025.xlsx",
        "El Salvador": "data/IDESV_06122025.xlsx",
        "Guatemala": "data/IDEGT_06122025.xlsx",
        "Honduras": "data/IDEHN_06122025.xlsx",
        "Nicaragua": "data/IDENI_06122025.xlsx",
        "Panamá": "data/IDEPA_06122025.xlsx",
        "República Dominicana": "data/IDEDO_06122025.xlsx"
    }

    # Interfaz lateral (Sidebar)
    
    st.sidebar.image("FullLogo.png", use_container_width=True)
  
    st.sidebar.title("Configuración")
    pais_seleccionado = st.sidebar.selectbox("Seleccione un país", list(paises.keys()))
    
    st.sidebar.markdown("---")
    st.sidebar.info("Este dashboard calcula y visualiza el Índice de Desempeño Económico de Bienestar (IDEW) para los países de la región CAPRD.")

    # Encabezado principal
    st.title(f"📊 Monitor IDEW: {pais_seleccionado}")
    st.markdown("---")

    ruta_archivo = paises[pais_seleccionado]

    try:
        # Carga y procesamiento
        df = cargar_datos(ruta_archivo)
        theta, _ = calcular_theta_sdh(df)

        # 1. Gráfico de Termómetro y Componentes
        st.subheader("Comparativo Reciente (2023 - 2025)")
        fig1 = plt.figure(figsize=(22, 10))
        gs = fig1.add_gridspec(nrows=2, ncols=3,
                              height_ratios=[0.6, 3],
                              width_ratios=[1, 1, 1],
                              hspace=0.1, wspace=0.3)

        for col, año in enumerate([2023, 2024, 2025]):
            # Validar si el año existe en los datos
            if año in df.index:
                idew, datos = calcular_idew(df, theta, año)
                calificacion = obtener_calificacion(idew)
                
                ax_term = fig1.add_subplot(gs[0, col])
                crear_termometro(ax_term, idew, año, calificacion)
                
                ax_comp = fig1.add_subplot(gs[1, col])
                crear_componentes(ax_comp, datos, año)
            else:
                ax_vacio = fig1.add_subplot(gs[:, col])
                ax_vacio.axis('off')
                ax_vacio.text(0.5, 0.5, f"Datos de {año} no disponibles", 
                              ha='center', va='center', fontsize=14, color='gray')

        plt.subplots_adjust(left=0.07, right=0.93, top=0.88, bottom=0.12)
        plt.suptitle(f"{pais_seleccionado.upper()}: COMPOSICIÓN DEL IDEW",
                     x=0.5, y=1.03, fontsize=20, fontweight='bold', color='#2c3e50')
        
        # Renderizar en Streamlit
        st.pyplot(fig1)

        st.markdown("---")

        # 2. Serie Histórica (Gráfico de línea)
        st.subheader("Evolución Histórica (2000 - 2030)")
        serie_idew = calcular_idew_vectorizado(df, theta)
        inicio, fin = 2000, 2030
        
        fig2 = graficar_linea_idew(serie_idew, pais=pais_seleccionado, inicio=inicio, fin=fin)
        st.pyplot(fig2)

        st.markdown("---")

        # 3. Tabla de Consistencia
        st.subheader("Tabla de Consistencia de Datos")
        idx_full = pd.Index(range(inicio, fin+1), name='Fecha')
        tabla = pd.DataFrame(index=idx_full)
        tabla['IDEW'] = serie_idew.reindex(idx_full).round(1)
        tabla['Calificación'] = tabla['IDEW'].apply(lambda x: obtener_calificacion(x) if pd.notna(x) else np.nan)
        
        # Usamos columnas de Streamlit para que la tabla no ocupe todo el ancho de forma antiestética
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.dataframe(tabla, use_container_width=True)

    except FileNotFoundError:
        st.error(f"❌ No se encontró el archivo de datos para {pais_seleccionado}. Verifica que '{ruta_archivo}' exista en tu repositorio.")
    except Exception as e:
        st.error(f"❌ Ocurrió un error al procesar los datos de {pais_seleccionado}: {e}")

if __name__ == "__main__":
    main()
