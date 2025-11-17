"""
Aplicación Principal de Análisis Financiero
Módulo principal que coordina todas las tabs
"""
import streamlit as st
from utils import initialize_session_state
from base_scenario import render_base_scenario
from calculations import render_calculations
from sensitivity import render_sensitivity
from two_variable_sensitivity import render_two_variable_sensitivity
from three_variable_sensitivity import render_three_variable_sensitivity  # NEW

# Configuración de la página
st.set_page_config(
    page_title="Financial Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar session_state
initialize_session_state()

# Título principal
st.title("📊 Financial Analysis")

# Main tabs
tab1, tab2, tab3, tab4, tab5= st.tabs([
    "🎯 Base Scenario",
    "📊 Calculations",
    "📈 1-Variable",
    "🎲 2-Variable",
    "🎯 3-Variable",  # NEW
])

with tab1:
    render_base_scenario()

with tab2:
    render_calculations()

with tab3:
    render_sensitivity()

with tab4:
    render_two_variable_sensitivity()

with tab5:
    render_three_variable_sensitivity()  # NEW

# Footer
st.divider()
st.caption("Built with Streamlit | v2.0")