"""
Shared utilities and constants
"""
import streamlit as st

# Default scenario values
DEFAULT_VALUES = {
    'num_years': 5,
    'sales_year1': 36000000.0,
    'sales_growth': 0.0,
    'cogs_pct': 36.0,
    'cost_growth': 0.0,
    'depreciation': 10000000.0,
    # General Expenses components (fixed costs) - NOT percentages
    'labor_cost': 2160000.0,  # Labor Cost annual (fixed)
    'energy_cost': 600000.0,  # Energy Cost annual (fixed)
    'sga': 300000.0,  # SG&A annual (fixed)
    'overhead_cost': 360000.0,  # Overhead Cost annual (fixed) - 1% of 36M sales
    'ebt_year0': 0.0,
    'tax_year0': 0.0,
    'tax_rate': 30.0,
    'tax_type': 'percentage',
    'tax_fixed_amount': 0.0,
    'capex_year0': 50000000.0,
    'ar_days': 45,
    'ap_days': 36,
    'inventory_days': 30,
    'days_in_year': 360,
    'covariance': 0.0,
    'variance': 1.0,
    'beta': 1.0,
    'wacc': 18.2,
    # Additional adjustments
    'salvage_value': 4000000.0,
    'erosion_cost': 800000.0,
    'opportunity_cost': 42000.0
}

def initialize_session_state():
    """Initializes all values in session_state if they don't exist"""
    if 'scenario_initialized' not in st.session_state:
        st.session_state.scenario_initialized = True
        for key, value in DEFAULT_VALUES.items():
            st.session_state[key] = value
        
        st.session_state.generate_calculations = False
        st.session_state.df_calculated = None

def load_default_values():
    """Loads default values into session_state"""
    for key, value in DEFAULT_VALUES.items():
        st.session_state[key] = value
    
    st.rerun()

def clear_values():
    """Clears all values to zero/empty"""
    st.session_state.num_years = 5
    for key in DEFAULT_VALUES.keys():
        if key != 'num_years' and key != 'tax_type':
            if isinstance(DEFAULT_VALUES[key], float):
                st.session_state[key] = 0.0
            elif isinstance(DEFAULT_VALUES[key], int):
                st.session_state[key] = 0
    st.rerun()