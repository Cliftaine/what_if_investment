"""
Tab 1: Base Scenario
Inputs to configure project parameters
"""
import streamlit as st
from utils import load_default_values, clear_values

def render_base_scenario():
    """Renders the Base Scenario tab"""

    st.info("💡 Enter your project's base parameters")

    # Layout in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        _investment_inputs()
    
    with col2:
        _sales_inputs()
        _costs_inputs()
    
    with col3:
        _ratio_inputs()

    st.divider()
    
    # Action buttons
    _render_action_buttons()

def _sales_inputs():
    """Renders sales inputs"""
    st.subheader("💰 Sales")
    sales_year1 = st.number_input(
        "Sales (Year 1)",
        min_value=0.0,
        value=st.session_state.sales_year1,
        step=100000.0,
        format="%.2f",
        key="input_sales_year1"
    )
    st.session_state.sales_year1 = sales_year1
    
    sales_growth = st.number_input(
        "Sales Growth after Inflation (%) - Year 2 onwards",
        min_value=0.0,
        max_value=100.0,
        value=st.session_state.sales_growth,
        step=0.1,
        format="%.2f",
        key="input_sales_growth"
    )
    st.session_state.sales_growth = sales_growth
    
    st.divider()

def _costs_inputs():
    """Renders costs inputs"""
    st.subheader("📦 Costs")
    cogs_pct = st.number_input(
        "Cost of Goods Sold (% of Sales)",
        min_value=0.0,
        max_value=100.0,
        value=st.session_state.cogs_pct,
        step=0.1,
        format="%.2f",
        key="input_cogs_pct"
    )
    st.session_state.cogs_pct = cogs_pct
    
    cost_growth = st.number_input(
        "Cost Growth (%) - Year 2 onwards",
        min_value=0.0,
        max_value=100.0,
        value=st.session_state.cost_growth,
        step=0.1,
        format="%.2f",
        key="input_cost_growth"
    )
    st.session_state.cost_growth = cost_growth
    
    depreciation = st.number_input(
        "Depreciation & Amortization (annual)",
        min_value=0.0,
        value=st.session_state.depreciation,
        step=10000.0,
        format="%.2f",
        key="input_depreciation"
    )
    st.session_state.depreciation = depreciation
    
    st.divider()

    
    labor_cost = st.number_input(
        "Labor Cost (Annual)",
        min_value=0.0,
        value=st.session_state.labor_cost,
        step=10000.0,
        format="%.2f",
        key="input_labor_cost",
        help="Annual labor costs"
    )
    st.session_state.labor_cost = labor_cost
    
    energy_cost = st.number_input(
        "Energy Cost (Annual)",
        min_value=0.0,
        value=st.session_state.energy_cost,
        step=10000.0,
        format="%.2f",
        key="input_energy_cost",
        help="Annual energy costs"
    )
    st.session_state.energy_cost = energy_cost
    
    sga = st.number_input(
        "SG&A (Annual)",
        min_value=0.0,
        value=st.session_state.sga,
        step=10000.0,
        format="%.2f",
        key="input_sga",
        help="Selling, General & Administrative expenses"
    )
    st.session_state.sga = sga
    
    overhead_cost = st.number_input(
        "Overhead Cost (Annual)",
        min_value=0.0,
        value=st.session_state.overhead_cost,
        step=10000.0,
        format="%.2f",
        key="input_overhead_cost",
        help="Annual overhead costs"
    )
    st.session_state.overhead_cost = overhead_cost
    
    # Calculate and show total General Expenses
    total_general_expenses = labor_cost + energy_cost + sga + overhead_cost
    st.info(f"💡 Total General Expenses: ${total_general_expenses:,.2f}")

def _investment_inputs():
    """Renders investment and initial values inputs"""
    st.subheader("📊 Initial Values")
    
    num_years = st.number_input(
        "Projection Years",
        min_value=1,
        max_value=20,
        value=st.session_state.num_years,
        step=1,
        key="input_num_years"
    )
    st.session_state.num_years = num_years
    
    wacc = st.number_input(
        "WACC (%)",
        min_value=0.0,
        max_value=100.0,
        value=st.session_state.get('wacc', 15.0),
        step=0.1,
        format="%.2f",
        key="input_wacc",
        help="Weighted Average Cost of Capital - Discount rate for free cash flows"
    )
    st.session_state.wacc = wacc
    
    st.info(f"💡 Current WACC: {wacc:.2f}%")

    ebt_year0 = st.number_input(
        "Earnings Before Tax (Year 0)",
        value=st.session_state.ebt_year0,
        step=1000.0,
        format="%.2f",
        key="input_ebt_year0"
    )
    st.session_state.ebt_year0 = ebt_year0
    
    tax_year0 = st.number_input(
        "Tax (Year 0)",
        value=st.session_state.tax_year0,
        step=1000.0,
        format="%.2f",
        key="input_tax_year0"
    )
    st.session_state.tax_year0 = tax_year0
    
    st.divider()

    tax_type = st.radio(
        "Tax Type",
        options=['percentage', 'fixed'],
        format_func=lambda x: "Percentage of EBIT" if x == 'percentage' else "Fixed Amount",
        key="input_tax_type",
        horizontal=True,
        index=0 if st.session_state.get('tax_type', 'percentage') == 'percentage' else 1,
        help="Choose whether tax is calculated as a percentage of EBIT or as a fixed annual amount"
    )
    st.session_state.tax_type = tax_type
    
    if tax_type == 'percentage':
        tax_rate = st.number_input(
            "Tax Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=st.session_state.tax_rate,
            step=0.1,
            format="%.2f",
            key="input_tax_rate",
            help="Tax rate applied to EBIT (only if EBIT > 0)"
        )
        st.session_state.tax_rate = tax_rate
        st.caption(f"📌 Tax will be calculated as: EBIT × {tax_rate}%")
    else:
        tax_fixed_amount = st.number_input(
            "Fixed Tax Amount (Annual)",
            min_value=0.0,
            value=st.session_state.get('tax_fixed_amount', 0.0),
            step=10000.0,
            format="%.2f",
            key="input_tax_fixed_amount",
            help="Fixed tax amount paid annually regardless of EBIT"
        )
        st.session_state.tax_fixed_amount = tax_fixed_amount
        st.caption(f"📌 Tax will be: ${tax_fixed_amount:,.2f} per year")
    
    st.divider()
    
    capex_year0 = st.number_input(
        "CAPEX (Year 0)",
        value=st.session_state.capex_year0,
        step=100000.0,
        format="%.2f",
        key="input_capex_year0"
    )
    st.session_state.capex_year0 = capex_year0
    
    salvage_value = st.number_input(
        f"Salvage Value (Year {st.session_state.num_years})",
        min_value=0.0,
        value=st.session_state.salvage_value,
        step=10000.0,
        format="%.2f",
        key="input_salvage_value",
        help="Resale value of assets at end of project"
    )
    st.session_state.salvage_value = salvage_value

def _ratio_inputs():
    """Renders turnover ratios inputs"""
    st.subheader("🔄 Turnover Ratios (days)")

    days_in_year = st.number_input(
        "Days in Year (for turnover calculations)",
        min_value=1,
        max_value=365,
        value=st.session_state.days_in_year,
        step=1,
        help="Use 365 for calendar days, 360 for financial year, or 252 for business days",
        key="input_days_in_year"
    )
    st.session_state.days_in_year = days_in_year

    ar_days = st.number_input(
        "Accounts Receivable Days",
        min_value=0,
        value=st.session_state.ar_days,
        step=1,
        key="input_ar_days"
    )
    st.session_state.ar_days = ar_days
    
    ap_days = st.number_input(
        "Accounts Payable Days",
        min_value=0,
        value=st.session_state.ap_days,
        step=1,
        key="input_ap_days"
    )
    st.session_state.ap_days = ap_days
    
    inventory_days = st.number_input(
        "Inventory Turnover Days",
        min_value=0,
        value=st.session_state.inventory_days,
        step=1,
        key="input_inventory_days"
    )
    st.session_state.inventory_days = inventory_days

    st.divider()
    st.subheader("📉 Additional Adjustments")
    
    erosion_cost = st.number_input(
        "Erosion Cost (Annual After-Tax)",
        min_value=0.0,
        value=st.session_state.erosion_cost,
        step=10000.0,
        format="%.2f",
        key="input_erosion_cost",
        help="Annual cash flow erosion from cannibalization of existing products"
    )
    st.session_state.erosion_cost = erosion_cost
    
    opportunity_cost = st.number_input(
        "Opportunity Cost (Annual)",
        min_value=0.0,
        value=st.session_state.opportunity_cost,
        step=1000.0,
        format="%.2f",
        key="input_opportunity_cost",
        help="Annual opportunity cost (e.g., foregone lease revenue)"
    )
    st.session_state.opportunity_cost = opportunity_cost

def _render_action_buttons():
    """Renders action buttons"""
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("🎯 Load Default Values", key="btn_load_defaults"):
            load_default_values()
    
    with col_btn2:
        if st.button("🔄 Clear Values", key="btn_clear_values"):
            clear_values()
    
    with col_btn3:
        if st.button("➡️ Generate Calculations Table", key="btn_generate_calc"):
            st.session_state.generate_calculations = True
            st.success("✅ Go to 'Calculations' tab to see the results")