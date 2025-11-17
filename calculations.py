"""
Tab 2: Calculations
Generates and displays the financial projections table
"""
import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
from io import BytesIO
from decimal import Decimal, ROUND_HALF_UP

def _to_decimal(value):
    """Converts a value to Decimal safely"""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))

def _from_decimal(decimal_value, decimals=2):
    """Converts Decimal back to float with specified precision"""
    return float(decimal_value.quantize(Decimal(10) ** -decimals, rounding=ROUND_HALF_UP))


def render_calculations():
    """Renders the Calculations tab"""
    st.header("📊 Calculations Table")
    
    if st.button("Calculate / Update", key="btn_calc_update_main"):
        st.session_state.generate_calculations = True
    
    if st.session_state.get('generate_calculations', False):
        _generate_and_display_calculations()
    else:
        st.info("👈 Configure parameters in 'Base Scenario' and press the 'Calculate' button")

def _generate_and_display_calculations():
    """Generates calculations and displays the table"""
    try:
        df_calc = _calculate_projection()
        st.session_state.df_calculated = df_calc
        
        # Configure columns for better display
        column_config = {
            "Concept": st.column_config.TextColumn(
                "Concept",
                width="small"
            )
        }
        
        # Add number columns with thousand separator formatting for each year
        for col in df_calc.columns[1:]:
            column_config[col] = st.column_config.NumberColumn(
                col,
                format="accounting",
                width="small"
            )
        
        # Display table with formatting
        st.dataframe(
            df_calc,
            use_container_width=True,
            height=700,
            hide_index=True,
            column_config=column_config
        )

        # Summary Metrics
        _render_summary_metrics(df_calc)
        
        st.success("✅ Calculations table generated successfully")

        # Download button
        st.divider()
        _render_download_button(df_calc)
        
    except Exception as e:
        st.error(f"❌ Error generating calculations: {str(e)}")
        st.exception(e)

def _calculate_projection():
    """Calculates the complete financial projection"""
    n_years = st.session_state.num_years
    
    # Define all concepts for the table
    concepts = [
        "Sales",
        "Cost of Goods Sold",
        "Gross Profit",
        "Labor Cost",
        "Energy Cost",
        "SG&A",
        "Overhead Cost",
        "General Expenses",
        "Depreciation & Amortization",
        "EBIT",
        "Tax",
        "NOPAT",
        "Depreciation & Amortization",
        "CAPEX",
        "Salvage Value (After-Tax)",
        "CXC",
        "CXP",
        "Inventory",
        "OPEX",
        "Var OPEX",
        "Erosion Cost",  # SEPARATED
        "Opportunity Cost",  # SEPARATED
        "Free Cash Flow",
        "Present Value (PV)",
        "Cumulative PV",
    ]
    
    # Create DataFrame with concepts as rows and years as columns
    columns = ["Concept", "Year 0"] + [f"Year {i}" for i in range(1, n_years + 1)]
    df_calc = pd.DataFrame(columns=columns)
    df_calc["Concept"] = concepts
    
    # Fill Year 0 values
    df_calc.loc[df_calc["Concept"] == "CAPEX", "Year 0"] = -st.session_state.capex_year0
    df_calc.loc[df_calc["Concept"] == "Free Cash Flow", "Year 0"] = -st.session_state.capex_year0
    df_calc.loc[df_calc["Concept"] == "Present Value (PV)", "Year 0"] = -st.session_state.capex_year0
    df_calc.loc[df_calc["Concept"] == "Cumulative PV", "Year 0"] = -st.session_state.capex_year0
    
    # Fill all other Year 0 values with 0
    df_calc["Year 0"] = df_calc["Year 0"].fillna(0)
    
    # Calculate each year
    for year in range(1, n_years + 1):
        _calculate_year(df_calc, year)
    
    # Fill remaining NaN with 0
    df_calc = df_calc.fillna(0)
    
    return df_calc

def _calculate_year0(df_calc):
    """Calculates Year 0 values (initial values)"""
    # Year 0 - Initial values
    # Most concepts are 0 in Year 0, only CAPEX and FCF are relevant
    
    # CAPEX (negative because it's an outflow)
    capex_year0 = -st.session_state.capex_year0
    df_calc.loc[df_calc["Concept"] == "CAPEX", "Year 0"] = capex_year0
    
    # Free Cash Flow for Year 0
    # FCF = NOPAT + Depreciation - CAPEX - Var OPEX
    # For Year 0: NOPAT = 0, Depreciation = 0, CAPEX = value, Var OPEX = 0
    # So FCF = 0 + 0 + (-CAPEX) + 0 = -CAPEX
    nopat_year0 = 0
    depreciation_year0 = 0
    var_opex_year0 = 0
    
    fcf_year0_decimal = (_to_decimal(nopat_year0) + 
                        _to_decimal(depreciation_year0) + 
                        _to_decimal(capex_year0) + 
                        _to_decimal(var_opex_year0))
    fcf_year0 = _from_decimal(fcf_year0_decimal)
    
    df_calc.loc[df_calc["Concept"] == "Free Cash Flow", "Year 0"] = fcf_year0
    df_calc.loc[df_calc["Concept"] == "Present Value (PV)", "Year 0"] = fcf_year0
    df_calc.loc[df_calc["Concept"] == "Cumulative PV", "Year 0"] = fcf_year0


def _calculate_year(df_calc, year):
    """Calculates values for a specific year"""
    current_col = f"Year {year}"
    previous_col = f"Year {year-1}"
    
    # ========== SALES ==========
    if year == 1:
        sales = st.session_state.sales_year1
    else:
        previous_sales = df_calc.loc[df_calc["Concept"] == "Sales", previous_col].values[0]
        growth_decimal = _to_decimal(st.session_state.sales_growth) / Decimal('100')
        sales_decimal = _to_decimal(previous_sales) * (Decimal('1') + growth_decimal)
        sales = _from_decimal(sales_decimal)
    
    df_calc.loc[df_calc["Concept"] == "Sales", current_col] = sales
    
    # ========== COST OF GOODS SOLD ==========
    if year == 1:
        cogs_pct_decimal = _to_decimal(st.session_state.cogs_pct) / Decimal('100')
        cogs_decimal = _to_decimal(sales) * cogs_pct_decimal
        cogs = _from_decimal(cogs_decimal)
    else:
        previous_cogs = df_calc.loc[df_calc["Concept"] == "Cost of Goods Sold", previous_col].values[0]
        cost_growth_decimal = _to_decimal(st.session_state.cost_growth) / Decimal('100')
        cogs_decimal = _to_decimal(previous_cogs) * (Decimal('1') + cost_growth_decimal)
        cogs = _from_decimal(cogs_decimal)

    df_calc.loc[df_calc["Concept"] == "Cost of Goods Sold", current_col] = cogs
    
    # ========== GROSS PROFIT ==========
    gross_profit_decimal = _to_decimal(sales) - _to_decimal(cogs)
    gross_profit = _from_decimal(gross_profit_decimal)
    df_calc.loc[df_calc["Concept"] == "Gross Profit", current_col] = gross_profit
    
    # ========== GENERAL EXPENSES (Fixed Costs) ==========
    labor_cost = st.session_state.labor_cost
    energy_cost = st.session_state.energy_cost
    sga = st.session_state.sga
    overhead_cost = st.session_state.overhead_cost
    
    general_expenses = labor_cost + energy_cost + sga + overhead_cost
    
    df_calc.loc[df_calc["Concept"] == "Labor Cost", current_col] = labor_cost
    df_calc.loc[df_calc["Concept"] == "Energy Cost", current_col] = energy_cost
    df_calc.loc[df_calc["Concept"] == "SG&A", current_col] = sga
    df_calc.loc[df_calc["Concept"] == "Overhead Cost", current_col] = overhead_cost
    df_calc.loc[df_calc["Concept"] == "General Expenses", current_col] = general_expenses
    
    # ========== DEPRECIATION & AMORTIZATION ==========
    depreciation = st.session_state.depreciation
    df_calc.loc[df_calc["Concept"] == "Depreciation & Amortization", current_col] = depreciation
    
    # ========== EBIT ==========
    ebit_decimal = (_to_decimal(gross_profit) - 
                    _to_decimal(general_expenses) - 
                    _to_decimal(depreciation))
    ebit = _from_decimal(ebit_decimal)
    df_calc.loc[df_calc["Concept"] == "EBIT", current_col] = ebit
    
    # ========== TAX ==========
    tax_type = st.session_state.get('tax_type', 'percentage')
    
    if tax_type == 'percentage':
        if ebit > 0:
            tax_rate_decimal = _to_decimal(st.session_state.tax_rate) / Decimal('100')
            tax_decimal = _to_decimal(ebit) * tax_rate_decimal
            tax = _from_decimal(tax_decimal)
        else:
            tax = 0
    else:
        tax = st.session_state.tax_fixed_amount
    
    df_calc.loc[df_calc["Concept"] == "Tax", current_col] = tax
    
    # ========== NOPAT ==========
    nopat_decimal = _to_decimal(ebit) - _to_decimal(tax)
    nopat = _from_decimal(nopat_decimal)
    df_calc.loc[df_calc["Concept"] == "NOPAT", current_col] = nopat
    
    # ========== CAPEX ==========
    capex = 0
    df_calc.loc[df_calc["Concept"] == "CAPEX", current_col] = capex
    
    # ========== SALVAGE VALUE (only in final year) ==========
    n_years = st.session_state.num_years
    if year == n_years:
        salvage_value = st.session_state.salvage_value
        salvage_tax = salvage_value * (st.session_state.tax_rate / 100)
        salvage_after_tax = salvage_value - salvage_tax
    else:
        salvage_after_tax = 0
    df_calc.loc[df_calc["Concept"] == "Salvage Value (After-Tax)", current_col] = salvage_after_tax
    
    # ========== CXC (Accounts Receivable) ==========
    ar_days_decimal = _to_decimal(st.session_state.ar_days)
    days_in_year_decimal = _to_decimal(st.session_state.days_in_year)
    cxc_decimal = (_to_decimal(sales) / days_in_year_decimal) * ar_days_decimal
    cxc = -_from_decimal(cxc_decimal)  # NEGATIVE: it's money tied up
    df_calc.loc[df_calc["Concept"] == "CXC", current_col] = cxc
    
    # ========== CXP (Accounts Payable) ==========
    ap_days_decimal = _to_decimal(st.session_state.ap_days)
    cxp_decimal = (_to_decimal(cogs) / days_in_year_decimal) * ap_days_decimal
    cxp = _from_decimal(cxp_decimal)  # POSITIVE: it's financing from suppliers
    df_calc.loc[df_calc["Concept"] == "CXP", current_col] = cxp
    
    # ========== INVENTORY ==========
    inventory_days_decimal = _to_decimal(st.session_state.inventory_days)
    inventory_decimal = (_to_decimal(cogs) / days_in_year_decimal) * inventory_days_decimal
    inventory = -_from_decimal(inventory_decimal)  # NEGATIVE: it's money tied up
    df_calc.loc[df_calc["Concept"] == "Inventory", current_col] = inventory
    
    # ========== OPEX (Operating Working Capital) ==========
    # Now it's simply the sum since signs are already correct
    opex_decimal = _to_decimal(cxc) + _to_decimal(cxp) + _to_decimal(inventory)
    opex = _from_decimal(opex_decimal)
    df_calc.loc[df_calc["Concept"] == "OPEX", current_col] = opex
    
    # ========== VAR OPEX (Change in Operating Working Capital) ==========
    if year == 1:
        previous_opex = 0
    else:
        previous_opex = df_calc.loc[df_calc["Concept"] == "OPEX", previous_col].values[0]
    
    var_opex_decimal = _to_decimal(opex) - _to_decimal(previous_opex)
    var_opex = _from_decimal(var_opex_decimal)
    df_calc.loc[df_calc["Concept"] == "Var OPEX", current_col] = var_opex
    
    # ========== EROSION COST (separate) ==========
    erosion_cost = -st.session_state.erosion_cost  # Negative because it reduces cash flow
    df_calc.loc[df_calc["Concept"] == "Erosion Cost", current_col] = erosion_cost
    
    # ========== OPPORTUNITY COST (separate) ==========
    opportunity_cost = -st.session_state.opportunity_cost  # Negative because it reduces cash flow
    df_calc.loc[df_calc["Concept"] == "Opportunity Cost", current_col] = opportunity_cost
    
    # ========== FREE CASH FLOW ==========
    fcf_decimal = (_to_decimal(nopat) + 
                   _to_decimal(depreciation) + 
                   _to_decimal(capex) + 
                   _to_decimal(var_opex) +
                   _to_decimal(salvage_after_tax) +
                   _to_decimal(erosion_cost) +
                   _to_decimal(opportunity_cost))
    fcf = _from_decimal(fcf_decimal)
    df_calc.loc[df_calc["Concept"] == "Free Cash Flow", current_col] = fcf

    # ========== PRESENT VALUE (PV) ==========
    wacc_decimal = _to_decimal(st.session_state.wacc) / Decimal('100')
    discount_factor_decimal = (Decimal('1') + wacc_decimal) ** year
    pv_decimal = _to_decimal(fcf) / discount_factor_decimal
    pv = _from_decimal(pv_decimal)
    df_calc.loc[df_calc["Concept"] == "Present Value (PV)", current_col] = pv

    # ========== CUMULATIVE PV (NPV) ==========
    if year == 1:
        pv_year0 = df_calc.loc[df_calc["Concept"] == "Present Value (PV)", "Year 0"].values[0]
        cumulative_pv_decimal = _to_decimal(pv_year0) + _to_decimal(pv)
    else:
        previous_cumulative_pv = df_calc.loc[df_calc["Concept"] == "Cumulative PV", previous_col].values[0]
        cumulative_pv_decimal = _to_decimal(previous_cumulative_pv) + _to_decimal(pv)
    
    cumulative_pv = _from_decimal(cumulative_pv_decimal)
    df_calc.loc[df_calc["Concept"] == "Cumulative PV", current_col] = cumulative_pv

def _render_summary_metrics(df_calc):
    """Renders summary financial metrics"""
    st.divider()
    st.subheader("📈 Summary Metrics")
    
    try:
        # Extract data for calculations
        n_years = st.session_state.num_years
        
        # Get all PV values for NPV calculation
        pv_values = []
        for i in range(n_years + 1):
            col_name = f"Year {i}"
            pv = df_calc.loc[df_calc["Concept"] == "Present Value (PV)", col_name].values[0]
            pv_values.append(pv)
        
        # Get all FCF values for IRR calculation
        fcf_values = []
        for i in range(n_years + 1):
            col_name = f"Year {i}"
            fcf = df_calc.loc[df_calc["Concept"] == "Free Cash Flow", col_name].values[0]
            fcf_values.append(fcf)
        
        # ========== NPV (Net Present Value) ==========
        npv = sum(pv_values)
        
        # ========== IRR (Internal Rate of Return) ==========
        try:
            irr = npf.irr(fcf_values)
            irr_pct = irr * 100 if not np.isnan(irr) else 0
        except:
            irr_pct = 0
        
        # ========== PAYBACK PERIOD ==========
        # Calculate cumulative FCF (not discounted initially)
        cumulative_fcf = 0
        payback_period = None
        initial_investment = abs(fcf_values[0])  # Year 0 FCF (absolute value)
        
        for i in range(1, len(fcf_values)):
            cumulative_fcf += fcf_values[i]
            if cumulative_fcf >= initial_investment:
                # Interpolate to find exact payback period
                if i == 1:
                    payback_period = initial_investment / fcf_values[i] if fcf_values[i] > 0 else None
                else:
                    previous_cumulative = cumulative_fcf - fcf_values[i]
                    remaining = initial_investment - previous_cumulative
                    payback_period = (i - 1) + (remaining / fcf_values[i]) if fcf_values[i] > 0 else None
                break
        
        # If payback not achieved, set to None
        if payback_period is None and cumulative_fcf < initial_investment:
            payback_period = None
        
        # ========== DISPLAY METRICS ==========
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="💰 Net Present Value (NPV)",
                value=f"${npv:,.2f}",
                delta="Positive" if npv > 0 else "Negative",
                delta_color="normal" if npv > 0 else "inverse"
            )
            st.caption(f"Sum of all Present Values")
        
        with col2:
            st.metric(
                label="📊 Internal Rate of Return (IRR)",
                value=f"{irr_pct:.2f}%",
                delta=f"vs WACC {st.session_state.wacc:.2f}%",
                delta_color="normal" if irr_pct > st.session_state.wacc else "inverse"
            )
            st.caption(f"Discount rate where NPV = 0")
        
        with col3:
            if payback_period is not None:
                st.metric(
                    label="⏱️ Payback Period",
                    value=f"{payback_period:.4f} years",
                )
                st.caption(f"Time to recover initial investment")
            else:
                st.metric(
                    label="⏱️ Payback Period",
                    value="Not achieved",
                )
                st.caption(f"Project does not recover investment")
        
        # Additional details in expander
        with st.expander("📋 Detailed Calculations"):
            st.write("**NPV Calculation:**")
            st.write(f"- Sum of all PV values: ${npv:,.2f}")
            
            st.write("\n**IRR Calculation:**")
            st.write(f"- Internal Rate of Return: {irr_pct:.4f}%")
            st.write(f"- WACC (hurdle rate): {st.session_state.wacc:.2f}%")
            if irr_pct > st.session_state.wacc:
                st.success(f"✅ IRR ({irr_pct:.2f}%) > WACC ({st.session_state.wacc:.2f}%) - Project is viable")
            else:
                st.error(f"❌ IRR ({irr_pct:.2f}%) < WACC ({st.session_state.wacc:.2f}%) - Project may not be viable")
            
            st.write("\n**Payback Period Calculation:**")
            st.write(f"- Initial Investment: ${initial_investment:,.2f}")
            if payback_period is not None:
                st.write(f"- Payback achieved in: {payback_period:.2f} years")
            else:
                st.write(f"- Cumulative FCF: ${cumulative_fcf:,.2f}")
                st.warning(f"⚠️ Payback not achieved within {n_years} years")
        
    except Exception as e:
        st.error(f"❌ Error calculating metrics: {str(e)}")
        st.exception(e)


def _render_download_button(df_calc):
    """Renders the Excel download button"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_calc.to_excel(writer, index=False, sheet_name='Calculations')
    
    st.download_button(
        label="📥 Download Table",
        data=output.getvalue(),
        file_name="financial_calculations.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )