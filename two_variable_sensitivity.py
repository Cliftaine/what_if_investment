"""
Tab 4: Two-Variable Sensitivity Analysis
Sensitivity analysis of two variables simultaneously on NPV, IRR, and Payback
"""
import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def render_two_variable_sensitivity():
    """Renders the Two-Variable Sensitivity Analysis tab"""
    st.header("🎲 Two-Variable Sensitivity Analysis")
    
    # Styled container
    with st.container():
        st.markdown("""
        <style>
        .stSlider > div > div > div > div {
            background: linear-gradient(90deg, #ff6b6b 0%, #4ecdc4 50%, #45b7d1 100%);
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.info("💡 Analyze how simultaneous changes in TWO variables affect NPV, IRR, and Payback Period")
    
    # Variable selection
    st.markdown("### 🎯 Configuration")
    
    # Available variables (same expanded list)
    variables = {
        # Revenue variables
        "💰 Sales (Year 1)": "sales_year1",
        "📈 Sales Growth (%)": "sales_growth",
        
        # Cost variables
        "📦 COGS (%)": "cogs_pct",
        "📊 Cost Growth (%)": "cost_growth",
        "🔧 Depreciation & Amortization": "depreciation",
        
        # General Expenses variables
        "👷 Labor Cost": "labor_cost",
        "⚡ Energy Cost": "energy_cost",
        "📋 SG&A": "sga",
        "🏢 Overhead Cost": "overhead_cost",
        
        # Financial variables
        "🏛️ Tax Rate (%)": "tax_rate",
        "💹 WACC (%)": "wacc",
        
        # Investment variables
        "🏗️ CAPEX (Year 0)": "capex_year0",
        "💎 Salvage Value": "salvage_value",
        
        # Working Capital variables
        "📅 AR Days": "ar_days",
        "📅 AP Days": "ap_days",
        "📅 Inventory Days": "inventory_days",
        
        # Additional costs
        "🔻 Erosion Cost": "erosion_cost",
        "⏰ Opportunity Cost": "opportunity_cost",
    }
    
    # Variable selection in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📌 Variable 1 (X-Axis)")
        selected_variable_1 = st.selectbox(
            "Select First Variable",
            options=list(variables.keys()),
            key="two_var_variable_1",
            help="This variable will be on the X-axis"
        )
        
        variable_key_1 = variables[selected_variable_1]
        base_value_1 = st.session_state[variable_key_1]
        st.metric(
            label="Current Base Value",
            value=f"{base_value_1:,.2f}",
            help="This is the current value from Base Scenario"
        )
    
    with col2:
        st.markdown("#### 📌 Variable 2 (Y-Axis)")
        selected_variable_2 = st.selectbox(
            "Select Second Variable",
            options=list(variables.keys()),
            key="two_var_variable_2",
            index=1,
            help="This variable will be on the Y-axis"
        )
        
        variable_key_2 = variables[selected_variable_2]
        base_value_2 = st.session_state[variable_key_2]
        st.metric(
            label="Current Base Value",
            value=f"{base_value_2:,.2f}",
            help="This is the current value from Base Scenario"
        )
    
    # Check if same variable selected
    if variable_key_1 == variable_key_2:
        st.error("⚠️ Please select two different variables for analysis")
        return
    
    # Sliders for Variable 1
    st.markdown("---")
    st.markdown("#### 🎚️ Variable 1 Range")
    
    col1_min, col1_max, col1_step = st.columns(3)
    
    with col1_min:
        variation_1_min = st.slider(
            "🔻 Minimum Variation (%)",
            min_value=-100,
            max_value=0,
            value=-30,
            step=1,
            key="two_var_variation_1_min",
            help="How much lower to test Variable 1"
        )
    
    with col1_max:
        variation_1_max = st.slider(
            "🔺 Maximum Variation (%)",
            min_value=0,
            max_value=200,
            value=30,
            step=1,
            key="two_var_variation_1_max",
            help="How much higher to test Variable 1"
        )
    
    with col1_step:
        step_1_size = st.slider(
            "⚡ Step Size (%)",
            min_value=1,
            max_value=30,
            value=10,
            step=1,
            key="two_var_step_1_size",
            help="Increment between each test point for Variable 1"
        )
    
    # Sliders for Variable 2
    st.markdown("#### 🎚️ Variable 2 Range")
    
    col2_min, col2_max, col2_step = st.columns(3)
    
    with col2_min:
        variation_2_min = st.slider(
            "🔻 Minimum Variation (%)",
            min_value=-100,
            max_value=0,
            value=-30,
            step=1,
            key="two_var_variation_2_min",
            help="How much lower to test Variable 2"
        )
    
    with col2_max:
        variation_2_max = st.slider(
            "🔺 Maximum Variation (%)",
            min_value=0,
            max_value=200,
            value=30,
            step=1,
            key="two_var_variation_2_max",
            help="How much higher to test Variable 2"
        )
    
    with col2_step:
        step_2_size = st.slider(
            "⚡ Step Size (%)",
            min_value=1,
            max_value=30,
            value=10,
            step=1,
            key="two_var_step_2_size",
            help="Increment between each test point for Variable 2"
        )
    
    # Visual preview
    st.markdown("---")
    st.markdown("#### 📊 Preview of Analysis Range")
    
    col_prev1, col_prev2 = st.columns(2)
    
    with col_prev1:
        st.write(f"**{selected_variable_1}:**")
        min_val_1 = base_value_1 * (1 + variation_1_min / 100)
        max_val_1 = base_value_1 * (1 + variation_1_max / 100)
        num_points_1 = len(np.arange(variation_1_min, variation_1_max + 1, step_1_size))
        
        st.info(f"Range: {min_val_1:,.2f} to {max_val_1:,.2f}")
        st.caption(f"Number of points: {num_points_1}")
    
    with col_prev2:
        st.write(f"**{selected_variable_2}:**")
        min_val_2 = base_value_2 * (1 + variation_2_min / 100)
        max_val_2 = base_value_2 * (1 + variation_2_max / 100)
        num_points_2 = len(np.arange(variation_2_min, variation_2_max + 1, step_2_size))
        
        st.info(f"Range: {min_val_2:,.2f} to {max_val_2:,.2f}")
        st.caption(f"Number of points: {num_points_2}")
    
    total_scenarios = num_points_1 * num_points_2
    st.warning(f"⚙️ Total scenarios to calculate: **{total_scenarios}**")
    
    # Run analysis button
    st.markdown("---")
    col_btn = st.columns([1, 2, 1])
    with col_btn[1]:
        run_button = st.button(
            "🚀 Run Two-Variable Analysis",
            key="btn_run_two_var_sensitivity",
            use_container_width=True,
            type="primary"
        )
    
    if run_button:
        st.session_state.run_two_var_sensitivity = True
    
    if st.session_state.get('run_two_var_sensitivity', False):
        st.markdown("---")
        _run_two_variable_analysis(
            variable_1_name=selected_variable_1,
            variable_1_key=variable_key_1,
            var_1_min=variation_1_min,
            var_1_max=variation_1_max,
            step_1=step_1_size,
            variable_2_name=selected_variable_2,
            variable_2_key=variable_key_2,
            var_2_min=variation_2_min,
            var_2_max=variation_2_max,
            step_2=step_2_size
        )

def _run_two_variable_analysis(variable_1_name, variable_1_key, var_1_min, var_1_max, step_1,
                                variable_2_name, variable_2_key, var_2_min, var_2_max, step_2):
    """Runs two-variable sensitivity analysis"""
    try:
        # Get base values
        base_value_1 = st.session_state[variable_1_key]
        base_value_2 = st.session_state[variable_2_key]
        
        # Create variations
        variations_1 = np.arange(var_1_min, var_1_max + 1, step_1)
        variations_2 = np.arange(var_2_min, var_2_max + 1, step_2)
        
        # Initialize results storage
        results = []
        
        # Progress tracking
        st.markdown("### ⚙️ Running Analysis...")
        total_scenarios = len(variations_1) * len(variations_2)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        scenario_count = 0
        
        # Store original values
        original_value_1 = st.session_state[variable_1_key]
        original_value_2 = st.session_state[variable_2_key]
        
        # Loop through all combinations
        for var_pct_1 in variations_1:
            for var_pct_2 in variations_2:
                scenario_count += 1
                status_text.markdown(f"**Calculating scenario {scenario_count}/{total_scenarios}:** Var1: `{var_pct_1:+.0f}%` | Var2: `{var_pct_2:+.0f}%`")
                
                # Apply variations
                varied_value_1 = base_value_1 * (1 + var_pct_1 / 100)
                varied_value_2 = base_value_2 * (1 + var_pct_2 / 100)
                
                st.session_state[variable_1_key] = varied_value_1
                st.session_state[variable_2_key] = varied_value_2
                
                # Calculate metrics
                npv, irr, payback = _calculate_metrics_for_two_var_sensitivity()
                
                results.append({
                    f'{variable_1_name} (%)': var_pct_1,
                    f'{variable_2_name} (%)': var_pct_2,
                    f'{variable_1_name} Value': varied_value_1,
                    f'{variable_2_name} Value': varied_value_2,
                    'NPV': npv,
                    'IRR (%)': irr * 100 if irr is not None else 0,
                    'Payback (years)': payback if payback is not None else np.nan
                })
                
                # Update progress
                progress_bar.progress(scenario_count / total_scenarios)
        
        # Restore original values
        st.session_state[variable_1_key] = original_value_1
        st.session_state[variable_2_key] = original_value_2
        
        progress_bar.empty()
        status_text.empty()
        
        # Create DataFrame
        df_results = pd.DataFrame(results)
        
        st.success("✅ Two-variable sensitivity analysis completed successfully!")
        
        # Display results
        st.markdown("---")
        _display_two_var_results(df_results, variable_1_name, variable_2_name, variations_1, variations_2)
        
    except Exception as e:
        st.error(f"❌ Error in two-variable analysis: {str(e)}")
        st.exception(e)

def _calculate_metrics_for_two_var_sensitivity():
    """Calculates NPV, IRR, and Payback for current scenario"""
    try:
        n_years = st.session_state.num_years
        
        # Calculate FCF for all years
        fcf_values = []
        
        # Year 0
        fcf_year0 = -st.session_state.capex_year0
        fcf_values.append(fcf_year0)
        
        # Years 1 onwards
        previous_opex = 0
        sales = st.session_state.sales_year1
        cogs = sales * (st.session_state.cogs_pct / 100)
        labor_cost = st.session_state.labor_cost
        energy_cost = st.session_state.energy_cost
        
        for year in range(1, n_years + 1):
            # Apply growth from Year 2 onwards
            if year > 1:
                sales = sales * (1 + st.session_state.sales_growth / 100)
                cogs = cogs * (1 + st.session_state.cost_growth / 100)
            
            # Gross Profit
            gross_profit = sales - cogs
            
            # General Expenses
            sga = st.session_state.sga
            overhead_cost = st.session_state.overhead_cost
            general_expenses = labor_cost + energy_cost + sga + overhead_cost
            
            # EBIT
            ebit = gross_profit - general_expenses - st.session_state.depreciation
            
            # Tax
            tax_type = st.session_state.get('tax_type', 'percentage')
            if tax_type == 'percentage':
                tax = ebit * (st.session_state.tax_rate / 100) if ebit > 0 else 0
            else:
                tax = st.session_state.tax_fixed_amount
            
            # NOPAT
            nopat = ebit - tax
            
            # Working Capital
            days_in_year = st.session_state.days_in_year
            cxc = -((sales / days_in_year) * st.session_state.ar_days)
            cxp = (cogs / days_in_year) * st.session_state.ap_days
            inventory = -((cogs / days_in_year) * st.session_state.inventory_days)
            
            opex = cxc + cxp + inventory
            var_opex = opex - previous_opex
            previous_opex = opex
            
            # Salvage value
            if year == n_years:
                salvage_value = st.session_state.salvage_value
                salvage_tax = salvage_value * (st.session_state.tax_rate / 100)
                salvage_after_tax = salvage_value - salvage_tax
            else:
                salvage_after_tax = 0
            
            # Erosion & Opportunity
            erosion_cost = -st.session_state.erosion_cost
            opportunity_cost = -st.session_state.opportunity_cost
            
            # FCF
            fcf = (nopat + st.session_state.depreciation + var_opex + 
                   salvage_after_tax + erosion_cost + opportunity_cost)
            fcf_values.append(fcf)
        
        # Calculate NPV
        wacc = st.session_state.wacc / 100
        pv_values = [fcf / ((1 + wacc) ** i) for i, fcf in enumerate(fcf_values)]
        npv = sum(pv_values)
        
        # Calculate IRR
        try:
            irr = npf.irr(fcf_values)
            if np.isnan(irr):
                irr = None
        except:
            irr = None
        
        # Calculate Payback
        cumulative_fcf = 0
        payback = None
        initial_investment = abs(fcf_values[0])
        
        for i in range(1, len(fcf_values)):
            cumulative_fcf += fcf_values[i]
            if cumulative_fcf >= initial_investment:
                if i == 1:
                    payback = initial_investment / fcf_values[i] if fcf_values[i] > 0 else None
                else:
                    previous_cumulative = cumulative_fcf - fcf_values[i]
                    remaining = initial_investment - previous_cumulative
                    payback = (i - 1) + (remaining / fcf_values[i]) if fcf_values[i] > 0 else None
                break
        
        return npv, irr, payback
        
    except Exception as e:
        return 0, None, None

def _display_two_var_results(df_results, var_1_name, var_2_name, variations_1, variations_2):
    """Displays results with heatmaps"""
    st.subheader("📊 Heatmap Results")
    
    # Create pivot tables for heatmaps
    pivot_npv = df_results.pivot_table(
        values='NPV',
        index=f'{var_2_name} (%)',
        columns=f'{var_1_name} (%)'
    )
    
    pivot_irr = df_results.pivot_table(
        values='IRR (%)',
        index=f'{var_2_name} (%)',
        columns=f'{var_1_name} (%)'
    )
    
    pivot_payback = df_results.pivot_table(
        values='Payback (years)',
        index=f'{var_2_name} (%)',
        columns=f'{var_1_name} (%)'
    )
    
    # Create heatmaps
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('NPV Heatmap', 'IRR Heatmap', 'Payback Heatmap'),
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}]]
    )
    
    # NPV Heatmap
    fig.add_trace(
        go.Heatmap(
            z=pivot_npv.values,
            x=pivot_npv.columns,
            y=pivot_npv.index,
            colorscale='RdYlGn',
            name='NPV',
            hovertemplate=f'{var_1_name}: %{{x}}%<br>{var_2_name}: %{{y}}%<br>NPV: $%{{z:,.2f}}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # IRR Heatmap
    fig.add_trace(
        go.Heatmap(
            z=pivot_irr.values,
            x=pivot_irr.columns,
            y=pivot_irr.index,
            colorscale='RdYlGn',
            name='IRR',
            hovertemplate=f'{var_1_name}: %{{x}}%<br>{var_2_name}: %{{y}}%<br>IRR: %{{z:.2f}}%<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Payback Heatmap
    fig.add_trace(
        go.Heatmap(
            z=pivot_payback.values,
            x=pivot_payback.columns,
            y=pivot_payback.index,
            colorscale='RdYlGn_r',
            name='Payback',
            hovertemplate=f'{var_1_name}: %{{x}}%<br>{var_2_name}: %{{y}}%<br>Payback: %{{z:.2f}} years<extra></extra>'
        ),
        row=1, col=3
    )
    
    fig.update_xaxes(title_text=f"{var_1_name} Variation (%)", row=1, col=1)
    fig.update_xaxes(title_text=f"{var_1_name} Variation (%)", row=1, col=2)
    fig.update_xaxes(title_text=f"{var_1_name} Variation (%)", row=1, col=3)
    
    fig.update_yaxes(title_text=f"{var_2_name} Variation (%)", row=1, col=1)
    fig.update_yaxes(title_text=f"{var_2_name} Variation (%)", row=1, col=2)
    fig.update_yaxes(title_text=f"{var_2_name} Variation (%)", row=1, col=3)
    
    fig.update_layout(
        height=500,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='gray')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show raw data table
    with st.expander("📋 View Raw Data"):
        st.dataframe(df_results, use_container_width=True, height=400)