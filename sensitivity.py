"""
Tab 3: Sensitivity Analysis
Single-variable sensitivity analysis on NPV, IRR, and Payback
"""
import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def render_sensitivity():
    """Renders the Sensitivity Analysis tab"""
    st.header("📈 Sensitivity Analysis")
    
    # Styled container for inputs
    with st.container():
        st.markdown("""
        <style>
        .stSlider > div > div > div > div {
            background: linear-gradient(90deg, #ff6b6b 0%, #4ecdc4 50%, #45b7d1 100%);
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.info("💡 Analyze how changes in ONE variable affect NPV, IRR, and Payback Period")
    
    # Variable selection
    st.markdown("### 🎯 Select Variable for Analysis")
    
    # Expanded list of available variables
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
    
    selected_variable = st.selectbox(
        "Choose a variable to analyze",
        options=list(variables.keys()),
        key="sensitivity_variable",
        help="Select which variable you want to test for sensitivity"
    )
    
    variable_key = variables[selected_variable]
    base_value = st.session_state[variable_key]
    
    # Display current value
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.metric(
            label="Current Base Value",
            value=f"{base_value:,.2f}",
            help="This is the current value from your Base Scenario"
        )
    
    with col_info2:
        if variable_key in ['sales_growth', 'cost_growth', 'cogs_pct', 'tax_rate', 'wacc']:
            st.caption(f"📌 Variable type: **Percentage**")
        else:
            st.caption(f"📌 Variable type: **Absolute Value**")
    
    # Sensitivity range configuration
    st.markdown("---")
    st.markdown("### 🎚️ Configure Sensitivity Range")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        variation_min = st.slider(
            "🔻 Minimum Variation (%)",
            min_value=-100,
            max_value=0,
            value=-50,
            step=5,
            key="sensitivity_variation_min",
            help="How much lower to test the variable"
        )
    
    with col2:
        variation_max = st.slider(
            "🔺 Maximum Variation (%)",
            min_value=0,
            max_value=200,
            value=50,
            step=5,
            key="sensitivity_variation_max",
            help="How much higher to test the variable"
        )
    
    with col3:
        step_size = st.slider(
            "⚡ Step Size (%)",
            min_value=5,
            max_value=30,
            value=10,
            step=5,
            key="sensitivity_step_size",
            help="Increment between each test point"
        )
    
    # Visual preview of range
    st.markdown("---")
    st.markdown("### 📊 Preview of Analysis Range")
    
    col_prev1, col_prev2, col_prev3 = st.columns(3)
    
    min_value = base_value * (1 + variation_min / 100)
    max_value = base_value * (1 + variation_max / 100)
    num_points = len(np.arange(variation_min, variation_max + 1, step_size))
    
    with col_prev1:
        st.metric("Minimum Value", f"{min_value:,.2f}", f"{variation_min}%")
    
    with col_prev2:
        st.metric("Maximum Value", f"{max_value:,.2f}", f"{variation_max}%")
    
    with col_prev3:
        st.metric("Number of Points", num_points)
    
    # Run analysis button
    st.markdown("---")
    col_btn = st.columns([1, 2, 1])
    with col_btn[1]:
        run_button = st.button(
            "🚀 Run Sensitivity Analysis",
            key="btn_run_sensitivity",
            use_container_width=True,
            type="primary"
        )
    
    if run_button:
        st.session_state.run_sensitivity = True
    
    # Run analysis
    if st.session_state.get('run_sensitivity', False):
        st.markdown("---")
        _run_sensitivity_analysis(
            variable_name=selected_variable,
            variable_key=variable_key,
            base_value=base_value,
            var_min=variation_min,
            var_max=variation_max,
            step=step_size
        )

def _run_sensitivity_analysis(variable_name, variable_key, base_value, var_min, var_max, step):
    """Runs sensitivity analysis for selected variable"""
    try:
        # Create variations
        variations = np.arange(var_min, var_max + 1, step)
        
        # Store original value
        original_value = st.session_state[variable_key]
        
        # Initialize results storage
        results = []
        
        # Progress tracking
        st.markdown("### ⚙️ Running Analysis...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Loop through variations
        for i, var_pct in enumerate(variations):
            status_text.markdown(f"**Calculating:** Variation `{var_pct:+.0f}%`")
            
            # Apply variation
            varied_value = base_value * (1 + var_pct / 100)
            st.session_state[variable_key] = varied_value
            
            # Calculate metrics
            npv, irr, payback = _calculate_metrics_for_sensitivity()
            
            results.append({
                'Variation (%)': var_pct,
                'Value': varied_value,
                'NPV': npv,
                'IRR (%)': irr * 100 if irr is not None else 0,
                'Payback (years)': payback if payback is not None else np.nan
            })
            
            # Update progress
            progress_bar.progress((i + 1) / len(variations))
        
        # Restore original value
        st.session_state[variable_key] = original_value
        
        progress_bar.empty()
        status_text.empty()
        
        # Create DataFrame
        df_sensitivity = pd.DataFrame(results)
        
        st.success("✅ Sensitivity analysis completed successfully!")
        
        # Display results
        st.markdown("---")
        _display_results_table(df_sensitivity, variable_name)
        _create_sensitivity_charts(df_sensitivity, variable_name, base_value)
        
    except Exception as e:
        st.error(f"❌ Error in sensitivity analysis: {str(e)}")
        st.exception(e)

def _calculate_metrics_for_sensitivity():
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
            
            # General Expenses (fixed costs)
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
            
            # Working Capital components
            days_in_year = st.session_state.days_in_year
            cxc = -((sales / days_in_year) * st.session_state.ar_days)
            cxp = (cogs / days_in_year) * st.session_state.ap_days
            inventory = -((cogs / days_in_year) * st.session_state.inventory_days)
            
            # OPEX
            opex = cxc + cxp + inventory
            
            # Var OPEX
            var_opex = opex - previous_opex
            previous_opex = opex
            
            # Salvage value (only in final year)
            if year == n_years:
                salvage_value = st.session_state.salvage_value
                salvage_tax = salvage_value * (st.session_state.tax_rate / 100)
                salvage_after_tax = salvage_value - salvage_tax
            else:
                salvage_after_tax = 0
            
            # Erosion & Opportunity Costs
            erosion_cost = -st.session_state.erosion_cost
            opportunity_cost = -st.session_state.opportunity_cost
            
            # FCF
            fcf = (nopat + st.session_state.depreciation + var_opex + 
                   salvage_after_tax + erosion_cost + opportunity_cost)
            fcf_values.append(fcf)
        
        # Calculate NPV
        wacc = st.session_state.wacc / 100
        pv_values = []
        for year, fcf in enumerate(fcf_values):
            pv = fcf / ((1 + wacc) ** year)
            pv_values.append(pv)
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
        return -np.inf, None, None

def _display_results_table(df_sensitivity, variable_name):
    """Displays results table with heatmap styling"""
    st.subheader(f"📊 Results: {variable_name}")
    
    # Create styled dataframe
    def style_heatmap(val, column_name):
        """Apply color gradient based on value"""
        if pd.isna(val) or val is None:
            return 'background-color: transparent'
        
        if column_name == 'NPV':
            if val > 0:
                intensity = min(abs(val) / df_sensitivity['NPV'].max() * 100, 100) if df_sensitivity['NPV'].max() > 0 else 0
                return f'background-color: rgba(76, 175, 80, {intensity/100}); color: white'
            else:
                intensity = min(abs(val) / abs(df_sensitivity['NPV'].min()) * 100, 100) if df_sensitivity['NPV'].min() < 0 else 0
                return f'background-color: rgba(244, 67, 54, {intensity/100}); color: white'
        
        elif column_name == 'IRR (%)':
            wacc = st.session_state.wacc
            if val > wacc:
                intensity = min((val - wacc) / 50 * 100, 100)
                return f'background-color: rgba(76, 175, 80, {intensity/100}); color: white'
            else:
                intensity = min((wacc - val) / 50 * 100, 100)
                return f'background-color: rgba(244, 67, 54, {intensity/100}); color: white'
        
        elif column_name == 'Payback (years)':
            max_payback = df_sensitivity['Payback (years)'].max()
            if not pd.isna(max_payback) and max_payback > 0:
                intensity = (1 - val / max_payback) * 100
                return f'background-color: rgba(76, 175, 80, {intensity/100}); color: white'
        
        return 'background-color: transparent'
    
    # Apply styling
    styled_df = df_sensitivity.style.apply(
        lambda x: [style_heatmap(v, x.name) for v in x],
        axis=0,
        subset=['NPV', 'IRR (%)', 'Payback (years)']
    ).format({
        'Variation (%)': '{:+.0f}%',
        'Value': '{:,.2f}',
        'NPV': lambda x: '${:,.2f}'.format(x) if not pd.isna(x) else 'N/A',
        'IRR (%)': lambda x: '{:.2f}%'.format(x) if not pd.isna(x) else 'N/A',
        'Payback (years)': lambda x: '{:.2f}'.format(x) if not pd.isna(x) else 'N/A'
    })
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=400
    )
    
    # Legend
    st.caption("🟢 Green: Favorable | 🔴 Red: Unfavorable | N/A: Not Achieved")

def _create_sensitivity_charts(df_sensitivity, variable_name, base_value):
    """Creates interactive charts for sensitivity analysis"""
    st.subheader("📈 Interactive Charts")
    
    # Create subplot with 3 charts
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('💰 NPV Sensitivity', '📊 IRR Sensitivity', '⏱️ Payback Sensitivity'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # NPV Chart
    fig.add_trace(
        go.Scatter(
            x=df_sensitivity['Variation (%)'],
            y=df_sensitivity['NPV'],
            mode='lines+markers',
            name='NPV',
            line=dict(color='#4CAF50', width=4),
            marker=dict(size=10, symbol='diamond'),
            fill='tozeroy',
            fillcolor='rgba(76, 175, 80, 0.1)',
            hovertemplate='<b>Variation:</b> %{x:+.0f}%<br><b>NPV:</b> $%{y:,.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1, opacity=0.7, line_width=2)
    
    # IRR Chart
    fig.add_trace(
        go.Scatter(
            x=df_sensitivity['Variation (%)'],
            y=df_sensitivity['IRR (%)'],
            mode='lines+markers',
            name='IRR',
            line=dict(color='#2196F3', width=4),
            marker=dict(size=10, symbol='diamond'),
            fill='tozeroy',
            fillcolor='rgba(33, 150, 243, 0.1)',
            hovertemplate='<b>Variation:</b> %{x:+.0f}%<br><b>IRR:</b> %{y:.2f}%<extra></extra>'
        ),
        row=1, col=2
    )
    
    wacc = st.session_state.wacc
    fig.add_hline(y=wacc, line_dash="dash", line_color="orange", row=1, col=2, opacity=0.7, line_width=2,
                  annotation_text=f"WACC: {wacc:.2f}%", annotation_position="top right")
    
    # Payback Chart
    fig.add_trace(
        go.Scatter(
            x=df_sensitivity['Variation (%)'],
            y=df_sensitivity['Payback (years)'],
            mode='lines+markers',
            name='Payback',
            line=dict(color='#FF9800', width=4),
            marker=dict(size=10, symbol='diamond'),
            fill='tozeroy',
            fillcolor='rgba(255, 152, 0, 0.1)',
            hovertemplate='<b>Variation:</b> %{x:+.0f}%<br><b>Payback:</b> %{y:.2f} years<extra></extra>'
        ),
        row=1, col=3
    )
    
    # Update layout
    fig.update_xaxes(title_text="Variation (%)", gridcolor='rgba(128, 128, 128, 0.2)', row=1, col=1)
    fig.update_xaxes(title_text="Variation (%)", gridcolor='rgba(128, 128, 128, 0.2)', row=1, col=2)
    fig.update_xaxes(title_text="Variation (%)", gridcolor='rgba(128, 128, 128, 0.2)', row=1, col=3)
    
    fig.update_yaxes(title_text="NPV ($)", gridcolor='rgba(128, 128, 128, 0.2)', row=1, col=1)
    fig.update_yaxes(title_text="IRR (%)", gridcolor='rgba(128, 128, 128, 0.2)', row=1, col=2)
    fig.update_yaxes(title_text="Payback (years)", gridcolor='rgba(128, 128, 128, 0.2)', row=1, col=3)
    
    fig.update_layout(
        height=500,
        showlegend=False,
        title_text=f"<b>Sensitivity Analysis: {variable_name}</b>",
        title_font_size=20,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='gray')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.markdown("---")
    st.subheader("🔍 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        npv_range = df_sensitivity['NPV'].max() - df_sensitivity['NPV'].min()
        npv_positive_pct = (df_sensitivity['NPV'] > 0).sum() / len(df_sensitivity) * 100
        st.metric(
            label="💰 NPV Range",
            value=f"${npv_range:,.2f}",
            delta=f"{npv_positive_pct:.0f}% scenarios are positive"
        )
        st.caption(f"Max: ${df_sensitivity['NPV'].max():,.2f} | Min: ${df_sensitivity['NPV'].min():,.2f}")
    
    with col2:
        irr_range = df_sensitivity['IRR (%)'].max() - df_sensitivity['IRR (%)'].min()
        irr_above_wacc_pct = (df_sensitivity['IRR (%)'] > st.session_state.wacc).sum() / len(df_sensitivity) * 100
        st.metric(
            label="📊 IRR Range",
            value=f"{irr_range:.2f}%",
            delta=f"{irr_above_wacc_pct:.0f}% scenarios beat WACC"
        )
        st.caption(f"Max: {df_sensitivity['IRR (%)'].max():.2f}% | Min: {df_sensitivity['IRR (%)'].min():.2f}%")
    
    with col3:
        payback_valid = df_sensitivity['Payback (years)'].dropna()
        if len(payback_valid) > 0:
            payback_range = payback_valid.max() - payback_valid.min()
            payback_achieved_pct = len(payback_valid) / len(df_sensitivity) * 100
            st.metric(
                label="⏱️ Payback Range",
                value=f"{payback_range:.2f} years",
                delta=f"{payback_achieved_pct:.0f}% achieve payback"
            )
            st.caption(f"Best: {payback_valid.min():.2f} yrs | Worst: {payback_valid.max():.2f} yrs")
        else:
            st.metric(
                label="⏱️ Payback Range",
                value="N/A",
                delta="No scenarios achieve payback"
            )