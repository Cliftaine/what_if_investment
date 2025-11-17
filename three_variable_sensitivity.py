"""
Tab: Three-Variable Sensitivity Analysis
Sensitivity analysis of three variables simultaneously on NPV
"""
import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.graph_objects as go

def render_three_variable_sensitivity():
    """Renders the Three-Variable Sensitivity Analysis tab"""
    st.header("🎯 Three-Variable Sensitivity Analysis")
    
    st.info("💡 Analyze how simultaneous changes in THREE variables affect NPV. Uses Monte Carlo sampling for efficiency.")
    
    st.warning("⚠️ Due to computational complexity, this analysis uses sampling instead of exhaustive grid search.")
    
    # Variable selection
    st.markdown("### 🎯 Select Variables")
    
    # Available variables
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
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📌 Variable 1")
        selected_var_1 = st.selectbox("Select First Variable", list(variables.keys()), key="three_var_1")
        var_key_1 = variables[selected_var_1]
        base_val_1 = st.session_state[var_key_1]
        st.metric("Base Value", f"{base_val_1:,.2f}")
        
        var_1_min = st.slider("Min Variation (%)", -100, 0, -30, 5, key="three_var_1_min")
        var_1_max = st.slider("Max Variation (%)", 0, 200, 30, 5, key="three_var_1_max")
    
    with col2:
        st.markdown("#### 📌 Variable 2")
        selected_var_2 = st.selectbox("Select Second Variable", list(variables.keys()), index=1, key="three_var_2")
        var_key_2 = variables[selected_var_2]
        base_val_2 = st.session_state[var_key_2]
        st.metric("Base Value", f"{base_val_2:,.2f}")
        
        var_2_min = st.slider("Min Variation (%)", -100, 0, -30, 5, key="three_var_2_min")
        var_2_max = st.slider("Max Variation (%)", 0, 200, 30, 5, key="three_var_2_max")
    
    with col3:
        st.markdown("#### 📌 Variable 3")
        selected_var_3 = st.selectbox("Select Third Variable", list(variables.keys()), index=2, key="three_var_3")
        var_key_3 = variables[selected_var_3]
        base_val_3 = st.session_state[var_key_3]
        st.metric("Base Value", f"{base_val_3:,.2f}")
        
        var_3_min = st.slider("Min Variation (%)", -100, 0, -30, 5, key="three_var_3_min")
        var_3_max = st.slider("Max Variation (%)", 0, 200, 30, 5, key="three_var_3_max")
    
    # Check for duplicates
    if len({var_key_1, var_key_2, var_key_3}) < 3:
        st.error("⚠️ Please select three different variables")
        return
    
    # Sampling configuration
    st.markdown("---")
    st.markdown("### ⚙️ Sampling Configuration")
    
    n_samples = st.slider(
        "Number of Samples",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        help="More samples = better coverage but slower computation"
    )
    
    st.info(f"💡 Will generate {n_samples} random combinations of the three variables")
    
    # Run button
    st.markdown("---")
    col_btn = st.columns([1, 2, 1])
    with col_btn[1]:
        if st.button("🚀 Run Three-Variable Analysis", use_container_width=True, type="primary"):
            st.session_state.run_three_var = True
    
    if st.session_state.get('run_three_var', False):
        st.markdown("---")
        _run_three_var_analysis(
            selected_var_1, var_key_1, base_val_1, var_1_min, var_1_max,
            selected_var_2, var_key_2, base_val_2, var_2_min, var_2_max,
            selected_var_3, var_key_3, base_val_3, var_3_min, var_3_max,
            n_samples
        )

def _run_three_var_analysis(name_1, key_1, base_1, min_1, max_1,
                            name_2, key_2, base_2, min_2, max_2,
                            name_3, key_3, base_3, min_3, max_3,
                            n_samples):
    """Runs three-variable sensitivity analysis using Monte Carlo sampling"""
    try:
        st.markdown("### ⚙️ Running Analysis...")
        
        # Store originals
        orig_1 = st.session_state[key_1]
        orig_2 = st.session_state[key_2]
        orig_3 = st.session_state[key_3]
        
        # Generate random samples
        np.random.seed(42)
        samples_1 = np.random.uniform(min_1, max_1, n_samples)
        samples_2 = np.random.uniform(min_2, max_2, n_samples)
        samples_3 = np.random.uniform(min_3, max_3, n_samples)
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(n_samples):
            status_text.markdown(f"**Sample {i+1}/{n_samples}**")
            
            # Apply variations
            val_1 = base_1 * (1 + samples_1[i] / 100)
            val_2 = base_2 * (1 + samples_2[i] / 100)
            val_3 = base_3 * (1 + samples_3[i] / 100)
            
            st.session_state[key_1] = val_1
            st.session_state[key_2] = val_2
            st.session_state[key_3] = val_3
            
            # Calculate NPV
            npv, irr, payback = _calculate_metrics()
            
            results.append({
                f'{name_1} (%)': samples_1[i],
                f'{name_2} (%)': samples_2[i],
                f'{name_3} (%)': samples_3[i],
                f'{name_1} Value': val_1,
                f'{name_2} Value': val_2,
                f'{name_3} Value': val_3,
                'NPV': npv,
                'IRR (%)': irr * 100 if irr else 0,
                'Payback (years)': payback if payback else np.nan
            })
            
            progress_bar.progress((i + 1) / n_samples)
        
        # Restore originals
        st.session_state[key_1] = orig_1
        st.session_state[key_2] = orig_2
        st.session_state[key_3] = orig_3
        
        progress_bar.empty()
        status_text.empty()
        
        df_results = pd.DataFrame(results)
        
        st.success("✅ Three-variable analysis completed!")
        
        # Display results
        _display_three_var_results(df_results, name_1, name_2, name_3)
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

def _calculate_metrics():
    """Calculate NPV, IRR, Payback"""
    try:
        n_years = st.session_state.num_years
        fcf_values = [-st.session_state.capex_year0]
        
        previous_opex = 0
        sales = st.session_state.sales_year1
        cogs = sales * (st.session_state.cogs_pct / 100)
        labor_cost = st.session_state.labor_cost
        energy_cost = st.session_state.energy_cost
        
        for year in range(1, n_years + 1):
            if year > 1:
                sales *= (1 + st.session_state.sales_growth / 100)
                cogs *= (1 + st.session_state.cost_growth / 100)
            
            gross_profit = sales - cogs
            general_expenses = labor_cost + energy_cost + st.session_state.sga + st.session_state.overhead_cost
            ebit = gross_profit - general_expenses - st.session_state.depreciation
            
            tax_type = st.session_state.get('tax_type', 'percentage')
            tax = (ebit * st.session_state.tax_rate / 100) if (tax_type == 'percentage' and ebit > 0) else (st.session_state.tax_fixed_amount if tax_type == 'fixed' else 0)
            
            nopat = ebit - tax
            
            days = st.session_state.days_in_year
            cxc = -((sales / days) * st.session_state.ar_days)
            cxp = (cogs / days) * st.session_state.ap_days
            inventory = -((cogs / days) * st.session_state.inventory_days)
            
            opex = cxc + cxp + inventory
            var_opex = opex - previous_opex
            previous_opex = opex
            
            salvage = (st.session_state.salvage_value * (1 - st.session_state.tax_rate / 100)) if year == n_years else 0
            erosion = -st.session_state.erosion_cost
            opportunity = -st.session_state.opportunity_cost
            
            fcf = nopat + st.session_state.depreciation + var_opex + salvage + erosion + opportunity
            fcf_values.append(fcf)
        
        wacc = st.session_state.wacc / 100
        npv = sum(fcf / ((1 + wacc) ** i) for i, fcf in enumerate(fcf_values))
        
        try:
            irr = npf.irr(fcf_values)
            irr = irr if not np.isnan(irr) else None
        except:
            irr = None
        
        cumulative = 0
        payback = None
        initial = abs(fcf_values[0])
        
        for i in range(1, len(fcf_values)):
            cumulative += fcf_values[i]
            if cumulative >= initial:
                prev_cum = cumulative - fcf_values[i]
                remaining = initial - prev_cum
                payback = (i - 1) + (remaining / fcf_values[i]) if fcf_values[i] > 0 else None
                break
        
        return npv, irr, payback
    except:
        return 0, None, None

def _display_three_var_results(df, name_1, name_2, name_3):
    """Display 3D scatter plot and statistics"""
    st.subheader("📊 Results Visualization")
    
    # 3D Scatter Plot
    fig = go.Figure(data=[go.Scatter3d(
        x=df[f'{name_1} (%)'],
        y=df[f'{name_2} (%)'],
        z=df[f'{name_3} (%)'],
        mode='markers',
        marker=dict(
            size=5,
            color=df['NPV'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="NPV ($)")
        ),
        text=[f'NPV: ${npv:,.0f}' for npv in df['NPV']],
        hovertemplate=f'{name_1}: %{{x:.1f}}%<br>{name_2}: %{{y:.1f}}%<br>{name_3}: %{{z:.1f}}%<br>%{{text}}<extra></extra>'
    )])
    
    fig.update_layout(
        scene=dict(
            xaxis_title=f"{name_1} Variation (%)",
            yaxis_title=f"{name_2} Variation (%)",
            zaxis_title=f"{name_3} Variation (%)"
        ),
        height=700,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.markdown("---")
    st.subheader("📈 Statistical Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean NPV", f"${df['NPV'].mean():,.2f}")
        st.metric("Median NPV", f"${df['NPV'].median():,.2f}")
    
    with col2:
        st.metric("Max NPV", f"${df['NPV'].max():,.2f}")
        st.metric("Min NPV", f"${df['NPV'].min():,.2f}")
    
    with col3:
        positive_pct = (df['NPV'] > 0).sum() / len(df) * 100
        st.metric("Positive NPV", f"{positive_pct:.1f}%")
        st.metric("Std Dev", f"${df['NPV'].std():,.2f}")
    
    with col4:
        mean_irr = df['IRR (%)'].mean()
        wacc = st.session_state.wacc
        st.metric("Mean IRR", f"{mean_irr:.2f}%", f"{mean_irr - wacc:+.2f}% vs WACC")
    
    # Raw data
    with st.expander("📋 View Sample Data"):
        st.dataframe(df.head(100), use_container_width=True)