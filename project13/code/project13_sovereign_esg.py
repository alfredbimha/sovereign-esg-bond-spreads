"""
===============================================================================
PROJECT 13: Sovereign ESG Scores and Bond Spreads
===============================================================================
RESEARCH QUESTION:
    Do better governance/ESG indicators lower sovereign borrowing costs?
METHOD:
    Panel regression with Fixed Effects using World Bank WGI indicators
DATA:
    World Bank WGI (free via wbgapi), simulated bond spreads calibrated
    to real-world sovereign debt literature
===============================================================================
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wbgapi as wb
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import warnings, os

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
for d in ['output/figures','output/tables','data']:
    os.makedirs(d, exist_ok=True)

print("STEP 1: Downloading World Bank Governance Indicators (WGI)...")

countries = ['BRA','MEX','ZAF','IND','IDN','TUR','COL','PHL','THA',
             'MYS','CHL','PER','POL','HUN','CZE','ROU','EGY','NGA',
             'KEN','MAR','ARG','VNM','PAK','BGD','GHA','SEN','URY',
             'PAN','CRI','BWA']

# WGI indicators
indicators = {
    'CC.EST': 'Control of Corruption',
    'GE.EST': 'Government Effectiveness', 
    'RL.EST': 'Rule of Law',
    'RQ.EST': 'Regulatory Quality',
    'VA.EST': 'Voice and Accountability',
    'PS.EST': 'Political Stability'
}

wgi_data = []
for code, name in indicators.items():
    try:
        data = wb.data.DataFrame(code, countries, time=range(2012, 2023), labels=True)
        data = data.reset_index()
        data = data.melt(id_vars=['economy','Country'], var_name='year', value_name=name)
        data['year'] = data['year'].str.replace('YR','').astype(int)
        if not wgi_data:
            wgi_data.append(data[['economy','Country','year',name]])
        else:
            wgi_data.append(data[[name]])
    except Exception as e:
        print(f"  Error downloading {name}: {e}")

if wgi_data:
    wgi = pd.concat(wgi_data, axis=1)
    wgi = wgi.loc[:, ~wgi.columns.duplicated()]
else:
    # Fallback
    wgi = pd.DataFrame()

if not wgi.empty:
    print(f"  Downloaded WGI for {wgi['economy'].nunique()} countries, {wgi['year'].nunique()} years")
    
    # Composite governance score
    gov_cols = [v for v in indicators.values() if v in wgi.columns]
    wgi['governance_score'] = wgi[gov_cols].mean(axis=1)
    
    # Simulate bond spreads (calibrated to real EM spread levels)
    np.random.seed(42)
    wgi['spread_bps'] = (
        500  # base spread
        - wgi['governance_score'] * 150  # better governance → lower spreads
        + np.random.normal(0, 50, len(wgi))
        + (2023 - wgi['year']) * 5  # time effect
    ).clip(50, 1500).round(0)
    
    # Add macro controls
    wgi['gdp_growth'] = np.random.normal(3.5, 2, len(wgi)).round(2)
    wgi['inflation'] = np.random.normal(5, 3, len(wgi)).clip(0.5, 25).round(2)
    wgi['debt_gdp'] = np.random.normal(55, 20, len(wgi)).clip(15, 120).round(1)
    
    wgi.to_csv('data/sovereign_panel.csv', index=False)
    
    # =========================================================================
    # Regressions
    # =========================================================================
    print("\nSTEP 2: Running panel regressions...")
    clean = wgi.dropna(subset=['governance_score','spread_bps'])
    
    # Pooled OLS
    X = add_constant(clean[['governance_score','gdp_growth','inflation','debt_gdp']])
    pooled = OLS(clean['spread_bps'], X).fit()
    print(f"  Pooled OLS: Gov coeff={pooled.params['governance_score']:.2f} (p={pooled.pvalues['governance_score']:.4f})")
    
    # Country FE
    fm = clean.groupby('economy')[['spread_bps','governance_score','gdp_growth','inflation','debt_gdp']].transform('mean')
    dm = clean[['spread_bps','governance_score','gdp_growth','inflation','debt_gdp']] - fm
    X_fe = add_constant(dm[['governance_score','gdp_growth','inflation','debt_gdp']])
    fe = OLS(dm['spread_bps'], X_fe).fit()
    print(f"  Country FE:  Gov coeff={fe.params['governance_score']:.2f} (p={fe.pvalues['governance_score']:.4f})")
    
    pd.DataFrame({
        'Model':['Pooled OLS','Country FE'],
        'Gov_coeff':[pooled.params['governance_score'], fe.params['governance_score']],
        'Gov_pvalue':[pooled.pvalues['governance_score'], fe.pvalues['governance_score']],
        'R2':[pooled.rsquared, fe.rsquared]
    }).to_csv('output/tables/regression_results.csv', index=False)
    
    # =========================================================================
    # Visualizations
    # =========================================================================
    print("\nSTEP 3: Visualizations...")
    
    # Fig 1: Governance vs Spreads scatter
    fig, ax = plt.subplots(figsize=(12, 7))
    latest = clean[clean['year']==clean['year'].max()]
    ax.scatter(latest['governance_score'], latest['spread_bps'], s=60, alpha=0.7, c='steelblue')
    for _, row in latest.iterrows():
        ax.annotate(row['economy'], (row['governance_score'], row['spread_bps']), fontsize=7)
    z = np.polyfit(latest['governance_score'], latest['spread_bps'], 1)
    xl = np.linspace(latest['governance_score'].min(), latest['governance_score'].max(), 50)
    ax.plot(xl, np.poly1d(z)(xl), 'r--', lw=2)
    ax.set_title('Governance Score vs Sovereign Bond Spread', fontweight='bold')
    ax.set_xlabel('Governance Score (WGI Composite)')
    ax.set_ylabel('Bond Spread (bps)')
    plt.tight_layout()
    plt.savefig('output/figures/fig1_governance_vs_spreads.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Fig 2: Governance heatmap
    pivot = clean.pivot_table(values='governance_score', index='Country', columns='year')
    if pivot.shape[0] > 0:
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(pivot.iloc[:20], annot=True, fmt='.2f', cmap='RdYlGn', ax=ax, linewidths=0.3)
        ax.set_title('Governance Scores Over Time (WGI)', fontweight='bold')
        plt.tight_layout()
        plt.savefig('output/figures/fig2_governance_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Fig 3: WGI pillars comparison
    if len(gov_cols) >= 3:
        latest_pillars = latest.groupby('Country')[gov_cols[:6]].mean().sort_values(gov_cols[0])
        fig, ax = plt.subplots(figsize=(14, 8))
        latest_pillars.iloc[:15].plot(kind='barh', ax=ax, width=0.8)
        ax.set_title('WGI Governance Pillars by Country', fontweight='bold')
        ax.set_xlabel('Score (-2.5 to +2.5)')
        ax.legend(fontsize=7, loc='lower right')
        plt.tight_layout()
        plt.savefig('output/figures/fig3_governance_pillars.png', dpi=150, bbox_inches='tight')
        plt.close()

print("  COMPLETE!")
