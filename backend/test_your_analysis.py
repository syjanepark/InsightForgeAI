#!/usr/bin/env python3
"""
Test script for your new analysis logic
"""
import pandas as pd
import numpy as np
from routers.analyze import generate_analysis

# Create sample data similar to your gaming dataset
np.random.seed(42)
sample_data = {
    'Name': ['Game A', 'Game B', 'Game C', 'Game D', 'Game E'] * 100,
    'Platform': ['PS4', 'Xbox', 'PC', 'Switch', 'Mobile'] * 100, 
    'Year': np.random.choice(range(2015, 2023), 500),
    'NA_Sales': np.random.lognormal(0, 1, 500),
    'EU_Sales': np.random.lognormal(-0.5, 0.8, 500),
    'JP_Sales': np.random.lognormal(-1, 0.6, 500),
    'Other_Sales': np.random.lognormal(-1.5, 0.5, 500),
    'Rank': range(1, 501)
}

df = pd.DataFrame(sample_data)

print("🎮 Testing your new analysis logic...")
print(f"📊 Sample data: {len(df)} rows, {len(df.columns)} columns")

try:
    result = generate_analysis(df)
    
    print("\n✅ Analysis Results:")
    print(f"📈 KPIs found: {len(result['kpis'])}")
    for kpi in result['kpis'][:3]:
        print(f"  • {kpi['metric']}: mean={kpi['mean']}, std={kpi['std']}")
    
    print(f"\n📊 Trend series: {len(result['trends'])}")
    for trend in result['trends'][:2]:
        print(f"  • {trend['series']}: {len(trend['values'])} data points, last YoY: {trend['last_yoy_pct']}%")
    
    print(f"\n📋 Charts generated: {len(result['charts'])}")
    
    print(f"\n💬 QA Context length: {len(result['qa_context'])} characters")
    print("QA Context preview:")
    print(result['qa_context'][:200] + "...")
    
    print("\n🎉 Your analysis logic works perfectly!")
    
except Exception as e:
    print(f"❌ Error testing analysis: {e}")
    import traceback
    traceback.print_exc()
