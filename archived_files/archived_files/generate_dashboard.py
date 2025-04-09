import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import os

# Create output directory
output_dir = 'dashboard'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_data():
    """Load the DEA results"""
    dea_results = pd.read_csv('dea_results.csv')
    # Drop duplicates to avoid showing the same hospital multiple times
    dea_results = dea_results.drop_duplicates(subset=['Hospital_Name'])
    return dea_results

def generate_dashboard_html():
    """Generate an HTML dashboard for DEA results"""
    data = load_data()
    
    # Create a new HTML file
    with open(f'{output_dir}/dea_dashboard.html', 'w') as f:
        # Write HTML header
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Hospital Efficiency Analysis Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/5.16.0/d3.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Open Sans', sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f8f9fa;
            color: #333;
        }
        .dashboard-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background-color: #3498db;
            color: white;
            padding: 20px;
            text-align: center;
            border-radius: 5px 5px 0 0;
            margin-bottom: 20px;
        }
        .dashboard-title {
            margin: 0;
            font-size: 28px;
            font-weight: 600;
        }
        .dashboard-subtitle {
            margin: 10px 0 0;
            font-weight: 300;
            font-size: 16px;
        }
        .card {
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 15px;
            margin-bottom: 20px;
        }
        .card-title {
            margin-top: 0;
            margin-bottom: 15px;
            font-size: 18px;
            font-weight: 600;
            color: #2c3e50;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .full-width {
            grid-column: 1 / -1;
        }
        .metric-container {
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
        }
        .metric-box {
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 15px;
            text-align: center;
            flex: 1;
            min-width: 150px;
            margin-right: 10px;
        }
        .metric-box:last-child {
            margin-right: 0;
        }
        .metric-value {
            font-size: 24px;
            font-weight: 600;
            color: #3498db;
            margin: 5px 0;
        }
        .metric-label {
            font-size: 14px;
            color: #7f8c8d;
        }
        .table-container {
            overflow-x: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #e1e1e1;
        }
        th {
            background-color: #f1f1f1;
            font-weight: 600;
        }
        tr:hover {
            background-color: #f9f9f9;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            color: #7f8c8d;
            font-size: 14px;
        }
        @media (max-width: 768px) {
            .grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="header">
            <h1 class="dashboard-title">Hospital Efficiency Analysis Dashboard</h1>
            <p class="dashboard-subtitle">Data Envelopment Analysis (DEA) Results</p>
        </div>
        ''')
        
        # Add key metrics section
        avg_te_vrs = data['TE_VRS'].mean()
        avg_te_crs = data['TE_CRS'].mean()
        avg_scale_eff = data['Scale_Efficiency'].mean()
        
        f.write('''
        <div class="card">
            <h2 class="card-title">Key Efficiency Metrics</h2>
            <div class="metric-container">
                <div class="metric-box">
                    <div class="metric-label">Avg Technical Efficiency (VRS)</div>
                    <div class="metric-value">{:.2f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Avg Technical Efficiency (CRS)</div>
                    <div class="metric-value">{:.2f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Avg Scale Efficiency</div>
                    <div class="metric-value">{:.2f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Total Hospitals</div>
                    <div class="metric-value">{}</div>
                </div>
            </div>
        </div>
        '''.format(avg_te_vrs, avg_te_crs, avg_scale_eff, len(data)))
        
        # Create grid layout for charts
        f.write('''
        <div class="grid">
        ''')
        
        # Add efficiency distribution chart
        fig1 = px.histogram(data, x='TE_VRS', nbins=20, 
                        title='Distribution of Technical Efficiency (VRS)',
                        labels={'TE_VRS': 'Technical Efficiency (VRS)'},
                        color_discrete_sequence=['#3498db'])
        fig1.update_layout(
            xaxis_title='Technical Efficiency (VRS)',
            yaxis_title='Number of Hospitals',
            plot_bgcolor='white',
            margin=dict(l=40, r=20, t=60, b=40),
        )
        
        f.write('''
        <div class="card">
            <h2 class="card-title">Efficiency Distribution</h2>
            <div id="efficiency-distribution"></div>
        </div>
        ''')
        
        # Add efficiency by size chart
        fig2 = px.box(data, x='Size_Category', y='TE_VRS', 
                    color='Size_Category',
                    title='Efficiency by Hospital Size',
                    labels={'TE_VRS': 'Technical Efficiency (VRS)', 'Size_Category': 'Hospital Size'},
                    color_discrete_sequence=px.colors.qualitative.Set2)
        fig2.update_layout(
            xaxis_title='Hospital Size',
            yaxis_title='Technical Efficiency (VRS)',
            plot_bgcolor='white',
            margin=dict(l=40, r=20, t=60, b=40),
        )
        
        f.write('''
        <div class="card">
            <h2 class="card-title">Efficiency by Hospital Size</h2>
            <div id="efficiency-by-size"></div>
        </div>
        ''')
        
        # Add scale efficiency categories chart
        scale_counts = data['Scale_Efficiency_Category'].value_counts()
        fig3 = px.pie(names=scale_counts.index, values=scale_counts.values, 
                    title='Scale Efficiency Categories',
                    color_discrete_sequence=px.colors.qualitative.Set3)
        fig3.update_layout(
            margin=dict(l=20, r=20, t=60, b=20),
        )
        
        f.write('''
        <div class="card">
            <h2 class="card-title">Scale Efficiency Categories</h2>
            <div id="scale-efficiency-categories"></div>
        </div>
        ''')
        
        # Add efficiency vs quality chart
        fig4 = px.scatter(data, x='Patient_Satisfaction', y='TE_VRS',
                        color='Size_Category', size='Bed_Occupancy_Rate',
                        hover_name='Hospital_Name',
                        title='Efficiency vs Patient Satisfaction',
                        labels={'TE_VRS': 'Technical Efficiency (VRS)', 
                                'Patient_Satisfaction': 'Patient Satisfaction',
                                'Size_Category': 'Hospital Size',
                                'Bed_Occupancy_Rate': 'Bed Occupancy Rate'},
                        color_discrete_sequence=px.colors.qualitative.Set1)
        fig4.update_layout(
            xaxis_title='Patient Satisfaction',
            yaxis_title='Technical Efficiency (VRS)',
            plot_bgcolor='white',
            margin=dict(l=40, r=20, t=60, b=40),
        )
        
        f.write('''
        <div class="card">
            <h2 class="card-title">Efficiency vs Patient Satisfaction</h2>
            <div id="efficiency-vs-satisfaction"></div>
        </div>
        ''')
        
        # Add correlation heatmap
        cols_to_include = ['TE_VRS', 'TE_CRS', 'Scale_Efficiency', 
                        'Patient_Satisfaction', 'Recommendation_Score', 
                        'Licensed_Beds', 'Bed_Occupancy_Rate']
        correlation_data = data[cols_to_include].corr().round(2)
        
        fig5 = px.imshow(correlation_data,
                        labels=dict(color="Correlation"),
                        x=correlation_data.columns,
                        y=correlation_data.columns,
                        color_continuous_scale='RdBu_r',
                        title='Correlation Matrix')
        fig5.update_layout(
            margin=dict(l=40, r=20, t=60, b=40),
        )
        
        f.write('''
        <div class="card full-width">
            <h2 class="card-title">Correlation Matrix</h2>
            <div id="correlation-matrix"></div>
        </div>
        ''')
        
        # Add top performers table
        top_hospitals = data.sort_values('TE_VRS', ascending=False).head(10)
        top_table = top_hospitals[['Hospital_Name', 'TE_VRS', 'Scale_Efficiency', 
                                'Size_Category', 'Occupancy_Category']].copy()
        top_table.columns = ['Hospital Name', 'Technical Efficiency (VRS)', 
                            'Scale Efficiency', 'Size Category', 'Occupancy Category']
        top_table_html = top_table.to_html(index=False, classes='display')
        
        f.write('''
        <div class="card full-width">
            <h2 class="card-title">Top 10 Most Efficient Hospitals</h2>
            <div class="table-container">
                {}
            </div>
        </div>
        '''.format(top_table_html))
        
        # Close grid
        f.write('''
        </div>
        ''')
        
        # Add footer
        f.write('''
        <div class="footer">
            <p>Hospital Efficiency Analysis Dashboard | Generated with Python and Plotly</p>
        </div>
        ''')
        
        # Add Plotly JavaScript
        f.write('''
        <script>
            // Efficiency Distribution Chart
            var efficiencyDistribution = {};
            Plotly.newPlot('efficiency-distribution', efficiencyDistribution.data, efficiencyDistribution.layout);
            
            // Efficiency by Size Chart
            var efficiencyBySize = {};
            Plotly.newPlot('efficiency-by-size', efficiencyBySize.data, efficiencyBySize.layout);
            
            // Scale Efficiency Categories Chart
            var scaleEfficiencyCategories = {};
            Plotly.newPlot('scale-efficiency-categories', scaleEfficiencyCategories.data, scaleEfficiencyCategories.layout);
            
            // Efficiency vs Satisfaction Chart
            var efficiencyVsSatisfaction = {};
            Plotly.newPlot('efficiency-vs-satisfaction', efficiencyVsSatisfaction.data, efficiencyVsSatisfaction.layout);
            
            // Correlation Matrix Chart
            var correlationMatrix = {};
            Plotly.newPlot('correlation-matrix', correlationMatrix.data, correlationMatrix.layout);
            
            // Make charts responsive
            window.onresize = function() {
                Plotly.relayout('efficiency-distribution', {
                    width: document.getElementById('efficiency-distribution').offsetWidth
                });
                Plotly.relayout('efficiency-by-size', {
                    width: document.getElementById('efficiency-by-size').offsetWidth
                });
                Plotly.relayout('scale-efficiency-categories', {
                    width: document.getElementById('scale-efficiency-categories').offsetWidth
                });
                Plotly.relayout('efficiency-vs-satisfaction', {
                    width: document.getElementById('efficiency-vs-satisfaction').offsetWidth
                });
                Plotly.relayout('correlation-matrix', {
                    width: document.getElementById('correlation-matrix').offsetWidth
                });
            };
        </script>
        '''.format(
            fig1.to_json(),
            fig2.to_json(),
            fig3.to_json(),
            fig4.to_json(),
            fig5.to_json()
        ))
        
        # Close HTML
        f.write('''
    </div>
</body>
</html>
        ''')
    
    print(f"Dashboard HTML file created at '{output_dir}/dea_dashboard.html'")

def main():
    print("Generating interactive dashboard...")
    generate_dashboard_html()
    print("Dashboard generation completed.")

if __name__ == "__main__":
    main() 