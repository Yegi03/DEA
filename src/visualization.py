import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mtick
import os
import squarify
import matplotlib.patches as patches
from matplotlib.sankey import Sankey

class DEAVisualizer:
    def __init__(self):
        # Set the style for all visualizations
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("viridis")
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['savefig.pad_inches'] = 0.3
        plt.rcParams['axes.titlesize'] = 18
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['figure.titlesize'] = 22
        
        self.colors = {
            'Small': '#2ecc71',
            'Medium': '#3498db',
            'Large': '#e74c3c',
            'Low': '#2ecc71',
            'High': '#e74c3c',
            'IRS': '#2ecc71',
            'DRS': '#e74c3c',
            'CRS': '#3498db'
        }
        
        # Create results directory and subdirectories
        self.results_dir = 'visualization_results'
        self.individual_dir = os.path.join(self.results_dir, 'individual_plots')
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        if not os.path.exists(self.individual_dir):
            os.makedirs(self.individual_dir)
    
    def plot_vrs_distribution(self, data):
        """Plot VRS efficiency distribution separately"""
        plt.figure(figsize=(10, 6))
        sns.histplot(data=data, x='TE_VRS', bins=20, color='#3498db')
        plt.title('VRS Efficiency Distribution', fontweight='bold')
        plt.xlabel('Efficiency Score', fontweight='bold')
        plt.ylabel('Count', fontweight='bold')
        
        mean_vrs = data['TE_VRS'].mean()
        plt.axvline(mean_vrs, color='red', linestyle='--', alpha=0.8)
        plt.text(mean_vrs + 0.1, plt.ylim()[1]*0.9, f'Mean: {mean_vrs:.2f}', 
                rotation=0, color='red')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.individual_dir, 'vrs_distribution.png'))
        plt.close()

    def plot_crs_distribution(self, data):
        """Plot CRS efficiency distribution separately"""
        plt.figure(figsize=(10, 6))
        sns.histplot(data=data, x='TE_CRS', bins=20, color='#2ecc71')
        plt.title('CRS Efficiency Distribution', fontweight='bold')
        plt.xlabel('Efficiency Score', fontweight='bold')
        plt.ylabel('Count', fontweight='bold')
        
        mean_crs = data['TE_CRS'].mean()
        plt.axvline(mean_crs, color='red', linestyle='--', alpha=0.8)
        plt.text(mean_crs + 0.1, plt.ylim()[1]*0.9, f'Mean: {mean_crs:.2f}', 
                rotation=0, color='red')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.individual_dir, 'crs_distribution.png'))
        plt.close()

    def plot_efficiency_scatter(self, data):
        """Plot VRS vs CRS efficiency scatter separately"""
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(data['TE_CRS'], data['TE_VRS'], 
                            c=data['Scale_Efficiency'], cmap='viridis', 
                            alpha=0.6)
        plt.xlabel('CRS Efficiency Score', fontweight='bold')
        plt.ylabel('VRS Efficiency Score', fontweight='bold')
        plt.title('VRS vs CRS Efficiency', fontweight='bold')
        
        plt.colorbar(scatter, label='Scale Efficiency')
        
        # Add reference line
        max_val = max(data['TE_CRS'].max(), data['TE_VRS'].max())
        plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.individual_dir, 'efficiency_scatter.png'))
        plt.close()

    def plot_scale_efficiency_dist(self, data):
        """Plot scale efficiency distribution separately"""
        plt.figure(figsize=(10, 6))
        sns.histplot(data=data, x='Scale_Efficiency', bins=20, color='#3498db')
        plt.title('Scale Efficiency Distribution', fontweight='bold')
        plt.xlabel('Scale Efficiency Score', fontweight='bold')
        plt.ylabel('Count', fontweight='bold')
        
        mean_se = data['Scale_Efficiency'].mean()
        plt.axvline(mean_se, color='red', linestyle='--', alpha=0.8)
        plt.text(mean_se + 0.1, plt.ylim()[1]*0.9, f'Mean: {mean_se:.2f}', 
                rotation=0, color='red')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.individual_dir, 'scale_efficiency_dist.png'))
        plt.close()

    def plot_scale_categories(self, data):
        """Plot scale efficiency categories separately"""
        plt.figure(figsize=(10, 8))
        category_counts = data['Scale_Efficiency_Category'].value_counts()
        
        # Define colors with lighter tones
        colors = {
            'CRS': '#7bed9f',  # Lighter green
            'IRS': '#74b9ff',  # Lighter blue
            'DRS': '#ff7675'   # Lighter red
        }
        
        plt.pie(category_counts, labels=category_counts.index, 
               colors=[colors[cat] for cat in category_counts.index],
               autopct='%1.1f%%', explode=[0.05]*len(category_counts),
               alpha=0.7)  # Add transparency
        plt.title('Scale Efficiency Categories', fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.individual_dir, 'scale_categories.png'))
        plt.close()

    def plot_efficiency_by_size(self, data):
        """Plot efficiency by hospital size separately"""
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=data, x='Size_Category', y='TE_VRS', 
                   order=['Small', 'Medium', 'Large'],
                   palette=[self.colors[cat] for cat in ['Small', 'Medium', 'Large']])
        plt.title('Efficiency by Hospital Size', fontweight='bold')
        plt.xlabel('Size Category', fontweight='bold')
        plt.ylabel('VRS Efficiency Score', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.individual_dir, 'efficiency_by_size.png'))
        plt.close()

    def plot_efficiency_by_occupancy(self, data):
        """Plot efficiency by occupancy rate separately"""
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=data, x='Occupancy_Category', y='TE_VRS',
                   order=['Low', 'Medium', 'High'],
                   palette=[self.colors[cat] for cat in ['Low', 'Medium', 'High']])
        plt.title('Efficiency by Occupancy Rate', fontweight='bold')
        plt.xlabel('Occupancy Category', fontweight='bold')
        plt.ylabel('VRS Efficiency Score', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.individual_dir, 'efficiency_by_occupancy.png'))
        plt.close()

    def plot_efficiency_by_quality(self, data):
        """Plot efficiency by quality rating separately"""
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=data, x='Quality_Category', y='TE_VRS',
                   order=['Low', 'Medium', 'High'],
                   palette=[self.colors[cat] for cat in ['Low', 'Medium', 'High']])
        plt.title('Efficiency by Quality Rating', fontweight='bold')
        plt.xlabel('Quality Category', fontweight='bold')
        plt.ylabel('VRS Efficiency Score', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.individual_dir, 'efficiency_by_quality.png'))
        plt.close()

    def plot_scale_efficiency_by_size(self, data):
        """Plot scale efficiency by hospital size separately"""
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=data, x='Size_Category', y='Scale_Efficiency',
                      order=['Small', 'Medium', 'Large'],
                      palette=[self.colors[cat] for cat in ['Small', 'Medium', 'Large']])
        plt.title('Scale Efficiency by Hospital Size', fontweight='bold')
        plt.xlabel('Size Category', fontweight='bold')
        plt.ylabel('Scale Efficiency', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.individual_dir, 'scale_efficiency_by_size.png'))
        plt.close()

    def plot_efficiency_size_categories(self, data):
        """Plot efficiency vs size with scale categories separately"""
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=data, x='Licensed_Beds', y='TE_VRS', 
                       hue='Scale_Efficiency_Category', size='Bed_Occupancy_Rate',
                       palette=[self.colors[cat] for cat in ['CRS', 'DRS', 'IRS']])
        plt.title('Efficiency vs Size with Scale Categories', fontweight='bold')
        plt.xlabel('Licensed Beds', fontweight='bold')
        plt.ylabel('VRS Efficiency Score', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.individual_dir, 'efficiency_size_categories.png'))
        plt.close()

    def plot_efficiency_distribution(self, data):
        """Plot enhanced distribution of efficiency scores"""
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, figure=fig)
        
        # VRS Efficiency Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        sns.histplot(data=data, x='TE_VRS', bins=20, ax=ax1, color='#3498db')
        ax1.set_title('VRS Efficiency Distribution', fontweight='bold')
        ax1.set_xlabel('Efficiency Score', fontweight='bold')
        ax1.set_ylabel('Count', fontweight='bold')
        
        # Add mean line and annotation
        mean_vrs = data['TE_VRS'].mean()
        ax1.axvline(mean_vrs, color='red', linestyle='--', alpha=0.8)
        ax1.text(mean_vrs + 0.1, ax1.get_ylim()[1]*0.9, f'Mean: {mean_vrs:.2f}', 
                rotation=0, color='red')
        
        # CRS Efficiency Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        sns.histplot(data=data, x='TE_CRS', bins=20, ax=ax2, color='#2ecc71')
        ax2.set_title('CRS Efficiency Distribution', fontweight='bold')
        ax2.set_xlabel('Efficiency Score', fontweight='bold')
        ax2.set_ylabel('Count', fontweight='bold')
        
        # Add mean line and annotation
        mean_crs = data['TE_CRS'].mean()
        ax2.axvline(mean_crs, color='red', linestyle='--', alpha=0.8)
        ax2.text(mean_crs + 0.1, ax2.get_ylim()[1]*0.9, f'Mean: {mean_crs:.2f}', 
                rotation=0, color='red')
        
        # Efficiency Scatter Plot
        ax3 = fig.add_subplot(gs[1, :])
        scatter = ax3.scatter(data['TE_CRS'], data['TE_VRS'], 
                            c=data['Scale_Efficiency'], cmap='viridis', 
                            alpha=0.6)
        ax3.set_xlabel('CRS Efficiency Score', fontweight='bold')
        ax3.set_ylabel('VRS Efficiency Score', fontweight='bold')
        ax3.set_title('VRS vs CRS Efficiency', fontweight='bold')
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax3, label='Scale Efficiency')
        
        # Add reference line
        ax3.plot([0, max(data['TE_CRS'].max(), data['TE_VRS'].max())], 
                 [0, max(data['TE_CRS'].max(), data['TE_VRS'].max())], 
                 'r--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'efficiency_distribution.png'))
        plt.close()
    
    def plot_scale_efficiency(self, data):
        """Plot enhanced scale efficiency analysis"""
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, figure=fig)
        
        # Scale Efficiency Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        sns.histplot(data=data, x='Scale_Efficiency', bins=20, ax=ax1, color='#3498db')
        ax1.set_title('Scale Efficiency Distribution', fontweight='bold')
        ax1.set_xlabel('Scale Efficiency Score', fontweight='bold')
        ax1.set_ylabel('Count', fontweight='bold')
        
        # Add mean line
        mean_se = data['Scale_Efficiency'].mean()
        ax1.axvline(mean_se, color='red', linestyle='--', alpha=0.8)
        ax1.text(mean_se + 0.1, ax1.get_ylim()[1]*0.9, f'Mean: {mean_se:.2f}', 
                rotation=0, color='red')
        
        # Scale Efficiency Categories
        ax2 = fig.add_subplot(gs[0, 1])
        category_counts = data['Scale_Efficiency_Category'].value_counts()
        wedges, texts, autotexts = ax2.pie(category_counts, labels=category_counts.index, 
                                         colors=[self.colors[cat] for cat in category_counts.index],
                                         autopct='%1.1f%%', explode=[0.05]*len(category_counts))
        ax2.set_title('Scale Efficiency Categories', fontweight='bold')
        
        # Scale Efficiency vs Size
        ax3 = fig.add_subplot(gs[1, :])
        scatter = ax3.scatter(data['Licensed_Beds'], data['Scale_Efficiency'],
                            c=data['TE_VRS'], cmap='viridis', alpha=0.6)
        ax3.set_xlabel('Licensed Beds', fontweight='bold')
        ax3.set_ylabel('Scale Efficiency', fontweight='bold')
        ax3.set_title('Scale Efficiency vs Hospital Size', fontweight='bold')
        plt.colorbar(scatter, ax=ax3, label='VRS Efficiency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'scale_efficiency.png'))
        plt.close()
    
    def plot_category_analysis(self, data):
        """Plot enhanced efficiency by different categories"""
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 3, figure=fig)
        
        # Size Category Analysis
        ax1 = fig.add_subplot(gs[0, 0])
        sns.boxplot(data=data, x='Size_Category', y='TE_VRS', 
                   order=['Small', 'Medium', 'Large'],
                   palette=[self.colors[cat] for cat in ['Small', 'Medium', 'Large']],
                   ax=ax1)
        ax1.set_title('Efficiency by Hospital Size', fontweight='bold')
        ax1.set_xlabel('Size Category', fontweight='bold')
        ax1.set_ylabel('VRS Efficiency Score', fontweight='bold')
        
        # Occupancy Category Analysis
        ax2 = fig.add_subplot(gs[0, 1])
        sns.boxplot(data=data, x='Occupancy_Category', y='TE_VRS',
                   order=['Low', 'Medium', 'High'],
                   palette=[self.colors[cat] for cat in ['Low', 'Medium', 'High']],
                   ax=ax2)
        ax2.set_title('Efficiency by Occupancy Rate', fontweight='bold')
        ax2.set_xlabel('Occupancy Category', fontweight='bold')
        ax2.set_ylabel('VRS Efficiency Score', fontweight='bold')
        
        # Quality Category Analysis
        ax3 = fig.add_subplot(gs[0, 2])
        sns.boxplot(data=data, x='Quality_Category', y='TE_VRS',
                   order=['Low', 'Medium', 'High'],
                   palette=[self.colors[cat] for cat in ['Low', 'Medium', 'High']],
                   ax=ax3)
        ax3.set_title('Efficiency by Quality Rating', fontweight='bold')
        ax3.set_xlabel('Quality Category', fontweight='bold')
        ax3.set_ylabel('VRS Efficiency Score', fontweight='bold')
        
        # Violin plots for additional insight
        ax4 = fig.add_subplot(gs[1, 0])
        sns.violinplot(data=data, x='Size_Category', y='Scale_Efficiency',
                      order=['Small', 'Medium', 'Large'],
                      palette=[self.colors[cat] for cat in ['Small', 'Medium', 'Large']],
                      ax=ax4)
        ax4.set_title('Scale Efficiency by Hospital Size', fontweight='bold')
        ax4.set_xlabel('Size Category', fontweight='bold')
        ax4.set_ylabel('Scale Efficiency', fontweight='bold')
        
        ax5 = fig.add_subplot(gs[1, 1:])
        sns.scatterplot(data=data, x='Licensed_Beds', y='TE_VRS', 
                       hue='Scale_Efficiency_Category', size='Bed_Occupancy_Rate',
                       palette=[self.colors[cat] for cat in ['CRS', 'DRS', 'IRS']],
                       ax=ax5)
        ax5.set_title('Efficiency vs Size with Scale Categories', fontweight='bold')
        ax5.set_xlabel('Licensed Beds', fontweight='bold')
        ax5.set_ylabel('VRS Efficiency Score', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'category_analysis.png'))
        plt.close()
    
    def plot_correlation_heatmap(self, data):
        """Plot enhanced correlation heatmap"""
        # Select relevant columns
        columns_to_include = [
            'TE_VRS', 'TE_CRS', 'Scale_Efficiency',
            'Licensed_Beds', 'Staffed_Beds', 'Total_Surveys', 'Bed_Occupancy_Rate',
            'Patient_Satisfaction', 'Overall_Rating', 'Recommendation_Score',
            'Staff_Responsiveness', 'Nurse_Communication', 'Doctor_Communication',
            'Cleanliness', 'Quietness'
        ]
        
        # Filter columns that actually exist in the data
        available_cols = [col for col in columns_to_include if col in data.columns]
        corr_matrix = data[available_cols].corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        plt.figure(figsize=(15, 12))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, fmt='.2f', square=True, linewidths=0.5,
                   cbar_kws={"shrink": 0.8})
        
        plt.title('Correlation Matrix of Key Performance Indicators', fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Add annotation about interpretation
        plt.figtext(0.5, 0.02, 
                   "Correlation values range from -1 (strong negative) to +1 (strong positive)", 
                   ha="center", fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'correlation_heatmap.png'))
        plt.close()
    
    def create_efficiency_scorecard(self, data):
        """Create a summary scorecard visualization"""
        plt.figure(figsize=(16, 10))
        
        # Calculate key statistics
        avg_te_vrs = data['TE_VRS'].mean()
        avg_te_crs = data['TE_CRS'].mean()
        avg_scale_eff = data['Scale_Efficiency'].mean()
        
        # Count by scale category
        scale_counts = data['Scale_Efficiency_Category'].value_counts()
        
        # Create a figure with grid layout
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 4, figure=fig)
        
        # Create the main title
        fig.suptitle('Hospital Efficiency Analysis Scorecard', fontsize=24, fontweight='bold')
        
        # Add summary metrics in first row
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        ax1.text(0.5, 0.6, f"{avg_te_vrs:.2f}", fontsize=36, ha='center', fontweight='bold', color='#74b9ff')
        ax1.text(0.5, 0.4, "Avg Technical Efficiency (VRS)", fontsize=14, ha='center')
        ax1.set_facecolor('#f8f9fa')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')
        ax2.text(0.5, 0.6, f"{avg_te_crs:.2f}", fontsize=36, ha='center', fontweight='bold', color='#7bed9f')
        ax2.text(0.5, 0.4, "Avg Technical Efficiency (CRS)", fontsize=14, ha='center')
        ax2.set_facecolor('#f8f9fa')
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        ax3.text(0.5, 0.6, f"{avg_scale_eff:.2f}", fontsize=36, ha='center', fontweight='bold', color='#ff7675')
        ax3.text(0.5, 0.4, "Avg Scale Efficiency", fontsize=14, ha='center')
        ax3.set_facecolor('#f8f9fa')
        
        # Create pie chart of scale efficiency categories with consistent colors
        ax4 = fig.add_subplot(gs[0, 3])
        colors = {
            'CRS': '#7bed9f',  # Lighter green
            'IRS': '#74b9ff',  # Lighter blue
            'DRS': '#ff7675'   # Lighter red
        }
        wedges, texts, autotexts = ax4.pie(scale_counts, labels=scale_counts.index, autopct='%1.1f%%',
                                      colors=[colors[cat] for cat in scale_counts.index], 
                                      startangle=90,
                                      alpha=0.7)  # Add transparency
        ax4.set_title('Scale Efficiency Categories', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'efficiency_scorecard.png'))
        plt.close()
    
    def create_poster_summary(self, data):
        """Create a visually appealing summary for poster presentation"""
        plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 3, figure=plt.gcf())
        
        # Main title
        plt.suptitle('Hospital Efficiency Analysis Overview', 
                    fontsize=28, fontweight='bold', y=0.95)
        
        # Key Metrics with Circular Progress Bars
        ax1 = plt.subplot(gs[0, 0])
        self._create_circular_progress(ax1, data['TE_VRS'].mean(), 
                                     'VRS\nEfficiency', '#3498db')
        
        ax2 = plt.subplot(gs[0, 1])
        self._create_circular_progress(ax2, data['TE_CRS'].mean(), 
                                     'CRS\nEfficiency', '#2ecc71')
        
        ax3 = plt.subplot(gs[0, 2])
        self._create_circular_progress(ax3, data['Scale_Efficiency'].mean(), 
                                     'Scale\nEfficiency', '#e74c3c')
        
        # Distribution Plot
        ax4 = plt.subplot(gs[1, :2])
        sns.kdeplot(data=data, x='TE_VRS', ax=ax4, color='#3498db', 
                   label='VRS', fill=True, alpha=0.3)
        sns.kdeplot(data=data, x='TE_CRS', ax=ax4, color='#2ecc71', 
                   label='CRS', fill=True, alpha=0.3)
        ax4.set_title('Efficiency Score Distribution', fontsize=16, fontweight='bold')
        ax4.set_xlabel('Efficiency Score', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=12)
        
        # Scale Categories Donut Chart
        ax5 = plt.subplot(gs[1, 2])
        self._create_donut_chart(ax5, data)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'poster_summary.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def create_efficiency_radar_chart(self, data):
        """Create a radar chart comparing different efficiency metrics"""
        # Calculate mean values for each metric
        metrics = ['Patient_Satisfaction', 'Staff_Responsiveness', 
                  'Nurse_Communication', 'Doctor_Communication', 
                  'Cleanliness', 'Quietness']
        
        # Prepare data for different efficiency categories
        high_eff = data[data['TE_VRS'] > data['TE_VRS'].median()]
        low_eff = data[data['TE_VRS'] <= data['TE_VRS'].median()]
        
        high_values = [high_eff[m].mean() for m in metrics]
        low_values = [low_eff[m].mean() for m in metrics]
        
        # Create the radar chart
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        
        # Close the plot by appending first value
        high_values += [high_values[0]]
        low_values += [low_values[0]]
        angles = np.concatenate((angles, [angles[0]]))
        
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, high_values, 'o-', linewidth=2, label='High Efficiency', 
                color='#2ecc71')
        ax.fill(angles, high_values, alpha=0.25, color='#2ecc71')
        ax.plot(angles, low_values, 'o-', linewidth=2, label='Low Efficiency', 
                color='#e74c3c')
        ax.fill(angles, low_values, alpha=0.25, color='#e74c3c')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, size=10, fontweight='bold')
        ax.set_title('Efficiency Performance Radar', fontsize=20, fontweight='bold', 
                    pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'efficiency_radar.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def create_efficiency_treemap(self, data):
        """Create a treemap visualization of hospital efficiency categories"""
        plt.figure(figsize=(15, 10))
        
        # Calculate sizes and colors based on efficiency scores
        size_categories = data.groupby('Size_Category')['TE_VRS'].agg(['mean', 'count'])
        sizes = size_categories['count'] * size_categories['mean']
        
        # Create treemap
        colors = [self.colors[cat] for cat in size_categories.index]
        squarify.plot(sizes=sizes, label=size_categories.index, 
                     color=colors, alpha=0.6, text_kwargs={'fontsize':12, 
                     'fontweight':'bold'})
        
        plt.title('Hospital Efficiency by Size Category', 
                 fontsize=20, fontweight='bold', pad=20)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'efficiency_treemap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def create_efficiency_flow(self, data):
        """Create a flow diagram showing relationships between categories"""
        plt.figure(figsize=(15, 10))
        
        # Create subplots for different views
        gs = GridSpec(2, 2, figure=plt.gcf())
        
        # Size to Scale Efficiency
        ax1 = plt.subplot(gs[0, :])
        size_scale = pd.crosstab(data['Size_Category'], data['Scale_Efficiency_Category'])
        size_scale.plot(kind='bar', stacked=True, ax=ax1,
                       color=[self.colors['CRS'], self.colors['DRS'], self.colors['IRS']])
        ax1.set_title('Scale Efficiency Distribution by Hospital Size', 
                     fontsize=16, fontweight='bold')
        ax1.set_xlabel('Hospital Size', fontweight='bold')
        ax1.set_ylabel('Number of Hospitals', fontweight='bold')
        ax1.legend(title='Scale Efficiency', bbox_to_anchor=(1.05, 1))
        
        # Efficiency by Size
        ax2 = plt.subplot(gs[1, 0])
        sns.boxplot(data=data, x='Size_Category', y='TE_VRS',
                   order=['Small', 'Medium', 'Large'],
                   palette=[self.colors[cat] for cat in ['Small', 'Medium', 'Large']],
                   ax=ax2)
        ax2.set_title('VRS Efficiency by Size', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Hospital Size', fontweight='bold')
        ax2.set_ylabel('VRS Efficiency', fontweight='bold')
        
        # Efficiency by Scale Category
        ax3 = plt.subplot(gs[1, 1])
        sns.boxplot(data=data, x='Scale_Efficiency_Category', y='TE_VRS',
                   order=['CRS', 'DRS', 'IRS'],
                   palette=[self.colors[cat] for cat in ['CRS', 'DRS', 'IRS']],
                   ax=ax3)
        ax3.set_title('VRS Efficiency by Scale Category', 
                     fontsize=14, fontweight='bold')
        ax3.set_xlabel('Scale Efficiency Category', fontweight='bold')
        ax3.set_ylabel('VRS Efficiency', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'efficiency_relationships.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def create_advanced_radar(self, data):
        """Create an advanced radar chart with multiple efficiency metrics"""
        # Define metrics for comparison
        metrics = ['Patient_Satisfaction', 'Staff_Responsiveness', 
                  'Nurse_Communication', 'Doctor_Communication', 
                  'Cleanliness', 'Quietness']
        
        # Create categories based on both VRS and Scale efficiency
        data['Efficiency_Category'] = 'Medium'
        data.loc[data['TE_VRS'] > data['TE_VRS'].quantile(0.67), 'Efficiency_Category'] = 'High'
        data.loc[data['TE_VRS'] < data['TE_VRS'].quantile(0.33), 'Efficiency_Category'] = 'Low'
        
        # Calculate mean values for each category
        categories = ['High', 'Medium', 'Low']
        values = {cat: [data[data['Efficiency_Category'] == cat][m].mean() 
                       for m in metrics] for cat in categories}
        
        # Number of variables
        num_vars = len(metrics)
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]
        
        # Initialize the spider plot
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))
        
        # Plot data
        for cat, color in zip(categories, ['#2ecc71', '#3498db', '#e74c3c']):
            values[cat] += values[cat][:1]
            ax.plot(angles, values[cat], 'o-', linewidth=2, label=cat, color=color)
            ax.fill(angles, values[cat], alpha=0.25, color=color)
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        
        # Add legend
        ax.legend(title='Efficiency Level', loc='upper right', 
                 bbox_to_anchor=(0.1, 0.1))
        
        plt.title('Hospital Performance Metrics by Efficiency Level', 
                 fontsize=20, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'advanced_radar.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def create_efficiency_matrix(self, data):
        """Create a matrix visualization of efficiency relationships"""
        plt.figure(figsize=(15, 10))
        
        # Create a grid of subplots
        gs = GridSpec(2, 3, figure=plt.gcf())
        
        # 1. Efficiency Score Distribution
        ax1 = plt.subplot(gs[0, 0])
        sns.kdeplot(data=data, x='TE_VRS', ax=ax1, fill=True, color='#3498db')
        ax1.set_title('VRS Efficiency Distribution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Efficiency Score')
        
        # 2. Scale Efficiency Distribution
        ax2 = plt.subplot(gs[0, 1])
        sns.kdeplot(data=data, x='Scale_Efficiency', ax=ax2, fill=True, color='#2ecc71')
        ax2.set_title('Scale Efficiency Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Scale Efficiency')
        
        # 3. Occupancy Rate Distribution
        ax3 = plt.subplot(gs[0, 2])
        sns.kdeplot(data=data, x='Bed_Occupancy_Rate', ax=ax3, fill=True, color='#e74c3c')
        ax3.set_title('Bed Occupancy Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Occupancy Rate')
        
        # 4. Efficiency vs Size Scatter
        ax4 = plt.subplot(gs[1, 0])
        sns.scatterplot(data=data, x='Licensed_Beds', y='TE_VRS', 
                       hue='Scale_Efficiency_Category', ax=ax4,
                       palette=[self.colors[cat] for cat in ['CRS', 'DRS', 'IRS']])
        ax4.set_title('Efficiency vs Size', fontsize=12, fontweight='bold')
        ax4.legend(title='Scale Category', bbox_to_anchor=(1.05, 1))
        
        # 5. Efficiency vs Occupancy
        ax5 = plt.subplot(gs[1, 1:])
        scatter = ax5.scatter(data['Bed_Occupancy_Rate'], data['TE_VRS'],
                            c=data['Scale_Efficiency'], cmap='viridis', 
                            s=100*data['Licensed_Beds'], alpha=0.6)
        ax5.set_title('Efficiency vs Occupancy', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Bed Occupancy Rate')
        ax5.set_ylabel('VRS Efficiency')
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax5, label='Scale Efficiency')
        
        plt.suptitle('Hospital Efficiency Analysis Matrix', 
                    fontsize=20, fontweight='bold', y=1.05)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'efficiency_matrix.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _create_circular_progress(self, ax, value, label, color):
        """Helper method to create circular progress indicators"""
        ax.set_aspect('equal')
        
        # Create the circular progress
        wedge = patches.Wedge(center=(0.5, 0.5), r=0.3,
                             theta1=0, theta2=value*360,
                             facecolor=color, alpha=0.6)
        ax.add_patch(wedge)
        
        # Add background circle
        bg = patches.Circle((0.5, 0.5), 0.3, fill=False, 
                          color='gray', alpha=0.3)
        ax.add_patch(bg)
        
        # Add text
        ax.text(0.5, 0.5, f'{value:.2f}', ha='center', va='center',
                fontsize=20, fontweight='bold')
        ax.text(0.5, 0.2, label, ha='center', va='center',
                fontsize=14, fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    def _create_donut_chart(self, ax, data):
        """Helper method to create a donut chart"""
        category_counts = data['Scale_Efficiency_Category'].value_counts()
        colors = [self.colors[cat] for cat in category_counts.index]
        
        # Create donut chart
        wedges, texts, autotexts = ax.pie(category_counts, labels=category_counts.index,
                                         colors=colors, autopct='%1.1f%%',
                                         pctdistance=0.75,
                                         wedgeprops=dict(width=0.5))
        
        # Customize text properties
        plt.setp(autotexts, size=10, weight="bold")
        plt.setp(texts, size=12, weight="bold")
        
        ax.set_title('Scale Efficiency Categories', fontsize=16, fontweight='bold')

    def create_bcc_ccr_comparison(self, data):
        """Create visualization comparing BCC (VRS) and CCR (CRS) efficiency scores"""
        plt.figure(figsize=(15, 10))
        
        # Create subplots
        gs = GridSpec(2, 2, figure=plt.gcf(), height_ratios=[1, 1.2])
        
        # 1. Scatter plot of BCC vs CCR
        ax1 = plt.subplot(gs[0, :])
        scatter = ax1.scatter(data['TE_CRS'], data['TE_VRS'], 
                            c=data['Scale_Efficiency'],
                            s=100*data['Licensed_Beds']/data['Licensed_Beds'].max(),
                            cmap='viridis', alpha=0.7)
        
        # Add diagonal line
        lims = [
            np.min([ax1.get_xlim(), ax1.get_ylim()]),
            np.max([ax1.get_xlim(), ax1.get_ylim()]),
        ]
        ax1.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
        
        # Labels and title
        ax1.set_xlabel('CCR (CRS) Efficiency Score', fontweight='bold')
        ax1.set_ylabel('BCC (VRS) Efficiency Score', fontweight='bold')
        ax1.set_title('BCC vs CCR Efficiency Scores', fontsize=14, fontweight='bold')
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax1, label='Scale Efficiency')
        
        # 2. Distribution comparison
        ax2 = plt.subplot(gs[1, 0])
        sns.kdeplot(data=data, x='TE_VRS', label='BCC (VRS)', ax=ax2)
        sns.kdeplot(data=data, x='TE_CRS', label='CCR (CRS)', ax=ax2)
        ax2.set_xlabel('Efficiency Score', fontweight='bold')
        ax2.set_ylabel('Density', fontweight='bold')
        ax2.set_title('Efficiency Score Distributions', fontsize=12, fontweight='bold')
        ax2.legend()
        
        # 3. Scale efficiency by returns to scale
        ax3 = plt.subplot(gs[1, 1])
        sns.boxplot(data=data, x='Scale_Efficiency_Category', y='Scale_Efficiency',
                   order=['CRS', 'IRS', 'DRS'],
                   palette=[self.colors[cat] for cat in ['CRS', 'IRS', 'DRS']],
                   ax=ax3)
        ax3.set_xlabel('Returns to Scale Category', fontweight='bold')
        ax3.set_ylabel('Scale Efficiency', fontweight='bold')
        ax3.set_title('Scale Efficiency by Returns to Scale', fontsize=12, fontweight='bold')
        
        plt.suptitle('BCC and CCR Efficiency Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(self.results_dir, 'bcc_ccr_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def generate_efficiency_table(self, data):
        """Generate a comprehensive efficiency analysis table"""
        # Calculate summary statistics by category
        stats = []
        
        # Overall statistics
        overall_stats = {
            'Category': 'Overall',
            'Count': len(data),
            'Avg_VRS': data['TE_VRS'].mean(),
            'Avg_CRS': data['TE_CRS'].mean(),
            'Avg_Scale': data['Scale_Efficiency'].mean(),
            'Most_Efficient_VRS': data['TE_VRS'].max(),
            'Least_Efficient_VRS': data['TE_VRS'].min(),
            'CRS_Count': len(data[data['Scale_Efficiency_Category'] == 'CRS']),
            'IRS_Count': len(data[data['Scale_Efficiency_Category'] == 'IRS']),
            'DRS_Count': len(data[data['Scale_Efficiency_Category'] == 'DRS'])
        }
        stats.append(overall_stats)
        
        # Statistics by size category
        for size in ['Small', 'Medium', 'Large']:
            size_data = data[data['Size_Category'] == size]
            size_stats = {
                'Category': f'Size: {size}',
                'Count': len(size_data),
                'Avg_VRS': size_data['TE_VRS'].mean(),
                'Avg_CRS': size_data['TE_CRS'].mean(),
                'Avg_Scale': size_data['Scale_Efficiency'].mean(),
                'Most_Efficient_VRS': size_data['TE_VRS'].max(),
                'Least_Efficient_VRS': size_data['TE_VRS'].min(),
                'CRS_Count': len(size_data[size_data['Scale_Efficiency_Category'] == 'CRS']),
                'IRS_Count': len(size_data[size_data['Scale_Efficiency_Category'] == 'IRS']),
                'DRS_Count': len(size_data[size_data['Scale_Efficiency_Category'] == 'DRS'])
            }
            stats.append(size_stats)
        
        # Convert to DataFrame
        df_stats = pd.DataFrame(stats)
        
        # Round numerical columns
        numeric_cols = ['Avg_VRS', 'Avg_CRS', 'Avg_Scale', 'Most_Efficient_VRS', 'Least_Efficient_VRS']
        df_stats[numeric_cols] = df_stats[numeric_cols].round(3)
        
        # Save to CSV
        df_stats.to_csv(os.path.join(self.results_dir, 'efficiency_summary.csv'), index=False)
        return df_stats

    def generate_poster_visualizations(self, data):
        """Generate all poster-specific visualizations"""
        # Create a copy of the data to avoid modifying the original
        plot_data = data.copy()
        
        # Generate poster visualizations
        self.create_poster_summary(plot_data)
        self.create_efficiency_radar_chart(plot_data)
        self.create_efficiency_treemap(plot_data)
        self.create_efficiency_flow(plot_data)
        self.create_advanced_radar(plot_data)
        self.create_efficiency_matrix(plot_data)
        self.create_bcc_ccr_comparison(plot_data)
        
        print(f"All poster visualizations have been generated and saved to the '{self.results_dir}' directory.")

    def create_efficiency_waterfall(self, data):
        """Create a waterfall chart showing efficiency decomposition"""
        plt.figure(figsize=(12, 6))
        
        # Calculate average efficiencies
        avg_vrs = data['TE_VRS'].mean()
        avg_crs = data['TE_CRS'].mean()
        avg_scale = data['Scale_Efficiency'].mean()
        
        # Create waterfall chart
        names = ['VRS\nEfficiency', 'Scale\nEffect', 'CRS\nEfficiency']
        values = [avg_vrs, avg_crs - avg_vrs, avg_crs]
        
        # Plot bars
        plt.bar(0, values[0], bottom=0, color='#2ecc71', width=0.5)
        plt.bar(1, values[1], bottom=values[0], color='#3498db', width=0.5)
        plt.bar(2, values[2], color='#e74c3c', width=0.5)
        
        # Add connecting lines
        plt.plot([0.25, 0.75], [values[0], values[0]], 'k--', alpha=0.3)
        plt.plot([1.25, 1.75], [values[2], values[2]], 'k--', alpha=0.3)
        
        # Customize plot
        plt.xticks(range(3), names, fontweight='bold')
        plt.ylabel('Efficiency Score', fontweight='bold')
        plt.title('Efficiency Decomposition Analysis', fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, v in enumerate(values):
            if i != 1:  # Skip the middle bar
                plt.text(i, v, f'{v:.3f}', ha='center', va='bottom')
        plt.text(1, values[0] + values[1]/2, f'{values[1]:.3f}', ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'efficiency_waterfall.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def create_efficiency_quadrants(self, data):
        """Create quadrant analysis of efficiency scores"""
        plt.figure(figsize=(12, 8))
        
        # Calculate medians for quadrant lines
        vrs_median = data['TE_VRS'].median()
        scale_median = data['Scale_Efficiency'].median()
        
        # Create scatter plot
        plt.scatter(data['TE_VRS'], data['Scale_Efficiency'],
                   c=data['Bed_Occupancy_Rate'], cmap='viridis',
                   s=100*data['Licensed_Beds']/data['Licensed_Beds'].max(),
                   alpha=0.6)
        
        # Add quadrant lines
        plt.axhline(y=scale_median, color='gray', linestyle='--', alpha=0.3)
        plt.axvline(x=vrs_median, color='gray', linestyle='--', alpha=0.3)
        
        # Add quadrant labels
        plt.text(data['TE_VRS'].max(), data['Scale_Efficiency'].max(),
                'High Efficiency\nOptimal Scale', ha='right', va='top')
        plt.text(data['TE_VRS'].min(), data['Scale_Efficiency'].max(),
                'Low Efficiency\nOptimal Scale', ha='left', va='top')
        plt.text(data['TE_VRS'].max(), data['Scale_Efficiency'].min(),
                'High Efficiency\nSuboptimal Scale', ha='right', va='bottom')
        plt.text(data['TE_VRS'].min(), data['Scale_Efficiency'].min(),
                'Low Efficiency\nSuboptimal Scale', ha='left', va='bottom')
        
        # Customize plot
        plt.xlabel('VRS Technical Efficiency', fontweight='bold')
        plt.ylabel('Scale Efficiency', fontweight='bold')
        plt.title('Hospital Efficiency Quadrant Analysis', fontsize=14, fontweight='bold')
        
        # Add colorbar
        plt.colorbar(label='Bed Occupancy Rate')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'efficiency_quadrants.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def create_efficiency_network(self, data):
        """Create a network visualization of efficiency relationships"""
        plt.figure(figsize=(15, 10))
        
        # Create subplots
        gs = GridSpec(2, 3, figure=plt.gcf())
        
        # 1. Efficiency Network
        ax1 = plt.subplot(gs[:, :2])
        categories = ['Small', 'Medium', 'Large']
        efficiency_levels = ['High', 'Medium', 'Low']
        
        # Create node positions
        pos = {}
        for i, cat in enumerate(categories):
            for j, level in enumerate(efficiency_levels):
                pos[f'{cat}_{level}'] = (i, j)
        
        # Calculate node sizes based on hospital counts
        sizes = {}
        for cat in categories:
            size_data = data[data['Size_Category'] == cat]
            for level in efficiency_levels:
                count = len(size_data[size_data['Efficiency_Category'] == level])
                sizes[f'{cat}_{level}'] = count
        
        # Plot nodes
        for node, position in pos.items():
            size = sizes.get(node, 0)
            if size > 0:
                ax1.scatter(position[0], position[1], s=size*100,
                          alpha=0.6, label=f'{node}: {size}')
        
        # Customize network plot
        ax1.set_xticks(range(len(categories)))
        ax1.set_yticks(range(len(efficiency_levels)))
        ax1.set_xticklabels(categories, fontweight='bold')
        ax1.set_yticklabels(efficiency_levels, fontweight='bold')
        ax1.set_title('Hospital Efficiency Network', fontsize=14, fontweight='bold')
        
        # 2. Efficiency Flow
        ax2 = plt.subplot(gs[:, 2])
        efficiency_flow = pd.crosstab(data['Size_Category'],
                                    data['Efficiency_Category'])
        efficiency_flow.plot(kind='bar', stacked=True, ax=ax2,
                           color=['#2ecc71', '#3498db', '#e74c3c'])
        ax2.set_title('Efficiency Distribution\nby Hospital Size',
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('Hospital Size Category', fontweight='bold')
        ax2.set_ylabel('Number of Hospitals', fontweight='bold')
        ax2.legend(title='Efficiency Level')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'efficiency_network.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def create_combined_efficiency_analysis(self, data):
        """Create a comprehensive combined efficiency analysis visualization"""
        plt.figure(figsize=(15, 10))
        
        # Create subplot grid
        gs = GridSpec(2, 2, figure=plt.gcf(), height_ratios=[1, 1])
        
        # 1. Bar plot comparing VRS and CRS efficiencies
        ax1 = plt.subplot(gs[0, 0])
        efficiency_means = [
            data['TE_VRS'].mean(),
            data['TE_CRS'].mean(),
            data['Scale_Efficiency'].mean()
        ]
        efficiency_std = [
            data['TE_VRS'].std(),
            data['TE_CRS'].std(),
            data['Scale_Efficiency'].std()
        ]
        bars = ax1.bar(['VRS\nEfficiency', 'CRS\nEfficiency', 'Scale\nEfficiency'],
                      efficiency_means,
                      yerr=efficiency_std,
                      capsize=5,
                      color=['#2ecc71', '#3498db', '#e74c3c'])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        ax1.set_title('Average Efficiency Scores', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Efficiency Score', fontweight='bold')
        
        # 2. Returns to Scale Distribution
        ax2 = plt.subplot(gs[0, 1])
        rts_counts = data['Scale_Efficiency_Category'].value_counts()
        colors = [self.colors[cat] for cat in ['CRS', 'IRS', 'DRS']]
        wedges, texts, autotexts = ax2.pie(rts_counts,
                                          labels=[f'{cat}\n({count} hospitals)' 
                                                 for cat, count in rts_counts.items()],
                                          colors=colors,
                                          autopct='%1.1f%%',
                                          explode=[0.05]*len(rts_counts))
        ax2.set_title('Returns to Scale Distribution', fontsize=12, fontweight='bold')
        
        # 3. Efficiency Score Distribution
        ax3 = plt.subplot(gs[1, :])
        for col, label, color in zip(['TE_VRS', 'TE_CRS', 'Scale_Efficiency'],
                                   ['VRS Efficiency', 'CRS Efficiency', 'Scale Efficiency'],
                                   ['#2ecc71', '#3498db', '#e74c3c']):
            sns.kdeplot(data=data, x=col, label=label, ax=ax3, color=color)
            # Add vertical lines for mean and median
            mean_val = data[col].mean()
            median_val = data[col].median()
            ax3.axvline(mean_val, color=color, linestyle='--', alpha=0.5)
            ax3.axvline(median_val, color=color, linestyle=':', alpha=0.5)
            
            # Add text annotations for mean and median
            ax3.text(mean_val, ax3.get_ylim()[1],
                    f'Mean\n{mean_val:.3f}',
                    ha='center', va='bottom', color=color)
            ax3.text(median_val, ax3.get_ylim()[1]*0.8,
                    f'Median\n{median_val:.3f}',
                    ha='center', va='bottom', color=color)
        
        ax3.set_title('Efficiency Score Distributions', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Efficiency Score', fontweight='bold')
        ax3.set_ylabel('Density', fontweight='bold')
        ax3.legend()
        
        # Add overall title
        plt.suptitle('Comprehensive Efficiency Analysis', fontsize=16, fontweight='bold', y=1.02)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(self.results_dir, 'combined_efficiency_analysis.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def create_simple_efficiency_comparison(self, data):
        """Create a simple and clear efficiency comparison visualization"""
        plt.figure(figsize=(15, 8))
        
        # Create two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Left plot: Simple bar comparison
        efficiency_types = ['VRS (BCC)', 'CRS (CCR)', 'Scale']
        efficiency_values = [
            data['TE_VRS'].mean(),
            data['TE_CRS'].mean(),
            data['Scale_Efficiency'].mean()
        ]
        
        bars = ax1.bar(efficiency_types, efficiency_values,
                      color=['#2ecc71', '#3498db', '#e74c3c'])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')
        
        ax1.set_title('Average Efficiency Scores', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Efficiency Score', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Returns to Scale composition
        rts_counts = data['Scale_Efficiency_Category'].value_counts()
        colors = ['#3498db', '#2ecc71', '#e74c3c']  # CRS, IRS, DRS
        
        wedges, texts, autotexts = ax2.pie(rts_counts,
                                          labels=[f'{cat}\n{count} hospitals' 
                                                 for cat, count in rts_counts.items()],
                                          colors=colors,
                                          autopct='%1.1f%%',
                                          explode=[0.05, 0.05, 0.05])
        
        # Make the percentage labels larger and bold
        plt.setp(autotexts, size=12, weight="bold")
        plt.setp(texts, size=12)
        
        ax2.set_title('Returns to Scale Distribution', fontsize=14, fontweight='bold')
        
        # Add overall title
        plt.suptitle('Hospital Efficiency Analysis Summary', 
                    fontsize=16, fontweight='bold', y=1.05)
        
        # Add text box with key statistics
        stats_text = (
            f'Number of Hospitals: {len(data)}\n'
            f'VRS Efficiency Range: {data["TE_VRS"].min():.3f} - {data["TE_VRS"].max():.3f}\n'
            f'Scale Efficiency Range: {data["Scale_Efficiency"].min():.3f} - {data["Scale_Efficiency"].max():.3f}'
        )
        plt.figtext(0.5, -0.05, stats_text, ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                   fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'simple_efficiency_comparison.png'),
                   dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()

    def create_performance_pairs_plot(self, data):
        """
        Creates a pairs plot showing relationships between key hospital performance indicators.
        """
        # Select key variables for the pairs plot
        vars_to_plot = ['TE_VRS', 'Licensed_Beds', 'Bed_Occupancy_Rate', 
                        'Patient_Satisfaction', 'Overall_Rating']
        
        # Create the pairs plot
        plt.figure(figsize=(12, 12))
        g = sns.PairGrid(data[vars_to_plot], diag_sharey=False)
        
        # Add scatter plots on the off-diagonal
        g.map_offdiag(plt.scatter, alpha=0.7, color='#4FB3A8', s=50)
        
        # Add distribution plots on the diagonal
        g.map_diag(sns.kdeplot, fill=True, color='#4FB3A8', alpha=0.5)
        
        # Customize the plot
        plt.suptitle('Relationships Between Key Hospital Performance Indicators', 
                    y=1.02, fontsize=14, fontweight='bold')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'performance_pairs_plot.png'))
        plt.close()

    def create_compact_relationship_matrix(self, data):
        """
        Creates a compact 2x2 matrix showing relationships between key performance indicators.
        """
        # Select the most important variables with better labels
        vars_to_plot = ['TE_VRS', 'Licensed_Beds', 'Patient_Satisfaction', 'Bed_Occupancy_Rate']
        labels = {
            'TE_VRS': 'Technical Efficiency',
            'Licensed_Beds': 'Hospital Size',
            'Patient_Satisfaction': 'Patient Satisfaction',
            'Bed_Occupancy_Rate': 'Bed Occupancy'
        }
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        
        # Create plots
        for i in range(2):
            for j in range(2):
                ax = axes[i][j]
                var1 = vars_to_plot[i*2 + j]
                var2 = vars_to_plot[j*2 + i]
                
                if i == j:  # Diagonal
                    sns.kdeplot(data=data, x=var1, ax=ax, fill=True, color='#4FB3A8', alpha=0.5)
                    ax.set_title(labels[var1], fontsize=10, fontweight='bold', pad=10)
                else:  # Off-diagonal
                    ax.scatter(data[var2], data[var1], alpha=0.7, color='#4FB3A8', s=30)
                    ax.set_xlabel(labels[var2], fontsize=9, fontweight='bold')
                    ax.set_ylabel(labels[var1], fontsize=9, fontweight='bold')
                
                # Add grid for better readability
                ax.grid(True, linestyle='--', alpha=0.3)
                
                # Ensure all tick labels are visible
                ax.tick_params(axis='both', labelsize=8)
                
                # Rotate x-axis labels if needed for better fit
                ax.tick_params(axis='x', labelrotation=45 if len(str(ax.get_xlim()[1])) > 4 else 0)
        
        plt.suptitle('Key Hospital Performance Relationships', y=1.02, fontsize=12, fontweight='bold')
        plt.savefig(os.path.join(self.results_dir, 'compact_performance_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def generate_all_visualizations(self, data):
        """Generate all visualizations"""
        # Create a copy of the data to avoid modifying the original
        plot_data = data.copy()
        
        # Generate simple comparison visualization
        self.create_simple_efficiency_comparison(plot_data)
        
        # Generate efficiency summary table
        summary_table = self.generate_efficiency_table(plot_data)
        
        # Generate performance pairs plot
        self.create_performance_pairs_plot(plot_data)
        
        # Generate compact relationship matrix
        self.create_compact_relationship_matrix(data)
        
        print(f"All visualizations have been generated and saved to the '{self.results_dir}' directory.")
        print("Efficiency summary table has been saved as 'efficiency_summary.csv'")
        
        return summary_table 