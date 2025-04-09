import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path

class AdvancedAnalyzer:
    def __init__(self):
        self.results_dir = 'analysis_results'
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            
        # Create subdirectories for different types of results
        self.tables_dir = os.path.join(self.results_dir, 'tables')
        self.plots_dir = os.path.join(self.results_dir, 'plots')
        os.makedirs(self.tables_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        plt.style.use('default')
    
    def generate_summary_tables(self, data):
        """Generate detailed summary tables for different aspects of the analysis"""
        
        # 1. Efficiency Statistics by Size Category
        size_stats = data.groupby('Size_Category').agg({
            'TE_VRS': ['count', 'mean', 'std', 'min', 'max'],
            'TE_CRS': ['count', 'mean', 'std', 'min', 'max'],
            'Scale_Efficiency': ['mean', 'std']
        }).round(4)
        size_stats.to_csv(os.path.join(self.tables_dir, 'efficiency_by_size.csv'))
        
        # 2. Efficiency Statistics by Occupancy Category
        occupancy_stats = data.groupby('Occupancy_Category').agg({
            'TE_VRS': ['count', 'mean', 'std', 'min', 'max'],
            'TE_CRS': ['count', 'mean', 'std', 'min', 'max'],
            'Scale_Efficiency': ['mean', 'std']
        }).round(4)
        occupancy_stats.to_csv(os.path.join(self.tables_dir, 'efficiency_by_occupancy.csv'))
        
        # 3. Efficiency Statistics by Quality Category
        quality_stats = data.groupby('Quality_Category').agg({
            'TE_VRS': ['count', 'mean', 'std', 'min', 'max'],
            'TE_CRS': ['count', 'mean', 'std', 'min', 'max'],
            'Scale_Efficiency': ['mean', 'std']
        }).round(4)
        quality_stats.to_csv(os.path.join(self.tables_dir, 'efficiency_by_quality.csv'))
        
        # 4. Scale Efficiency Analysis
        scale_stats = data.groupby('Scale_Efficiency_Category').agg({
            'Scale_Efficiency': ['count', 'mean', 'std', 'min', 'max'],
            'Licensed_Beds': ['mean', 'std'],
            'Bed_Occupancy_Rate': ['mean', 'std']
        }).round(4)
        scale_stats.to_csv(os.path.join(self.tables_dir, 'scale_efficiency_analysis.csv'))
        
        # 5. Top and Bottom Performers
        top_performers = data.nlargest(10, 'TE_VRS')[
            ['Hospital_Name', 'TE_VRS', 'TE_CRS', 'Scale_Efficiency', 'Size_Category', 'Quality_Category']
        ]
        top_performers.to_csv(os.path.join(self.tables_dir, 'top_performers.csv'))
        
        bottom_performers = data.nsmallest(10, 'TE_VRS')[
            ['Hospital_Name', 'TE_VRS', 'TE_CRS', 'Scale_Efficiency', 'Size_Category', 'Quality_Category']
        ]
        bottom_performers.to_csv(os.path.join(self.tables_dir, 'bottom_performers.csv'))
        
        print("Summary tables have been generated and saved to the 'tables' directory.")
    
    def generate_additional_plots(self, data):
        """Generate additional visualizations for deeper analysis"""
        
        # 1. Efficiency Scores vs Hospital Size
        plt.figure(figsize=(12, 6))
        plt.scatter(data['Licensed_Beds'], data['TE_VRS'], alpha=0.6)
        plt.xlabel('Licensed Beds')
        plt.ylabel('VRS Efficiency Score')
        plt.title('Hospital Size vs Efficiency')
        plt.savefig(os.path.join(self.plots_dir, 'size_vs_efficiency.png'))
        plt.close()
        
        # 2. Efficiency Scores vs Occupancy Rate
        plt.figure(figsize=(12, 6))
        plt.scatter(data['Bed_Occupancy_Rate'], data['TE_VRS'], alpha=0.6)
        plt.xlabel('Bed Occupancy Rate (%)')
        plt.ylabel('VRS Efficiency Score')
        plt.title('Occupancy Rate vs Efficiency')
        plt.savefig(os.path.join(self.plots_dir, 'occupancy_vs_efficiency.png'))
        plt.close()
        
        # 3. Quality Metrics Correlation Plot
        quality_cols = ['Patient_Satisfaction', 'Overall_Rating', 'Recommendation_Score', 
                       'Staff_Responsiveness', 'Nurse_Communication', 'Doctor_Communication', 
                       'Cleanliness', 'Quietness']
        quality_corr = data[quality_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(quality_corr, annot=True, cmap='RdYlBu', center=0)
        plt.title('Correlation Matrix of Quality Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'quality_metrics_correlation.png'))
        plt.close()
        
        # 4. Scale Efficiency Distribution by Category
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Size Category
        sns.boxplot(data=data, x='Size_Category', y='Scale_Efficiency', ax=axes[0])
        axes[0].set_title('Scale Efficiency by Hospital Size')
        
        # Occupancy Category
        sns.boxplot(data=data, x='Occupancy_Category', y='Scale_Efficiency', ax=axes[1])
        axes[1].set_title('Scale Efficiency by Occupancy Rate')
        
        # Quality Category
        sns.boxplot(data=data, x='Quality_Category', y='Scale_Efficiency', ax=axes[2])
        axes[2].set_title('Scale Efficiency by Quality Rating')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'scale_efficiency_by_category.png'))
        plt.close()
        
        # 5. Efficiency Score Distribution by Scale Category
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=data, x='Scale_Efficiency_Category', y='TE_VRS')
        plt.title('VRS Efficiency Scores by Scale Efficiency Category')
        plt.savefig(os.path.join(self.plots_dir, 'efficiency_by_scale_category.png'))
        plt.close()
        
        print("Additional plots have been generated and saved to the 'plots' directory.")
    
    def run_advanced_analysis(self, data):
        """Run all advanced analyses"""
        self.generate_summary_tables(data)
        self.generate_additional_plots(data)
        print("\nAdvanced analysis completed successfully!") 