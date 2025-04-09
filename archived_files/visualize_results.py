import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import os

class DEAVisualizer:
    def __init__(self, results_df: pd.DataFrame):
        """
        Initialize visualizer with DEA results
        
        Args:
            results_df: DataFrame containing DEA results
        """
        self.results_df = results_df
        self.output_dir = 'visualizations'
        os.makedirs(self.output_dir, exist_ok=True)
        
    def plot_efficiency_distribution(self):
        """Plot distribution of technical efficiency scores"""
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.results_df, x='Technical_Efficiency', bins=20)
        plt.title('Distribution of Technical Efficiency Scores')
        plt.xlabel('Efficiency Score')
        plt.ylabel('Number of Hospitals')
        plt.axvline(x=self.results_df['Technical_Efficiency'].mean(), 
                   color='r', linestyle='--', 
                   label=f'Mean: {self.results_df["Technical_Efficiency"].mean():.2f}')
        plt.legend()
        plt.savefig(f'{self.output_dir}/efficiency_distribution.png')
        plt.close()
        
    def plot_scale_efficiency(self):
        """Plot scale efficiency vs technical efficiency"""
        plt.figure(figsize=(10, 6))
        plt.scatter(self.results_df['Technical_Efficiency'], 
                   self.results_df['Scale_Efficiency'],
                   alpha=0.6)
        plt.title('Scale Efficiency vs Technical Efficiency')
        plt.xlabel('Technical Efficiency')
        plt.ylabel('Scale Efficiency')
        
        # Add reference lines
        plt.axhline(y=1, color='r', linestyle='--', label='Optimal Scale')
        plt.axvline(x=1, color='g', linestyle='--', label='Technical Efficiency = 1')
        
        plt.legend()
        plt.savefig(f'{self.output_dir}/scale_efficiency.png')
        plt.close()
        
    def plot_efficiency_by_size(self):
        """Plot efficiency scores by hospital size"""
        plt.figure(figsize=(12, 6))
        
        # Ensure all categories are present
        categories = ['Small', 'Medium', 'Large']
        data = []
        positions = []
        labels = []
        
        for i, category in enumerate(categories):
            if category in self.results_df['Size_Category'].unique():
                scores = self.results_df[self.results_df['Size_Category'] == category]['Technical_Efficiency']
                if len(scores) > 0:
                    data.append(scores)
                    positions.append(i)
                    labels.append(category)
                    
                    # Print summary statistics
                    print(f"\n{category} Hospitals:")
                    print(f"  Count: {len(scores)}")
                    print(f"  Mean: {scores.mean():.4f}")
                    print(f"  Std: {scores.std():.4f}")
                    print(f"  Min: {scores.min():.4f}")
                    print(f"  Max: {scores.max():.4f}")
        
        # Create boxplot
        plt.boxplot(data, positions=positions)
        plt.xticks(positions, labels)
        plt.title('Technical Efficiency by Hospital Size')
        plt.xlabel('Hospital Size Category')
        plt.ylabel('Technical Efficiency Score')
        plt.savefig(f'{self.output_dir}/efficiency_by_size.png')
        plt.close()
        
    def plot_malmquist_components(self, time_periods: List[str]):
        """Plot Malmquist Index components over time"""
        plt.figure(figsize=(12, 6))
        
        # Prepare data
        periods = []
        efficiency_changes = []
        technical_changes = []
        
        for i in range(len(time_periods) - 1):
            period = f"{time_periods[i]}-{time_periods[i+1]}"
            periods.append(period)
            efficiency_changes.append(
                self.results_df[f'Malmquist_{period}_Efficiency_Change'].mean()
            )
            technical_changes.append(
                self.results_df[f'Malmquist_{period}_Technical_Change'].mean()
            )
        
        # Plot
        x = np.arange(len(periods))
        width = 0.35
        
        plt.bar(x - width/2, efficiency_changes, width, label='Efficiency Change')
        plt.bar(x + width/2, technical_changes, width, label='Technical Change')
        
        plt.xlabel('Time Period')
        plt.ylabel('Change Index')
        plt.title('Malmquist Index Components Over Time')
        plt.xticks(x, periods)
        plt.legend()
        plt.savefig(f'{self.output_dir}/malmquist_components.png')
        plt.close()
        
    def plot_efficiency_frontier(self, input_var: str, output_var: str):
        """Plot efficiency frontier for a given input-output pair"""
        plt.figure(figsize=(10, 6))
        
        # Plot all hospitals
        plt.scatter(self.results_df[input_var], 
                   self.results_df[output_var],
                   alpha=0.6,
                   label='All Hospitals')
        
        # Highlight efficient hospitals
        efficient_mask = self.results_df['Technical_Efficiency'] == 1
        plt.scatter(self.results_df.loc[efficient_mask, input_var],
                   self.results_df.loc[efficient_mask, output_var],
                   color='red',
                   label='Efficient Hospitals')
        
        plt.title(f'Efficiency Frontier: {input_var} vs {output_var}')
        plt.xlabel(input_var)
        plt.ylabel(output_var)
        plt.legend()
        plt.savefig(f'{self.output_dir}/efficiency_frontier_{input_var}_{output_var}.png')
        plt.close()
        
    def plot_correlation_heatmap(self):
        """Plot correlation heatmap of all variables"""
        plt.figure(figsize=(12, 10))
        
        # Calculate correlation matrix
        corr_matrix = self.results_df.select_dtypes(include=[np.number]).corr()
        
        # Create heatmap
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   fmt='.2f')
        
        plt.title('Correlation Matrix of DEA Variables')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/correlation_heatmap.png')
        plt.close()
        
    def plot_context_efficiency(self, context_factors: Dict[str, List[str]]):
        """Plot efficiency scores by context factors"""
        for factor, categories in context_factors.items():
            plt.figure(figsize=(12, 6))
            
            # Prepare data for boxplot
            data = []
            positions = []
            labels = []
            
            for i, category in enumerate(categories):
                if category in self.results_df[factor].unique():
                    scores = self.results_df[self.results_df[factor] == category]['Technical_Efficiency']
                    if len(scores) > 0:
                        data.append(scores)
                        positions.append(i)
                        labels.append(category)
                        
                        # Print summary statistics
                        print(f"\n{factor} - {category}:")
                        print(f"  Count: {len(scores)}")
                        print(f"  Mean: {scores.mean():.4f}")
                        print(f"  Std: {scores.std():.4f}")
                        print(f"  Min: {scores.min():.4f}")
                        print(f"  Max: {scores.max():.4f}")
            
            if data:
                # Create boxplot
                plt.boxplot(data, positions=positions)
                plt.xticks(positions, labels)
                plt.title(f'Technical Efficiency by {factor}')
                plt.xlabel(factor)
                plt.ylabel('Technical Efficiency Score')
                plt.savefig(f'{self.output_dir}/efficiency_by_{factor.lower()}.png')
            else:
                print(f"\nNo valid data for {factor}")
            
            plt.close()

def main():
    # Load results
    results_df = pd.read_csv('dea_results.csv')
    
    # Initialize visualizer
    visualizer = DEAVisualizer(results_df)
    
    # Generate all visualizations
    print("Generating visualizations...")
    
    # Basic efficiency plots
    visualizer.plot_efficiency_distribution()
    visualizer.plot_scale_efficiency()
    visualizer.plot_efficiency_by_size()
    
    # Correlation analysis
    visualizer.plot_correlation_heatmap()
    
    # Efficiency frontiers for key input-output pairs
    visualizer.plot_efficiency_frontier('Licensed_Beds', 'Patient_Satisfaction')
    visualizer.plot_efficiency_frontier('Total_Surveys', 'Overall_Rating')
    
    # Context-dependent analysis
    context_factors = {
        'Size_Category': ['Small', 'Medium', 'Large'],
        'Location': ['Urban', 'Rural'],
        'Specialty': ['General', 'Specialized']
    }
    visualizer.plot_context_efficiency(context_factors)
    
    print("Visualizations completed and saved to 'visualizations' directory")

if __name__ == "__main__":
    main() 