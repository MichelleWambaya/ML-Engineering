"""
Visualization Module for Soil Richness Analysis
Creates charts, maps, and interactive visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import folium
from folium.plugins import HeatMap

class SoilVisualizer:
    """Creates visualizations for soil data"""
    
    def __init__(self, data_path="data/processed/soil_data_processed.csv",
                 viz_dir="visualizations"):
        self.data_path = Path(data_path)
        self.viz_dir = Path(viz_dir)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.df = None
    
    def load_data(self):
        """Load processed data"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data not found at {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        return self.df
    
    def plot_richness_distribution(self):
        """Plot distribution of soil richness scores"""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(self.df['soil_richness_score'], bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('Soil Richness Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Soil Richness Scores')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        self.df.boxplot(column='soil_richness_score', ax=plt.gca())
        plt.ylabel('Soil Richness Score')
        plt.title('Box Plot of Soil Richness Scores')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'richness_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved richness distribution plot")
    
    def plot_nutrient_analysis(self):
        """Plot nutrient analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Nitrogen
        axes[0, 0].scatter(self.df['nitrogen_ppm'], self.df['soil_richness_score'], alpha=0.6)
        axes[0, 0].set_xlabel('Nitrogen (ppm)')
        axes[0, 0].set_ylabel('Soil Richness Score')
        axes[0, 0].set_title('Nitrogen vs Soil Richness')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Phosphorus
        axes[0, 1].scatter(self.df['phosphorus_ppm'], self.df['soil_richness_score'], alpha=0.6, color='orange')
        axes[0, 1].set_xlabel('Phosphorus (ppm)')
        axes[0, 1].set_ylabel('Soil Richness Score')
        axes[0, 1].set_title('Phosphorus vs Soil Richness')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Potassium
        axes[1, 0].scatter(self.df['potassium_ppm'], self.df['soil_richness_score'], alpha=0.6, color='green')
        axes[1, 0].set_xlabel('Potassium (ppm)')
        axes[1, 0].set_ylabel('Soil Richness Score')
        axes[1, 0].set_title('Potassium vs Soil Richness')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Organic Matter
        axes[1, 1].scatter(self.df['organic_matter_percent'], self.df['soil_richness_score'], alpha=0.6, color='red')
        axes[1, 1].set_xlabel('Organic Matter (%)')
        axes[1, 1].set_ylabel('Soil Richness Score')
        axes[1, 1].set_title('Organic Matter vs Soil Richness')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'nutrient_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved nutrient analysis plot")
    
    def plot_correlation_heatmap(self):
        """Plot correlation heatmap"""
        numeric_cols = [
            'ph', 'organic_matter_percent', 'nitrogen_ppm', 'phosphorus_ppm',
            'potassium_ppm', 'cec_meq_per_100g', 'rainfall_mm_per_year',
            'temperature_celsius', 'elevation_meters', 'soil_richness_score'
        ]
        
        corr_matrix = self.df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Heatmap of Soil Parameters')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved correlation heatmap")
    
    def plot_top_counties(self, n=10):
        """Plot top N counties by soil richness"""
        top_counties = self.df.nlargest(n, 'soil_richness_score')
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(n), top_counties['soil_richness_score'], color='green', alpha=0.7)
        plt.yticks(range(n), top_counties['county'])
        plt.xlabel('Soil Richness Score')
        plt.title(f'Top {n} Counties by Soil Richness')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'top_counties.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved top {n} counties plot")
    
    def create_interactive_map(self):
        """Create interactive map of soil richness"""
        # Approximate coordinates for Kenyan counties (simplified)
        # In a real project, you'd use actual county boundaries
        np.random.seed(42)
        county_coords = {
            county: [np.random.uniform(-4.5, 5.5), np.random.uniform(33.5, 42.0)]
            for county in self.df['county'].unique()
        }
        
        # Create base map centered on Kenya
        m = folium.Map(location=[0.0236, 37.9062], zoom_start=6)
        
        # Add markers for each county
        for _, row in self.df.iterrows():
            county = row['county']
            richness = row['soil_richness_score']
            coords = county_coords.get(county, [0.0236, 37.9062])
            
            # Color based on richness
            if richness >= 70:
                color = 'green'
            elif richness >= 50:
                color = 'yellow'
            elif richness >= 30:
                color = 'orange'
            else:
                color = 'red'
            
            folium.CircleMarker(
                location=coords,
                radius=richness / 5,
                popup=f"{county}: {richness:.1f}",
                color='black',
                fill=True,
                fillColor=color,
                fillOpacity=0.6
            ).add_to(m)
        
        m.save(str(self.viz_dir / 'soil_richness_map.html'))
        print("Saved interactive map")
    
    def create_interactive_dashboard(self):
        """Create interactive Plotly dashboard"""
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=self.df['nitrogen_ppm'],
            y=self.df['soil_richness_score'],
            mode='markers',
            marker=dict(
                size=self.df['organic_matter_percent'] * 3,
                color=self.df['phosphorus_ppm'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Phosphorus (ppm)")
            ),
            text=self.df['county'],
            hovertemplate='<b>%{text}</b><br>' +
                         'Nitrogen: %{x:.1f} ppm<br>' +
                         'Richness: %{y:.1f}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title='Interactive Soil Richness Analysis',
            xaxis_title='Nitrogen (ppm)',
            yaxis_title='Soil Richness Score',
            hovermode='closest',
            width=1200,
            height=700
        )
        
        fig.write_html(str(self.viz_dir / 'interactive_dashboard.html'))
        print("Saved interactive dashboard")
    
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        self.load_data()
        print("Generating visualizations...")
        
        self.plot_richness_distribution()
        self.plot_nutrient_analysis()
        self.plot_correlation_heatmap()
        self.plot_top_counties()
        self.create_interactive_map()
        self.create_interactive_dashboard()
        
        print("\nAll visualizations generated!")

if __name__ == "__main__":
    visualizer = SoilVisualizer()
    visualizer.generate_all_visualizations()

