import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

def plot_threads(folder_name):
    results_dir = os.path.join('./results', folder_name)
    output_image = os.path.join('./plots', f"{folder_name}_peak_trend.png")
    
    all_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    df = pd.concat([pd.read_csv(os.path.join(results_dir, f), skipinitialspace=True) for f in all_files])

    stats = df.groupby('Strategy')['TotalTime'].agg(['mean', 'max']).reset_index()
    stats = stats.sort_values(by='mean', ascending=False)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))
    
    plot_df = stats.melt(id_vars='Strategy', var_name='Metric', value_name='Time')
    plot_df['Metric'] = plot_df['Metric'].replace({'mean': 'Average', 'max': 'Maximum'})
    ax = sns.barplot(x='Strategy', y='Time', hue='Metric', data=plot_df, palette='viridis')

    x_coords = []
    y_coords = []
    
    for i, strategy in enumerate(stats['Strategy']):
        x = i - 0.2  
        y = stats.loc[stats['Strategy'] == strategy, 'mean'].values[0]
        x_coords.append(x)
        y_coords.append(y)

    plt.plot(x_coords, y_coords, color='red', marker='o', linestyle='--', 
             linewidth=2, label='Performance Trend', markersize=8)

    plt.yscale('log')
    plt.title(f'Performance Scaling and Stability: {folder_name}', fontsize=16, pad=20)
    plt.ylabel('Total Time (Seconds) - Log Scale', fontsize=13)
    plt.xlabel('Implementation Strategy', fontsize=13)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Trend chart saved: {output_image}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot.py <folder_name>")
    else:
        os.makedirs('./plots', exist_ok=True)
        plot_threads(sys.argv[1])
