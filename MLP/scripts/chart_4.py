import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate():
    results_dir = './results/inter-sample-vectorized'
    output_path = './plots/chart_4.png'

    if not os.path.exists(results_dir):
        print(f"Error: Directory '{results_dir}' not found.")
        return

    all_files = [f for f in os.listdir(results_dir) if f.endswith('.csv') and not f.startswith('1')]
    dataframes = []
    for f in all_files:
        df = pd.read_csv(os.path.join(results_dir, f), skipinitialspace=True)
        dataframes.append(df)
    
    master_df = pd.concat(dataframes, ignore_index=True)

    sns.set_theme(style="whitegrid")
    sns.set_context("talk")

    plt.figure(figsize=(12, 7))
    
    ax = sns.barplot(x='Strategy', y='TotalTime', data=master_df, palette='viridis',
                    errorbar=('ci', 95), capsize=.15, alpha=0.8,
                    err_kws={'linewidth': 2.5, 'color': 'black'})

    plt.yscale('log')

    plt.title(f'Performance Analysis: Fit under different thread numbers', fontsize=20, pad=20)
    plt.ylabel('Total Time (Seconds) - Log Scale', fontsize=22)
    plt.xlabel('Strategy', fontsize=22)

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f'{h:.2f}s', (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=16, fontweight='bold',
                        xytext=(0, 10), textcoords='offset points')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"generated: {output_path}")

os.makedirs('./plots', exist_ok=True)
generate()
