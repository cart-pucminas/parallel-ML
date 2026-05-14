import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate():
    results_dir = './results/fit8'
    output_path = './images/plots/chart_3.png'    

    if not os.path.exists(results_dir):
        print(f"Error: Directory '{results_dir}' not found.")
        return

    all_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    dataframes = []
    for f in all_files:
        df = pd.read_csv(os.path.join(results_dir, f), skipinitialspace=True)
        dataframes.append(df)
    
    master_df = pd.concat(dataframes, ignore_index=True)

    sns.set_theme(style="whitegrid")
    sns.set_context("talk")

    plt.figure(figsize=(12, 6))

    ax = sns.lineplot(data=master_df, x='Instance', y='Accuracy', hue='Strategy', 
                      marker='o', linewidth=2.5, alpha=0.8)

    plt.title(f'Accuracy Analysis: Fit routine', fontsize=25, pad=20)
    plt.ylabel('Validation Accuracy', fontsize=25)
    plt.xlabel('Training Session (Instance)', fontsize=25)
    plt.xticks(range(10))
    
    plt.ylim(0.85, 1.0) 

    plt.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"generated: {output_path}")

os.makedirs('./plots', exist_ok=True)
generate()
