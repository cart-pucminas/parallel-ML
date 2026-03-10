import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_master_plot(results_dir, output_path):
    all_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    dataframes = []

    for file in all_files:
        path = os.path.join(results_dir, file)
        df = pd.read_csv(path, skipinitialspace=True)
        dataframes.append(df)

    master_df = pd.concat(dataframes, ignore_index=True)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))

    ax = sns.boxplot(x='Strategy', y='Throughput', data=master_df, 
                     palette='viridis', width=0.6)

    plt.yscale('log') 

    plt.title('Performance Comparison: Parallelization Strategies (Log Scale)', fontsize=16)
    plt.ylabel('Throughput (Samples/sec)', fontsize=13)
    plt.xlabel('Implementation Strategy', fontsize=13)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Master Chart generated: {output_path}")

os.makedirs('./plots', exist_ok=True)
generate_master_plot('./results/ff', './plots/master_comparison.png')
