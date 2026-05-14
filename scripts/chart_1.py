import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate():
    results_dir = './results/ff'
    output_path = './images/plots/chart_1.png'

    all_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    dataframes = []

    for file in all_files:
        path = os.path.join(results_dir, file)
        df = pd.read_csv(path, skipinitialspace=True)
        dataframes.append(df)

    master_df = pd.concat(dataframes, ignore_index=True)

    master_df['Strategy'] = master_df['Strategy'].str.replace('_OpenMP', '', case=False)

    master_df['Strategy'] = master_df['Strategy'].str.replace('_', ' ')

    seq_data = master_df[master_df['Strategy'] == 'Sequential']
    if seq_data.empty:
        print("Erro: Dados sequenciais não encontrados para calcular o Speedup.")
        return
    
    avg_seq_time = seq_data['TotalTime'].mean()

    master_df['Speedup'] = avg_seq_time / master_df['TotalTime']

    plot_df = master_df[master_df['Strategy'] != 'Sequential'].copy()

    sns.set_theme(style="whitegrid")
    sns.set_context("talk") 

    plt.figure(figsize=(12, 8)) 

    ax = sns.boxplot(x='Strategy', y='Speedup', data=plot_df, 
                     palette='viridis', width=0.5)

    plt.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Sequential Baseline')

    plt.title('Performance Comparison: Parallelization Speedup', fontsize=25, pad=20)
    
    plt.ylabel('Speedup', fontsize=25, labelpad=15)
    plt.xlabel('Implementation Strategy', fontsize=25, labelpad=15)
    
    plt.xticks(fontsize=25) 
    plt.yticks(fontsize=25)
    
    plt.legend(fontsize=25, loc='upper right', frameon=True)

    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"generated: {output_path}")

os.makedirs('./plots', exist_ok=True)
generate()
