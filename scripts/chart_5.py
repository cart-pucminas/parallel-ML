import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate():
    results_dir = './results/inter-sample-vectorized'
    output_path = './plots/chart_5.png'

    all_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    if not all_files:
        print(f"Erro: Nenhum arquivo CSV encontrado em {results_dir}")
        return

    dataframes = []
    for file in all_files:
        path = os.path.join(results_dir, file)
        df = pd.read_csv(path, skipinitialspace=True)
        dataframes.append(df)

    master_df = pd.concat(dataframes, ignore_index=True)

    master_df['Threads'] = master_df['Strategy'].str.extract(r'(\d+)').astype(int)

    avg_seq_time = master_df[master_df['Threads'] == 1]['TotalTime'].mean()
    master_df['Speedup'] = avg_seq_time / master_df['TotalTime']

    plot_df = master_df[master_df['Threads'] != 1].copy()
    plot_df = plot_df.sort_values(by='Threads', ascending=True)

    plot_df['Threads'] = plot_df['Threads'].astype(str)
    thread_order = plot_df['Threads'].unique() 

    sns.set_theme(style="whitegrid")
    sns.set_context("talk") 
    plt.figure(figsize=(14, 10)) 

    ax = sns.barplot(x='Threads', y='Speedup', data=plot_df, 
                     palette='viridis', order=thread_order,
                     errorbar=('ci', 95), capsize=.15, alpha=0.8,
                     err_kws={'linewidth': 2.5, 'color': 'black'})

    plt.axhline(1.0, color='red', linestyle='--', linewidth=3, label='Sequential Baseline')

    plt.title('Performance Comparison: Fit Routine Speedup', fontsize=30, pad=30)
    plt.ylabel('Speedup', fontsize=25, labelpad=20)
    plt.xlabel('Number of Threads', fontsize=25, labelpad=20)
    
    plt.xticks(fontsize=25) 
    plt.yticks(fontsize=25)
    
    plt.legend(fontsize=22, loc='upper left', frameon=True, shadow=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"generated: {output_path}")

os.makedirs('./plots', exist_ok=True)
generate()
