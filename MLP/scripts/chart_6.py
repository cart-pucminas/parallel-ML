import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate():
    results_dir = './results/inter-sample-vectorized'
    output_path = './plots/chart_6.png'

    all_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    if not all_files: return

    dataframes = [pd.read_csv(os.path.join(results_dir, f), skipinitialspace=True) for f in all_files]
    master_df = pd.concat(dataframes, ignore_index=True)

    master_df['Threads'] = master_df['Strategy'].str.extract(r'(\d+)').astype(int)
    avg_seq_time = master_df[master_df['Threads'] == 1]['TotalTime'].mean()
    master_df['Speedup'] = avg_seq_time / master_df['TotalTime']
    master_df['Efficiency'] = master_df['Speedup'] / master_df['Threads'].clip(upper=4)

    plot_df = master_df[master_df['Threads'] != 1].copy()
    plot_df = plot_df.sort_values(by='Threads', ascending=True)
    
    plot_df['Threads_Str'] = plot_df['Threads'].astype(str)
    thread_order = plot_df['Threads_Str'].unique()

    max_efficiency = plot_df['Efficiency'].max()
    y_upper_limit = max(1.2, max_efficiency * 1.15) 

    sns.set_theme(style="whitegrid")
    sns.set_context("talk") 
    plt.figure(figsize=(14, 10)) 

    ax = sns.barplot(x='Threads_Str', y='Efficiency', data=plot_df, 
                     palette='magma', order=thread_order,
                     errorbar=('ci', 95), capsize=.15, alpha=0.8,
                     err_kws={'linewidth': 2.5, 'color': 'black'})

    plt.axhline(1.0, color='green', linestyle='--', linewidth=3, label='Ideal Efficiency (1.0)')

    plt.title('Parallel Efficiency: Thread Scalability', fontsize=30, pad=30)
    plt.ylabel('Efficiency ($\eta = S_p / N_{cores}$)', fontsize=25, labelpad=20)
    plt.xlabel('Number of Threads', fontsize=25, labelpad=20)
   
    plt.xticks(fontsize=25) 
    plt.yticks(fontsize=25)
    
    plt.ylim(0, y_upper_limit)

    plt.legend(fontsize=22, loc='upper left', frameon=True, shadow=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"generated: {output_path}")

os.makedirs('./plots', exist_ok=True)
generate()
