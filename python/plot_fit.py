import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def merge_and_plot_avg_max(results_dir, output_image):
    # 1. Gather all CSVs in the directory
    all_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    dataframes = []

    for f in all_files:
        path = os.path.join(results_dir, f)
        df = pd.read_csv(path, skipinitialspace=True)
        dataframes.append(df)

    master_df = pd.concat(dataframes, ignore_index=True)

    df_melted = master_df.melt(id_vars='Strategy', var_name='Metric', value_name='Time')
    df_melted['Metric'] = df_melted['Metric'].replace({'AvgTime': 'Average Latency', 'MaxTime': 'Maximum Latency'})

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))

    ax = sns.barplot(x='Strategy', y='Time', hue='Metric', data=df_melted, palette='viridis')

    plt.title('Training Time Analysis', fontsize=16, pad=20)
    plt.ylabel('Time (Seconds)', fontsize=13)
    plt.xlabel('Parallelization Strategy', fontsize=13)
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', fontsize=9, fontweight='bold', 
                        xytext=(0, 5), textcoords='offset points')

    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Master Grouped Chart saved: {output_image}")

os.makedirs('./plots', exist_ok=True)
merge_and_plot_avg_max('./results/fit4', './plots/fit4.png')
