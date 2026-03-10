import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def plot_accuracy(folder_name):
    results_dir = os.path.join('./results', folder_name)
    output_image = os.path.join('./plots', f"{folder_name}_convergence.png")
    
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
    plt.figure(figsize=(12, 6))

    ax = sns.lineplot(data=master_df, x='Instance', y='Accuracy', hue='Strategy', 
                      marker='o', linewidth=2.5, alpha=0.8)

    plt.title(f'Accuracy Analysis: Fit routine', fontsize=16, pad=20)
    plt.ylabel('Validation Accuracy', fontsize=13)
    plt.xlabel('Training Session (Instance)', fontsize=13)
    plt.xticks(range(10))
    
    plt.ylim(0.85, 1.0) 

    plt.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Convergence plot saved: {output_image}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_convergence.py <folder_name>")
    else:
        os.makedirs('./plots', exist_ok=True)
        plot_accuracy(sys.argv[1])
