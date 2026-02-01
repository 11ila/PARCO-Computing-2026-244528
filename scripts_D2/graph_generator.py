import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def create_graphs(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract base filename without extension for saving graphs
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    
    # Create output directory if it doesn't exist
    output_dir = 'graphs'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Line plot: Time vs Number of Processes
    plt.figure(figsize=(10, 6))
    plt.plot(df['num_processes'], df['time'], marker='o', linewidth=2, markersize=8)
    plt.xlabel('Number of Processes', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Execution Time vs Number of Processes', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{base_name}_line.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Bar plot: Time vs Number of Processes
    plt.figure(figsize=(10, 6))
    plt.bar(df['num_processes'].astype(str), df['time'], color='steelblue', edgecolor='black')
    plt.xlabel('Number of Processes', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Execution Time vs Number of Processes', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{base_name}_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Speedup plot
    baseline_time = df['time'].iloc[0]  # Time for first number of processes
    df['speedup'] = baseline_time / df['time']
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['num_processes'], df['speedup'], marker='s', linewidth=2, markersize=8, color='green', label='Actual Speedup')
    plt.plot(df['num_processes'], df['num_processes'] / df['num_processes'].iloc[0], 
             linestyle='--', color='red', label='Ideal Speedup')
    plt.xlabel('Number of Processes', fontsize=12)
    plt.ylabel('Speedup', fontsize=12)
    plt.title('Speedup vs Number of Processes', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{base_name}_speedup.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Efficiency plot
    df['efficiency'] = df['speedup'] / (df['num_processes'] / df['num_processes'].iloc[0])
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['num_processes'], df['efficiency'] * 100, marker='D', linewidth=2, markersize=8, color='orange')
    plt.xlabel('Number of Processes', fontsize=12)
    plt.ylabel('Efficiency (%)', fontsize=12)
    plt.title('Parallel Efficiency vs Number of Processes', fontsize=14, fontweight='bold')
    plt.ylim(0, 110)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{base_name}_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Heatmap: Process vs Time with percentage speedup
    pivot_data = df.pivot_table(values='speedup', index='num_processes', aggfunc='first')
    pivot_data_percentage = (pivot_data * 100).round(1)
    
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(pivot_data_percentage, annot=True, fmt='.1f', cmap='YlOrRd', 
                     cbar_kws={'label': 'Speedup (%)'}, 
                     annot_kws={'color': 'black', 'fontsize': 10},
                     linewidths=0)
    plt.xlabel('', fontsize=12)
    plt.ylabel('Number of Processes', fontsize=12)
    plt.title('Speedup Heatmap (%)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{base_name}_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Graphs saved in '{output_dir}' directory with prefix '{base_name}'")
    print("\nSummary Statistics:")
    print(df.to_string(index=False))

# Process both CSV files
if __name__ == "__main__":
    csv_files = ["weak_scaling_results.csv", "strong_scaling_results.csv"]
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            print(f"\n{'='*60}")
            print(f"Processing: {csv_file}")
            print('='*60)
            create_graphs(csv_file)
        else:
            print(f"\nWarning: File '{csv_file}' not found. Skipping...")
    
    print(f"\n{'='*60}")
    print("All processing complete!")
    print('='*60)