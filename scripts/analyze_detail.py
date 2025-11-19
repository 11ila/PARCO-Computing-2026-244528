#!/usr/bin/env python3
"""

Generates 4 graphs:
  - performance_overview.png: Comparison of average times by schedule type
  - heatmap_chunk_size.png: Heatmap speedup by schedule type and chunk size
  - efficiency.png: Parallel efficiency (speedup/threads)
  - 90percent.png: 90th percentile of times by configuration

Usage: python3 analyse_detail.py <file.csv>
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

# Configurazione stile grafici
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data(csv_file):
    """Load and prepare data from CSV with individual runs."""
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found!")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File '{csv_file}' empty!")
        sys.exit(1)
    
    # Check required columns (new format with individual runs)
    required_cols = ['run', 'threads', 'schedule_type', 'chunk_size', 'time_ms', 'speedup']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        print(f"Error: columns not fund in CSV: {missing_cols}")
        print(f"columns found: {list(df.columns)}")
        sys.exit(1)
    
    # Convert chunk_size from string to categorical
    df['chunk_size'] = df['chunk_size'].astype(str)
    
    # Create readable labels for configurations
    df['config'] = df['schedule_type'] + '-' + df['chunk_size']
    
    # Map the full names of schedule types
    schedule_map = {'S': 'Static', 'D': 'Dynamic', 'G': 'Guided'}
    df['schedule_name'] = df['schedule_type'].map(schedule_map)
    
    print(f" Data created : {len(df)} rows, {df['run'].nunique()} run")
    print(f" Threads tested: {sorted(df['threads'].unique())}")
    print(f" Configurations: {df['config'].nunique()}")
    
    return df

def aggregate_statistics(df):
    """
    Calculates aggregate statistics from individual runs.
    Returns a DataFrame with averages, min, max, and standard deviation for each configuration.
    """
    agg_df = df.groupby(['threads', 'schedule_type', 'chunk_size', 'schedule_name', 'config']).agg({
        'time_ms': ['mean', 'min', 'max', 'std'],
        'speedup': ['mean', 'min', 'max', 'std'],
        'p90_ms': 'last', 
        'run': 'count'
    }).reset_index()
    
    # simplify cols name
    agg_df.columns = ['threads', 'schedule_type', 'chunk_size', 'schedule_name', 'config',
                      'avg_time_ms', 'min_time_ms', 'max_time_ms', 'stddev_time_ms',
                      'avg_speedup', 'min_speedup', 'max_speedup', 'stddev_speedup',
                      'p90_ms', 'num_runs']
    
    return agg_df

def plot_performance_overview(df, output_file='performance_overview.png'):
    """
    Chart 1: Performance Overview with area plot
    Shows performance vs threads by schedule type
    """
    #  Aggregates for schedule type and threads
    perf_data = df.groupby(['threads', 'schedule_name']).agg({
        'time_ms': 'mean',
        'speedup': 'mean'
    }).reset_index()
    
    # Create two subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- Subplot 1: Performance vs Thread ---
    threads_list = sorted(perf_data['threads'].unique())
    
    # Prepare data for plot area
    for sched in ['Static', 'Dynamic', 'Guided']:
        data = perf_data[perf_data['schedule_name'] == sched].sort_values('threads')
        color_map = {'Static': '#ff9933', 'Dynamic': '#33cccc', 'Guided': '#3399ff'}
        ax1.plot(data['threads'], data['time_ms'], marker='o', linewidth=2, 
                 label=sched[0], color=color_map[sched])
        ax1.fill_between(data['threads'], 0, data['time_ms'], alpha=0.3, color=color_map[sched])
    
    ax1.set_xlabel('Number of Thread(s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('avarage time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Performance vs Thread per Scheduling', fontsize=14, fontweight='bold')
    ax1.legend(title='Schedule Type', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(threads_list)
    
    # --- Subplot 2: Speedup Related ---
    for sched in ['Static', 'Dynamic', 'Guided']:
        data = perf_data[perf_data['schedule_name'] == sched].sort_values('threads')
        color_map = {'Static': '#ff9933', 'Dynamic': '#33cccc', 'Guided': '#3399ff'}
        ax2.plot(data['threads'], data['speedup'], marker='o', linewidth=2,
                 label=sched[0], color=color_map[sched])
    
    # line: ideal speedup 
    ax2.plot(threads_list, threads_list, '--', color='gray', linewidth=2, 
             label='Ideal Speedup', alpha=0.7)
    
    ax2.set_xlabel('Number of Thread', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Speedup', fontsize=12, fontweight='bold')
    ax2.set_title('Relative Speedup ', fontsize=14, fontweight='bold')
    ax2.legend(title='Schedule Type', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(threads_list)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f" Graphic saved: {output_file}")
    plt.close()

def plot_heatmap_chunk_size(df, output_file='heatmap_chunk_size.png'):
    """
     Graph 2: Heatmap Chunk Size - 4 subplots per threads (1, 4, 16, 32)
    Shows speedup as a function of schedule type and chunk size
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Thread counts to analyze
    thread_counts = [1, 4, 16, 32]
    
    for idx, t in enumerate(thread_counts):
        ax = axes[idx]
        
        # Filter data for this number of threads
        df_t = df[df['threads'] == t]
        
        if len(df_t) == 0:
            ax.text(0.5, 0.5, f'No data for {t} thread(s)', 
                    ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Calculate avarage speedup  for schedule type and chunk size
        heatmap_data = df_t.groupby(['schedule_name', 'chunk_size'])['speedup'].mean().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='schedule_name', columns='chunk_size', values='speedup')
        
        #  sort the columns: first '-', then with the numbers in order
        chunk_order = ['-'] + sorted([c for c in heatmap_pivot.columns if c != '-'], 
                                       key=lambda x: int(x))
        heatmap_pivot = heatmap_pivot[[c for c in chunk_order if c in heatmap_pivot.columns]]
        
        # sort the rows
        row_order = [r for r in ['Static', 'Dynamic', 'Guided'] if r in heatmap_pivot.index]
        heatmap_pivot = heatmap_pivot.reindex(row_order)
        
        # Crea heatmap
        sns.heatmap(heatmap_pivot, annot=True, fmt='.2f', cmap='RdYlGn', 
                    cbar_kws={'label': 'Speedup'}, ax=ax, linewidths=0.5,
                    vmin=0, vmax=df['speedup'].max())
        
        ax.set_xlabel('Chunk Size', fontsize=11, fontweight='bold')
        ax.set_ylabel('Schedule Type', fontsize=11, fontweight='bold')
        ax.set_title(f'{t} Thread(s)', fontsize=13, fontweight='bold', pad=10)
    
    plt.suptitle('Heatmap: Speedup for Schedule Type and Chunk Size', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f" Graphic saved: {output_file}")
    plt.close()

def plot_efficiency(df_agg, output_file='efficiency.png'):
    """
    Graph 3: Parallel Efficiency
    Shows efficiency (speedup/threads) for each configuration
    """
    # Calculate efficiency from aggregate statistics
    df_eff = df_agg.copy()
    df_eff['efficiency'] = (df_eff['avg_speedup'] / df_eff['threads']) * 100
    
    # Calculate average efficiency per configuration (aggregated across all threads)
    eff_stats = df_eff.groupby(['schedule_name', 'chunk_size']).agg({
        'efficiency': 'mean'
    }).reset_index()
    
    eff_stats.columns = ['schedule_name', 'chunk_size', 'eff_mean']
    eff_stats['label'] = eff_stats['schedule_name'] + '-' + eff_stats['chunk_size']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # sets colours
    colors = {'Static': '#3498db', 'Dynamic': '#e74c3c', 'Guided': '#2ecc71'}
    bar_colors = [colors[name] for name in eff_stats['schedule_name']]
    
    # bar chart
    bars = ax.bar(range(len(eff_stats)), eff_stats['eff_mean'],
                   color=bar_colors, alpha=0.8)
    
    # 100% reference line (ideal efficiency)
    ax.axhline(y=100, color='red', linestyle='--', linewidth=2, 
               label='Ideal Efficiency(100%)', alpha=0.7)
    
    ax.set_xlabel('Configuration (Schedule Type - Chunk Size)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Parallel Efficiency (%)', fontsize=12, fontweight='bold')
    ax.set_title('Parallel Efficiency: Speedup / Number of Threads × 100\n(data near 100% indicates good scaling)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(eff_stats)))
    ax.set_xticklabels(eff_stats['label'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[sched], label=sched) 
                       for sched in ['Static', 'Dynamic', 'Guided']]
    legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', 
                                       linewidth=2, label='Efficienza Ideale'))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f" graphic saved: {output_file}")
    plt.close()

def plot_90_percentile(df_agg, output_file='percentile_90_analysis.png'):
    """
    Graph 4: 90th Percentile
    Shows the 90th percentile of times for each configuration of all threads.
    """
    p90_stats = df_agg[['threads', 'schedule_name', 'chunk_size', 'p90_ms']].copy()
    
    # sort for threads, schedule_name e chunk_size
    p90_stats = p90_stats.sort_values(['threads', 'schedule_name', 'chunk_size'])
    
    # create labels
    p90_stats['label'] = (p90_stats['threads'].astype(str) + 't-' + 
                          p90_stats['schedule_name'].str[0] + '-' +
                          p90_stats['chunk_size'])
    
    fig, ax = plt.subplots(figsize=(20, 8))
    
    #colours
    colors = {'Static': '#3498db', 'Dynamic': '#e74c3c', 'Guided': '#2ecc71'}
    bar_colors = [colors[name] for name in p90_stats['schedule_name']]
    
    bars = ax.bar(range(len(p90_stats)), p90_stats['p90_ms'],
                   color=bar_colors, alpha=0.8, width=0.8)
    
    # add values on the bars
    for i, (bar, p90) in enumerate(zip(bars, p90_stats['p90_ms'])):
        height = bar.get_height()
        if height > 0.05:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{p90:.2f}',
                    ha='center', va='bottom', fontsize=6, fontweight='bold')
    
    ax.set_xlabel('Configuration (Threads - Schedule Type - Chunk Size)', fontsize=12, fontweight='bold')
    ax.set_ylabel('90° Percentile Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('90° Percentile of the execution times \n(Indicates the time under which 90% of executions are found)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(p90_stats)))
    ax.set_xticklabels(p90_stats['label'], rotation=90, ha='center', fontsize=6)
    ax.grid(axis='y', alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[sched], label=sched) 
                       for sched in ['Static', 'Dynamic', 'Guided']]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f" Graphic saved: {output_file}")
    plt.close()

def print_summary(df, df_agg):
    """Print a summary of the main statistics."""
    print("\n" + "="*60)
    print("SUMMARY OF THE STATISTICS")
    print("="*60)
    
    print(f"\nNumber of total run: {df['run'].max()}")
    print(f"Number of threads tested: {sorted(df['threads'].unique())}")
    print(f"Number di configuration tested: {df['config'].nunique()}")
    
    # Configuration with the best avarage speedup
    best_idx = df_agg['avg_speedup'].idxmax()
    best_row = df_agg.loc[best_idx]
    print(f"\nBest Configuration (avarage speedup): {best_row['config']} with {best_row['threads']}t ({best_row['avg_speedup']:.2f}x)")
    
    # Configuration with minimum avarage time
    fastest_idx = df_agg['avg_time_ms'].idxmin()
    fastest_row = df_agg.loc[fastest_idx]
    print(f"Fastes Configuration (avarage time): {fastest_row['config']} with {fastest_row['threads']}t ({fastest_row['avg_time_ms']:.3f} ms)")
    
    # Statistic for schedule type
    print("\n" + "-"*60)
    print("Avarage Speedup for Schedule Type (all the threads):")
    print("-"*60)
    for sched in ['Static', 'Dynamic', 'Guided']:
        mean_speedup = df_agg[df_agg['schedule_name'] == sched]['avg_speedup'].mean()
        print(f"  {sched:10s}: {mean_speedup:.2f}x")
    
    print("\n" + "-"*60)
    print("Avarage Speedup for number of threads:")
    print("-"*60)
    for t in sorted(df_agg['threads'].unique()):
        mean_speedup = df_agg[df_agg['threads'] == t]['avg_speedup'].mean()
        print(f"  {t:2d} threads: {mean_speedup:.2f}x")
    
    # Top 5 configurazioni per speedup
    print("\n" + "-"*60)
    print("Top 5 configuration for avarage speedup:")
    print("-"*60)
    top5 = df_agg.nlargest(5, 'avg_speedup')[['threads', 'config', 'avg_speedup', 'avg_time_ms']]
    for idx, row in top5.iterrows():
        print(f"  {row['threads']:2d}t {row['config']:10s}: {row['avg_speedup']:.2f}x ({row['avg_time_ms']:.3f} ms)")
    
    print("\n" + "="*60 + "\n")

def main():
    if len(sys.argv) < 2:
        print("Use: python3 analyze_detail.py <file.csv>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    print(f"\n{'='*60}")
    print(f"ANALISIS OF BENCHMARK RESULTS SpMV")
    print(f"{'='*60}")
    print(f"File CSV: {csv_file}\n")
    
    # load data (single run)
    df = load_data(csv_file)
    
    # Aggregate statistics
    print("\nCalcolo statistiche aggregate...")
    df_agg = aggregate_statistics(df)
    print(f" Statistics have been calculated for {len(df_agg)} configurations")
    
    # generate all the graphs
    print("\nGeneration of graphs...")
    plot_performance_overview(df)
    plot_heatmap_chunk_size(df)
    plot_efficiency(df_agg)
    plot_90_percentile(df_agg)
    
    # SUM-UP
    print_summary(df, df_agg)
    
    print(" Analisis completed with success!")
    print(f"\nGraphs generated:")
    print("  - performance_overview.png")
    print("  - heatmap_chunk_size.png")
    print("  - efficiency.png")
    print("  - 90percent.png")

if __name__ == "__main__":
    main()