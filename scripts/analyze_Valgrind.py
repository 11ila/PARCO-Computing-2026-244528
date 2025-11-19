import re
import os
import matplotlib.pyplot as plt
import numpy as np
import sys

print("=" * 80)
print("ADVANCED ANALYSIS WITH VALGRIND - Performance metrics")
print("=" * 80)

# Directory with results
results_dir = "."

print(f"\n Reading files from the current directory: {os.getcwd()}\n")

# Thread to analyze
threads = [1, 4, 16, 32]

def parse_cachegrind_file(filepath):
    """Reads a cachegrind file and extracts the data PROGRAM TOTALS"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Search for the PROGRAM TOTALS rows
        pattern = r'(\d+(?:,\d+)*)\s+(\d+(?:,\d+)*)\s+(\d+(?:,\d+)*)\s+(\d+(?:,\d+)*)\s+(\d+(?:,\d+)*)\s+(\d+(?:,\d+)*)\s+(\d+(?:,\d+)*)\s+(\d+(?:,\d+)*)\s+(\d+(?:,\d+)*)\s+PROGRAM TOTALS'
        match = re.search(pattern, content)
        
        if not match:
            return None
        
        values = [int(v.replace(',', '')) for v in match.groups()]
        
        return {
            'Ir': values[0],
            'I1mr': values[1],
            'ILmr': values[2],
            'Dr': values[3],
            'D1mr': values[4],
            'DLmr': values[5],
            'Dw': values[6],
            'D1mw': values[7],
            'DLmw': values[8]
        }
    except Exception as e:
        print(f" Error: {e}")
        return None

data_by_threads = {}

for t in threads:
    filepath = f"cachegrind_{t}t.txt"
    if not os.path.exists(filepath):
        filepath = f"cachegrind_{t}t.out"
    
    print(f" Thread Processing {t}...")
    
    if not os.path.exists(filepath):
        print(f" File not found")
        continue
    
    data = parse_cachegrind_file(filepath)
    if data:
        data_by_threads[t] = data
        print(f" extracted data")

if not data_by_threads:
    print("\n ERROR: no data extracted!")
    sys.exit(1)

sorted_threads = sorted(data_by_threads.keys())

print(f"\n Date extracted from {len(data_by_threads)} file\n")
print("=" * 80)
print("ADVANCED GRAPHICS GENERATION")
print("=" * 80)

# ============================================================================
# 1. CACHE MISS RATE (%)
# ============================================================================
print("\n1. Cache Miss Rate...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# L1 Data Cache Miss Rate
l1_miss_rate = [(data_by_threads[t]['D1mr'] / data_by_threads[t]['Dr']) * 100 
                for t in sorted_threads]

ax1.plot(sorted_threads, l1_miss_rate, marker='o', linewidth=2.5, 
         markersize=10, color='#E63946', markeredgecolor='white', 
         markeredgewidth=2)
ax1.set_xlabel('Number of Thread(s)', fontsize=13, fontweight='bold')
ax1.set_ylabel('L1 Miss Rate (%)', fontsize=13, fontweight='bold')
ax1.set_title('L1 Data Cache Miss Rate', fontsize=15, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xticks(sorted_threads)
ax1.set_facecolor('#F8F9FA')

for t, v in zip(sorted_threads, l1_miss_rate):
    ax1.annotate(f'{v:.3f}%', xy=(t, v), xytext=(0, 10),
                 textcoords='offset points', ha='center', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                          edgecolor='#E63946', alpha=0.9))

# LL Cache Miss Rate
ll_miss_rate = [(data_by_threads[t]['DLmr'] / data_by_threads[t]['Dr']) * 100 
                for t in sorted_threads]

ax2.plot(sorted_threads, ll_miss_rate, marker='o', linewidth=2.5, 
         markersize=10, color='#06A77D', markeredgecolor='white', 
         markeredgewidth=2)
ax2.set_xlabel('Number of Thread(s)', fontsize=13, fontweight='bold')
ax2.set_ylabel('LL Miss Rate (%)', fontsize=13, fontweight='bold')
ax2.set_title('Last Level Cache Miss Rate', fontsize=15, fontweight='bold', pad=20)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xticks(sorted_threads)
ax2.set_facecolor('#F8F9FA')

for t, v in zip(sorted_threads, ll_miss_rate):
    ax2.annotate(f'{v:.4f}%', xy=(t, v), xytext=(0, 10),
                 textcoords='offset points', ha='center', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                          edgecolor='#06A77D', alpha=0.9))

plt.tight_layout()
plt.savefig('cache_miss_rate.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: cache_miss_rate.png")
plt.close()

# ============================================================================
# 2. INSTRUCTIONS PER CACHE MISS
# ============================================================================
print("2. Cache Miss instruction...")

fig, ax = plt.subplots(figsize=(12, 7))

# Istruction for L1 miss
inst_per_l1_miss = [data_by_threads[t]['Ir'] / data_by_threads[t]['D1mr'] 
                    for t in sorted_threads]

# Istruction for LL miss
inst_per_ll_miss = [data_by_threads[t]['Ir'] / data_by_threads[t]['DLmr'] 
                    for t in sorted_threads]

x = np.arange(len(sorted_threads))
width = 0.35

bars1 = ax.bar(x - width/2, inst_per_l1_miss, width, label='Instr per L1 Miss',
               color='#E63946', edgecolor='white', linewidth=2)
bars2 = ax.bar(x + width/2, inst_per_ll_miss, width, label='Instr per LL Miss',
               color='#06A77D', edgecolor='white', linewidth=2)

ax.set_xlabel('Number of Thread(s)', fontsize=13, fontweight='bold')
ax.set_ylabel('Instructions for Miss', fontsize=13, fontweight='bold')
ax.set_title('Instructions Executed for Cache Misfits\n(More high = better the efficiency)', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(sorted_threads)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.set_facecolor('#F8F9FA')

#bars values
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:,.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('instructions_per_miss.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: instructions_per_miss.png")
plt.close()

# ============================================================================
# 3. OVERHEAD DI PARALLELIZZAZIONE
# ============================================================================
print("3. Parallelisation Overhead...")

baseline = data_by_threads[1]['Ir']
overhead = [(data_by_threads[t]['Ir'] - baseline) / baseline * 100 
            for t in sorted_threads]

plt.figure(figsize=(12, 7))
colors = ['#2E86AB' if o <= 0 else '#E63946' for o in overhead]

bars = plt.bar(sorted_threads, overhead, color=colors, edgecolor='white', 
               linewidth=2, alpha=0.8)

plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
plt.xlabel('Number of Thread(s)', fontsize=13, fontweight='bold')
plt.ylabel('Overhead (%)', fontsize=13, fontweight='bold')
plt.title('Parallelization Overhead compared to 1 Thread\n(Istruzioni Extra Eseguite)', 
          fontsize=15, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, axis='y', linestyle='--')
plt.xticks(sorted_threads)

ax = plt.gca()
ax.set_facecolor('#F8F9FA')

# Aggiungi valori
for i, (t, v) in enumerate(zip(sorted_threads, overhead)):
    plt.annotate(f'{v:+.2f}%', xy=(t, v), 
                 xytext=(0, 10 if v >= 0 else -20),
                 textcoords='offset points', ha='center', fontsize=10,
                 fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                          edgecolor=colors[i], alpha=0.9))

plt.tight_layout()
plt.savefig('parallelization_overhead.png', dpi=300, bbox_inches='tight', facecolor='white')
print(" Saved: parallelization_overhead.png")
plt.close()

# ============================================================================
# 4. MEMORY BANDWIDTH UTILIZATION
# ============================================================================
print("4. Memory Bandwidth Utilization...")

# Total memory accesses (Read + Write)
total_mem_ops = [data_by_threads[t]['Dr'] + data_by_threads[t]['Dw'] 
                 for t in sorted_threads]

# Total cache misses (going to main memory)
total_misses = [data_by_threads[t]['DLmr'] + data_by_threads[t]['DLmw'] 
                for t in sorted_threads]

# Percentage of accesses going to RAM
ram_access_rate = [(total_misses[i] / total_mem_ops[i]) * 100 
                   for i in range(len(sorted_threads))]

plt.figure(figsize=(12, 7))

plt.plot(sorted_threads, ram_access_rate, marker='o', linewidth=2.5, 
         markersize=12, color='#6A4C93', markeredgecolor='white', 
         markeredgewidth=2)

plt.xlabel('Number of Thread(s)', fontsize=13, fontweight='bold')
plt.ylabel('RAM Access Rate (%)', fontsize=13, fontweight='bold')
plt.title('Percentage of Memory Accesses Reaching RAM\n(Lower = Better cache efficiency)', 
          fontsize=15, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(sorted_threads)

ax = plt.gca()
ax.set_facecolor('#F8F9FA')

for t, v in zip(sorted_threads, ram_access_rate):
    plt.annotate(f'{v:.4f}%', xy=(t, v), xytext=(0, 12),
                 textcoords='offset points', ha='center', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                          edgecolor='#6A4C93', alpha=0.9))

plt.tight_layout()
plt.savefig('memory_bandwidth_utilization.png', dpi=300, bbox_inches='tight', facecolor='white')
print(" Salved: memory_bandwidth_utilization.png")
plt.close()

# ============================================================================
# 5. CACHE EFFICIENCY COMPARISON
# ============================================================================
print("5. Cache Efficiency Comparison...")

fig, ax = plt.subplots(figsize=(14, 8))

# L1 Hit Rate
l1_hit_rate = [(1 - data_by_threads[t]['D1mr'] / data_by_threads[t]['Dr']) * 100 
               for t in sorted_threads]

# LL Hit Rate (among those who missed L1)
ll_hit_rate = [(1 - data_by_threads[t]['DLmr'] / data_by_threads[t]['D1mr']) * 100 
               for t in sorted_threads]

x = np.arange(len(sorted_threads))
width = 0.35

bars1 = ax.bar(x - width/2, l1_hit_rate, width, label='L1 Hit Rate',
               color='#E63946', edgecolor='white', linewidth=2)
bars2 = ax.bar(x + width/2, ll_hit_rate, width, label='LL Hit Rate (dopo L1 miss)',
               color='#06A77D', edgecolor='white', linewidth=2)

ax.set_xlabel('Number of Thread(s)', fontsize=13, fontweight='bold')
ax.set_ylabel('Hit Rate (%)', fontsize=13, fontweight='bold')
ax.set_title('Cache Hit Rate: L1 vs Last Level\n(higher = better performance)', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(sorted_threads)
ax.legend(fontsize=12, loc='lower left')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.set_facecolor('#F8F9FA')
ax.set_ylim([99, 100])  # Focus sulla parte alta

# Aggiungi valori
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('cache_efficiency_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
print(" Saved: cache_efficiency_comparison.png")
plt.close()

# ============================================================================
# 6.SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY OF METRICS")
print("=" * 80)

print(f"\n{'Thread':<8} {'L1 Miss%':<12} {'LL Miss%':<12} {'Overhead%':<12} {'RAM Access%':<12}")
print("-" * 80)
for i, t in enumerate(sorted_threads):
    print(f"{t:<8} {l1_miss_rate[i]:<12.3f} {ll_miss_rate[i]:<12.4f} "
          f"{overhead[i]:<12.2f} {ram_access_rate[i]:<12.4f}")
print("-" * 80)

print("\n" + "=" * 80)
print("COMPLETED!")
print("=" * 80)
print(f"\n5 graph generated:")
print("  1. cache_miss_rate.png - Miss rate L1 and LL")
print("  2. instructions_per_miss.png - Code Efficiency")
print("  3. parallelization_overhead.png - Parallelization Overhead ")
print("  4. memory_bandwidth_utilization.png -  memoria RAM usage ")
print("  5. cache_efficiency_comparison.png - Hit rate cache\n")

print("EXPLANATION:")
print("-" * 80)
print("✓ L1 Miss Rate low (<1%) = Excellent location")
print("✓ LL Miss Rate very low (<0.01%) = Excellent use of cache")
print("✓ Overhead <5% = Efficient Parallelization ")
print("✓ low RAM Access Rate  = Few expensive accesses to RAM")
print("✓ High instructions for Miss  = Compute-intensive code (good!)")
print("-" * 80)