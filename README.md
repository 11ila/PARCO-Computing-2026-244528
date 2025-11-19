# Sparse Matrix-Vector Multiplication (SpMV)
## Parallel Performance Analysis

**Author:** Ilaria Basanisi  
**Student ID:** 244528  
**Email:** ilaria.basanisi@studenti.unitn.it

---

## Overview

This project implements and benchmarks parallel sparse matrix-vector multiplication (SpMV) using OpenMP. The implementation compares different scheduling strategies (static, dynamic, guided) with various chunk sizes across multiple thread configurations. Test matrices are taken from the SuiteSparse collection with different sparsity patterns and sizes.

---

## Repository Structure

```
repo/
├── README.md
├── src/
│   └── code.cpp                    # Main implementation
├── scripts/
│   ├── bash_code.sh                # Local execution script
│   ├── run_complete.pbs            # Cluster job submission
│   ├── analyze_detail.py           # Results analysis 
│   ├── analyze_Valgrind.py         # Results analysis (Valgrind)
│   └── matrix/
│       ├── matrix.txt              # Matrix list
│       ├── node_type.xlsx          # List of node where each matrix run
│       ├── 10K.mtx                 # matrix file
│       ├── 20K.mtx
│       ├── 40K.mtx
│       ├── 50K.mtx
│       ├── 62K.mtx
│       ├── 90K.mtx
│       └── 500K.mtx
├── results/                        # Benchmark outputs
│   ├── 10/                         # results of the 10K matrix run
│   │   ├── output_spmv.log
│   │   ├── benchmark_detail_10.csv
│   │   ├── benchmark_log_10.txt              
│   ├── 20/                         # results of the 20K matrix run
│   │   ├── output_spmv.log
│   │   ├── benchmark_detail_20.csv
│   │   └── benchmark_log_20.txt
│   ├── 40/                         # results of the 40K matrix run
│   ├── 50/                         # results of the 50K matrix run
│   ├── 62/                         # results of the 620K matrix run
│   ├── 90/                         # results of the 90K matrix run
│   └── 500/                        # results of the 500K matrix run
└── plots/                          # Performance figures
    ├── 10/
    │   ├── efficiency.png
    │   ├── heatmap_chunk_size.png
    │   └── performance_overview.png
    │   └── percentile_90_analysis.png
    │   └── cache_efficiency_comparison.png
    │   └── cache_miss_rate.png
    │   └── memory_bandwidth_utilization.png
    │   └── parallelization_overhead.png
    │   └── instructions_per_miss.png
    ├── 20/
    ├── 40/
    ├── 50/
    ├── 62/
    ├── 90/
    └── 500/
```

---

## Code Overview

### Input
- **Format:** Matrix Market (.mtx) sparse matrices
- **Available matrices:** 10K, 20K, 40K, 50K, 62K, 90K, 500K

NB: In the github repository the files 90K.mtx and 500K.mtx are not present because they are too heavy

### Output
For each run, the program generates:

**two log files:**
- `error_spmv.log` - Error messages
- `output_spmv.log` - Standard output

**Results directory containing:**
- `benchmark_detail.csv` - All run timings (run,threads,schedule_type,chunk_size,time_ms,speedup,p90_ms)
- `INDEX.txt` - scheme of the generated files
- `benchmark_log.txt` - Statistical summary
- `benchmark_output.txt` - check summary
- `valgrind/` - Memory profiling (1, 4, 16, 32 threads) (create 3 file for each thread: cachegrind.txt - memcheck.log - cachegrind.out)

---

## Implementation Details

### Key Functions

#### `COOMatrix read_matrix_market(const string& filename)` (line 68)
- Reads Matrix Market (.mtx) files
- Returns matrix in COO format

#### `CSRMatrix convert_coo_to_csr(const COOMatrix& coo)` (line 102)
- Converts COO to CSR format
- Optimizes for SpMV operations

#### `vector<double> spmv_sequential(const CSRMatrix& A, const vector<double>& x)` (line 150)
- Sequential baseline implementation
- Multiplies CSR matrix by random vector (A*x)

#### `vector<double> spmv_parallel(const CSRMatrix& A, const vector<double>& x, const string& schedule_clause)` (line 162)
- Parallel SpMV with OpenMP
- Tests different scheduling strategies and chunk sizes

---

## Compilation

### Required Modules (Cluster)
```bash
module load gcc91
module load python-3.10.14_gcc91
```

### Compiler Flags
```bash
-O3 -fopenmp -std=c++11 -march=native
```

### Local Compilation
```bash
$ g++ -O3 -fopenmp -std=c++11 -march=native code.cpp -o code
$ bash bash_code.sh 10K.mtx
```

> **Note:** `code.cpp`, `code` (executable) and `10K.mtx` must be in the same directory

### Cluster Execution
```bash
$ qsub run_complete.pbs
```

---

## Configuration Parameters

### Matrix Selection
To change the matrix, modify line 12 in the PBS file:
```bash
MATRIX_FILE="10K.mtx"
```

### Walltime
To change the walltime, modify line 4 in the PBS file:
```bash
#PBS -l walltime=02:00:00
```
(currently set to 2 hours)

---

## Results Analysis

The `analyze_detail.py` script generates performance plots from benchmark data.

### Generated Outputs
- `efficiency.png` - Parallel efficiency
- `heatmap_chunk_size.png` - Performance heatmap
- `performance_overview.png` - Time and speedup vs threads
- `percentile_90_analysis.png` - schedule & chucksize vs 90th percentile (time)


### Usage

 Specify the file in the terminal
```bash
$ python3 analyze_benchmark.py benchmark_detail_10.csv
```

The `analyze_Valgrind.py` script generates performance plots from valgrind data.

### Generated Outputs
- `cache_efficiency_comparison.png` - L1 vs LL hit rate
- `cache_miss_rate.png` - L1 and LL miss rate per thread
- `instruction_per_miss.png` - instruction per miss vs num of thread
- `memory_bandwidth_utilization.png` - RAM access rate per num of thread
- `parallelization_overhead.png` - overhead of parallelization per num of thread

### Usage

 Specify the directory in the terminal
```bash
$ python3 analyze_benchmark.py .
```
NB:the file: analyze_benchmark.py has to be in the directory (valgrind)
