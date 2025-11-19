#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <tuple>
#include <map>
#include <string>
#include <numeric>
#include <omp.h>

using namespace std;

//------------CONFIGURAZIONE: SCHEDULING + CHUNK SIZE---------------

struct ScheduleConfig {
    string name;
    string type;  // S, D, G
    string chunk;
    string omp_clause;
};

vector<ScheduleConfig> generate_schedule_configs() {
    return {
        // STATIC
        {"Static-Default", "S", "-", "static"},
        {"Static-10", "S", "10", "static,10"},
        {"Static-100", "S", "100", "static,100"},
        {"Static-1000", "S", "1000", "static,1000"},
        
        // DYNAMIC
        {"Dynamic-Default", "D", "-", "dynamic"},
        {"Dynamic-10", "D", "10", "dynamic,10"},
        {"Dynamic-100", "D", "100", "dynamic,100"},
        {"Dynamic-1000", "D", "1000", "dynamic,1000"},
        
        // GUIDED
        {"Guided-Default", "G", "-", "guided"},
        {"Guided-10", "G", "10", "guided,10"},
        {"Guided-100", "G", "100", "guided,100"},
        {"Guided-1000", "G", "1000", "guided,1000"}
    };
}

//-----------STRUTTURA COO & CSR Matrix----------------

struct COOMatrix {
    vector<int> row, col;
    vector<double> val;
    int num_rows, num_cols, nnz;
};

struct CSRMatrix {
    vector<double> values;
    vector<int> col_indices, row_ptr;
    int num_rows, num_cols;
};

//------------TIMING------------------------

inline double now_sec() {
    return omp_get_wtime();
}

//------------I/O MATRIX MARKET (lettura mtx)------------

COOMatrix read_matrix_market(const string& filename) {
    COOMatrix coo = {vector<int>(), vector<int>(), vector<double>(), 0, 0, 0};
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "ERROR: File " << filename << " not found!" << endl;
        return coo;
    }
    
    string line;
    while (getline(file, line) && line[0] == '%');
    
    istringstream(line) >> coo.num_rows >> coo.num_cols >> coo.nnz;
    
    coo.row.reserve(coo.nnz);
    coo.col.reserve(coo.nnz);
    coo.val.reserve(coo.nnz);
    
    int r, c;
    double v;
    for (int i = 0; i < coo.nnz; ++i) {
        if (!(file >> r >> c >> v)) {
            cerr << "ERROR:  triplete(r,c,v) reading " << i + 1 << endl;
            coo.nnz = i;
            break;
        }
        coo.row.push_back(r - 1);
        coo.col.push_back(c - 1);
        coo.val.push_back(v);
    }
    
    return coo;
}

//-----------CONVERSIONE COO -> CSR--------

CSRMatrix convert_coo_to_csr(const COOMatrix& coo) {
    CSRMatrix csr;
    csr.num_rows = coo.num_rows;
    csr.num_cols = coo.num_cols;

    vector<tuple<int, int, double>> triplets;
    triplets.reserve(coo.row.size());

    for (size_t i = 0; i < coo.row.size(); ++i) {
        triplets.emplace_back(coo.row[i], coo.col[i], coo.val[i]);
    }
    
    sort(triplets.begin(), triplets.end());
    
    vector<int> counts(csr.num_rows, 0);
    for (const auto& t : triplets) {
        counts[get<0>(t)]++;
    }

    csr.row_ptr.resize(csr.num_rows + 1);
    csr.row_ptr[0] = 0;
    for (int i = 0; i < csr.num_rows; ++i) {
        csr.row_ptr[i + 1] = csr.row_ptr[i] + counts[i];
    }
    
    csr.values.resize(triplets.size());
    csr.col_indices.resize(triplets.size());
    vector<int> pos = csr.row_ptr;
    
    for (const auto& t : triplets) {
        int idx = pos[get<0>(t)]++;;
        csr.col_indices[idx] = get<1>(t);
        csr.values[idx] = get<2>(t);
    }
    
    return csr;
}

//-----------VETTORE RANDOM (x) ----------------------------------

vector<double> vec_rand(int n) {
    vector<double> x(n);
    for (int i = 0; i < n; ++i) {
        x[i] = 1.0 + (i % 10);
    }
    return x;
}

//---------------------MOLTIPLICAZIONE VET*MAT SEQUENZIALE (A*x=y)------------------

vector<double> spmv_sequential(const CSRMatrix& A, const vector<double>& x) {
    vector<double> y(A.num_rows, 0.0);
    
    for (int i = 0; i < A.num_rows; ++i) {
        for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
            y[i] += A.values[k] * x[A.col_indices[k]];
        }
    }
    return y;
}

//--------------------MOLTIPLICAZIONE VET*MAT PARALLELO OpenMP-----------------

vector<double> spmv_parallel(const CSRMatrix& A, const vector<double>& x, 
                             const string& schedule_clause) {
    vector<double> y(A.num_rows, 0.0);
    
    string schedule_type;
    int chunk_size = 0;
    
    size_t comma_pos = schedule_clause.find(',');
    if (comma_pos != string::npos) {
        schedule_type = schedule_clause.substr(0, comma_pos);
        chunk_size = stoi(schedule_clause.substr(comma_pos + 1));
    } else {
        schedule_type = schedule_clause;
    }
    
    if (schedule_type == "static") {
        if (chunk_size > 0) {
            #pragma omp parallel for schedule(static, chunk_size)
            for (int i = 0; i < A.num_rows; ++i) {
                double local_sum = 0.0;
                #pragma omp simd reduction(+:local_sum)
                for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
                    local_sum += A.values[k] * x[A.col_indices[k]];
                }
                y[i] = local_sum;
            }
        } else {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < A.num_rows; ++i) {
                double local_sum = 0.0;
                #pragma omp simd reduction(+:local_sum)
                for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
                    local_sum += A.values[k] * x[A.col_indices[k]];
                }
                y[i] = local_sum;
            }
        }
    } else if (schedule_type == "dynamic") {
        if (chunk_size > 0) {
            #pragma omp parallel for schedule(dynamic, chunk_size)
            for (int i = 0; i < A.num_rows; ++i) {
                double local_sum = 0.0;
                #pragma omp simd reduction(+:local_sum)
                for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
                    local_sum += A.values[k] * x[A.col_indices[k]];
                }
                y[i] = local_sum;
            }
        } else {
            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < A.num_rows; ++i) {
                double local_sum = 0.0;
                #pragma omp simd reduction(+:local_sum)
                for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
                    local_sum += A.values[k] * x[A.col_indices[k]];
                }
                y[i] = local_sum;
            }
        }
    } else if (schedule_type == "guided") {
        if (chunk_size > 0) {
            #pragma omp parallel for schedule(guided, chunk_size)
            for (int i = 0; i < A.num_rows; ++i) {
                double local_sum = 0.0;
                #pragma omp simd reduction(+:local_sum)
                for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
                    local_sum += A.values[k] * x[A.col_indices[k]];
                }
                y[i] = local_sum;
            }
        } else {
            #pragma omp parallel for schedule(guided)
            for (int i = 0; i < A.num_rows; ++i) {
                double local_sum = 0.0;
                #pragma omp simd reduction(+:local_sum)
                for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
                    local_sum += A.values[k] * x[A.col_indices[k]];
                }
                y[i] = local_sum;
            }
        }
    }
    
    return y;
}

//------------CALCOLO 90° PERCENTILE------------------------------------

double calculate_percentile_90(vector<double> times) {
    if (times.empty()) return 0.0;
    
    sort(times.begin(), times.end());
    size_t p90_idx = static_cast<size_t>(ceil(0.90 * times.size())) - 1;
    if (p90_idx >= times.size()) p90_idx = times.size() - 1;
    
    return times[p90_idx];
}

//-----------BENCHMARK CON OUTPUT CSV DETTAGLIATO PER TUTTE LE RUN--------------------

void benchmark(const CSRMatrix& A, const string& matrix_name, int num_runs) {
    int num_threads;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }

    // Cache del tempo sequenziale (calcolato una sola volta per num_threads)
    static map<int, double> seq_time_cache;
    double time_ref;
    
    if (seq_time_cache.find(num_threads) == seq_time_cache.end()) {
        vector<double> x = vec_rand(A.num_cols);
        double start_ref = now_sec();
        vector<double> y_ref = spmv_sequential(A, x);
        double end_ref = now_sec();
        time_ref = (end_ref - start_ref) * 1000.0;
        seq_time_cache[num_threads] = time_ref;
    } else {
        time_ref = seq_time_cache[num_threads];
    }
    
    vector<double> x = vec_rand(A.num_cols);
    vector<ScheduleConfig> configs = generate_schedule_configs();
    
    // Accumulatore dei tempi per ogni configurazione (per calcolo p90)
    static map<string, vector<double>> time_accumulator;
    
    // Stampa header CSV solo la prima volta
    static bool csv_header_printed = false;
    if (!csv_header_printed) {
        cout << "run,threads,schedule_type,chunk_size,time_ms,speedup,p90_ms" << endl;
        csv_header_printed = true;
    }
    
    // Esegui tutte le run per ogni configurazione
    for (int run = 1; run <= num_runs; ++run) {
        for (const auto& config : configs) {
            double start = now_sec();
            spmv_parallel(A, x, config.omp_clause);
            double end = now_sec();
            double t = (end - start) * 1000.0;
            double s = time_ref / t;
            
            // Accumula tempi per il 90° percentile
            string key = to_string(num_threads) + "_" + config.type + "_" + config.chunk;
            time_accumulator[key].push_back(t);
            
            // Calcola p90 corrente
            double p90 = calculate_percentile_90(time_accumulator[key]);
            
            // Stampa risultato della singola run
            cout << run << ","
                 << num_threads << ","
                 << config.type << ","
                 << config.chunk << ","
                 << fixed << setprecision(6) << t << ","
                 << setprecision(4) << s << ","
                 << setprecision(6) << p90
                 << endl;
        }
    }
    
    // Stampa su stderr il riepilogo per il log (mantenendo il formato originale)
    cerr << "\n========================================" << endl;
    cerr << ">>> TESTING WITH " << num_threads << " THREAD(S)" << endl;
    cerr << "========================================" << endl;
    cerr << "\nCONFIG         | AVG TIME (ms) | MIN TIME (ms) | MAX TIME (ms) | SPEEDUP" << endl;
    cerr << "--------------------------------------------------------------------" << endl;
    
    // Calcola statistiche per il riepilogo
    string best_config;
    double best_time = 1e9;
    double best_speedup = 0.0;
    
    for (const auto& config : configs) {
        string key = to_string(num_threads) + "_" + config.type + "_" + config.chunk;
        vector<double>& times = time_accumulator[key];
        
        double avg_time = accumulate(times.begin(), times.end(), 0.0) / times.size();
        double min_time = *min_element(times.begin(), times.end());
        double max_time = *max_element(times.begin(), times.end());
        double avg_speedup = time_ref / avg_time;
        
        string config_label = config.type + "-" + config.chunk;
        
        cerr << left << setw(15) << config_label
             << "| " << fixed << setprecision(3) << setw(13) << avg_time
             << "| " << setw(13) << min_time
             << "| " << setw(13) << max_time
             << "| " << setprecision(4) << avg_speedup << "x"
             << endl;
        
        if (avg_time < best_time) {
            best_time = avg_time;
            best_config = config_label;
            best_speedup = avg_speedup;
        }
    }
    
    cerr << "\n[OK] Best config: " << best_config 
         << " (avg: " << fixed << setprecision(3) << best_time << " ms, "
         << "speedup: " << setprecision(4) << best_speedup << "x)" << endl;
}

//-----------MAIN------------------------------------

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <matrix.mtx> [num_runs]" << endl;
        cerr << "  Default num_runs: 50" << endl;
        return 1;
    }
    
    string filename = argv[1];
    int num_runs = (argc >= 3) ? atoi(argv[2]) : 50;
    
    if (num_runs < 1) {
        cerr << "Error: num_runs must be >= 1" << endl;
        return 1;
    }
    
    cerr << "\n============================================" << endl;
    cerr << "SpMV Benchmark" << endl;
    cerr << "============================================" << endl;
    cerr << "Matrix file: " << filename << endl;
    cerr << "Number of runs: " << num_runs << endl;
    cerr << "============================================" << endl;
    
    COOMatrix coo = read_matrix_market(filename);
    
    if (coo.nnz == 0) {
        cerr << "Error: empty or invalid matrix" << endl;
        return 1;
    }
    
    cerr << "Matrix loaded:" << endl;
    cerr << "  Rows: " << coo.num_rows << endl;
    cerr << "  Cols: " << coo.num_cols << endl;
    cerr << "  NNZ:  " << coo.nnz << endl;
    
    CSRMatrix csr = convert_coo_to_csr(coo);
    
    benchmark(csr, filename, num_runs);
    
    return 0;
}