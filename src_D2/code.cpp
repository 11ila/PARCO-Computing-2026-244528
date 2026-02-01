#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <unordered_set>
#include <unordered_map>

using namespace std;

//-----------STRUCTURE: COO & CSR Matrix---------------------------------

struct CSRMatrix {
    int rows = 0, cols = 0, global_rows = 0;
    vector<int> rowptr, col;
    vector<double> val;
};

//Performance metrics container
struct Metrics {
    double spmv_time, comm_time, comp_time, total_time;
    long long flops, memory_bytes;
    int local_nnz;
};

//-------------INDEX MAPPING AND PARTIONING UTILS-----------------------------

// Cyclic distribution mapping
inline bool is_my_row(int global_row, int rank, int size) { return (global_row % size) == rank; }
inline int get_owner(int global_col, int size) { return global_col % size; }
inline int global_to_local(int global_idx, int size) { return global_idx / size; }


//----------------------READ MATRIX MARKET FILE--------------------------

CSRMatrix read_matrix_distributed(const string& file, int rank, int size) {
    CSRMatrix local;
    ifstream fin(file);
    if (!fin) return local;

    string line;
    while (getline(fin, line) && line[0] == '%'); // Skip Market Market header comments

    int M, N, nnz_total;
    stringstream ss(line);
    ss >> M >> N >> nnz_total;
    
    local.global_rows = M;
    local.cols = N;
    //Calculate local row count based on cyclic distribution
    local.rows = M / size + (rank < (M % size) ? 1 : 0);
    local.rowptr.assign(local.rows + 1, 0);

    vector<vector<pair<int, double>>> temp(local.rows);
    int r, c; double v;
    while (fin >> r >> c >> v) {
        r--; c--; // Convert 1-based Market Market to 0-based
        if (is_my_row(r, rank, size)) temp[global_to_local(r, size)].push_back({c, v});
    }

    // Convert temporary storage to CSR format
    for (int i = 0; i < local.rows; i++) {
        local.rowptr[i+1] = local.rowptr[i] + temp[i].size();
        for (auto& p : temp[i]) {
            local.col.push_back(p.first);
            local.val.push_back(p.second);
        }
    }
    return local;
}

//----------------------PARALLEL SPMV EXECUTION--------------------------

class OptimizedSpMV {
private:
    const CSRMatrix& A;
    const vector<double>& x_local;
    int rank, size;
    vector<int> send_counts, recv_counts, sdispls, rdispls, recv_local_pos, x_idx;
    vector<char> x_is_remote;
    vector<double> send_buf, recv_buf;

public:
    OptimizedSpMV(const CSRMatrix& A_, const vector<double>& x_local_, int rank_, int size_)
        : A(A_), x_local(x_local_), rank(rank_), size(size_) {
        send_counts.resize(size, 0); recv_counts.resize(size, 0);
        sdispls.resize(size, 0); rdispls.resize(size, 0);
    }

    //Identify which remote elements of vector 'x' are needed for local rows 
    // and establish the Alltoallv communication schedule
    void setup() {
        vector<unordered_set<int>> needed(size);
        for (int j : A.col) {
            int owner = get_owner(j, size);
            if (owner != rank) needed[owner].insert(j);
        }

        vector<int> send_indices;
        unordered_map<int, int> remote_map;
        int current_remote_idx = 0;

        for (int p = 0; p < size; p++) {
            send_counts[p] = needed[p].size();
            for (int idx : needed[p]) {
                send_indices.push_back(idx);
                remote_map[idx] = current_remote_idx++;
            }
        }

        // Exchange counts to define the receive buffer structure
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        
        /*int stot = 0, rtot = 0;
        for (int p = 0; p < size; p++) {
            sdispls[p] = stot; rdispls[p] = rtot;
            stot += send_counts[p]; rtot += recv_counts[p];
            sdispls[p] = (p == 0) ? 0 : sdispls[p-1] + send_counts[p-1]; // Correzione sdispls
        }*/

        // Calculate displacements for the Alltoallv buffers
        sdispls[0] = 0; rdispls[0] = 0;
        for(int p=1; p<size; p++) {
            sdispls[p] = sdispls[p-1] + send_counts[p-1];
            rdispls[p] = rdispls[p-1] + recv_counts[p-1];
        }
        int total_recv = rdispls[size-1] + recv_counts[size-1];
        vector<int> recv_indices(total_recv);

        // Exchange indices of vector x to finalize the mapping
       // vector<int> recv_indices(rdispls[size-1] + recv_counts[size-1]);
        MPI_Alltoallv(send_indices.data(), send_counts.data(), sdispls.data(), MPI_INT,
                      recv_indices.data(), recv_counts.data(), rdispls.data(), MPI_INT, MPI_COMM_WORLD);

        recv_local_pos.resize(recv_indices.size());
        for (size_t i = 0; i < recv_indices.size(); i++) 
            recv_local_pos[i] = global_to_local(recv_indices[i], size);

        send_buf.resize(recv_indices.size());
        recv_buf.resize(send_indices.size());

        // Optimize row computation by pre-identifying local vs remote access
        x_is_remote.resize(A.col.size());
        x_idx.resize(A.col.size());
        for (size_t k = 0; k < A.col.size(); k++) {
            int j = A.col[k];
            if (get_owner(j, size) == rank) {
                x_is_remote[k] = 0; x_idx[k] = global_to_local(j, size);
            } else {
                x_is_remote[k] = 1; x_idx[k] = remote_map[j];
            }
        }
    }

    void run(vector<double>& y, double& comm_t, double& comp_t) {
        double t_start = MPI_Wtime();
        // Pack send buffer
        for (size_t i = 0; i < recv_local_pos.size(); i++) 
            send_buf[i] = x_local[recv_local_pos[i]];

        // Perform non-blocking-like collective communication
        MPI_Alltoallv(send_buf.data(), recv_counts.data(), rdispls.data(), MPI_DOUBLE,
                      recv_buf.data(), send_counts.data(), sdispls.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        double t_mid = MPI_Wtime();
        
        
        // CSR Matrix Product
        for (int i = 0; i < A.rows; i++) {
            double sum = 0;
            for (int k = A.rowptr[i]; k < A.rowptr[i+1]; k++) {
                sum += A.val[k] * (x_is_remote[k] ? recv_buf[x_idx[k]] : x_local[x_idx[k]]);
            }
            y[i] = sum;
        }
        double t_end = MPI_Wtime();
        comm_t += (t_mid - t_start);
        comp_t += (t_end - t_mid); 
    }
};

//----------------------MAIN PROGRAM--------------------------

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) cout << "Usage: " << argv[0] << " <matrix.mtx>" << endl;
        MPI_Finalize(); return 0;
    }

    // Step 1: Matrix Loading
    CSRMatrix A = read_matrix_distributed(argv[1], rank, size);
    vector<double> x_local(A.rows, 1.0);
    vector<double> y_local(A.rows, 0.0);

    // Step 2: SpMV Setup (Communication Patterns)
    OptimizedSpMV spmv(A, x_local, rank, size);
    spmv.setup();

    double comm_t = 0, comp_t = 0;
    //Warmup run to initialize caches and MPI buffers (doesn't count in timing)
    double dummy_comm = 0, dummy_comp = 0;
    spmv.run(y_local, dummy_comm, dummy_comp);

    // Step 3: Performance Benchmark (Average over 50 iterations)
    double t0 = MPI_Wtime();
    for(int i=0; i<50; i++) spmv.run(y_local, comm_t, comp_t);
    double t1 = MPI_Wtime();

    double avg_total = (t1 - t0) / 50.0;
    double avg_comm = comm_t / 50.0;
    double avg_comp = comp_t / 50.0;
    
    // Step 4: Output for PBS Log Extraction
    if (rank == 0) {
        cout << "Matrix: " << argv[1] << endl;
        cout << "Strategy: OPTIMIZED ALLTOALLV" << endl;
        cout << "SpMV time: " << avg_total * 1000.0 << " ms" << endl; 
        cout << "Comp(ms): " << avg_comp * 1000.0 << endl;
        cout << "Comm(ms): " << avg_comm * 1000.0 << endl;
    }

   // Step 50: CSV Logging for Strong Scaling Analysis
    if (rank == 0) {
        const string csv_filename = "weak_scaling_results.csv";
        
        
        ifstream check_file(csv_filename);
        bool file_exists = check_file.good();
        check_file.close();

        ofstream csv(csv_filename, ios::app);
        

        if (!file_exists) {
            csv << "matrix,type,procs,time,comp,comm,speedup,eff,flops,mem,imb,min,max" << endl;
        }

        // Clean matrix name for CSV (remove path and extension)
        string m_name = argv[1];
        size_t last_dot = m_name.find_last_of(".");
        if (last_dot != string::npos) m_name = m_name.substr(0, last_dot);

        csv << m_name << ",opt," << size << "," 
            << avg_total * 1000.0 << "," 
            << avg_comp * 1000.0 << "," 
            << avg_comm * 1000.0 << ","
            << "0,0,0,0,0,0,0" << endl; 
        
        csv.close();
    }

    MPI_Finalize();
    return 0;
}