#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <random>
#include <ctime>
#include <iomanip>
#include <cstring>

using namespace std;

// Structure to represent a non-zero element of the matrix
struct MatrixElement {
    int row;
    int col;
    double value;
};

// Function to generate a random sparse matrix
vector<MatrixElement> generateSparseMatrix(int rows, int cols, int nnz) {
    vector<MatrixElement> elements;
    elements.reserve(nnz);
    
    // Random number generator
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> rowDist(1, rows);
    uniform_int_distribution<> colDist(1, cols);
    uniform_real_distribution<> valDist(-100.0, 100.0);
    
    // Set to keep track of already used positions
    set<pair<int, int>> usedPositions;
    
    while (static_cast<int>(elements.size()) < nnz) {
        int row = rowDist(gen);
        int col = colDist(gen);
        
        // Verify that the position has not already been used
        if (usedPositions.find({row, col}) == usedPositions.end()) {
            double value = valDist(gen);
            elements.push_back({row, col, value});
            usedPositions.insert({row, col});
        }
    }
    
    return elements;
}

// Function to save the matrix in Matrix Market (.mtx) format
bool saveMatrixMTX(const string& filename, int rows, int cols, 
                   const vector<MatrixElement>& elements) {
    ofstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return false;
    }
    
    // Matrix Market format header
    file << "%%MatrixMarket matrix coordinate real general\n";
    file << "% Auto-generated - Weak Scaling Matrix\n";
    
    // Matrix dimensions and number of non-zero elements
    file << rows << " " << cols << " " << elements.size() << "\n";
    
    // Matrix elements (row col value)
    file << fixed << setprecision(6);
    for (const auto& elem : elements) {
        file << elem.row << " " << elem.col << " " << elem.value << "\n";
    }
    
    file.close();
    return true;
}

void printUsage(const char* progName) {
    cout << "Usage: " << progName << " [mode] [args...]" << endl;
    cout << endl;
    cout << "Modes:" << endl;
    cout << "  1. Batch mode (generates all weak scaling matrices):" << endl;
    cout << "     " << progName << " batch" << endl;
    cout << endl;
    cout << "  2. Custom mode (generates a single matrix):" << endl;
    cout << "     " << progName << " random <rows> <avg_nnz_per_row>" << endl;
    cout << "     Output: synthetic_random_<rows>.mtx" << endl;
    cout << endl;
    cout << "Examples:" << endl;
    cout << "  " << progName << " batch" << endl;
    cout << "  " << progName << " random 10000 5" << endl;
}

void generateBatchMatrices() {
    // Configuration of matrices according to the weak scaling scheme
    struct MatrixConfig {
        int processes;
        int rows;
        int nnz_target;
        string filename;
    };
    
    vector<MatrixConfig> configs = {
        {1,   10000,   50000,   "synthetic_random_10K.mtx"},
        {2,   20000,   100000,  "synthetic_random_20K.mtx"},
        {4,   40000,   200000,  "synthetic_random_40K.mtx"},
       {8,   80000,   400000,  "synthetic_random_80K.mtx"},
        {16,  160000,  800000,  "synthetic_random_160K.mtx"},
        {32,  320000,  1600000, "synthetic_random_320K.mtx"},
        {64,  640000,  3200000, "synthetic_random_640K.mtx"},
        {128, 1280000, 6400000, "synthetic_random_1280K.mtx"},
        {256, 2560000, 12800000, "synthetic_random_2560K.mtx"}
    };
    
    cout << "=== Weak Scaling Matrix Generation (Batch Mode) ===" << endl;
    cout << "Format: Matrix Market (.mtx)" << endl << endl;
    
    for (const auto& config : configs) {
        cout << "Generating matrix for " << config.processes << " processes..." << endl;
        cout << "  Dimensions: " << config.rows << " x " << config.rows << endl;
        cout << "  NNZ Target: " << config.nnz_target << endl;
        cout << "  File: " << config.filename << endl;
        
        auto startTime = clock();
        
        // Generate the sparse matrix
        vector<MatrixElement> elements = generateSparseMatrix(
            config.rows, 
            config.rows, 
            config.nnz_target
        );
        
        // Save in MTX format
        if (saveMatrixMTX(config.filename, config.rows, config.rows, elements)) {
            auto endTime = clock();
            double elapsed = double(endTime - startTime) / CLOCKS_PER_SEC;
            
            cout << "  Completed in " << fixed << setprecision(2) 
                 << elapsed << " seconds" << endl;
            cout << "  NNZ generated: " << elements.size() << endl;
        } else {
            cout << "  Error during generation" << endl;
        }
        
        cout << endl;
    }
    
    cout << "=== Generation Complete ===" << endl;
    cout << "Total files generated: " << configs.size() << endl;
}

void generateCustomMatrix(int rows, int avg_nnz_per_row) {
    // Calculate the total number of non-zero elements
    int total_nnz = rows * avg_nnz_per_row;
    
    // Output file name
    string filename = "synthetic_random_" + to_string(rows) + ".mtx";
    
    cout << "=== Custom Matrix Generation ===" << endl;
    cout << "Dimensions: " << rows << " x " << rows << endl;
    cout << "Avg NNZ per row: " << avg_nnz_per_row << endl;
    cout << "Total NNZ: " << total_nnz << endl;
    cout << "File: " << filename << endl;
    cout << endl;
    
    auto startTime = clock();
    
    // Generate the sparse matrix
    vector<MatrixElement> elements = generateSparseMatrix(rows, rows, total_nnz);
    
    // Save in MTX format
    if (saveMatrixMTX(filename, rows, rows, elements)) {
        auto endTime = clock();
        double elapsed = double(endTime - startTime) / CLOCKS_PER_SEC;
        
        cout << " Completed in " << fixed << setprecision(2) 
             << elapsed << " seconds" << endl;
        cout << "NNZ generated: " << elements.size() << endl;
        cout << endl;
        cout << "File created: " << filename << endl;
    } else {
        cerr << " Error during file generation" << endl;
        exit(1);
    }
}

int main(int argc, char* argv[]) {
    // No arguments or "batch" → batch mode
    if (argc == 1 || (argc == 2 && strcmp(argv[1], "batch") == 0)) {
        generateBatchMatrices();
        return 0;
    }
    
    // Custom mode: random <rows> <avg_nnz_per_row>
    if (argc == 4 && strcmp(argv[1], "random") == 0) {
        try {
            int rows = stoi(argv[2]);
            int avg_nnz_per_row = stoi(argv[3]);
            
            if (rows <= 0 || avg_nnz_per_row <= 0) {
                cerr << "Error: rows and avg_nnz_per_row must be positive" << endl;
                return 1;
            }
            
            generateCustomMatrix(rows, avg_nnz_per_row);
            return 0;
            
        } catch (const exception& e) {
            cerr << "Parameter error: " << e.what() << endl;
            printUsage(argv[0]);
            return 1;
        }
    }
    
    // Invalid arguments
    cerr << "Error: invalid arguments" << endl;
    cout << endl;
    printUsage(argv[0]);
    return 1;
}