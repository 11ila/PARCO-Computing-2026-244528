#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Uso: $0 <matrix_file> [num_runs] [threads...]"
    exit 1
fi

MATRIX_FILE=$1
NUM_RUNS=${2:-50}
EXECUTABLE="./code"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MATRIX_STEM="${MATRIX_FILE%.*}"

# Se non sono specificati thread, usa default
if [ $# -le 2 ]; then
    THREADS=(1 2 4 8 16 32)
else
    shift 2
    THREADS=("$@")
fi

# Verifica eseguibile
if [ ! -f "$EXECUTABLE" ]; then
    echo "ERRORE: $EXECUTABLE non trovato!"
    echo "Compila prima con: g++ -std=c++11 -fopenmp -O3 -march=native -o code code.cpp"
    exit 1
fi

# Verifica matrice
if [ ! -f "$MATRIX_FILE" ]; then
    echo "ERRORE: File matrice $MATRIX_FILE non trovato!"
    exit 1
fi

# File output
CSV_DETAIL="benchmark_detail_${MATRIX_STEM}.csv"
TXT_LOG="benchmark_log_${MATRIX_STEM}.txt"

echo "============================================"
echo "BENCHMARK SpMV OpenMP"
echo "============================================"
echo "Matrix:        $MATRIX_FILE"
echo "Executable:    $EXECUTABLE"
echo "Runs/config:   $NUM_RUNS"
echo "Threads:       ${THREADS[@]}"
echo "Timestamp:     $TIMESTAMP"
echo "============================================"
echo ""

# Header CSV viene stampato dal C++ nella prima esecuzione
first_run=true

# Esegui benchmark per ogni configurazione di thread
for t in "${THREADS[@]}"; do
    export OMP_NUM_THREADS=$t
    
    echo ""
    echo "========================================" | tee -a "$TXT_LOG"
    echo ">>> TESTING WITH $t THREAD(S)" | tee -a "$TXT_LOG"
    echo "========================================" | tee -a "$TXT_LOG"
    echo ""
    
    # Esegui il programma
    # stdout -> CSV data, stderr -> log messages
    if [ "$first_run" = true ]; then
        # Prima esecuzione: includi header CSV
        $EXECUTABLE "$MATRIX_FILE" $NUM_RUNS >> "$CSV_DETAIL" 2>&1 | tee -a "$TXT_LOG"
        first_run=false
    else
        # Successive esecuzioni: salta header CSV (che viene stampato dal C++)
        $EXECUTABLE "$MATRIX_FILE" $NUM_RUNS 2>&1 | tee -a "$TXT_LOG" | tail -n +2 >> "$CSV_DETAIL"
    fi
    
    # Verifica che l'esecuzione sia andata a buon fine
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "[OK] Test completato per $t thread(s)"
    else
        echo "[ERROR] Test fallito per $t thread(s)"
    fi
    
    echo ""
done

echo ""
echo "============================================"
echo ">>> RIEPILOGO GLOBALE"
echo "============================================"
echo ""

# Calcola statistiche dalle singole run usando awk
echo "THREADS    | BEST CONFIG          | TIME (ms)    | SPEEDUP   "
echo "----------------------------------------------------------------"

for t in "${THREADS[@]}"; do
    # Estrai tutte le righe per questo numero di threads
    # Trova la configurazione con tempo minimo medio
    best_line=$(awk -F',' -v threads="$t" '
        NR > 1 && $2 == threads {
            key = $2 "," $3 "," $4
            sum[key] += $5
            count[key]++
            speedup[key] = $6
            config[key] = $3 "-" $4
        }
        END {
            min_avg = 999999
            for (k in sum) {
                avg = sum[k] / count[k]
                if (avg < min_avg) {
                    min_avg = avg
                    best_key = k
                }
            }
            if (best_key != "") {
                split(best_key, parts, ",")
                printf "%s,%s,%.3f,%.4f\n", parts[1], config[best_key], min_avg, speedup[best_key]
            }
        }
    ' "$CSV_DETAIL")
    
    if [ -n "$best_line" ]; then
        echo "$best_line" | awk -F',' '{printf "%-10s | %-20s | %-12s | %-10s\n", $1, $2, $3, $4"x"}'
    fi
done

echo "================================================================"
echo ""
echo "[OK] Benchmark completato con successo!"
echo ""
echo "File generati:"
echo "  - $CSV_DETAIL     (tutte le run - $(wc -l < "$CSV_DETAIL") righe)"
echo "  - $TXT_LOG        (log completo)"
echo ""
echo "Per analizzare i risultati:"
echo "  python3 analyze_detail.py $CSV_DETAIL"
echo ""
echo "============================================"