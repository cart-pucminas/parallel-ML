#!/bin/bash

runs="${1:-1}"
metricsfile="${2:-metrics.csv}"
bitrainfile="${3:-datasets/preprocessing/treino_classe_binaria.csv}"
bitestfile="${4:-datasets/preprocessing/teste_classe_binaria.csv}"
multitrainfile="${5:-datasets/preprocessing/treino_multiclasse.csv}"
multitestfile="${6:-datasets/preprocessing/teste_multiclasse.csv}"

echo "Logging results to metrics.csv"
echo "Name, ThreadSetting, Multiple, Time, F1, Precision, Recall" > $metricsfile

make

if [[ "$runs" =~ ^[0-9]+$ ]]; then
    for ((i = 0; i < $runs; i++)); do
        echo ""
        echo "Run" $i
        echo ""

        sleep 2
        echo "Running parallel_sliding on CPU..."
        ./parallel_sliding $bitrainfile $bitestfile $metricsfile -1 0 >/dev/null
        #echo "Parallel sliding CPU skipped."

        sleep 2
        echo "Running parallel_sliding on GPU..."
        #./parallel_sliding $bitrainfile $bitestfile $metricsfile 0 0 >/dev/null
        echo "Parallel sliding GPU skipped."

        sleep 2
        echo "Running linear_sliding..."
        ./linear_sliding $bitrainfile $bitestfile $metricsfile -1 0 >/dev/null
        #echo "Linear sliding skipped."

        sleep 2
        echo "Running parallel_sliding_multiclass on CPU..."
        ./parallel_sliding_multiclass $multitrainfile $multitestfile $metricsfile -1 1 >/dev/null
        #echo "Parallel sliding CPU multiclass skipped."

        sleep 2
        echo "Running parallel_sliding_multiclass on GPU..."
        #./parallel_sliding_multiclass $multitrainfile $multitestfile $metricsfile 0 1 >/dev/null
        echo "Parallel sliding GPU multiclass skipped."

        sleep 2
        echo "Running linear_sliding_multiclass..."
        ./linear_sliding_multiclass $multitrainfile $multitestfile $metricsfile -1 1 >/dev/null
        #echo "Linear sliding multiclass skipped."
    done
else
    for ((i = 0; i < 10; i++)); do
        echo "Running batch" $i
        sleep 2
        export DPCPP_CPU_NUM_CUS=1
        echo "1 thread"
        sleep 2
        ./parallel_sliding $bitrainfile $bitestfile $metricsfile 1 0 >/dev/null
        sleep 2
        ./parallel_sliding_multiclass $multitrainfile $multitestfile $metricsfile 1 1 >/dev/null
        sleep 2
        export DPCPP_CPU_NUM_CUS=2
        echo "2 thread"
        ./parallel_sliding $bitrainfile $bitestfile $metricsfile 2 0 >/dev/null
        sleep 2
        ./parallel_sliding_multiclass $multitrainfile $multitestfile $metricsfile 2 1 >/dev/null
        sleep 2
        export DPCPP_CPU_NUM_CUS=4
        echo "4 thread"
        ./parallel_sliding $bitrainfile $bitestfile $metricsfile 4 0 >/dev/null
        sleep 2
        ./parallel_sliding_multiclass $multitrainfile $multitestfile $metricsfile 4 1 >/dev/null
        sleep 2
        export DPCPP_CPU_NUM_CUS=8
        echo "8 thread"
        ./parallel_sliding $bitrainfile $bitestfile $metricsfile 8 0 >/dev/null
        sleep 2
        ./parallel_sliding_multiclass $multitrainfile $multitestfile $metricsfile 8 1 >/dev/null
        sleep 2
        export DPCPP_CPU_NUM_CUS=16
        echo "16 thread"
        ./parallel_sliding $bitrainfile $bitestfile $metricsfile 16 0 >/dev/null
        sleep 2
        ./parallel_sliding_multiclass $multitrainfile $multitestfile $metricsfile 16 1 >/dev/null
    done
fi


echo ""
echo "___ Results: ___"
echo ""
cat $metricsfile
