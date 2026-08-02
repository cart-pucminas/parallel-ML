# CART Tree: Comparison between sequential and oneAPI/SYCL versions

> Undergraduate Research Project - performance evaluation of *CART Decision Trees* for IDS datasets using sequential C++ and oneAPI/SYCL

**Advisor:** Henrique Cota de Freitas

---

## Overview

This folder implements and compares a **CART Decision Tree** classifier running on:

* **Sequential C++** for binary classification
* **Sequential C++** for multiclass classification
* **oneAPI/SYCL** for binary classification
* **oneAPI/SYCL** for multiclass classification

The comparison focuses on **training time** and classification metrics for intrusion detection datasets.

> Note: the sequential and SYCL versions share the same entry point and evaluation flow. The main difference is the implementation used to find the best split during tree construction.

---

## Objectives

1. Compare sequential CART execution with a parallel oneAPI/SYCL implementation.
2. Evaluate the classifier on **binary** and **multiclass** IDS datasets.
3. Collect execution time, F1 score, precision and recall in a CSV metrics file.
4. Investigate strong scaling on CPU by changing the number of oneAPI CPU compute units.

---

## Requirements

* **Compilers**

  * GCC/G++ for the sequential implementations
  * oneAPI DPC++ (`icpx`) for the SYCL implementations

* **Libraries**

  * C++ standard library
  * oneAPI/SYCL
  * oneDPL, used by the SYCL split-search implementation

* **Input datasets**

  The default script expects preprocessed CSV files at:

```bash
datasets/preprocessing/treino_classe_binaria.csv
datasets/preprocessing/teste_classe_binaria.csv
datasets/preprocessing/treino_multiclasse.csv
datasets/preprocessing/teste_multiclasse.csv
```

> The CSV reader expects training files with a header. The target label is loaded by the shared dataset utility code.

---

## Build

From inside `CART-Tree`, build all main variants with:

```bash
make
```

This produces:

```bash
linear_sliding
parallel_sliding
linear_sliding_multiclass
parallel_sliding_multiclass
```

To remove object files:

```bash
make clean
```

To remove object files and generated binaries:

```bash
make clear
```

### Sequential binary CART

```bash
g++ -O3 -c main.cc
g++ -O3 -c util.cc
g++ -O3 -c linear_sliding.cc
g++ -O3 -o linear_sliding main.o util.o linear_sliding.o
./linear_sliding <train.csv> <test.csv> <metrics.csv> -1 0
```

### Sequential multiclass CART

```bash
g++ -O3 -c main.cc
g++ -O3 -c util.cc
g++ -O3 -c linear_sliding_multiclass.cc
g++ -O3 -o linear_sliding_multiclass main.o util.o linear_sliding_multiclass.o
./linear_sliding_multiclass <train.csv> <test.csv> <metrics.csv> -1 1
```

### oneAPI/SYCL binary CART

```bash
icpx -fsycl -O3 -c parallel_sliding.cc
icpx -fsycl -O3 -o parallel_sliding main.o util.o parallel_sliding.o
./parallel_sliding <train.csv> <test.csv> <metrics.csv> -1 0
```

### oneAPI/SYCL multiclass CART

```bash
icpx -fsycl -O3 -c parallel_sliding_multiclass.cc
icpx -fsycl -O3 -o parallel_sliding_multiclass main.o util.o parallel_sliding_multiclass.o
./parallel_sliding_multiclass <train.csv> <test.csv> <metrics.csv> -1 1
```

The fourth argument controls the device/thread setting recorded in the metrics file:

* `-1`: use the SYCL CPU selector in the parallel versions
* `0`: use the SYCL GPU selector in the parallel versions
* `1`, `2`, `4`, `8`, `16`: CPU scaling settings used by `run.sh`

The fifth argument selects the task:

* `0`: binary classification
* `1`: multiclass classification

---

## Run experiments

The helper script builds the project and runs the available experiments:

```bash
./run.sh
```

Default behavior:

* runs each enabled variant once
* writes results to `metrics.csv`
* uses the default dataset paths listed above
* skips GPU runs, because the GPU commands are currently commented in the script

The script accepts optional arguments:

```bash
./run.sh [runs] [metrics.csv] [binary_train.csv] [binary_test.csv] [multi_train.csv] [multi_test.csv]
```

Example:

```bash
./run.sh 5 results.csv \
  datasets/preprocessing/treino_classe_binaria.csv \
  datasets/preprocessing/teste_classe_binaria.csv \
  datasets/preprocessing/treino_multiclasse.csv \
  datasets/preprocessing/teste_multiclasse.csv
```

If the first argument is not numeric, the script runs the CPU scaling experiment for `1`, `2`, `4`, `8` and `16` compute units using `DPCPP_CPU_NUM_CUS`.

```bash
./run.sh scale results.csv
```

---

## Output

Results are appended to the metrics CSV using the following columns:

```csv
Name, ThreadSetting, Multiple, Time, F1, Precision, Recall
```

Where:

* `Name`: executable used in the run
* `ThreadSetting`: CPU/GPU/thread setting passed to the executable
* `Multiple`: `0` for binary classification and `1` for multiclass classification
* `Time`: training time in seconds
* `F1`, `Precision`, `Recall`: evaluation metrics on the test set

---

## Expected Results (guide)

* **Sequential CART** provides the baseline for binary and multiclass classification.
* **oneAPI/SYCL CART** can reduce split-search time when the dataset has enough samples and features to amortize parallel overhead.
* **CPU scaling** should be interpreted together with memory access patterns and the cost of sorting feature values at each tree node.
* **GPU execution** depends on having a compatible SYCL GPU backend and enabling the GPU commands in `run.sh`.

> Interpret speedups considering dataset size, tree depth, transfer overheads, device selection and the cost of recursive tree construction.

---

## Contact

* **Authors:** Antônio Drumond Cota de Sousa, João Victor F. Pena- *PUC Minas / CArT*
* **Email:** adcsousa@sga.pucminas.br, joao.pena.1470628@sga.pucminas.br

> Suggestions and issues are welcome.
