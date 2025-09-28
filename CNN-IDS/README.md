# CNN: Comparison between oneAPI (CPU) and OpenMP (CPU)

> Undergraduate Research Project – performance evaluation of *CNN* on CPU using OpenMP and oneAPI

**Advisor:** Henrique Cota de Freitas

---

## 📌 Overview

This repository implements and compares the **CNN** algorithm running on:

* **oneAPI/SYCL on CPU**
* **OpenMP on CPU** (classic parallelism)

The comparison focuses on **execution time** and **scalability (strong)**.

> ⚠️ Note: The CNN codebase is kept as unified as possible across variants, changing only the parallel primitives and *backend* details.

---

## 🎯 Objectives

1. Measure performance gains between CPU under **oneAPI** and **OpenMP**.
2. Investigate **strong scaling** (fixed problem size, increase *threads*).

---

## 🛠️ Requirements

* **Compilers**

  * oneAPI DPC++ (*icpx*) for SYCL
  * GCC for **OpenMP CPU**
  * **Libraries**: OpenMP, SYCL/oneAPI

---

## ⚙️ Build

#### oneAPI/SYCL

```bash
icpx -O3 -fsycl -ltbb cnn-oneapi-CPU.cpp -o cnn-oneapi-CPU
./cnn-oneapi-CPU
```

#### OpenMP 

```bash
g++ -O3 -fopenmp cnn-openmp-CPU -o cnn-oneapi-CPU
export OMP_NUM_THREADS = X
./cnn-oneapi-CPU
```

### Sequential
```bash
g++ -O3 cnn-seq.cpp -o cnn-seq
./cnn-seq
```
---

## ✅ Expected Results (guide)

* **oneAPI CPU** tends to outperform **OpenMP CPU** on large problems (higher parallelism and *throughput*), as long as data transfer doesn’t dominate.
* **OpenMP CPU** provides a good *baseline* and is easy to port.

> Interpret *speedups* considering allocation/copy *overheads*, memory access patterns (coalescing), and parallelization policies in each variant.

---

## 📬 Contact

* **Authors:** Rafael Rodrigues, Thiago Augusto – *Puc MINAS/ CArt*
* **E-mail:** [rafaelroliveira2003@gmail.com](mailto:rafaelroliveira2003@gmail.com) and [thadleao@gmail.com](mailto:thadleao@gmail.com)

> Suggestions and *issues* are welcome!
