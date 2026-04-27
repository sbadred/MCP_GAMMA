# Minimal Complete Pool (MCP)   Verifier

This repository contains the core C++ and CUDA-accelerated routines for evaluating the completeness and minimality of Pauli string operator pools. It provides the mathematical verification backend for the framework introduced in the paper on constructing Lie-algebra generator pools for Variational Quantum Eigensolvers (VQE).

This code allows researchers to systematically verify two critical properties of any given Pauli operator pool:
1. **The Rank Condition:** Evaluates the rank of the anti-commutation adjacency matrix ($\Gamma_\mathcal{A}$) over the binary field ($\mathbb{F}_2$) to guarantee theoretical completeness ($2N-4$).
2. **Bracket Independence**.
3. **Generates the Lie Algebra**
4. **Generates the Product Group**
   
## 📖 Citation
If you use this code in your research, please cite our associated manuscript:

> **An Optimal Framework for Constructing Lie-algebra Generator Pools: Application to variational quantum eigensolvers for Chemistry.** > *Yaromir Viswanathan, Olivier Adjoua, César Féniou, Siwar Badreddine, and Jean-Philip Piquemal.* (2026).

*(Note: The full quantum chemistry simulations presented in the paper were performed using the proprietary Hyperion emulator. This repository isolates the open-source algebraic pool verification framework.)*

---

## 📂 Repository Structure
* `CheckGamma.h` / `CheckGamma.cu` : The highly optimized, CUDA-accelerated backend for generating binary matrices and reducing them over $\mathbb{F}_2$.
* `main.cc` : The main execution script containing the wrapper functions and the user-defined pool definitions (e.g., the H8 molecule starters).
* `job_mcp.sh` : (Optional) Slurm/batch script for cluster job submission.

---

## ⚙️ Compilation and Execution

This repository utilizes GPU acceleration for matrix reductions over the binary field ($\mathbb{F}_2$). Therefore, the NVIDIA CUDA Compiler (`nvcc`) is required to compile the backend routines.

### 1. Environment Setup
If you are running this code on a High-Performance Computing (HPC) cluster, you will first need to load the appropriate NVIDIA HPC SDK module. For example:
```bash
module load Core/Portland/nvhpc/24.7```

### 2. Compiling the Source Code

Compile the main execution routine (main.cc) alongside the CUDA-accelerated backend (CheckGamma.cu) into a single executable named main:
```bash
nvcc -o main main.cc CheckGamma.cu

### 3.Running the Verification
You can execute the program directly by running ./main.
