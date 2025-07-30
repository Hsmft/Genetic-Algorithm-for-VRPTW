# Genetic Algorithm for the Vehicle Routing Problem with Time Windows (VRPTW)

![Language](https://img.shields.io/badge/language-C%2B%2B-blue.svg)

This repository contains a C++ implementation of a **Genetic Algorithm (GA)** designed to solve the **Vehicle Routing Problem with Time Windows (VRPTW)**. This project aims to find high-quality, near-optimal solutions by evolving a population of candidate solutions over multiple generations.

---

## 📋 Project Overview

The Vehicle Routing Problem with Time Windows (VRPTW) is a classic NP-hard optimization problem. This solver finds effective solutions using a Genetic Algorithm, a metaheuristic inspired by the process of natural selection.

The solver implements the following workflow:

1.  **Representation:** Each solution (chromosome) in the population is represented as a single sequence (a permutation) of all customer IDs.
2.  **Initial Population:** A diverse initial population is generated using a mix of greedy heuristics (e.g., farthest customer, earliest due date) and purely random permutations.
3.  **Fitness Evaluation:** The fitness of each chromosome is determined by a decoding function that converts the customer sequence into a set of feasible routes. The objective is to minimize the number of vehicles first, and then the total travel distance. A repair mechanism is used to handle infeasible solutions generated during crossover and mutation.
4.  **Evolutionary Cycle:** The algorithm evolves the population using the following genetic operators:
    * **Selection:** **Tournament Selection** is used to choose parent individuals for reproduction.
    * **Crossover:** **Ordered Crossover (OX)** is employed to create offspring by combining segments from two parent chromosomes while preserving the relative order of customers.
    * **Mutation:** **Inversion Mutation** is used to introduce new genetic material by reversing a random subsequence within a chromosome.
5.  **Elitism:** The best solutions from the current generation are automatically carried over to the next, ensuring that the quality of the population does not decrease over time.

---

## 🛠️ Technologies Used

* **Language:** C++ (utilizing C++11 features like `<chrono>` and `<random>`)
* **Libraries:** C++ Standard Library only. No external optimization libraries were used.

---

## 🚀 How to Compile and Run

### Compilation
You can compile the source code using a standard C++ compiler like g++.

```bash
g++ -std=c++11 -o solver GENETIC.cpp
```

### Execution
The program is run from the command line with the following arguments:

```bash
./solver [instance-file-path] [max-execution-time] [max-evaluations]
```
* `instance-file-path`: The path to the problem instance file (e.g., `instances/800-rh-61.txt`).
* `max-execution-time`: The maximum run time in seconds. Use `0` for no time limit.
* `max-evaluations`: The maximum number of objective function evaluations. Use `0` for no limit.

**Example:**
```bash
./solver instances/800-rh-61.txt 60 0
```
This command runs the solver on the `800-rh-61.txt` instance for a maximum of 60 seconds.

---

## 📄 License
This project is licensed under the custom **Hesameddin Fathi Non‑Commercial & Academic Co‑Authorship License 1.0**. Please see the [LICENSE](./LICENSE) file for full details.
