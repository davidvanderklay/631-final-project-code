# CSCE 631 Project: Comparative Analysis of CFR Variants in Leduc Hold'em

This repository contains the implementation and benchmarking framework for comparing deterministic and sampling-based Counterfactual Regret Minimization (CFR) agents in Leduc Hold'em.

## Project Structure

* `leduc_game.py`: The core game engine implementing Leduc Hold'em rules.
* `exploitability.py`: Algorithm to calculate the exact Best Response and Exploitability metric.
* `cfr_solvers.py`: Implementations of Vanilla CFR, CFR+, and Pruning CFR.
* `mccfr_solvers.py`: Implementations of External Sampling and Outcome Sampling MCCFR.
* `experiment_runner.py`: The main driver script to run the benchmark with time/memory tracking.
* `generate_plots.py`: Visualization script to generate the analysis graphs.

## Setup Instructions

### 1. Prerequisites

Ensure you have Python 3.8 or higher installed.

### 2. Create a Virtual Environment

It is recommended to use a virtual environment to manage dependencies.

**macOS/Linux:**

```
python3 -m venv venv
source venv/bin/activate
```

**Windows:**

```
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

Install the required libraries using the provided requirements file.

```
pip install -r requirements.txt
```

## Running the Experiments

To replicate the results presented in the report, follow these steps:

### Step 1: Verify Game Logic (Optional)

Run the game engine directly to ensure the rules and state transitions are working correctly.

```
python leduc_game.py
```

### Step 2: Run the Benchmark

This script will train all 5 agents for a set time budget (default 30s per agent). It tracks wall-clock time, nodes touched, memory usage, and exploitability.

```
python experiment_runner.py
```

*Output: This will generate a file named `cfr_benchmark_results.csv` containing the raw data.*

### Step 3: Generate Plots

Once the CSV file is generated, run the visualization script to produce the figures for the report.

```
python generate_plots.py
```

*Output: This will create four images:*

* `plot_1_time_convergence.png`
* `plot_2_node_efficiency.png`
* `plot_3_pruning_speedup.png`
* `plot_4_memory_profile.png`
