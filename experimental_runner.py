import time
import tracemalloc
import pandas as pd
import numpy as np
from collections import defaultdict

from leduc_game import LeducGame
from exploitability import ExploitabilityCalculator
from cfr_solvers import VanillaCFR, CFRPlus, PruningCFR
from mccfr_solvers import ExternalSamplingMCCFR, OutcomeSamplingMCCFR


class ExperimentRunner:
    def __init__(self, game_class, time_budget=60, eval_interval=5.0):
        """
        game_class: Class reference to instantiate new games.
        time_budget: Max seconds to train.
        eval_interval: Seconds between exploitability checks.
        """
        self.game_class = game_class
        self.time_budget = time_budget
        self.eval_interval = eval_interval
        self.results = []

    def estimate_nodes_touched(self, solver_name, iterations):
        """
        Rough estimates of nodes touched per iteration for Leduc Hold'em.
        Used for logging
        """
        # Full tree walk roughly ~1300-1500 nodes per player pass
        if "Vanilla" in solver_name or "CFR+" in solver_name:
            return iterations * 2800
        # Pruning skips branches, so this is upper bound
        elif "Pruning" in solver_name:
            return iterations * 2800  # Treating as upper bound
        # ES branches on player (a few) and samples opponent
        elif "External" in solver_name:
            return iterations * 150  # Approx avg nodes visited per ES iter
        # OS samples single trajectory
        elif "Outcome" in solver_name:
            return iterations * 8  # Avg depth of game
        return 0

    def run_solver(self, solver_name, solver_class, solver_kwargs, n_runs=1):
        print(f"\n>>> Running {solver_name} ({n_runs} runs) <<<")

        for run_id in range(n_runs):
            print(f"  Run {run_id + 1}/{n_runs}...")

            # 1. Setup
            # Handle seeding MCCFR
            if (
                "seed" in solver_kwargs
                or "Outcome" in solver_name
                or "External" in solver_name
            ):
                # Varies seed per run
                current_kwargs = solver_kwargs.copy()
                current_kwargs["seed"] = 42 + run_id
            else:
                current_kwargs = solver_kwargs

            game = self.game_class()
            solver = solver_class(game, **current_kwargs)
            calc = ExploitabilityCalculator(game)

            # 2. Init Metrics
            start_time = time.time()
            training_time_accumulated = 0.0
            next_eval_time = 0.0

            # Determine block size based on solver type
            # MCCFR is fast, so run big blocks. Vanilla slow, small blocks.
            if "Outcome" in solver_name:
                iter_block = 2000
            elif "External" in solver_name:
                iter_block = 500
            else:
                iter_block = 10

            # Memory Tracking
            tracemalloc.start()

            # 3. Training Loop
            while training_time_accumulated < self.time_budget:
                # TRAIN BLOCK (Timed)
                t0 = time.time()
                solver.train(iter_block)
                dt = time.time() - t0
                training_time_accumulated += dt

                # EVALUATION
                if training_time_accumulated >= next_eval_time:
                    # Memory snapshot
                    current_mem, peak_mem = tracemalloc.get_traced_memory()

                    # Compute exploitability
                    profile = solver.get_average_strategy_profile()
                    exp = calc.compute_exploitability(profile)

                    nodes = self.estimate_nodes_touched(solver_name, solver.iterations)

                    # Log
                    self.results.append(
                        {
                            "Solver": solver_name,
                            "Run_ID": run_id,
                            "Wall_Clock_Time": round(training_time_accumulated, 2),
                            "Iterations": solver.iterations,
                            "Nodes_Touched_Est": nodes,
                            "Exploitability": exp,
                            "Peak_Memory_MB": peak_mem / (1024 * 1024),
                        }
                    )

                    # Print status for sanity check (overwrite line)
                    print(
                        f"    Time: {training_time_accumulated:.1f}s | Iters: {solver.iterations} | Exp: {exp:.4f}",
                        end="\r",
                    )

                    next_eval_time += self.eval_interval

            # Final cleanup per run
            tracemalloc.stop()
            print("")

    def save_results(self, filename="cfr_benchmark_results.csv"):
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")
        print(
            df.groupby(["Solver", "Wall_Clock_Time"])
            .mean(numeric_only=True)[["Exploitability", "Iterations"]]
            .head(10)
        )
        return df


if __name__ == "__main__":
    # Configuration
    BUDGET_SECONDS = 30  # Short budget for testing purposes (Change to 300 for final)
    INTERVAL_SECONDS = 2.0

    runner = ExperimentRunner(
        LeducGame, time_budget=BUDGET_SECONDS, eval_interval=INTERVAL_SECONDS
    )

    # 1. Deterministic Solvers
    runner.run_solver("Vanilla CFR", VanillaCFR, {}, n_runs=1)
    runner.run_solver("CFR+", CFRPlus, {}, n_runs=1)
    runner.run_solver("Pruning CFR", PruningCFR, {"prune_threshold": -200}, n_runs=1)

    # 2. MCCFR Solvers
    # External Sampling
    runner.run_solver(
        "External Sampling MCCFR", ExternalSamplingMCCFR, {"seed": 0}, n_runs=3
    )

    # Outcome Sampling
    runner.run_solver(
        "Outcome Sampling MCCFR",
        OutcomeSamplingMCCFR,
        {"seed": 0, "linear_averaging": True, "epsilon": 0.6},
        n_runs=3,
    )

    # Save
    df = runner.save_results()
