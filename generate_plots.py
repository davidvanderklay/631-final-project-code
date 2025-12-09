import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def generate_plots(filename="cfr_benchmark_results.csv"):
    # Load Data
    df = pd.read_csv(filename)

    # Pre-processing
    df["Exploitability"] = df["Exploitability"] + 1e-9

    solvers = df["Solver"].unique()
    palette = sns.color_palette("husl", len(solvers))
    solver_colors = dict(zip(solvers, palette))

    # Plot 1: Exploitability vs. Wall-Clock Time
    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=df,
        x="Wall_Clock_Time",
        y="Exploitability",
        hue="Solver",
        palette=solver_colors,
        linewidth=2.0,
    )

    plt.yscale("log")
    plt.xlabel("Wall Clock Time (seconds)")
    plt.ylabel("Exploitability (Log Scale)")
    plt.title("Convergence: Exploitability vs. Time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("plot_1_time_convergence.png", dpi=300)
    print("Generated plot_1_time_convergence.png")

    # Plot 2: Exploitability vs. Nodes Touched (Efficiency)
    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=df,
        x="Nodes_Touched_Est",
        y="Exploitability",
        hue="Solver",
        palette=solver_colors,
        linewidth=2.0,
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Estimated Nodes Touched (Log Scale)")
    plt.ylabel("Exploitability (Log Scale)")
    plt.title("Algorithmic Efficiency: Exploitability vs. Nodes")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("plot_2_node_efficiency.png", dpi=300)
    print("Generated plot_2_node_efficiency.png")

    # Plot 3: Pruning Analysis (Speedup)

    plt.figure(figsize=(8, 5))

    # Filter for Vanilla and Pruning
    subset = df[df["Solver"].isin(["Vanilla CFR", "Pruning CFR"])].copy()

    subset["Iterations_Per_Second"] = subset["Iterations"] / subset["Wall_Clock_Time"]

    sns.lineplot(
        data=subset,
        x="Wall_Clock_Time",
        y="Iterations_Per_Second",
        hue="Solver",
        markers=True,
        dashes=False,
    )

    plt.xlabel("Wall Clock Time (seconds)")
    plt.ylabel("Iterations Per Second")
    plt.title("Impact of Pruning on Traversal Speed")
    plt.tight_layout()
    plt.savefig("plot_3_pruning_speedup.png", dpi=300)
    print("Generated plot_3_pruning_speedup.png")

    # Plot 4: Memory Profile
    plt.figure(figsize=(8, 5))

    # Aggregate max memory per solver
    mem_df = df.groupby("Solver")["Peak_Memory_MB"].max().reset_index()

    sns.barplot(data=mem_df, x="Solver", y="Peak_Memory_MB", palette=solver_colors)

    plt.ylabel("Peak Memory Usage (MB)")
    plt.title("Memory Footprint Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plot_4_memory_profile.png", dpi=300)
    print("Generated plot_4_memory_profile.png")


if __name__ == "__main__":
    generate_plots()
