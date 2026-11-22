"""Plot only measurements present in the revision evaluation CSV."""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def create_plots(input_path, output_dir):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(
            f"Evaluation results not found: {input_path}. "
            "Run evaluation/run_revision_eval.py first."
        )

    frame = pd.read_csv(input_path)
    required = {
        "system",
        "semantic_similarity",
        "refusal",
        "retrieval_time_seconds",
        "critic_time_seconds",
        "generation_time_seconds",
        "total_time_seconds",
        "trust_score",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Results CSV is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Results CSV contains no evaluation rows.")

    os.makedirs(output_dir, exist_ok=True)
    systems = list(frame["system"].dropna().unique())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    similarity = frame.groupby("system")["semantic_similarity"].agg(["mean", "std"])
    axes[0].bar(
        similarity.index,
        similarity["mean"],
        yerr=similarity["std"].fillna(0),
        capsize=4,
    )
    axes[0].set_ylabel("Cosine similarity to reference")
    axes[0].set_title("Semantic similarity (descriptive)")
    axes[0].tick_params(axis="x", rotation=25)

    refusal = frame.groupby("system")["refusal"].mean().reindex(systems)
    axes[1].bar(refusal.index, refusal.values)
    axes[1].set_ylabel("Refusal fraction")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Refusal rate")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "quality_and_refusal.png"), dpi=200)
    plt.close(fig)

    timing_columns = [
        "retrieval_time_seconds",
        "critic_time_seconds",
        "generation_time_seconds",
    ]
    timing_means = frame.groupby("system")[timing_columns].mean().reindex(systems)
    ax = timing_means.plot(kind="bar", figsize=(11, 5))
    ax.set_ylabel("Seconds per query (mean)")
    ax.set_title("Measured stage latency")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(
        ["Retrieval", "Critic", "Generation"],
        title="Pipeline stage",
    )
    ax.figure.tight_layout()
    ax.figure.savefig(os.path.join(output_dir, "stage_latency.png"), dpi=200)
    plt.close(ax.figure)

    trust_rows = frame.loc[
        (frame["system"] == "trust_aware_dense")
        & frame["trust_score"].notna()
    ]
    if not trust_rows.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(pd.to_numeric(trust_rows["trust_score"]), bins=15, range=(0, 1))
        ax.set_xlabel("Trust decision index (not calibrated probability)")
        ax.set_ylabel("Query count")
        ax.set_title("Trust-aware dense run")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "trust_score_distribution.png"), dpi=200)
        plt.close(fig)

    if "faithfulness_score" in frame.columns and "hallucination_rate" in frame.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        faith = frame.groupby("system")["faithfulness_score"].mean().reindex(systems)
        halluc = frame.groupby("system")["hallucination_rate"].mean().reindex(systems)
        
        axes[0].bar(faith.index, faith.values, color="#10b981")
        axes[0].set_ylabel("Faithfulness Score [0-1]")
        axes[0].set_ylim(0, 1.05)
        axes[0].set_title("Claim Groundedness / Faithfulness")
        axes[0].tick_params(axis="x", rotation=25)

        axes[1].bar(halluc.index, halluc.values, color="#ef4444")
        axes[1].set_ylabel("Hallucination Rate [0-1]")
        axes[1].set_ylim(0, 1.05)
        axes[1].set_title("Unsupported Claim / Hallucination Rate")
        axes[1].tick_params(axis="x", rotation=25)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "faithfulness_and_hallucination.png"), dpi=200)
        plt.close(fig)

    print(f"Generated measured-result plots in {output_dir}")
    print("No hallucination, OOD, calibration, or scaling estimates are inferred.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=os.path.join(BASE_DIR, "evaluation", "revision_results.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(BASE_DIR, "evaluation", "revision_plots"),
    )
    args = parser.parse_args()
    create_plots(args.input, args.output_dir)


if __name__ == "__main__":
    main()
