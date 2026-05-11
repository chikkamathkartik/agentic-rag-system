import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {'baseline': '#ef4444', 'agent': '#38bdf8'}

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
df = pd.read_csv(os.path.join(BASE_DIR, "evaluation", "results.csv"))

graph_dir = os.path.join(BASE_DIR, "graphs")
os.makedirs(graph_dir, exist_ok=True)

print("📊 Generating comprehensive evaluation plots in graphs/...")

# ---------- 1. SIMILARITY TREND WITH CONFIDENCE BAND ----------
fig, ax = plt.subplots(figsize=(12, 6))

queries = range(len(df))
baseline_sim = df["sim_baseline"].values
agent_sim = df["sim_agent"].values

ax.plot(queries, baseline_sim, label="Baseline", linewidth=2.5, marker='o', 
        markersize=5, color=colors['baseline'], alpha=0.8)
ax.plot(queries, agent_sim, label="Agent RAG", linewidth=2.5, marker='s', 
        markersize=5, color=colors['agent'], alpha=0.8)

# Add fill between for comparison
ax.fill_between(queries, baseline_sim, agent_sim, where=(agent_sim >= baseline_sim), 
                alpha=0.2, color=colors['agent'], label='Agent Better')
ax.fill_between(queries, baseline_sim, agent_sim, where=(agent_sim < baseline_sim), 
                alpha=0.2, color=colors['baseline'], label='Baseline Better')

ax.set_title("Similarity Score Across Queries", fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel("Query Index", fontsize=12)
ax.set_ylabel("Cosine Similarity", fontsize=12)
ax.legend(loc='best', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(os.path.join(graph_dir, "1_similarity_trend.png"), dpi=300)
plt.close()
print("✓ Similarity trend plot saved")


# ---------- 2. HALLUCINATION COMPARISON - BAR CHART ----------
fig, ax = plt.subplots(figsize=(8, 6))

categories = ["Baseline", "Agent"]
halluc_rates = [
    df["hallucination_baseline"].mean(),
    df["hallucination_agent"].mean()
]

bars = ax.bar(categories, halluc_rates, width=0.5, 
              color=[colors['baseline'], colors['agent']], alpha=0.8, edgecolor='black', linewidth=2)

# Add value labels on bars
for bar, val in zip(bars, halluc_rates):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_title("Hallucination Rate Comparison", fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel("Hallucination Rate", fontsize=12)
ax.set_ylim(0, max(halluc_rates) * 1.2)
ax.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(graph_dir, "2_hallucination_comparison.png"), dpi=300)
plt.close()
print("✓ Hallucination comparison plot saved")


# ---------- 3. EXECUTION TIME BREAKDOWN ----------
if 'agent_retrieval_time' in df.columns and 'agent_generation_time' in df.columns:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Timeline for each query
    ax1.plot(df.index, df['agent_total_time'], label='Total Time', linewidth=2.5, 
             marker='o', markersize=5, color=colors['agent'])
    ax1.plot(df.index, df['agent_retrieval_time'], label='Retrieval Time', linewidth=2, 
             marker='^', markersize=4, linestyle='--', alpha=0.7)
    ax1.plot(df.index, df['agent_generation_time'], label='Generation Time', linewidth=2, 
             marker='s', markersize=4, linestyle='--', alpha=0.7)
    
    ax1.set_title("Agent Execution Time Breakdown", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Query Index", fontsize=11)
    ax1.set_ylabel("Time (seconds)", fontsize=11)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Pie chart for average time composition
    avg_retrieval = df['agent_retrieval_time'].mean()
    avg_generation = df['agent_generation_time'].mean()
    avg_other = df['agent_total_time'].mean() - avg_retrieval - avg_generation
    
    times = [avg_retrieval, avg_generation, max(0, avg_other)]
    labels = ['Retrieval', 'Generation', 'Other']
    colors_pie = ['#f97316', '#38bdf8', '#8b5cf6']
    
    wedges, texts, autotexts = ax2.pie(times, labels=labels, autopct='%1.1f%%',
                                         colors=colors_pie, startangle=90, textprops={'fontsize': 11})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax2.set_title("Average Time Distribution", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, "3_execution_time.png"), dpi=300)
    plt.close()
    print("✓ Execution time plot saved")


# ---------- 4. COMPREHENSIVE METRICS COMPARISON ----------
fig, ax = plt.subplots(figsize=(10, 6))

metrics = ['Similarity', 'Hallucination\n(Lower is Better)']
baseline_vals = [
    df["sim_baseline"].mean(),
    df["hallucination_baseline"].mean()
]
agent_vals = [
    df["sim_agent"].mean(),
    df["hallucination_agent"].mean()
]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', 
               color=colors['baseline'], alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, agent_vals, width, label='Agent RAG', 
               color=colors['agent'], alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_title("Overall Performance Metrics", fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel("Score", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend(loc='best', fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(os.path.join(graph_dir, "4_metrics_comparison.png"), dpi=300)
plt.close()
print("✓ Metrics comparison plot saved")


# ---------- 5. RESPONSE LENGTH DISTRIBUTION ----------
fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(df["len_baseline"], bins=15, label="Baseline", alpha=0.7, 
        color=colors['baseline'], edgecolor='black', linewidth=1.5)
ax.hist(df["len_agent"], bins=15, label="Agent", alpha=0.7, 
        color=colors['agent'], edgecolor='black', linewidth=1.5)

ax.set_title("Response Length Distribution", fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel("Number of Words", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.legend(loc='best', fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.3)

# Add statistics text
stats_text = f"Baseline avg: {df['len_baseline'].mean():.0f} words\nAgent avg: {df['len_agent'].mean():.0f} words"
ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(graph_dir, "5_response_length.png"), dpi=300)
plt.close()
print("✓ Response length plot saved")


# ---------- 6. SCATTER PLOT: SIMILARITY vs HALLUCINATION ----------
fig, ax = plt.subplots(figsize=(10, 7))

scatter1 = ax.scatter(df["sim_baseline"], df["hallucination_baseline"], 
                     s=100, alpha=0.6, label="Baseline", color=colors['baseline'], edgecolors='black', linewidth=1.5)
scatter2 = ax.scatter(df["sim_agent"], df["hallucination_agent"], 
                     s=100, alpha=0.6, label="Agent", color=colors['agent'], edgecolors='black', linewidth=1.5)

ax.set_title("Similarity vs Hallucination Trade-off", fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel("Similarity Score", fontsize=12)
ax.set_ylabel("Hallucination Rate", fontsize=12)
ax.legend(loc='best', fontsize=11, scatterpoints=1)
ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(graph_dir, "6_similarity_hallucination_tradeoff.png"), dpi=300)
plt.close()
print("✓ Trade-off scatter plot saved")


# ---------- 7. QUERY-BY-QUERY DETAILED COMPARISON ----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Similarity improvement
improvement = df["sim_agent"] - df["sim_baseline"]
colors_list = [colors['agent'] if x > 0 else colors['baseline'] for x in improvement]

ax1.bar(range(len(improvement)), improvement, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax1.set_title("Per-Query Similarity Improvement (Agent - Baseline)", fontsize=14, fontweight='bold')
ax1.set_xlabel("Query Index", fontsize=11)
ax1.set_ylabel("Similarity Difference", fontsize=11)
ax1.grid(axis='y', linestyle='--', alpha=0.3)

# Hallucination difference
halluc_diff = df["hallucination_baseline"] - df["hallucination_agent"]
colors_list2 = [colors['agent'] if x > 0 else colors['baseline'] for x in halluc_diff]

ax2.bar(range(len(halluc_diff)), halluc_diff, color=colors_list2, alpha=0.8, edgecolor='black', linewidth=1)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax2.set_title("Per-Query Hallucination Reduction (Baseline - Agent)", fontsize=14, fontweight='bold')
ax2.set_xlabel("Query Index", fontsize=11)
ax2.set_ylabel("Hallucination Difference", fontsize=11)
ax2.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(graph_dir, "7_query_by_query_comparison.png"), dpi=300)
plt.close()
print("✓ Query-by-query comparison plot saved")


# ---------- 8. PERFORMANCE SUMMARY HEATMAP ----------
fig, ax = plt.subplots(figsize=(10, 6))

summary_data = np.array([
    [df["sim_baseline"].mean(), df["sim_agent"].mean()],
    [df["hallucination_baseline"].mean(), df["hallucination_agent"].mean()],
    [df["len_baseline"].mean() / 100, df["len_agent"].mean() / 100],  # Normalized
])

im = ax.imshow(summary_data, cmap='RdYlGn', aspect='auto')

ax.set_xticks([0, 1])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(['Baseline', 'Agent'], fontsize=11)
ax.set_yticklabels(['Similarity↑', 'Hallucination↓', 'Response Length\n(normalized)'], fontsize=11)

# Add text annotations
for i in range(3):
    for j in range(2):
        if i == 2:
            text = ax.text(j, i, f'{summary_data[i, j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold', fontsize=11)
        else:
            text = ax.text(j, i, f'{summary_data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontweight='bold', fontsize=11)

ax.set_title("Performance Summary Heatmap", fontsize=16, fontweight='bold', pad=20)
plt.colorbar(im, ax=ax, label='Score')

plt.tight_layout()
plt.savefig(os.path.join(graph_dir, "8_performance_heatmap.png"), dpi=300)
plt.close()
print("✓ Performance heatmap saved")


print("\n✅ All plots generated successfully in: graphs/")
print("\nGenerated plots:")
print("  1. similarity_trend.png - Similarity scores across queries")
print("  2. hallucination_comparison.png - Hallucination rates")
print("  3. execution_time.png - Time breakdown and distribution")
print("  4. metrics_comparison.png - Overall metrics")
print("  5. response_length.png - Response length distribution")
print("  6. similarity_hallucination_tradeoff.png - Trade-off analysis")
print("  7. query_by_query_comparison.png - Per-query improvements")
print("  8. performance_heatmap.png - Summary heatmap")

plt.close()

print("Plots saved in graphs/")