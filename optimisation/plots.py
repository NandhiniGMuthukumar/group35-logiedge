"""Export measured benchmark figures with explicit units and frontier axes."""

from pathlib import Path


def save_figures(rows, frontier, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    folder = Path(folder)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11,
                         "axes.spines.top":False,"axes.spines.right":False})
    colors = ["#2563eb","#0f766e","#b45309"]
    short_names={"M1_FP32":"M1 FP32","M2_INT8":"M2 INT8","M3_STRUCTURED_INT8":"M3 pruned INT8"}
    labels = [short_names.get(row["variant"],row["variant"]) for row in rows]
    fig, axes = plt.subplots(1,3,figsize=(13.5,4.3),layout="constrained")
    x = np.arange(len(rows))
    axes[0].bar(x-.18,[r["mean_latency_ms"] for r in rows],.36,label="Mean",color="#2563eb")
    axes[0].bar(x+.18,[r["p95_latency_ms"] for r in rows],.36,label="p95",color="#93c5fd")
    axes[0].set_ylabel("Invoke latency (ms)")
    axes[0].set_title("10 warm-ups, 200 timed calls")
    axes[0].legend(frameon=False)
    for ax,key,title,unit in [(axes[1],"model_KB_decimal","Actual model files","Size (decimal KB)"),
                              (axes[2],"energy_proxy_mJ","CPU/TDP energy proxy","Estimated energy (mJ)")]:
        bars=ax.bar(x,[r[key] for r in rows],color=colors[:len(rows)])
        ax.set_ylabel(unit);ax.set_title(title)
        for bar,row in zip(bars,rows):
            ax.annotate(f"{row[key]:.4g}",(bar.get_x()+bar.get_width()/2,bar.get_height()),
                        xytext=(0,5),textcoords="offset points",ha="center",fontsize=10)
        ax.margins(y=.2)
    for ax in axes:
        ax.set_xticks(x,labels,rotation=15,ha="right")
        ax.set_ylim(bottom=0);ax.grid(axis="y",alpha=.18);ax.set_axisbelow(True)
    fig.suptitle("LogiEdge | one CPU runtime, one thread, same held-out split",fontsize=15)
    for ext in ("png","svg"):
        fig.savefig(folder/f"benchmark_metrics.{ext}",dpi=180,bbox_inches="tight")
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(9.5,6.6))
    fig.subplots_adjust(left=.11,right=.96,top=.89,bottom=.22)
    feasible=[r for r in rows if r["deployment_metric_eligible"]]
    front=sorted([r for r in feasible if r["variant"] in frontier],key=lambda r:r["mean_latency_ms"])
    if len(front)>1:
        ax.plot([r["mean_latency_ms"] for r in front],[r["model_KB_decimal"] for r in front],
                color="#64748b",linestyle="--",linewidth=1,label="Feasible 2-objective frontier")
    offsets=[(14,26),(14,-40),(14,4)]
    for i,row in enumerate(rows):
        is_front=row["variant"] in frontier
        marker="*" if is_front else ("o" if row["deployment_metric_eligible"] else "x")
        ax.scatter(row["mean_latency_ms"],row["model_KB_decimal"],s=190 if is_front else 90,
                   color=colors[i%len(colors)],marker=marker,zorder=3,label=row["variant"])
        text=(f"{row['variant']}\naccuracy {row['accuracy_percent']:.2f}% | Critical {row['critical_recall_percent']:.2f}%")
        ax.annotate(text,(row["mean_latency_ms"],row["model_KB_decimal"]),xytext=offsets[i%3],
                    textcoords="offset points",fontsize=10,arrowprops={"arrowstyle":"-","color":"#94a3b8"})
    ax.set(xlabel="Mean invoke latency (ms) — minimize",ylabel="Model size (decimal KB) — minimize",
           title="Pareto comparison: mean latency and file size")
    ax.set_xlim(0,max(r["mean_latency_ms"] for r in rows)*1.8)
    ax.set_ylim(0,max(r["model_KB_decimal"] for r in rows)*1.25)
    ax.grid(alpha=.18);ax.legend(loc="lower right",frameon=False)
    fig.text(.01,.01,"Star = nondominated among models with accuracy >88% and Critical recall >95%.\n"
             "p95 and energy proxy are reported separately; they are not axes of this 2D frontier.",fontsize=9,color="#475569")
    for ext in ("png","svg"):
        fig.savefig(folder/f"pareto_latency_size.{ext}",dpi=180,bbox_inches="tight")
    plt.close(fig)
