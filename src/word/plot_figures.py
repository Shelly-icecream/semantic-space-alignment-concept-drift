import json
import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import matplotlib.pyplot as plt
import numpy as np

import paths

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    csv_path = paths.word_csv(paths.SHIFT_RESULTS_CSV)
    meta_path = paths.word_csv(paths.SHIFT_META_JSON)

    words = []
    shifts = []
    with open(csv_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                words.append(parts[0])
                shifts.append(float(parts[1]))

    shift_dict = {w: s for w, s in zip(words, shifts)}
    all_shifts = np.array(shifts)

    if meta_path.is_file():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        anchor_mean = float(meta["anchor_mean"])
        threshold = float(meta["threshold"])
    else:
        anchor_mean = float(np.mean(all_shifts))
        threshold = float(np.mean(all_shifts) + 2 * np.std(all_shifts))

    fig_dir = paths.fig_word_dir()

    plt.figure(figsize=(10, 5))
    plt.hist(all_shifts, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    plt.axvline(
        threshold,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Significance Threshold ({threshold:.4f})",
    )
    plt.axvline(
        anchor_mean,
        color="orange",
        linestyle="dashed",
        linewidth=2,
        label=f"Anchor Mean ({anchor_mean:.4f})",
    )
    plt.xlabel("Semantic Shift")
    plt.ylabel("Frequency")
    plt.title("Distribution of Semantic Shift")
    plt.legend()
    plt.tight_layout()
    p1 = fig_dir / "shift_distribution.png"
    plt.savefig(str(p1), dpi=150)
    print("Saved:", p1)

    top20_words = words[:20]
    top20_shifts = [shift_dict[w] for w in top20_words]
    colors = ["#d62728" if s > threshold else "#1f77b4" for s in top20_shifts]

    plt.figure(figsize=(12, 6))
    plt.bar(range(20), top20_shifts, color=colors, edgecolor="black")
    plt.axhline(
        threshold,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Significance Threshold ({threshold:.4f})",
    )
    plt.xticks(range(20), top20_words, rotation=45, ha="right", fontsize=10)
    plt.xlabel("Word")
    plt.ylabel("Semantic Shift")
    plt.title("Top 20 Words by Semantic Shift (Auto)")
    plt.legend()
    plt.tight_layout()
    p2 = fig_dir / "shift_top20_auto.png"
    plt.savefig(str(p2), dpi=150)
    print("Saved:", p2)

    selected_words = [
        "飘尘",
        "西柏林",
        "新地",
        "乙烯",
        "猛戳",
        "起用",
        "南都",
        "力荐",
        "电焊工",
        "黑子",
        "木鱼",
        "比格",
        "自耕农",
        "特力",
        "黑蚂蚁",
        "王国",
        "矮马",
        "震旦",
        "黑豹",
        "一马当先",
    ]

    selected_shifts = [shift_dict.get(w, 0) for w in selected_words]
    colors_sel = ["#d62728" if s > threshold else "#1f77b4" for s in selected_shifts]

    plt.figure(figsize=(14, 6))
    plt.bar(range(len(selected_words)), selected_shifts, color=colors_sel, edgecolor="black")
    plt.axhline(
        threshold,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Significance Threshold ({threshold:.4f})",
    )
    plt.xticks(range(len(selected_words)), selected_words, rotation=45, ha="right", fontsize=10)
    plt.xlabel("Word")
    plt.ylabel("Semantic Shift")
    plt.title("Selected High-Value Words by Semantic Shift")
    plt.legend()
    plt.tight_layout()
    p3 = fig_dir / "shift_top20_selected.png"
    plt.savefig(str(p3), dpi=150)
    print("Saved:", p3)

    print("\n完成！三张图已全部生成。")


if __name__ == "__main__":
    main()
