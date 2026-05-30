import json
import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

import paths

_CJK_FONT_PATHS = [
    Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
]

CJK_FONT = None


def _setup_cjk_font() -> font_manager.FontProperties | None:
    global CJK_FONT
    for path in _CJK_FONT_PATHS:
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            CJK_FONT = font_manager.FontProperties(fname=str(path))
            break
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    return CJK_FONT


_setup_cjk_font()

BAR_COLOR = "#1f77b4"


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
            json.load(f)

    fig_dir = paths.fig_word_dir()
    tick_font = {"fontproperties": CJK_FONT} if CJK_FONT else {}

    plt.figure(figsize=(10, 5))
    plt.hist(all_shifts, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    plt.xlabel("Semantic Shift")
    plt.ylabel("Frequency")
    plt.title("Shift Distribution (Top-5000 Non-Anchor Words)")
    plt.tight_layout()
    p1 = fig_dir / "shift_distribution.png"
    plt.savefig(str(p1), dpi=150)
    plt.close()
    print("Saved:", p1)

    top20_words = words[:20]
    top20_shifts = [shift_dict[w] for w in top20_words]

    plt.figure(figsize=(12, 6))
    plt.bar(range(20), top20_shifts, color=BAR_COLOR, edgecolor="black")
    plt.xticks(
        range(20),
        top20_words,
        rotation=45,
        ha="right",
        fontsize=10,
        **tick_font,
    )
    plt.xlabel("Word")
    plt.ylabel("Semantic Shift")
    plt.title("Top 20 Words by Semantic Shift (Automatic)")
    plt.tight_layout()
    p2 = fig_dir / "shift_top20_auto.png"
    plt.savefig(str(p2), dpi=150)
    plt.close()
    print("Saved:", p2)

    print("Done. Two figures generated.")


if __name__ == "__main__":
    main()
