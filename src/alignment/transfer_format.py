"""Convert word2vec text format to gensim KeyedVectors (.kv) under result/alignment/."""

import argparse
import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from gensim.models import KeyedVectors

import paths


def convert_one(txt_path: Path, out_kv: Path) -> None:
    print(f"Loading {txt_path} ...")
    model = KeyedVectors.load_word2vec_format(str(txt_path), binary=False)
    out_kv.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_kv))
    print(f"Saved: {out_kv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="word2vec txt -> gensim .kv in result/alignment/")
    parser.add_argument(
        "--renmin-txt",
        type=Path,
        help="Path to Renmin sgns text (e.g. sgns.renmin.word)",
    )
    parser.add_argument(
        "--weibo-txt",
        type=Path,
        help="Path to Weibo sgns text (e.g. sgns.weibo.word)",
    )
    args = parser.parse_args()
    paths.alignment_dir()

    if args.renmin_txt:
        convert_one(args.renmin_txt, paths.alignment_kv(paths.KV_RENMIN))
    if args.weibo_txt:
        convert_one(args.weibo_txt, paths.alignment_kv(paths.KV_WEIBO_RAW))

    if not args.renmin_txt and not args.weibo_txt:
        parser.print_help()
        print("\nProvide at least one of --renmin-txt / --weibo-txt.")


if __name__ == "__main__":
    main()
