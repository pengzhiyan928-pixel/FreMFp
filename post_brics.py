
# -*- coding: utf-8 -*-
import argparse, pathlib, sys, ast
from collections import Counter
import pandas as pd
import numpy as np

def parse_frag_lists(df, smiles_col, fragments_col="fragments", joined_col="frags_joined"):
    """优先解析 fragments 列（Python 列表字面量），备用 frags_joined（分号分隔）。"""
    frag_lists = []
    if fragments_col in df.columns:
        for s in df[fragments_col]:
            if pd.isna(s) or str(s).strip() == "":
                frag_lists.append([])
            else:
                try:
                    xs = ast.literal_eval(str(s))
                    frag_lists.append([str(x) for x in xs])
                except Exception:
                    frag_lists.append([])
    elif joined_col in df.columns:
        for s in df[joined_col].fillna(""):
            xs = [t for t in str(s).split(";") if t]
            frag_lists.append(xs)
    else:
        sys.exit("❌ 未找到 fragments / frags_joined 列，无法解析片段列表。")
    return frag_lists

def build_vocab(frag_lists):
    """全库词表：频次↓ + SMILES字典序↑，返回 vocab(list)、frag2id(dict)、freq(Counter)。"""
    freq = Counter()
    for xs in frag_lists:
        freq.update(xs)
    vocab = list(freq.keys())
    vocab.sort()
    vocab.sort(key=lambda x: -freq[x])
    frag2id = {s:i for i,s in enumerate(vocab)}
    return vocab, frag2id, freq

def write_vocab(vocab, freq, out_path):
    pd.DataFrame({
        "frag_id": range(len(vocab)),
        "fragment_smiles": vocab,
        "freq": [freq[s] for s in vocab],
    }).to_csv(out_path, index=False)

def write_index(names, frag_lists, frag2id, out_path):
    import numpy as np
    m = max((len(xs) for xs in frag_lists), default=0)
    rows = []
    for xs in frag_lists:
        ids = [frag2id[s] for s in xs]
        # 原来：ids = ids + [-1] * (m - len(ids))
        # 现在：补空（两种都行，选其一）

        # A. 补空字符串（Excel 里显示为空白）
        ids = ids + [""] * (m - len(ids))

        # B. 或者：补 NaN（to_csv 会写成空单元格）
        # ids = ids + [np.nan] * (m - len(ids))

        rows.append(ids)

    cols = [f"node_{i}" for i in range(m)]
    df = pd.DataFrame(rows, index=names, columns=cols)
    # 如果用了 NaN，可加 na_rep="" 强制空白显示
    df.to_csv(out_path)  # to_csv 默认就会把 NaN 写成空


def write_exploded(names, frag_lists, frag2id, out_path):
    recs = []
    for name, xs in zip(names, frag_lists):
        for j, s in enumerate(xs):
            recs.append((name, j, s, frag2id[s]))
    pd.DataFrame(recs, columns=["name","frag_idx","fragment_smiles","frag_id"]).to_csv(out_path, index=False)

def main():
    ap = argparse.ArgumentParser("BRICS 分割后处理：生成词表与索引矩阵")
    ap.add_argument("--input",  default="batchsplit2.csv", help="分割结果文件（含 fragments / frags_joined）")
    ap.add_argument("--name_col", default="name", help="分子名称列名")
    ap.add_argument("--smiles_col", default="smiles", help="SMILES 列名（仅用于携带）")
    ap.add_argument("--outdir", default="brics_out", help="输出目录")
    args = ap.parse_args()

    inp = pathlib.Path(args.input)
    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 读入
    if inp.suffix.lower() in {".csv",".tsv"}:
        sep = "," if inp.suffix.lower()==".csv" else "\t"
        df = pd.read_csv(inp, sep=sep)
    elif inp.suffix.lower() in {".xls",".xlsx"}:
        df = pd.read_excel(inp)
    else:
        sys.exit(f"❌ 不支持的文件格式: {inp}")

    if args.name_col not in df.columns:
        sys.exit(f"❌ 找不到名称列 {args.name_col}，表头：{list(df.columns)}")

    # 解析片段列表
    frag_lists = parse_frag_lists(df, args.smiles_col)
    names = df[args.name_col].astype(str).tolist()

    # 构建词表与映射
    vocab, frag2id, freq = build_vocab(frag_lists)

    # 写出
    write_vocab(vocab, freq, outdir / "ring_total_list.csv")
    write_index(names, frag_lists, frag2id, outdir / "index_data.csv")
    write_exploded(names, frag_lists, frag2id, outdir / "frags_exploded.csv")

    print(f"✅ 完成。词表 {len(vocab)} 个片段；index 行数 {len(names)}。")
    print(f"   - 词表：{(outdir/'ring_total_list.csv').resolve()}")
    print(f"   - 索引：{(outdir/'index_data.csv').resolve()}")
    print(f"   - 竖表：{(outdir/'frags_exploded.csv').resolve()}")

if __name__ == "__main__":
    main()
