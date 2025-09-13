
# -*- coding: utf-8 -*-
import argparse, pathlib
import pandas as pd
import numpy as np

def load_vocab(vocab_path):
    v = pd.read_csv(vocab_path)
    v["frag_id"] = v["frag_id"].astype(int)
    v = v.sort_values("frag_id")
    vocab_ids = v["frag_id"].tolist()
    return v, vocab_ids

def counts_from_exploded(exploded_path, vocab_ids):
    df = pd.read_csv(exploded_path)  # 需要列: name, frag_id
    df["frag_id"] = pd.to_numeric(df["frag_id"], errors="coerce")
    df = df.dropna(subset=["frag_id"])
    df["frag_id"] = df["frag_id"].astype(int)

    cnt = df.groupby(["name","frag_id"]).size().unstack(fill_value=0)
    # 补齐所有列，并按 frag_id 升序
    for fid in vocab_ids:
        if fid not in cnt.columns:
            cnt[fid] = 0
    cnt = cnt[vocab_ids]
    cnt.columns = ["frag_{}".format(i) for i in vocab_ids]
    onehot = (cnt > 0).astype(int)
    return cnt, onehot

def counts_from_index(index_path, vocab_ids):
    # index_data.csv 的第一列是 name 索引，后续列 node_0..node_m
    df = pd.read_csv(index_path, index_col=0, dtype=str)
    # 转长表，过滤空白
    long = (df.stack()
              .reset_index()
              .rename(columns={"level_0":"name", 0:"frag_id"}))
    long = long[long["frag_id"].notna() & (long["frag_id"] != "")]
    long["frag_id"] = pd.to_numeric(long["frag_id"], errors="coerce")
    long = long.dropna(subset=["frag_id"])
    long["frag_id"] = long["frag_id"].astype(int)

    cnt = long.groupby(["name","frag_id"]).size().unstack(fill_value=0)
    for fid in vocab_ids:
        if fid not in cnt.columns:
            cnt[fid] = 0
    cnt = cnt[vocab_ids]
    cnt.columns = ["frag_{}".format(i) for i in vocab_ids]
    onehot = (cnt > 0).astype(int)
    return cnt, onehot

def main():
    ap = argparse.ArgumentParser("BRICS 指纹生成")
    ap.add_argument("--vocab",   default="brics_out/ring_total_list.csv")
    ap.add_argument("--index",   default="brics_out/index_data.csv")
    ap.add_argument("--exploded",default="brics_out/frags_exploded.csv")
    ap.add_argument("--outdir",  default="brics_out")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    vocab_df, vocab_ids = load_vocab(args.vocab)

    expl = pathlib.Path(args.exploded)
    if expl.exists():
        counts, onehot = counts_from_exploded(expl, vocab_ids)
    else:
        counts, onehot = counts_from_index(args.index, vocab_ids)

    # 保存
    counts.to_csv(outdir / "counts.csv")
    onehot.to_csv(outdir / "one_hot.csv")

    # 友好版列名：frag_id|smiles
    id2smi = dict(zip(vocab_df["frag_id"], vocab_df["fragment_smiles"]))
    rename_map = {"frag_{}".format(fid): "frag_{}|{}".format(fid, id2smi[fid]) for fid in vocab_ids}
    counts.rename(columns=rename_map).to_csv(outdir / "counts_with_smiles.csv")

    print("✅ 指纹已生成：")
    print("   -", (outdir/"counts.csv").resolve())
    print("   -", (outdir/"one_hot.csv").resolve())
    print("   -", (outdir/"counts_with_smiles.csv").resolve())

if __name__ == "__main__":
    main()
