import argparse, pathlib, sys, re
from typing import List, Optional
from collections import Counter
import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS
from tqdm import tqdm

# ==== 兼容导入 RDKit 标准化模块 ====
try:
    from rdkit.Chem.MolStandardize import rdMolStandardize as std
except Exception:
    try:
        # 某些版本也可以这样导
        from rdkit.Chem import rdMolStandardize as std
    except Exception:
        std = None  # 没有可用的标准化模块则跳过

# ==== 文本清洗：去空白，不改内容 ====
_WS = re.compile(r"\s+")
def clean_smiles_text(s: str) -> Optional[str]:
    if s is None:
        return None
    s = str(s)
    s = _WS.sub("", s).strip()
    return s or None

# ==== 片段标准化（官能团规范 + 互变体归一），不去重 ====
if std is not None:
    _normalizer = std.Normalizer()
    _tautomer   = std.TautomerEnumerator()
else:
    _normalizer = None
    _tautomer   = None

def standardize_fragment_mol(m: Chem.Mol) -> Chem.Mol:
    """对片段做官能团规范化 + 互变体归一；若环境不支持则原样返回。"""
    if m is None:
        return None
    # 初步 sanitize（避免 Kekulize 报错）
    try:
        Chem.SanitizeMol(m)
    except Exception:
        Chem.SanitizeMol(
            m,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
        )
    # 规范化与互变体归一（若可用）
    if _normalizer is not None:
        m = _normalizer.normalize(m)
    if _tautomer is not None:
        m = _tautomer.Canonicalize(m)
    # 再次 sanitize
    try:
        Chem.SanitizeMol(m)
    except Exception:
        Chem.SanitizeMol(
            m,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
        )
    return m

def strip_stars(mol: Chem.Mol) -> Chem.Mol:
    em = Chem.EditableMol(mol)
    star_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == '*']
    for idx in sorted(star_idx, reverse=True):
        em.RemoveAtom(idx)
    return em.GetMol()

def _bond_idx(mol, i, j):
    bd = mol.GetBondBetweenAtoms(i, j)
    return bd.GetIdx() if bd is not None else None

def _collect_protected_bonds(mol):

    protected = set()
    smarts_list = [
        ("[CX3](=O)-[OX2H0]-[#6]", [(0,1), (1,2)]),
        ("[CX3](=O)-[SX2H0]-[#6]", [(0,1), (1,2)]),


        ("[#6;!a]-[OX2]-[#6;!a]", [(0,1), (1,2)]),
        ("[#6;!a]-[SX2]-[#6;!a]", [(0,1), (1,2)]),

        ("[#6;a]-[OX2H0]-[#6;!a]", [(1,2)]),  # Ar–O–Alkyl
        ("[#6;a]-[SX2H0]-[#6;!a]", [(1,2)]),  # Ar–S–Alkyl
        ("[#6;!a]-[OX2H0]-[#6;a]", [(0,1)]),  # Alkyl–O–Ar
        ("[#6;!a]-[SX2H0]-[#6;a]", [(0,1)]),  # Alkyl–S–Ar
    ]

    for smi, pairs in smarts_list:
        q = Chem.MolFromSmarts(smi)
        if q is None:
            continue
        for m in mol.GetSubstructMatches(q):
            for a, b in pairs:
                idx = _bond_idx(mol, m[a], m[b])
                if idx is not None:
                    protected.add(idx)
    return protected

def brics_core_fragments(smiles, allow_ring_ring=True):

    # ——输入 SMILES 清洗——
    smiles = clean_smiles_text(smiles)
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        raise ValueError("SMILES 解析失败: {}".format(smiles))

    protected = _collect_protected_bonds(mol)

    # BRICS 候选键
    cand = list(BRICS.FindBRICSBonds(mol))
    bond_ids = []
    for (ai, bi), _labels in cand:
        bd = Chem.Bond  # 仅用于类型提示
        bd = mol.GetBondBetweenAtoms(ai, bi)
        if bd is None:
            continue
        bidx = bd.GetIdx()

        # 1) 不切支链内部受保护的官能团键
        if bidx in protected:
            continue

        # 2) 可选：不切环-环（默认允许切，满足“单环/稠环要能断开”的需求）
        if not allow_ring_ring:
            if mol.GetAtomWithIdx(ai).IsInRing() and mol.GetAtomWithIdx(bi).IsInRing():
                continue

        bond_ids.append(bidx)

    # 按保留下来的键切割
    fragmol = Chem.FragmentOnBonds(mol, bond_ids, addDummies=True)
    frags = Chem.GetMolFrags(fragmol, asMols=True, sanitizeFrags=False)

    out = []
    for f in frags:
        # 去掉占位符(*)并标准化
        em = Chem.EditableMol(f)
        star_idx = [a.GetIdx() for a in f.GetAtoms() if a.GetAtomicNum() == 0]
        for idx in sorted(star_idx, reverse=True):
            em.RemoveAtom(idx)
        f2 = em.GetMol()

        # ——标准化：官能团规范 + 互变体归一（不去重）——
        f2 = standardize_fragment_mol(f2)

        out.append(Chem.MolToSmiles(f2, canonical=True, isomericSmiles=False))
    return out

def load_table(path: pathlib.Path, smiles_col: str) -> pd.DataFrame:
    # 读表（csv/tsv/xlsx）
    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "," if path.suffix.lower() == ".csv" else "\t"
        df = pd.read_csv(path, sep=sep)
    elif path.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(path)
    else:
        sys.exit(f"❌ 不支持的文件格式: {path}")

    # 如果指定列名不存在，兜底用第2列当 SMILES
    if smiles_col not in df.columns:
        if df.shape[1] >= 2:
            print(f"⚠️ 未找到列 {smiles_col}，将使用第2列作为SMILES。表头为：{list(df.columns)}")
            df = df.rename(columns={df.columns[1]: smiles_col})
        else:
            sys.exit(f"❌ 列 {smiles_col} 不存在，且无法兜底。表头为：{list(df.columns)}")
    return df

def save_table(df: pd.DataFrame, path: pathlib.Path):
    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "," if path.suffix.lower() == ".csv" else "\t"
        df.to_csv(path, index=False, sep=sep)
    elif path.suffix.lower() in {".xls", ".xlsx"}:
        df.to_excel(path, index=False)
    else:
        sys.exit(f"❌ 不支持的文件格式: {path}")

def main(args):
    inp  = pathlib.Path(args.input)
    outp = pathlib.Path(args.output)

    df = load_table(inp, args.smiles_col)

    # ——输入列先做一次文本清洗（不改化学、仅去空白）——
    df[args.smiles_col] = df[args.smiles_col].map(clean_smiles_text)

    frag_lists, errors = [], 0
    for smi in tqdm(df[args.smiles_col].astype(str), desc="BRICS 分割进度"):
        try:
            frag_lists.append(brics_core_fragments(smi))
        except ValueError as e:
            errors += 1
            frag_lists.append([])  # 解析失败留空
            print(e, file=sys.stderr)

    # 写结果（不做去重）
    df["fragments"]    = frag_lists
    df["frags_joined"] = df["fragments"].apply(lambda xs: ";".join(xs))
    save_table(df, outp)

    print(f"✅ 处理完成：共 {len(df)} 条；解析失败 {errors} 条")
    print(f"   输出已保存到 {outp.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量 BRICS 分割脚本")
    parser.add_argument("--input", default="test.xlsx", help="输入文件")
    parser.add_argument("--smiles_col", default="smiles", help="SMILES 列名")
    parser.add_argument("--output", default="batchsplit2.csv", help="输出文件")
    # 如需避免切环-环，可在这里加一个开关；当前保持默认 True
    main(parser.parse_args())
