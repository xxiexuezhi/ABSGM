import pandas as pd
import os

curdir = os.path.dirname(__file__)


def get_sabdab_details():
    f = f"20230207_0460443_summary.tsv"
    df = pd.read_csv(f, sep="\t").drop_duplicates()
   # df = df.query(
    #    "antigen_type == antigen_type and antigen_type.str.contains('protein')"
    #)
    df = df.query("Hchain != Lchain")
    df = df.query("Hchain == Hchain and Lchain == Lchain")
    print(f"SabDab\nAbAg Complexes: {len(df)}\nPDB Files: {len(set(df.pdb.values))}\n")
    return df


def read_pdb_line(line, pqr=False):
    aname = line[12:16].strip()
    anumb = int(line[5:11].strip())
    resname = line[17:21].strip()[:3]
    chain = line[21]
    resnumb = line[22:27]
    x = float(line[30:38])
    y = float(line[38:46])
    z = float(line[46:54])
    if pqr:
        return chain, (resname, resnumb), (aname, anumb), (x, y, z), line[63:70]
    return chain, (resname, resnumb), (aname, anumb), (x, y, z)


def pqr2xyzr(fin, fout, cdrnumb):
    xyzr = []
    atoms = {}
    with open(fin) as f:
        for line in f:
            if line.startswith("ATOM "):
                (
                    chain,
                    (resname, resnumb),
                    (aname, anumb),
                    (x, y, z),
                ) = read_pdb_line(line)
                r = line[63:70]
                newline = f"{x} {y} {z} {r}"
                xyzr.append(newline)
                atoms[anumb] = (chain, resnumb, resname, aname, cdrnumb)

    recalc = False
    if os.path.isfile(fout):
        with open(fout, "r") as f:
            print(fout)
            if len(f.readlines()) != len(xyzr):
                recalc = True

    with open(fout, "w") as f:
        f.write("".join(xyzr))

    return atoms, recalc


def remove_redundant(df):
    unique = set()
    keep = []
    discard = []
    for i in range(0, int(len(df)), 6):
        old_unique = len(unique)

        ab = df.iloc[i : i + 6]

        cdrs = ""
        for cdr in ab["cdr_seq"].values:
            cdrs += cdr

        unique.add(cdrs)
        new_unique = len(unique)

        if old_unique != new_unique:
            keep += list(range(i, i + 6))
        else:
            discard += list(range(i, i + 6))

    df_removed = df.iloc[discard]
    df = df.iloc[keep]
    return df, df_removed