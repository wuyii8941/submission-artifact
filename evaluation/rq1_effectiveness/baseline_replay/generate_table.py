#!/usr/bin/env python3
"""
Generate the LaTeX snippet for Table II from attribution JSON.

Usage:
  python generate_table.py [--json results/table2_attribution.json] [--output table2.tex]
"""
import argparse
import json
import os
import sys


COL_ORDER = ["FE/ctx", "AOT/bwd", "Low/IR", "Fusion", "Sched.", "Triton"]
ROW_ORDER = ["DeepOpFuzz", "Random", "FreeFuzz", "NNSmith", "WhiteFox"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json",
                    default=os.path.join(os.path.dirname(__file__), "results", "table2_attribution.json"))
    ap.add_argument("--output",
                    default=os.path.join(os.path.dirname(__file__), "results", "table2.tex"))
    args = ap.parse_args()

    with open(args.json) as f:
        data = json.load(f)

    per_run = data["per_run"]
    union = data["any_baseline_union"]["row"]

    lines = [
        r"\begin{table}[tbp]",
        r"\centering",
        r"\caption{4-hour replay coverage of the 30 PyTorch/Inductor root causes, broken down by Inductor stage. Each cell counts distinct root causes triggered (after fingerprint dedup) by the method's saved trigger programs.}",
        r"\label{tab:pipeline-gap}",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\renewcommand{\arraystretch}{1.06}",
        r"\begin{tabular}{@{}lrrrrrrr@{}}",
        r"\toprule",
        r"Method & FE/ctx & AOT/bwd & Low/IR & Fusion & Sched. & Triton & Total \\",
        r"\midrule",
        r"\emph{Root causes} & 3 & 5 & 6 & 3 & 4 & 9 & 30 \\",
        r"\midrule",
    ]
    for name in ROW_ORDER:
        if name not in per_run:
            continue
        r = per_run[name]["row"]
        cells = [str(r[c]) for c in COL_ORDER]
        lines.append(f"{name:11s} & " + " & ".join(cells) + f" & {r['Total']} \\\\")
    lines += [
        r"\midrule",
        r"\emph{Any baseline} & " + " & ".join(str(union[c]) for c in COL_ORDER) + f" & {union['Total']} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.25em}",
        r"\begin{minipage}{0.96\columnwidth}",
        r"\footnotesize",
        r"\emph{Any baseline} reports the union over the four baselines. ",
        r"Each method ran for 4 hours on PyTorch~2.11 (the corresponding bug version) with a single seed. ",
        r"All methods use the same eager-vs-compiled differential oracle. ",
        r"bug\_051 (scheduler prologue typo) is a code-level defect not triggerable by any test program; it is included in the denominator for consistency with the bug catalog but is unreachable by all methods. ",
        r"NNSmith's $0$ reflects a structural ceiling: its ONNX-stable opset on PT~2.11 contains no \texttt{slice\_scatter}, \texttt{diagonal\_scatter}, \texttt{cumsum}, \texttt{adaptive\_avg\_pool}, \texttt{nested\_tensor}, \texttt{inference\_mode}, backward graphs, complex dtypes, or \texttt{float8} primitives required by 17+ of the 30 root causes.",
        r"\end{minipage}",
        r"\vspace{-3mm}",
        r"\end{table}",
        ""
    ]

    text = "\n".join(lines)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(text)
    print(f"Wrote {args.output}")
    print()
    print(text)


if __name__ == "__main__":
    main()
