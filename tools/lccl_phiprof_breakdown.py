#!/usr/bin/env python3
"""
Per-step cost breakdown of a Vlasiator run with the LCCL field solver, from the phiprof report
(phiprof_0.txt, written at the end of the run).

The rows are disjoint and add up to the time per step of the "Simulation" timer:
  translation and acceleration of each species, velocity moments, boundary conditions,
  the field solve ("Propagate Fields") with its parts, field-grid coupling out, I/O, other.
Inside the field solve the table splits out the species tensors, the RME (Eq. 38) with
assembly, HYPRE assemble, BoomerAMG/GMRES setup and solve, Faraday, and the Gauss
correction with assembly, HYPRE assemble, BoomerAMG/PCG setup, solve and the operators of
the consistent form. Iterations per solve and the acceleration work (velocity blocks times
substeps) come from the work units of the timers that carry them.

Timers below 1 % of the total are missing from phiprof's default ("compact") print; for a
complete table run with
    export PHIPROF_PRINTS=groups,full
The script uses the "All timers" table if the report has one, otherwise the compact one.

  python3 lccl_phiprof_breakdown.py phiprof_0.txt [--steps N] [--latex] [--csv out.csv]
"""
import argparse
import re
import sys


class Node:
    def __init__(self, label, level, count, time, rate, unit):
        self.label, self.level, self.count, self.time = label, level, count, time
        self.rate, self.unit = rate, unit          # work units per second and their label, or None
        self.parent, self.children = None, []

    def ancestors(self):
        n = self.parent
        while n is not None:
            yield n
            n = n.parent

    def descendants(self):
        for c in self.children:
            yield c
            yield from c.descendants()


def parse(path):
    """Timer tree of the last 'All timers' table, or of the compact table if there is none."""
    lines = open(path).read().splitlines()
    tables, i = [], 0
    while i < len(lines):
        if re.match(r"\s*\|\s*Id\s*\|\s*Lvl\s*\|", lines[i]):
            title = ""
            for k in range(i - 1, max(i - 8, -1), -1):
                if "timers" in lines[k].lower():
                    title = lines[k].strip()
                    break
            rows, j = [], i + 1
            while j < len(lines) and lines[j].startswith("---"):
                j += 1
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                rows.append(lines[j])
                j += 1
            tables.append((title, rows))
            i = j
        else:
            i += 1
    if not tables:
        sys.exit(f"{path}: no phiprof timer table found")
    full = [t for t in tables if t[0].startswith("All timers")]
    title, rows = (full or tables)[-1]
    complete = bool(full)

    root = Node("(root)", 0, 0, 0.0, None, None)
    stack = [root]
    for r in rows:
        cells = [c for c in r.split("|")][1:-1]
        if len(cells) < 12:
            continue
        ident, lvl, _grp, name = (c.strip() for c in cells[:4])
        label = name
        if not lvl.isdigit():
            continue
        level = int(lvl)
        try:
            count, time = float(cells[4]), float(cells[5])
        except ValueError:
            continue
        rate = unit = None
        m = re.match(r"\s*([-+0-9.eEinf]+)\s+(.*)/s\s*$", cells[11])
        if m:
            rate, unit = float(m.group(1)), m.group(2)
        node = Node(label, level, count, time, rate, unit)
        while len(stack) > level:
            stack.pop()
        node.parent = stack[-1]
        stack[-1].children.append(node)
        stack.append(node)
        if ident == "":           # phiprof's "Other" rows: time not covered by the children
            node.label = "(other)"
    return root, complete


def find_nodes(root, pred, within=None):
    """Nodes matching pred, outermost only (a match nested in a match is not counted twice)."""
    pool = within.descendants() if within is not None else root.descendants()
    out = []
    for n in pool:
        if n.label != "(other)" and pred(n.label) and not any(
                pred(a.label) for a in n.ancestors() if within is None or a is not within):
            out.append(n)
    return out


def total(nodes):
    return sum(n.time for n in nodes)


def work_units(nodes):
    """Total work units of the nodes (rate x time), and their label."""
    units = [n for n in nodes if n.rate is not None]
    if not units:
        return None, None
    return sum(n.rate*n.time for n in units), units[0].unit


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("report", help="phiprof report, e.g. phiprof_0.txt")
    p.add_argument("--steps", type=float, help="number of steps (default: calls of the 'Propagate' timer)")
    p.add_argument("--latex", action="store_true", help="print the table as LaTeX rows")
    p.add_argument("--csv", help="also write the table to this CSV file")
    a = p.parse_args()

    root, complete = parse(a.report)
    eq = lambda s: (lambda l: l == s)
    starts = lambda s: (lambda l: l.startswith(s))

    # Everything is counted inside the main loop ("Simulation") only: the set-up before it calls some of
    # the same routines (moments, translation with dt = 0, the dt = 0 field-solver call, getFieldsFromFsGrid).
    sim = find_nodes(root, eq("Simulation"))
    scope = sim[0] if sim else None
    step_time = total(sim) if sim else None

    def find(r, pred, within=None):
        return find_nodes(r, pred, within if within is not None else scope)

    prop = find(root, eq("Propagate"))
    steps = a.steps or (sum(n.count for n in prop) if prop else None)
    if not steps:
        sys.exit("cannot tell the number of steps: no 'Propagate' timer; give --steps")

    rows = []          # (indent, name, seconds over the run or None, note)

    def add(indent, name, nodes, note=""):
        rows.append((indent, name, total(nodes) if nodes else None, note))
        return total(nodes) if nodes else 0.0

    accounted = 0.0
    species = sorted({n.label.split(" ", 1)[1] for n in (scope or root).descendants()
                      if n.label.startswith(("translate ", "accelerate "))})
    for sp in species:
        accounted += add(0, f"Translation, {sp}", find(root, eq(f"translate {sp}")))
    for sp in species:
        nodes = find(root, eq(f"accelerate {sp}"))
        wu, unit = work_units(nodes)
        note = ""
        if wu:
            note = f"{wu/steps:.3g} {unit} per step, {1e9*total(nodes)/wu:.3g} ns each"
        accounted += add(0, f"Acceleration, {sp}", nodes, note)
    accounted += add(0, "Velocity moments", find(root, eq("Compute interp moments")))
    accounted += add(0, "Boundary conditions", find(root, starts("Update system boundaries")))

    fields = find(root, eq("Propagate Fields"))
    accounted += add(0, "Field solve (Propagate Fields)", fields)
    if fields:
        f0 = fields[0]
        inside = 0.0
        inside += add(1, "field-grid coupling in", find(root, eq("fsgrid-coupling-in"), f0))
        inside += add(1, "species tensors (alpha_s, mu, J-hat)", find(root, eq("LCCL: species tensors"), f0))
        rme = find(root, eq("LCCL: RME (Eq. 38)"), f0)
        inside += add(1, "RME, Eq. 38", rme)
        if rme:
            r0 = rme[0]
            part = 0.0
            part += add(2, "assembly", find(root, lambda l: l in ("LCCL: assign DOF numbering",
                                                                   "LCCL: assemble Eq.38 system"), r0))
            part += add(2, "HYPRE assemble", find(root, eq("LCCL: RME HYPRE assemble"), r0))
            part += add(2, "BoomerAMG/GMRES setup", find(root, eq("LCCL: RME BoomerAMG/GMRES setup"), r0))
            solve = find(root, eq("LCCL: RME GMRES solve"), r0)
            wu, unit = work_units(solve)
            calls = sum(n.count for n in solve)
            part += add(2, "GMRES solve", solve, f"{wu/calls:.3g} {unit} per solve" if wu and calls else "")
            rows.append((2, "rest of the RME", r0.time - part, ""))
        inside += add(1, "Faraday, Eq. 23", find(root, eq("LCCL: Faraday (Eq. 23)"), f0))
        gauss = find(root, eq("LCCL: Gauss correction"), f0)
        inside += add(1, "Gauss correction", gauss)
        if gauss:
            g0 = gauss[0]
            part = 0.0
            part += add(2, "consistent operators (series, filter, mask)",
                        find(root, starts("LCCL: consistent Gauss"), g0))
            part += add(2, "assembly", find(root, lambda l: l in (
                "LCCL: assign phi DOF numbering", "LCCL: compute mean(rho)",
                "LCCL: assemble Gauss-correction system", "LCCL: hand Gauss-correction system to HYPRE IJ"), g0))
            part += add(2, "HYPRE assemble", find(root, eq("LCCL: Gauss HYPRE assemble"), g0))
            part += add(2, "BoomerAMG/PCG setup", find(root, eq("LCCL: Gauss BoomerAMG/PCG setup"), g0))
            solve = find(root, eq("LCCL: Gauss PCG solve"), g0)
            wu, unit = work_units(solve)
            calls = sum(n.count for n in solve)
            part += add(2, "PCG solve", solve, f"{wu/calls:.3g} {unit} per solve" if wu and calls else "")
            rows.append((2, "rest of the Gauss correction", g0.time - part, ""))
        rows.append((1, "rest of the field solve (filters, node <-> Yee, publish, ...)", f0.time - inside, ""))
    accounted += add(0, "Field-grid coupling out (getFieldsFromFsGrid)", find(root, eq("getFieldsFromFsGrid")))
    accounted += add(0, "I/O", find(root, eq("IO")))
    if step_time is not None:
        rows.append((0, "Other", step_time - accounted, ""))

    # ---- output ----
    ref = step_time if step_time else accounted
    def ms(t): return "-" if t is None else f"{1e3*t/steps:.4g}"
    def pct(t): return "-" if t is None else f"{100*t/ref:.3g}"
    if a.latex:
        for ind, name, t, note in rows:
            pad = "\\quad " * ind
            print(f"{pad}{name} & {ms(t)} & {pct(t)} & {note} \\\\")
    else:
        w = max(len("  "*ind + name) for ind, name, _, _ in rows)
        print(f"{'':{w}}  {'ms/step':>9}  {'% step':>7}  note")
        for ind, name, t, note in rows:
            print(f"{'  '*ind + name:{w}}  {ms(t):>9}  {pct(t):>7}  {note}")
        print(f"\n{steps:g} steps; {1e3*ref/steps:.4g} ms per step"
              + ("" if step_time else " (no 'Simulation' timer: sum of the rows)"))
        rme_t = total(find(root, eq("LCCL: RME (Eq. 38)")))  # inside the main loop
        solves = total(find(root, lambda l: l in ("LCCL: RME BoomerAMG/GMRES setup", "LCCL: RME GMRES solve",
                                                    "LCCL: Gauss BoomerAMG/PCG setup", "LCCL: Gauss PCG solve")))
        if fields:
            print(f"RME (Eq. 38): {100*rme_t/ref:.3g} % of a step; whole field solve: {100*total(fields)/ref:.3g} %; "
                  f"linear-solver setup and solve, both systems: {100*solves/ref:.3g} %")
    if not complete:
        print("\nNote: compact report; timers below 1 % of the total are missing (rows '-'). "
              "Run with PHIPROF_PRINTS=groups,full for all of them.", file=sys.stderr)
    if a.csv:
        with open(a.csv, "w") as f:
            f.write("level,category,ms_per_step,percent_of_step,note\n")
            for ind, name, t, note in rows:
                f.write(f"{ind},\"{name}\",{'' if t is None else 1e3*t/steps},"
                        f"{'' if t is None else 100*t/ref},\"{note}\"\n")


if __name__ == "__main__":
    main()
