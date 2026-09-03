#!/usr/bin/env python3
"""Re-derive the numbers runbook_zh.md's intro table and §9.7-§9.10 quote, from the logs.

usage: python3 docs/check_runbook_numbers.py      # -> N values checked, 0 MISMATCHES

Those sections are hand-written Markdown, and hand-transcribed benchmark numbers have
survived review in this repo before. A document-wide grep for a value is not enough: it
cannot tell a number pasted into the neighbouring row from a correct one, and it cannot
notice `results/*/logs/` changing while the prose stays put. So every claim here is
(section, which row, which values), the values are recomputed from the per-node logs
through the SAME generator that produced that campaign's tables.txt, and the row that is
supposed to carry them must actually carry them.

Four campaigns, four generators (each imports the first one's aggregation: all-rank
pooling from every node's log, mean over rotated reps, >25%-off-median reps excluded):

  results/p5en_3arm_20260831/   make_3arm_tables.py    -> §9.7
  results/p5en_stack_20260831/  make_stack_tables.py   -> §9.8 + the intro PR table
  results/p5en_ovlp_20260831/   make_ovlp_tables.py    -> §9.9
  results/b300_stack_20260903/  make_stack_tables.py   -> §9.10

§9.10 is the same four arms as §9.8 on b300, and its four cross-machine bullets quote a
p5en number next to every b300 one -- so those claims are checked against BOTH generators
at once, which is the only way a "b300 is 1.63x slower here" sentence can be verified.

Three documents quote these two campaigns, and every claim is registered against all of
the ones that carry it, from ONE value list: runbook_zh.md (§9.7-§9.10), this kit's
README (`### B300 results`), and -- when the kit sits inside the repo -- the repo-level
../README.md, whose decode table takes the p5en `EP_NUM_SUB_PARTS=1` cell and the b300
default cell as each machine's best-of, so its two cells are checked at DIFFERENT knobs.

NOT COVERED, and left that way on purpose: §9.1-§9.6 and §10.1-§10.2 come from the
2026-08-25 campaign under results/p5en_2n4n_20260825/, whose own generator publishes
comparison.md and is checked by results/p5en_3arm_20260831/check_comparison.py. Adding
them here means a fifth tag scheme; they are older and already reviewed.

Percentages are recomputed from the RAW means, not from the rounded µs printed in the
row, which is what tables.txt does -- so e.g. 167.6/178.9 reads -6.4%, not -6.3%.
Exits nonzero on any mismatch, so this can gate a commit.
"""
import importlib.util
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.dirname(HERE)
RESULTS = os.path.join(KIT, "results")
# overridable so the checker itself can be mutation-tested against a perturbed copy --
# a checker that passes on a doc with a wrong number is worse than none.
RUNBOOK = os.environ.get("RUNBOOK", os.path.join(HERE, "runbook_zh.md"))
# The English README carries the same b300 table as §9.10. Two documents quoting one
# campaign is exactly how a number drifts, so both are checked against the same values.
README = os.environ.get("README", os.path.join(KIT, "README.md"))
# The repo-level README publishes the same campaign a third time, in its cross-stack
# tables. It lives outside this kit, so it is checked only when present.
TOPREADME = os.environ.get("TOPREADME", os.path.join(os.path.dirname(KIT), "README.md"))
HAVE_TOP = os.path.exists(TOPREADME)


def _mod(path, name, env):
    """Load a generator as its own module object.

    Each generator resolves its log directory at import time from EPRUNS, so three
    importlib loads give three independent module objects with three log dirs. A plain
    `import` would not: the second one would reuse the first's already-executed module.
    """
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)           # import-safe: every generator has __main__
    return m


D3 = os.path.join(RESULTS, "p5en_3arm_20260831")
DS = os.path.join(RESULTS, "p5en_stack_20260831")
DO = os.path.join(RESULTS, "p5en_ovlp_20260831")
DB = os.path.join(RESULTS, "b300_stack_20260903")

g3 = _mod(os.path.join(D3, "make_3arm_tables.py"), "gen3",
          {"EPRUNS": os.path.join(D3, "logs")})
gs = _mod(os.path.join(DS, "make_stack_tables.py"), "genstack",
          {"EPRUNS_STACK": os.path.join(DS, "logs")})
go = _mod(os.path.join(DO, "make_ovlp_tables.py"), "genovlp",
          {"EPRUNS_OVLP": os.path.join(DO, "logs")})
# The b300 driver re-executes the p5en stack generator against the b300 logs, so gb.m is
# a SECOND, independent copy of gs's code with its own EPRUNS and its own CACHE. It must
# be loaded last and keyed off EPRUNS_B300, never EPRUNS: gs itself writes EPRUNS.
gb = _mod(os.path.join(DB, "make_stack_tables.py"), "genb300",
          {"EPRUNS_B300": os.path.join(DB, "logs")}).m
assert gs.g.EPRUNS != gb.g.EPRUNS, "the two stack generators collapsed onto one log dir"

# The b300 24 SM campaign, which owns the repo README's two b300 throughput cells. Its own
# driver loads gb as a side effect (for the ANCHOR section), so it must come after gb; it
# is keyed off EPRUNS_SM24 and its accessors take an explicit SM count.
D24 = os.path.join(RESULTS, "b300_sm24_20260903")
g24 = _mod(os.path.join(D24, "make_sm24_tables.py"), "gensm24",
           {"EPRUNS_SM24": os.path.join(D24, "logs")})
assert g24.g.EPRUNS not in (gs.g.EPRUNS, gb.g.EPRUNS), "the 24 SM generator reused a log dir"

MAIN, PR12, PR89, STACK = gs.MAIN, gs.PR12, gs.PR89, gs.STACK
DFLT, SUB1 = gs.DFLT, gs.SUB1
# same arm labels on both machines, by construction (the stack merge sha is pinned)
assert (gb.MAIN, gb.PR12, gb.PR89, gb.STACK) == (MAIN, PR12, PR89, STACK)


# ---------------------------------------------------------------- formatting
def N(x, nd=1):
    return None if x is None else "%.*f" % (nd, x)


def P(new, base):
    """the row's `d` cell: signed percent off base, from raw means"""
    return None if None in (new, base) else "%+.1f%%" % (100.0 * (new - base) / base)


def SHARE(part, whole):
    return None if None in (part, whole) else "%.0f%%" % (100.0 * part / whole)


def RATIO(a, b):
    return None if None in (a, b) else "%.1f×" % (a / b)


# ------------------------------------------------------- §9.7: the 3-arm campaign
def v3(arm, nodes, tok, op, knob=DFLT):
    return g3.stat(g3.cell(arm, nodes, tok, knob), op)[0]


def l3(arm, nodes, tok, knob=DFLT):
    d, c = v3(arm, nodes, tok, "dispatch", knob), v3(arm, nodes, tok,
                                                     "reduced combine", knob)
    return None if None in (d, c) else d + c


# ------------------------------------------------------- §9.8: the stack campaign
def basis(tok):
    """The rep set every cross-arm row in §9.8 is held to: main's own reps.

    make_stack_tables.best_of() uses main as the denominator, so main's reps at its own
    best knob set the basis, and each row intersects with it. main is not in TUNABLE, so
    its only knob is the default -- asserted below rather than assumed.
    """
    return {rep for rep, _ in gs.cell(MAIN, tok, DFLT)}


assert MAIN not in gs.TUNABLE, "main gained a knob: basis() must pick its best one"


def best_knob(arm, tok):
    """The knob §9.8's table would deploy this arm at -- PER TOKEN SIZE.

    Not a per-arm property: the stack's prefill is faster at the DEFAULT geometry and
    its decode at EP_NUM_SUB_PARTS=1, so carrying one arm's decode knob into its prefill
    row reads 5289.8 instead of 5198.8. Chosen on the unfiltered layer total, the same
    way make_stack_tables.best_of() chooses it.
    """
    ks = (DFLT, SUB1) if arm in gs.TUNABLE else (DFLT,)
    opts = [(k, gs.layer(arm, tok, k)) for k in ks]
    opts = [(k, v) for k, v in opts if v is not None]
    return min(opts, key=lambda kv: kv[1])[0] if opts else None


KNOB_LABEL = {DFLT: "默认", SUB1: "`EP_NUM_SUB_PARTS=1`"}


def bo(arm, tok, knob):
    """(arm's layer total, main's layer total) on the reps the two share -- §9.8's table.

    main is RE-averaged on each row's rep set; carrying one main mean across rows with
    different reps is exactly the drift the generator's rep filtering exists to stop.
    """
    only = basis(tok) & {rep for rep, _ in gs.cell(arm, tok, knob)}
    return gs.layer(arm, tok, knob, only), gs.layer(MAIN, tok, DFLT, only)


def sv(arm, tok, op, knob, only):
    return gs.us(arm, tok, op, knob, only)


# The additivity table is a difference of differences, so it uses the reps ALL FOUR arms
# share, which is a stricter set than basis().
ADD = gs.shared_reps(128, lambda _arm: DFLT)


def add_row(op):
    """(main, #1+#2, #8+#9, expected, stack, residual, share of larger single win)"""
    def val(arm):
        return (gs.layer(arm, 128, DFLT, ADD) if op == "layer"
                else sv(arm, 128, op, DFLT, ADD))
    m, a, b, s = val(MAIN), val(PR12), val(PR89), val(STACK)
    if None in (m, a, b, s):
        return (None,) * 7
    exp = m + (a - m) + (b - m)
    return m, a, b, exp, s, s - exp, 100.0 * (s - exp) / max(abs(a - m), abs(b - m))


# ------------------------------------------------------- §9.9: the overlap campaign
def vo(arm, tok, ovlp, op):
    return go.g.stat(go.cell(arm, tok, ovlp), op)[0]


def lo(arm, tok, ovlp):
    d, c = vo(arm, tok, ovlp, "dispatch"), vo(arm, tok, ovlp, "reduced combine")
    return None if None in (d, c) else d + c


# ----------------------------------------------------- §9.10: the b300 stack campaign
# Every row of §9.10's table is at the DEFAULT geometry -- one knob for the whole table,
# so the rows stay comparable to each other. The one cell where SUB1 would win (#1+#2's
# prefill) is quoted in the section's own prose and checked there, at the knob it names.
def vb(arm, tok, op, knob=DFLT):
    only = ({rep for rep, _ in gb.cell(MAIN, tok, DFLT)}
            & {rep for rep, _ in gb.cell(arm, tok, knob)})
    return gb.us(arm, tok, op, knob, only)


def lb(arm, tok, knob=DFLT):
    d, c = vb(arm, tok, "dispatch", knob), vb(arm, tok, "reduced combine", knob)
    return None if None in (d, c) else d + c


# The b300 24 SM campaign. n=1, so unlike vb() there is no shared-rep intersection to take
# -- but the SM count is an argument rather than a default, because this campaign's whole
# point is that 12 and 24 are different measurements of the same arm.
def v24(arm, op, sm=24, tok=8192):
    return g24.us(arm, sm, tok, op)


def s24(arm, op, sm=24, tok=8192):
    return g24.so(arm, sm, tok, op)


BADD = gb.shared_reps(128, lambda _arm: DFLT)


def badd_row(op):
    """b300 additivity at 128 tok: (residual us, residual as % of the larger win)"""
    def val(arm):
        return gb.us(arm, 128, op, DFLT, BADD)
    m, a, b, s = val(MAIN), val(PR12), val(PR89), val(STACK)
    if None in (m, a, b, s):
        return None, None
    res = s - (m + (a - m) + (b - m))
    return res, SHARE(res, max(abs(a - m), abs(b - m)))


def RATIO2(a, b):
    """§9.10's cross-machine ratios are quoted to 2 decimals (1.63x, not 1.6x)."""
    return None if None in (a, b) else "%.2f×" % (a / b)


# ---------------------------------------------------------------- the claims
CLAIMS = []


def claim(section, anchor, values, nth=1, span=1):
    """values must all appear in the `span` lines starting at the `nth` anchor match."""
    CLAIMS.append((section, anchor, nth, span, values))


# --- intro PR table. All four rows are ONE campaign (the stack one), each arm at the
#     knob §9.8's BEST OPERATING POINT picks for it, so the table is internally
#     comparable rather than assembled from whichever campaign flattered each arm.
_m128, _m8192 = bo(MAIN, 128, DFLT)[0], bo(MAIN, 8192, DFLT)[0]
claim("intro", r"^\| `main` \|", [N(_m128), N(_m8192)])
for pat, arm, toks in ((r"^\| `#1\+#2`", PR12, (128,)),
                       (r"^\| `#8\+#9`", PR89, (128, 8192)),
                       (r"^\| 四个叠加", STACK, (128, 8192))):
    vals = []
    for tok in toks:
        v, base = bo(arm, tok, best_knob(arm, tok))
        vals += [N(v), P(v, base)]
    claim("intro", pat, vals)

# --- §9.7: the 2N table, then the 4N table (same row labels, so nth selects which)
for nth, nodes in ((1, 2), (2, 4)):
    for op in ("dispatch", "cached dispatch", "combine", "reduced combine"):
        if nodes == 4 and op == "combine":
            continue                     # the 4N table prints four rows, not five
        m8, p8 = v3(MAIN, nodes, 8192, op), v3(PR89, nodes, 8192, op)
        m1, p1 = v3(MAIN, nodes, 128, op), v3(PR89, nodes, 128, op)
        claim("9.7", r"^\| %s \|" % re.escape(op),
              [N(m8), N(p8), P(p8, m8), N(m1), N(p1), P(p1, m1)],
              nth=1 if op == "combine" else nth)
    m8, p8 = l3(MAIN, nodes, 8192), l3(PR89, nodes, 8192)
    m1, p1 = l3(MAIN, nodes, 128), l3(PR89, nodes, 128)
    claim("9.7", r"^\| \*\*层总时间\*\* \|",
          [N(m8), N(p8), P(p8, m8), N(m1), N(p1), P(p1, m1)], nth=nth)

# §9.7 conclusion 2: #8+#9 vs #1+#2 at #1+#2's own best knob, 2N then 4N
claim("9.7", r"所以 2 节点 decode 上",
      [N(l3(PR89, 2, 128)), N(l3(PR12, 2, 128, SUB1)),
       N(v3(PR12, 2, 128, "dispatch", SUB1)),
       N(v3(PR12, 2, 128, "reduced combine", SUB1))], span=3)
claim("9.7", r"4 节点上两者打平",
      [N(v3(PR12, 4, 128, "dispatch", SUB1)),
       N(v3(PR12, 4, 128, "reduced combine", SUB1)),
       N(l3(PR12, 4, 128, SUB1)), N(l3(PR89, 4, 128))], span=2)

# §9.7 conclusion 3: every ratio is a shrinkage of a percentage, so both are checked
_sh = []
for what in ("layer8192", "layer128", "disp128"):
    if what == "layer8192":
        a2, b2 = l3(PR89, 2, 8192), l3(MAIN, 2, 8192)
        a4, b4 = l3(PR89, 4, 8192), l3(MAIN, 4, 8192)
    elif what == "layer128":
        a2, b2 = l3(PR89, 2, 128), l3(MAIN, 2, 128)
        a4, b4 = l3(PR89, 4, 128), l3(MAIN, 4, 128)
    else:
        a2, b2 = v3(PR89, 2, 128, "dispatch"), v3(MAIN, 2, 128, "dispatch")
        a4, b4 = v3(PR89, 4, 128, "dispatch"), v3(MAIN, 4, 128, "dispatch")
    r2, r4 = 100.0 * (a2 - b2) / b2, 100.0 * (a4 - b4) / b4
    _sh += [P(a2, b2), P(a4, b4), RATIO(abs(r2), abs(r4))]
_p2 = v3(PR12, 2, 128, "dispatch"), v3(MAIN, 2, 128, "dispatch")
_p4 = v3(PR12, 4, 128, "dispatch"), v3(MAIN, 4, 128, "dispatch")
_sh += [P(*_p2), P(*_p4),
        RATIO(abs(100.0 * (_p2[0] - _p2[1]) / _p2[1]),
              abs(100.0 * (_p4[0] - _p4[1]) / _p4[1]))]
claim("9.7", r"收益随规模缩水", _sh, span=4)

# --- §9.8 best operating point (128 tok)
for pat, arm in ((r"^\| `main` \|", MAIN), (r"^\| `#1\+#2` \|", PR12),
                 (r"^\| `#8\+#9` \|", PR89), (r"^\| \*\*叠加\*\* \|", STACK)):
    knob = best_knob(arm, 128)
    only = basis(128) & {rep for rep, _ in gs.cell(arm, 128, knob)}
    v, base = bo(arm, 128, knob)
    # the knob label is checked like a number: a row naming the wrong geometry is the
    # same class of error as one quoting the wrong microseconds.
    vals = [KNOB_LABEL[knob], N(sv(arm, 128, "dispatch", knob, only)),
            N(sv(arm, 128, "reduced combine", knob, only)), N(v)]
    if arm != MAIN:
        vals.append(P(v, base))
    claim("9.8", pat, vals)

# --- §9.8 additivity
for op, pat in (("dispatch", r"^\| dispatch \|"),
                ("reduced combine", r"^\| reduced combine \|"),
                ("layer", r"^\| 层总时间 \|")):
    m, a, b, exp, s, res, share = add_row(op)
    vals = [N(m), N(a), N(b), N(exp), N(s), N(res)]
    if op != "reduced combine":
        vals.append(SHARE(res, max(abs(a - m), abs(b - m))))
    claim("9.8", pat, vals)

# §9.8 prose: "best of each, not the sum", and the env knob only the stack can use
_d_stack_dflt = sv(STACK, 128, "dispatch", DFLT, ADD)
_d_stack_sub1 = sv(STACK, 128, "dispatch", SUB1, ADD)
_stack_bo = bo(STACK, 128, SUB1)[0]
claim("9.8", r"各取其优",
      [N(add_row("reduced combine")[5]), N(add_row("dispatch")[5]),
       SHARE(add_row("dispatch")[5],
             max(abs(add_row("dispatch")[1] - add_row("dispatch")[0]),
                 abs(add_row("dispatch")[2] - add_row("dispatch")[0]))),
       N(_d_stack_dflt), N(sv(PR12, 128, "dispatch", DFLT, ADD)),
       # each arm at its own published operating point, i.e. the intro table's rows
       P(_stack_bo, bo(PR12, 128, SUB1)[0]), N(bo(PR12, 128, SUB1)[0]),
       P(_stack_bo, bo(PR89, 128, DFLT)[0]), N(bo(PR89, 128, DFLT)[0]),
       N(_d_stack_dflt - _d_stack_sub1), N(_d_stack_sub1)], span=7)

# §9.8 prefill: the stack adds nothing on top of #8+#9
_pre = basis(8192)
claim("9.8", r"prefill 8192 tok 完全是",
      [N(_m8192), N(bo(PR89, 8192, DFLT)[0]), N(bo(STACK, 8192, DFLT)[0]),
       P(bo(PR89, 8192, DFLT)[0], _m8192),
       N(sv(MAIN, 8192, "reduced combine", DFLT, _pre)),
       N(sv(PR89, 8192, "reduced combine", DFLT, _pre)),
       N(sv(MAIN, 8192, "dispatch", DFLT, _pre)),
       N(sv(STACK, 8192, "dispatch", DFLT, _pre))], span=4)

# --- §9.9 bracket table: one row per (tok, op), each arm compared only within an ovlp
for tok in (128, 8192):
    for op in ("dispatch", "reduced combine"):
        m0, p0 = vo(MAIN, tok, 0, op), vo(PR89, tok, 0, op)
        m1, p1 = vo(MAIN, tok, 1, op), vo(PR89, tok, 1, op)
        claim("9.9", r"^\| %d %s \|" % (tok, re.escape(op)),
              [N(m0), N(p0), P(p0, m0), N(m1), N(p1), P(p1, m1)])

# §9.9 the split, which is the section's finding
_d0 = vo(PR89, 128, 0, "dispatch") - vo(MAIN, 128, 0, "dispatch")
_d1 = vo(PR89, 128, 1, "dispatch") - vo(MAIN, 128, 1, "dispatch")
claim("9.9", r"decode dispatch\*\*：",
      [N(-_d0), SHARE(_d1, _d0), N(-_d1), SHARE(_d0 - _d1, _d0), N(_d1 - _d0)], span=2)
_c0 = vo(PR89, 8192, 0, "reduced combine") - vo(MAIN, 8192, 0, "reduced combine")
_c1 = vo(PR89, 8192, 1, "reduced combine") - vo(MAIN, 8192, 1, "reduced combine")
claim("9.9", r"prefill reduced combine\*\*：",
      [N(-_c0), SHARE(_c0 - _c1, _c0), N(_c1 - _c0)])

# §9.9 the flag read as a configuration (within one arm, never across arms)
claim("9.9", r"顺带：这个 flag 本身是一根性能轴",
      [N(lo(MAIN, 8192, 0)), N(lo(MAIN, 8192, 1)),
       P(lo(MAIN, 8192, 1), lo(MAIN, 8192, 0)),
       N(lo(MAIN, 128, 0)), N(lo(MAIN, 128, 1)),
       P(lo(MAIN, 128, 1), lo(MAIN, 128, 0)),
       P(lo(PR89, 8192, 1), lo(PR89, 8192, 0)),
       P(lo(PR89, 128, 1), lo(PR89, 128, 0)),
       P(vo(MAIN, 8192, 1, "dispatch"), vo(MAIN, 8192, 0, "dispatch")),
       N(vo(MAIN, 8192, 0, "dispatch")), N(vo(MAIN, 8192, 1, "dispatch")),
       P(vo(MAIN, 8192, 1, "reduced combine"),
         vo(MAIN, 8192, 0, "reduced combine")),
       N(vo(MAIN, 8192, 0, "reduced combine")),
       N(vo(MAIN, 8192, 1, "reduced combine"))], span=5)

# --- §9.10: the b300 table, all four rows at the default geometry
def sp(arm, tok, op, knob=DFLT):
    """the p5en counterpart of vb(): same op, same rep basis rule, other machine"""
    return sv(arm, tok, op, knob, basis(tok) & {rep for rep, _ in gs.cell(arm, tok, knob)})


def b300(values, zh, en, span=1, zh_span=None, en_span=None):
    """One b300 claim, registered against §9.10 AND the README's b300 section.

    Same value list for both, so the two documents cannot disagree about the campaign --
    which is the failure mode of publishing one campaign in two languages.
    """
    claim("9.10", zh, values, span=zh_span or span)
    claim("README/b300", en, values, span=en_span or span)


ROWS = ((MAIN, r"^\| `main` `54fffef` \|", r"^\| `main` \|"),
        (PR12, r"^\| `#1\+#2` `bfbdd15` \|", r"^\| #1\+#2 \|"),
        (PR89, r"^\| `#8\+#9` `3c737dc` \|", r"^\| #8\+#9 \|"),
        (STACK, r"^\| \*\*叠加\*\* `a35285f` \|", r"^\| \*\*stack\*\* \|"))
for arm, zh, en in ROWS:
    vals = []
    for tok in (128, 8192):
        for get in (lambda a, t: vb(a, t, "dispatch"), lb):
            v, base = get(arm, tok), get(MAIN, tok)
            vals.append(N(v))
            if arm != MAIN:
                vals.append(P(v, base))
    b300(vals, zh, en)

# the one cell where the table's single knob is not that arm's optimum -- named in the
# caveat sentence above the table, so it is checked there too
b300([N(lb(PR12, 8192, SUB1)), P(lb(PR12, 8192, SUB1), lb(MAIN, 8192))],
     r"全表只有一格默认不是最优", r"would beat the default is", span=2)

# bullet 1: #1+#2 pays double here, because b300's *unpatched* decode dispatch is the
# slower one. Both machines' numbers are recomputed, so the 1.63x cannot drift.
_bd, _pd = vb(MAIN, 128, "dispatch"), sp(MAIN, 128, "dispatch")
b300([P(vb(PR12, 128, "dispatch"), _bd), N(_pd), N(sp(PR12, 128, "dispatch")),
      P(sp(PR12, 128, "dispatch"), _pd), RATIO2(_bd, _pd), N(_bd),
      N(vb(STACK, 128, "dispatch")), N(sp(STACK, 128, "dispatch"))],
     r"在 b300 上值双倍", r"is worth roughly double on b300", zh_span=4, en_span=5)

# bullet 2: the combine win halves, and the reason is in main's own combine
_brc, _prc = vb(MAIN, 8192, "reduced combine"), sp(MAIN, 8192, "reduced combine")
b300([P(vb(PR89, 128, "reduced combine"), vb(MAIN, 128, "reduced combine")),
      N(vb(MAIN, 128, "reduced combine")), N(vb(PR89, 128, "reduced combine")),
      P(sp(PR89, 128, "reduced combine"), sp(MAIN, 128, "reduced combine")),
      P(vb(PR89, 8192, "reduced combine"), _brc), N(_brc),
      N(vb(PR89, 8192, "reduced combine")),
      P(sp(PR89, 8192, "reduced combine"), _prc), N(_prc), RATIO2(_prc, _brc)],
     r"的 combine 收益腰斩", r"combine win roughly halves", zh_span=3, en_span=4)

# bullet 3: a prefill dispatch win that the p5en cell does not have
_bpd, _ppd = vb(MAIN, 8192, "dispatch"), sp(MAIN, 8192, "dispatch")
b300([N(_bpd), N(vb(PR89, 8192, "dispatch")), P(vb(PR89, 8192, "dispatch"), _bpd),
      P(sp(PR89, 8192, "dispatch"), _ppd), N(_ppd), N(sp(PR89, 8192, "dispatch"))],
     r"上不存在的 prefill", r"win that does not exist on p5en", span=3)

# bullet 4: the knob axis. Its sign flips between #1+#2 alone and the stack, so every one
# of the cells it is claimed from is checked, not just the headline.
_s_dflt, _s_sub1 = vb(STACK, 8192, "dispatch"), vb(STACK, 8192, "dispatch", SUB1)
_a_dflt, _a_sub1 = vb(PR12, 8192, "dispatch"), vb(PR12, 8192, "dispatch", SUB1)
b300([N(vb(STACK, 128, "dispatch", SUB1)), N(vb(STACK, 128, "dispatch")),
      N(vb(PR12, 128, "dispatch", SUB1)), N(vb(PR12, 128, "dispatch")),
      N(_d_stack_dflt - _d_stack_sub1),             # what it was worth on p5en
      N(_a_sub1), N(_a_dflt), N(_a_sub1 - _a_dflt),
      N(_s_sub1 - _s_dflt, 0), N(_s_sub1), N(_s_dflt)],
     r"在 b300 decode 上是零", r"buys nothing on b300 decode", zh_span=6, en_span=8)

# additivity + the negative control
_bres, _bshare = badd_row("dispatch")
_mc_dflt, _mc_sub1 = vb(MAIN, 128, "dispatch"), vb(MAIN, 128, "dispatch", SUB1)
b300([N(_bres), _bshare,
      SHARE(add_row("dispatch")[5],
            max(abs(add_row("dispatch")[1] - add_row("dispatch")[0]),
                abs(add_row("dispatch")[2] - add_row("dispatch")[0]))),
      N(badd_row("reduced combine")[0]),
      P(_mc_sub1, _mc_dflt), N(_mc_sub1), N(_mc_dflt)],
     r"^\*\*叠加性\*\*", r"^\*\*Additivity\.\*\*", zh_span=4, en_span=5)

# --- the repo-level README: the same two campaigns, quoted as a best-of instead of a
# four-arm decomposition. Its decode row is the p5en `EP_NUM_SUB_PARTS=1` cell and the
# b300 default cell, because those are each machine's optimum -- so unlike §9.10 this
# table is NOT one knob, and each cell is checked at the knob its own footnote names.
def G(x):
    """SO GB/s as the repo README prints it: 2 decimals, because the interesting b300
    dispatch difference (128.90 vs the older 24 SM 125) is smaller than 1 GB/s wide."""
    return None if x is None else "%.2f" % x


if HAVE_TOP:
    _bc8 = vb(MAIN, 8192, "combine")     # unweighted combine: the op the GB/s table uses
    _pc8 = sp(MAIN, 8192, "combine")
    _pd1 = sp(STACK, 128, "dispatch", SUB1)
    _pc1 = sp(STACK, 128, "reduced combine", SUB1)
    _bd1, _bc1 = vb(STACK, 128, "dispatch"), vb(STACK, 128, "reduced combine")
    _pstep0 = sp(MAIN, 128, "dispatch") + sp(MAIN, 128, "reduced combine")
    _bstep0 = vb(MAIN, 128, "dispatch") + vb(MAIN, 128, "reduced combine")
    claim("top", r"^\| \*\*DeepEP V2, released EFA 1\.50\.0 \+ 4 upstream PRs\*\* \(2026-09\)",
          [N(_pd1), N(_pc1), N(_bd1), N(_bc1)])

    # Cross-stack ratios: one side is ours and recomputed, the other is a constant from
    # the campaign that owns it. Declared here rather than inlined so that a ratio can
    # never be "verified" against a number this file also made up.
    KINETO_P5EN_D, KINETO_P5EN_C = 151.6, 189.6     # route B, bench_kineto op-level
    PPLX_P5EN, PPLX_B300 = 222.0 + 245.0, 140.0 + 149.0   # pplx-garden p50 steps
    UCCL_P5EN_D, PPLX_P5EN_D = 207.0, 222.0         # the two pre-GDAKI p5en dispatches
    UCCL_B300_STEP = 439.0                          # UCCL's own b300 dispatch+combine loop
    # Route B's b300 decode cells are NOT bench_kineto -- they are test_ep.py's `expanded
    # dispatch` / `reduced combine`, same clock and same op variant as our arm, so the two
    # rows are directly subtractable once the SM count matches. Re-parsed from
    # deepep-v2-efa-gdaki-b200/results/b300_20260813/b300_tok_{64,12}sm_128_rank0.log
    # (`#SM: 55` and `#SM: 12`, that node's 8 of 16 ranks). Constants because that
    # directory is local-only and gitignored; only OUR side of every ratio is recomputed.
    # Carried to 3 decimals so a published step is the sum of the MEANS, not of two
    # rounded cells -- 207.0 + 196.4 would print 403.4 where the run is 403.5.
    GDAKI_B300_55SM_D, GDAKI_B300_55SM_C = 200.445, 160.085
    GDAKI_B300_12SM_D, GDAKI_B300_12SM_C = 207.044, 196.414
    # Route B's 8192-token SM sweep, b300_pfsm_p1_{12,24,48,64}_rank{0,1}.log -- 16 ranks
    # pooled over both nodes, unweighted `combine` (the op the throughput table uses).
    # Its 24 SM cell is what makes the sweep quotable next to the published 8-rank
    # 131 GB/s / 1788.1 us cell: same arm, 0.3% apart.
    GDAKI_PF_12SM_D, GDAKI_PF_12SM_C = 1147.562, 2788.438
    GDAKI_PF_55SM_D = 822.109
    GDAKI_PF_24SM_C, GDAKI_PF_48SM_C, GDAKI_PF_55SM_C = 1783.000, 1883.500, 1760.688
    GDAKI_PF_24SM_D, GDAKI_PF_48SM_D = 981.092, 809.218
    GDAKI_PF_12SM_C_GB, GDAKI_PF_24SM_C_GB = 84.12, 131.62
    GDAKI_PF_24SM_D_GB = 124.44
    KINETO_B300_D, KINETO_B300_C = GDAKI_B300_55SM_D, GDAKI_B300_55SM_C
    _g12 = GDAKI_B300_12SM_D + GDAKI_B300_12SM_C
    _g55 = GDAKI_B300_55SM_D + GDAKI_B300_55SM_C
    _bed1 = vb(STACK, 128, "expanded dispatch")   # matched op for the 12 SM comparison

    claim("top", r"^‡ \*\*The `DeepEP V2` decode row above is obsolete",
          [RATIO2(KINETO_P5EN_D, _pd1), RATIO2(KINETO_B300_D, _bd1),
           N(GDAKI_B300_55SM_D), N(GDAKI_B300_55SM_C),
           N(GDAKI_B300_12SM_D), N(GDAKI_B300_12SM_C), N(_g12),
           RATIO2(UCCL_B300_STEP, _g12)], span=41)

    claim("top", r"^§ \*\*This is the fastest decode we have measured",
          [N(sp(STACK, 128, "combine", SUB1)), N(vb(STACK, 128, "combine")),
           N(_pc1 - sp(STACK, 128, "combine", SUB1)),
           N(_bc1 - vb(STACK, 128, "combine")),
           N(_bc1), N(_pd1), N(_pc1), N(_bd1),
           RATIO2(KINETO_P5EN_D, _pd1), RATIO2(KINETO_B300_D, _bd1),
           N(_bd1 + _bc1), N(_pd1 + _pc1), RATIO2(PPLX_P5EN, _pd1 + _pc1),
           N(_d_stack_dflt - _d_stack_sub1),
           N(vb(STACK, 128, "dispatch", SUB1)),
           N(_pd1 + _pc1), N(_bd1 + _bc1),
           P(_pd1, sp(MAIN, 128, "dispatch")), P(_bd1, vb(MAIN, 128, "dispatch")),
           N(sp(MAIN, 128, "dispatch")), N(vb(MAIN, 128, "dispatch")),
           P(_pd1 + _pc1, _pstep0), P(_bd1 + _bc1, _bstep0),
           RATIO2(vb(MAIN, 128, "dispatch"), sp(MAIN, 128, "dispatch")),
           # the matched-12 SM b300 comparison against route B
           N(GDAKI_B300_12SM_D), N(GDAKI_B300_12SM_C), N(_g12),
           N(_bed1), N(_bed1 + _bc1),
           P(_bed1, GDAKI_B300_12SM_D), P(_bc1, GDAKI_B300_12SM_C),
           P(_bed1 + _bc1, _g12),
           N(_bed1 - _bd1, 1),                       # expanded vs plain dispatch
           P(_bed1, GDAKI_B300_55SM_D), P(_bed1 + _bc1, _g55)],
          span=45)
    # the b300 step now leads pplx's, which is the strongest claim in the section: assert
    # the sign here so the prose cannot say "past pplx" off a number that no longer is.
    assert _bd1 + _bc1 < PPLX_B300, "b300 step no longer beats pplx: reword the highlight"
    assert _bc1 > 149.0, "b300 combine now beats pplx too: move that star as well"

    claim("top", r"^\*\*B300 highlights\*\*",
          [N(_bd1), N(_bd1 + _bc1), N(_bc1),
           P(_bed1, GDAKI_B300_12SM_D), P(_bed1 + _bc1, _g12)],
          nth=2, span=6)   # nth=2: the throughput table has a highlights block too
    claim("top", r"^\*\*On p5en, UCCL-EP and pplx-garden are tied",
          [N(_pd1), RATIO2(UCCL_P5EN_D, _pd1), RATIO2(PPLX_P5EN_D, _pd1)], span=4)

    # --- the throughput table's row: GB/s AND the us behind it, both generated. The row is
    # SM-matched to the row above it per machine, so the two halves come from DIFFERENT
    # campaigns -- p5en at 12 SM from gs, b300 at 24 SM from g24 -- and each half is read
    # from the generator that owns that SM count. Reading b300 from gb here would silently
    # republish the 12 SM cells under a 24 SM label.
    row = []
    for op in ("dispatch", "combine"):
        row += [G(gs.so(STACK, 8192, op, DFLT, basis(8192))), N(sp(STACK, 8192, op))]
    for op in ("dispatch", "combine"):
        row += [G(s24(STACK, op)), N(v24(STACK, op))]
    claim("top",
          r"^\| \*\*DeepEP V2, released EFA 1\.50\.0 \+ 4 upstream PRs\*\* \(24 SM b300",
          row)

    claim("top", r"^§ \*\*Same arm and same denominator",
          [N(gs.mb(STACK, 8192, "dispatch"), 1), N(gs.mb(STACK, 8192, "combine"), 1),
           N(gb.mb(STACK, 8192, "combine"), 1)], span=16)

    # The two b300 prefill columns at 24 SM, against route B at the same SM and against
    # this campaign's own unpatched main. Every "ours" value comes from g24; route B's side
    # stays a constant, so a ratio is never checked against a number this file invented.
    _24d, _24c = v24(STACK, "dispatch"), v24(STACK, "combine")
    _24md, _24mc = v24(MAIN, "dispatch"), v24(MAIN, "combine")
    claim("top", r"^\*\*At 24 SM this row wins both b300 columns",
          [G(GDAKI_PF_24SM_D_GB), N(GDAKI_PF_24SM_D),
           G(GDAKI_PF_24SM_C_GB), N(GDAKI_PF_24SM_C),
           N(_24d), P(_24d, GDAKI_PF_24SM_D),
           N(_24c), P(_24c, GDAKI_PF_24SM_C),
           N(_24d + _24c), N(GDAKI_PF_24SM_D + GDAKI_PF_24SM_C),
           P(_24d + _24c, GDAKI_PF_24SM_D + GDAKI_PF_24SM_C),
           N(_24md), P(_24d, _24md), N(_24mc), P(_24c, _24mc)],
          span=9)

    claim("top", r"^\*\*Combine's SM curve is not monotone",
          [N(GDAKI_PF_12SM_C), N(GDAKI_PF_24SM_C), N(GDAKI_PF_48SM_C), N(GDAKI_PF_55SM_C),
           N(_24c), P(_24c, GDAKI_PF_55SM_C),
           N(GDAKI_PF_48SM_D), N(_24d), P(_24d, GDAKI_PF_48SM_D),
           P(_24d, GDAKI_PF_24SM_D),
           # the cross-campaign 12 -> 24 cross-validation, both sides on `combine`
           P(_24mc, vb(MAIN, 8192, "combine")), N(vb(MAIN, 8192, "combine")), N(_24mc),
           P(GDAKI_PF_24SM_C, GDAKI_PF_12SM_C), N(GDAKI_PF_12SM_C), N(GDAKI_PF_24SM_C),
           # p5en is untouched on dispatch and all-combine on the win
           P(sp(STACK, 8192, "dispatch"), _ppd), N(_ppd), N(sp(STACK, 8192, "dispatch")),
           P(sp(STACK, 8192, "combine"), _pc8), N(_pc8), N(sp(STACK, 8192, "combine")),
           G(gs.so(MAIN, 8192, "combine", DFLT, basis(8192))),
           G(gs.so(STACK, 8192, "combine", DFLT, basis(8192))),
           # the decode-at-24-SM signal that is NOT yet a result
           P(v24(PR12, "dispatch", tok=128), vb(PR12, 128, "dispatch")),
           N(vb(PR12, 128, "dispatch")), N(v24(PR12, "dispatch", tok=128)),
           P(v24(MAIN, "dispatch", tok=128), vb(MAIN, 128, "dispatch")),
           N(vb(MAIN, 128, "dispatch")), N(v24(MAIN, "dispatch", tok=128))],
          span=22)
    # The row's two b300 cells are now claimed as wins on both ops. Assert the signs, so the
    # prose cannot survive a re-measurement that flips one of them.
    assert _24d < GDAKI_PF_24SM_D and _24c < GDAKI_PF_24SM_C, \
        "b300 no longer wins both prefill columns at 24 SM: reword the § footnote"
    assert _24c < GDAKI_PF_55SM_C, "route B's 55 SM combine now wins: fix the 'every point' claim"
    assert _24d > GDAKI_PF_48SM_D, "our dispatch now beats route B's optimum too: say so"

    # the two Recommendations cells that now quote a step
    claim("top", r"^\| \*\*MoE inference, decode\*\*",
          [N(_pd1), N(_pc1), N(_pd1 + _pc1), RATIO2(PPLX_P5EN, _pd1 + _pc1),
           N(_bd1), N(_bc1), N(_bd1 + _bc1)])
    claim("top", r"^\| \*\*MoE inference, prefill\*\*",
          [N(v24(STACK, "dispatch") / 1000.0, 2), N(v24(STACK, "combine") / 1000.0, 2),
           P(v24(STACK, "dispatch"), GDAKI_PF_24SM_D),
           P(v24(STACK, "combine"), GDAKI_PF_24SM_C)])
    claim("top", r"^\| \*\*Very large EP", [N(_pd1 + _pc1), N(_bd1 + _bc1)])

# §9.10's table is only comparable if the whole thing is one knob: assert the doc's own
# caveat, rather than trusting the prose.
assert all(gb.cell(a, t, DFLT) for a in (MAIN, PR12, PR89, STACK) for t in (128, 8192)), \
    "a b300 arm is missing its default-knob cell: the table cannot be one-knob"


# ---------------------------------------------------------------- checking
def _find(lines, pred, what, path):
    # A renamed section must fail with its own name, not a bare StopIteration:
    # sections HAVE been renumbered here (the b300 one moved 9.7 -> 9.10).
    for i, l in enumerate(lines):
        if pred(l):
            return i
    raise SystemExit("check_runbook_numbers.py: cannot find %s in %s -- was the section "
                     "renamed? update the claims with it." % (what, path))


def sections():
    """key -> (lines of its document, first line, last line).

    intro = everything above §1; 9.x = the heading down to the next one; README/b300 =
    the README's b300 heading down to its next `## `.
    """
    rb = open(RUNBOOK, errors="replace").read().split("\n")
    out = {"intro": (rb, 0, _find(rb, lambda l: l.startswith("## 1."),
                                  "the `## 1.` heading", RUNBOOK))}
    for key in ("9.7", "9.8", "9.9", "9.10"):
        s = _find(rb, lambda l, k=key: l.startswith("### %s " % k), "section %s" % key,
                  RUNBOOK)
        e = next((i for i in range(s + 1, len(rb)) if rb[i].startswith("### ")), len(rb))
        out[key] = (rb, s, e)

    rm = open(README, errors="replace").read().split("\n")
    s = _find(rm, lambda l: l.startswith("### B300 results"), "the B300 results heading",
              README)
    e = next((i for i in range(s + 1, len(rm)) if rm[i].startswith("## ")), len(rm))
    out["README/b300"] = (rm, s, e)

    if HAVE_TOP:
        # the whole file: this campaign's numbers are spread over two tables, two
        # footnotes and the recommendations, and the anchors are unique on their own.
        rt = open(TOPREADME, errors="replace").read().split("\n")
        out["top"] = (rt, 0, len(rt))
    return out


def norm(s):
    """The runbook writes minus as U+2212, the generators as '-'."""
    return s.replace("−", "-").replace("–", "-")


def main():
    sec = sections()
    bad = missing = checked = 0
    for section, anchor, nth, span, values in CLAIMS:
        lines, lo_, hi_ = sec[section]
        rx = re.compile(anchor)
        hits = [i for i in range(lo_, hi_) if rx.search(lines[i])]
        if len(hits) < nth:
            print("NO ROW    [%s] %-34s -> %d matches, wanted #%d"
                  % (section, anchor, len(hits), nth))
            missing += 1
            continue
        at = hits[nth - 1]
        blob = norm("\n".join(lines[at:at + span]))
        for v in values:
            checked += 1
            if v is None:
                print("NO DATA   [%s] %-34s -- a cell is missing from logs/"
                      % (section, anchor))
                missing += 1
            elif v not in blob:
                print("MISMATCH  [%s] %-34s expected %s\n    %s"
                      % (section, anchor, v, norm(lines[at]).strip()))
                bad += 1
    print("%d values checked across %d rows, %d MISMATCHES, %d unresolvable"
          % (checked, len(CLAIMS), bad, missing))
    return 1 if (bad or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
