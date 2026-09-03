#!/usr/bin/env python3
"""Build docs/slides_zh.html -- a self-contained Chinese slide deck.

usage: python3 docs/build_slides_html.py [-o out.html]

WHY THIS IS A GENERATOR AND NOT A HAND-WRITTEN FILE
---------------------------------------------------
Hand-copied benchmark numbers survive every review: the table looks plausible, so
nobody re-derives it, and a stale or mis-pasted cell outlives the campaign that
produced it. So every performance number in the deck is computed here from the
logs under results/, at build time:

  * the 3-arm campaign (main / PR #1+#2 / PR #8+#9, 2N and 4N, two knob
    settings) comes from results/p5en_3arm_20260831/ through that directory's own
    make_3arm_tables.py, i.e. the SAME aggregation its comparison.md and
    check_comparison.py use. There is no second implementation to drift.
  * the type-2-vs-type-5 backend table comes from results/p5en_2n4n_20260825/
    through THAT directory's make_tables.py, reusing its own arm definitions
    (O2_12_8192_T2 & co). Re-globbing those logs from a guessed tag pattern is a
    trap and was tried first: the 2N type-2 arm is `official_..._nodbg_rep1`, and
    a pattern without `_nodbg` silently picks up the EP_BUFFER_DEBUG-on runs
    instead (which printf inside dispatch's timed loop). The result is then
    CROSS-CHECKED against that campaign's published summary.txt; a mismatch
    aborts the build.

Anything that is genuinely external (version numbers, the dependency chain) is a
literal in EXTERNAL below, each carrying its own source string, and the source is
rendered onto the slide so a reader can check it.

Deliberately NOT in this deck: a flattened cross-stack table. README.md's
'Side-by-side results' section states that its row groups are different shapes and
different clocks and must not be compared column-wise, so flattening them into
one slide table would delete that warning. The cross-stack slide states the three
things to align and points at the README.
"""
import argparse
import glob
import html
import importlib.util
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R3ARM = os.path.join(ROOT, "results", "p5en_3arm_20260831")
R0825 = os.path.join(ROOT, "results", "p5en_2n4n_20260825")

# Point the 3-arm generator at its own logs BEFORE importing it: it reads EPRUNS
# at module level, so setting this afterwards would silently look in the wrong dir.
# Assigned, not setdefault: an inherited EPRUNS from the shell must not redirect it.
os.environ["EPRUNS"] = os.path.join(R3ARM, "logs")
_spec = importlib.util.spec_from_file_location(
    "gen3", os.path.join(R3ARM, "make_3arm_tables.py"))
g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g)


def _import_0825():
    """Import the 0825 generator for its arm definitions.

    It calls main() at module scope (it is a print-a-report script, not a
    library), so stdout is swallowed during import -- otherwise the whole of
    summary.txt lands in our build output.
    """
    saved, devnull = sys.stdout, open(os.devnull, "w")
    prev = os.environ.get("EPRUNS")
    os.environ["EPRUNS"] = os.path.join(R0825, "logs")
    try:
        sys.stdout = devnull
        s = importlib.util.spec_from_file_location(
            "gen0825", os.path.join(R0825, "make_tables.py"))
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
        return m
    finally:
        sys.stdout = saved
        devnull.close()
        if prev is None:
            del os.environ["EPRUNS"]
        else:
            os.environ["EPRUNS"] = prev


g8 = _import_0825()


def _import_campaign(subdir, script, envvar, name):
    """Import a campaign's own generator, pointed at its own logs.

    Each of these sets EPRUNS itself from `envvar` and then loads its OWN copy of
    make_3arm_tables.py, so the aggregation is shared by import while the log
    directories stay separate. They have __main__ guards, so importing prints nothing.
    """
    d = os.path.join(ROOT, "results", subdir)
    os.environ[envvar] = os.path.join(d, "logs")
    s = importlib.util.spec_from_file_location(name, os.path.join(d, script))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


gS = _import_campaign("p5en_stack_20260831", "make_stack_tables.py",
                      "EPRUNS_STACK", "genstack")
gO = _import_campaign("p5en_ovlp_20260831", "make_ovlp_tables.py",
                      "EPRUNS_OVLP", "genovlp")

MAIN, PR12, PR89 = "main54fffef", "pr12bfbdd15", "pr893c737dc"
STACK = gS.STACK
LABEL = {MAIN: "main", PR12: "PR #1+#2", PR89: "PR #8+#9", STACK: "四个叠加"}
DFLT, SUB1 = g.DEFAULT_KNOB, g.SUBPARTS1

EXTERNAL = {
    "chain": {
        "src": "EFA installer 1.50.0 / 镜像构建期 gate（build_image.sh）",
        "rows": [
            ("DeepEP V2 unordered kernels", "amazon-contributing/DeepEP", "—"),
            ("NCCL GIN", "≥ 2.30.4（实际 2.31.2）", "有 nccl_device.h"),
            ("aws-ofi-nccl", "<b>1.21.1</b>", "nm 里有 ncclGinPlugin_v14"),
            ("libfabric", "2.6.0amzn1.0", "16 个 fabric: efa-direct"),
            ("rdma-core", "64.0amzn0", "20 个 comp_cntr 符号"),
            ("efa 内核驱动", "<b>3.3.0</b>", "EFA_QUERY_DEVICE_CAPS_COMP_CNTR"),
        ],
    },
}


# ------------------------------------------------------------------ 3-arm data
def us(arm, n, tok, op, knob=DFLT):
    m, _kept, _short = g.stat(g.cell(arm, n, tok, knob), op)
    if m is None:
        raise SystemExit("3arm: no data for %s %dN %dtok %s %s"
                         % (arm, n, tok, knob, op))
    return m


def so_range(arm, n, tok, op, knob=DFLT):
    lo, hi = g.rng(g.cell(arm, n, tok, knob), op, 0)
    return "—" if lo is None else ("%d" % lo if lo == hi else "%d–%d" % (lo, hi))


def layer(arm, n, tok, knob=DFLT):
    """dispatch + reduced combine: the two calls one MoE layer actually makes."""
    return us(arm, n, tok, "dispatch", knob) + us(arm, n, tok, "reduced combine", knob)


def cfg_of(arm, n, tok, knob=DFLT):
    """(#SM, #QPs) as the logs printed it -- #QPs is how PR #9's bump is visible."""
    sms, qps = set(), set()
    for _rep, (_ops, _W, _bn, cfg, _ref, _t) in g.cell(arm, n, tok, knob):
        for sm, q, qtot in cfg:
            sms.add(sm)
            qps.add("%d/%d" % (q, qtot))
    return ("/".join(str(x) for x in sorted(sms)) or "—",
            ",".join(sorted(qps)) or "—")


def node_spread(arm, n, tok, op, knob=DFLT):
    """per-rep (max-min)/min across per-NODE means, in %.

    combine splits by machine, so this is the statistic PR #8 claims to move. It
    must be computed per node from that node's own log -- pooling all ranks first
    averages the layering away.
    """
    out = []
    for _rep, (_ops, _W, bynode, _c, _r, _t) in g.cell(arm, n, tok, knob):
        means = [st.mean(v[op]) for v in bynode.values() if op in v]
        if len(means) >= 2:
            out.append(100.0 * (max(means) - min(means)) / min(means))
    return out


def pct(new, base):
    return "—" if base in (None, 0) else "%+.1f%%" % (100.0 * (new - base) / base)


# ------------------------------------------------- stack campaign (4 arms, 2N)
# Its own campaign, never mixed with the 3-arm one: additivity is a difference of
# differences, the least forgiving thing to assemble across two node pairs and two
# days. Also cut short (15 of 30 cells ran), so every cross-arm value below is
# restricted to the reps the compared arms SHARE -- gS does that, we only ask for it.
def sbasis(tok):
    """the rep set §9.8's tables hold every row to: main's own reps"""
    return {rep for rep, _ in gS.cell(MAIN, tok, DFLT)}


def sknob(arm, tok):
    """the knob this arm would be DEPLOYED at -- per token size, not per arm.

    The stack's prefill is fastest at the default geometry and its decode at
    EP_NUM_SUB_PARTS=1; carrying one across would misreport the other by ~90 us.
    """
    ks = (DFLT, SUB1) if arm in gS.TUNABLE else (DFLT,)
    opts = [(k, gS.layer(arm, tok, k)) for k in ks]
    opts = [(k, v) for k, v in opts if v is not None]
    if not opts:
        raise SystemExit("stack: %s has no data at %d tok" % (arm, tok))
    return min(opts, key=lambda kv: kv[1])[0]


def sonly(arm, tok, knob=None):
    """(knob, the reps this arm shares with main) -- one basis per row."""
    knob = sknob(arm, tok) if knob is None else knob
    return knob, sbasis(tok) & {rep for rep, _ in gS.cell(arm, tok, knob)}


def sus(arm, tok, op, knob=None):
    knob, only = sonly(arm, tok, knob)
    v = gS.us(arm, tok, op, knob, only)
    if v is None:
        raise SystemExit("stack: no %s for %s %dtok %s" % (op, arm, tok, knob))
    return v


def slayer(arm, tok, knob=None):
    return sus(arm, tok, "dispatch", knob) + sus(arm, tok, "reduced combine", knob)


def sbase(arm, tok):
    """main's layer total RE-AVERAGED on this arm's rep basis.

    Not one main mean reused for every row: the arms have unequal rep counts here, and
    dividing an arm by a main mean taken over reps main-but-not-that-arm has would fold
    rep drift into the delta.
    """
    _knob, only = sonly(arm, tok)
    return gS.layer(MAIN, tok, DFLT, only)


# The additivity row needs a STRICTER basis: the reps all four arms share.
SADD = gS.shared_reps(128, lambda _arm: DFLT)


def add_row(op):
    """(main, #1+#2, #8+#9, expected, measured stack, residual) at 128 tok, one basis"""
    def val(arm):
        return (gS.layer(arm, 128, DFLT, SADD) if op == "layer"
                else gS.us(arm, 128, op, DFLT, SADD))
    m, a, b, s = (val(x) for x in (MAIN, PR12, PR89, STACK))
    if None in (m, a, b, s):
        raise SystemExit("stack: additivity not computable for %s" % op)
    exp = m + (a - m) + (b - m)
    return m, a, b, exp, s, s - exp


# ------------------------------------------- overlap campaign (the #8+#9 bracket)
def ous(arm, tok, ovlp, op):
    v = gO.g.stat(gO.cell(arm, tok, ovlp), op)[0]
    if v is None:
        raise SystemExit("ovlp: no %s for %s %dtok ovlp%d" % (op, arm, tok, ovlp))
    return v


def olayer(arm, tok, ovlp):
    return ous(arm, tok, ovlp, "dispatch") + ous(arm, tok, ovlp, "reduced combine")


# ------------------------------------------------------- 0825 backend-arm data
# The arms are the 0825 generator's OWN tag lists, not a pattern of our own. Its
# `arm()` is the same aggregation as the 3-arm module's: per-run all-rank mean,
# then mean over reps, pooled across every node's log.
A0825 = {
    ("type2", 2, 8192): g8.O2_12_8192_T2, ("type5", 2, 8192): g8.O2_12_8192_T5,
    ("type2", 2, 128):  g8.O2_12_128_T2,  ("type5", 2, 128):  g8.O2_12_128_T5,
    ("type2", 4, 8192): g8.O4_12_8192_T2, ("type5", 4, 8192): g8.O4_12_8192_T5,
    ("type2", 4, 128):  g8.O4_12_128_T2,  ("type5", 4, 128):  g8.O4_12_128_T5,
}
# summary.txt's published dispatch pairs, as an independent cross-check: if a key
# in A0825 is wired to the wrong arm, this fails instead of publishing quietly.
CHECK0825 = {("type2", 2, 8192): 1644.0, ("type5", 2, 8192): 1502.9,
             ("type2", 2, 128): 365.1,   ("type5", 2, 128): 169.1,
             ("type2", 4, 8192): 4315.0, ("type5", 4, 8192): 3955.3,
             ("type2", 4, 128): 1003.2,  ("type5", 4, 128): 184.3}


def us0825(arm, n, tok, op="dispatch"):
    v = g8.arm(A0825[(arm, n, tok)], op)[0]
    if v is None:
        raise SystemExit("0825: no %s for %s %dN %dtok" % (op, arm, n, tok))
    return v


def so0825(arm, n, tok, op="dispatch"):
    lo, hi = g8.rng(A0825[(arm, n, tok)], op, 0)
    return "—" if lo is None else ("%d" % lo if lo == hi else "%d–%d" % (lo, hi))


def verify0825():
    """Abort rather than publish if the re-derivation disagrees with summary.txt."""
    bad = []
    for key, claimed in sorted(CHECK0825.items()):
        got = us0825(*key)
        if abs(got - claimed) > 0.5:      # summary.txt prints one decimal
            bad.append("  %s: summary.txt %.1f us, logs %.1f us" % (key, claimed, got))
    if bad:
        raise SystemExit("0825 cross-check FAILED against %s/summary.txt:\n%s"
                         % (R0825, "\n".join(bad)))
    return len(CHECK0825)


# ------------------------------------------------------------------- HTML bits
def esc(s):
    return html.escape(str(s), quote=False)


def table(headers, rows, cls="", note=""):
    """rows may contain raw <b> markup, so cells are NOT escaped -- callers build
    them from formatted numbers and fixed labels, never from log text."""
    h = "".join("<th>%s</th>" % c for c in headers)
    body = "".join("<tr>%s</tr>" % "".join("<td>%s</td>" % c for c in r) for r in rows)
    out = '<table class="%s"><thead><tr>%s</tr></thead><tbody>%s</tbody></table>' % (
        cls, h, body)
    if note:
        out += '<p class="note">%s</p>' % note
    return out


SLIDES = []

# Slide numbers are ASSIGNED FROM THIS LIST, not typed into the titles. Inserting a
# slide used to mean renumbering every later title and every "见第 N 节" by hand, which
# is the same failure mode as hand-copying a benchmark number. Add a key here, in the
# position you want it, and both the titles and the cross-references follow.
ORDER = ["what", "chain", "env", "path", "host", "build", "first", "method",
         "overview", "prefill", "decode", "subparts", "stack", "ovlp",
         "layering", "scaling", "crossstack", "gotchas", "status"]
NUM = {k: i + 1 for i, k in enumerate(ORDER)}


def ref(*keys):
    """'第 13 节' / '第 8–16 节' -- cross-references are computed, never hand-typed.

    Concatenated into slide bodies rather than %-formatted, so adding a reference does
    not mean threading one more argument through a long % tuple in the right position.
    """
    return "第 %s 节" % "–".join(str(NUM[k]) for k in keys)


def slide(key, title, body, sub=""):
    """key=None for the unnumbered slides (cover, summary, refs)."""
    if key is not None:
        title = "%d · %s" % (NUM[key], title)
    SLIDES.append((key, title, sub, body))


# ===================================================================== slides
def build():
    nlogs = len(glob.glob(os.path.join(R3ARM, "logs", "*.log")))
    nclaims = len(re.findall(r"^\s*\(\"", open(
        os.path.join(R3ARM, "check_comparison.py")).read(), re.M))
    n0825 = verify0825()
    # How many runbook values the checker re-derives -- asked of the checker itself, so
    # the deck cannot quote a claim count the checker no longer makes.
    _s = importlib.util.spec_from_file_location(
        "runbookcheck", os.path.join(HERE, "check_runbook_numbers.py"))
    _chk = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(_chk)          # __main__-guarded: builds CLAIMS, checks nothing
    nrunbook = sum(len(vals) for _sec, _a, _n, _sp, vals in _chk.CLAIMS)

    def decay(knob):
        """how many times smaller the 4N win is than the 2N win, for PR #1+#2's
        dispatch lever -- computed, so the two knob settings stay comparable."""
        w2 = 1 - us(PR12, 2, 128, "dispatch", knob) / us(MAIN, 2, 128, "dispatch")
        w4 = 1 - us(PR12, 4, 128, "dispatch", knob) / us(MAIN, 4, 128, "dispatch")
        return w2 / w4

    def residual(op):
        """'+47.5 µs = 较大单臂收益的 85%' -- a residual in us alone is unreadable:
        47 us next to a 55 us win is a different statement than next to a 600 us one."""
        m, a, b, _exp, _s, res = add_row(op)
        return "%+.1f µs = 较大单臂收益的 %.0f%%" % (
            res, 100.0 * res / max(abs(a - m), abs(b - m)))

    def unguarded(tok, op):
        """share of #8+#9's win on this op that survives --prefer-overlap-with-compute=1.

        At =1 the channel-clamp removal and the forward-warp pairing are compiled out,
        so what is left is the QP bump and remote-first. A BRACKET, not an attribution:
        the flag also changes double-buffering, so arms are only ever compared inside
        one flag value -- both differences below are main-vs-#8+#9 at a fixed flag.
        """
        d0 = ous(PR89, tok, 0, op) - ous(MAIN, tok, 0, op)
        d1 = ous(PR89, tok, 1, op) - ous(MAIN, tok, 1, op)
        return "%.0f%%" % (100.0 * d1 / d0)

    # ---- 1 cover
    # Two ratios, each computed WITHIN one campaign. Dividing the 0825 type-2
    # number by a 0831 patched number would mix two code SHAs into one speedup.
    slide(None, "DeepEP V2 on AWS EFA",
          '<ul>'
          '<li>MoE all-to-all（dispatch + combine）跑在 <b>EFA</b> 上，而非 IB / RoCE</li>'
          '<li>只用发布包：不编 NCCL、不编 aws-ofi-nccl、不换内核模块</li>'
          '<li><b>后端</b>：两个环境变量，decode dispatch %.1f → %.1f µs（<b>%.2f×</b>，2 节点）</li>'
          '<li><b>补丁</b>：PR #8+#9 再把 decode 一层 %.1f → %.1f µs（<b>%s</b>，2 节点）</li>'
          '<li><b>叠加</b>：四个 PR 合并后 decode 一层 %.1f → %.1f µs（<b>%s</b>，另一条 campaign）</li>'
          '</ul>'
          '<p class="note">数据：<code>results/p5en_3arm_20260831/</code>（%d 份日志，%d 条断言由 '
          '<code>check_comparison.py</code> 校验）、<code>results/p5en_2n4n_20260825/</code>'
          '（%d 个 cell 与其 summary.txt 交叉核对）、<code>results/p5en_stack_20260831/</code> 与 '
          '<code>results/p5en_ovlp_20260831/</code>。各 campaign 的代码 SHA 与机器不同，'
          '每个比值只在本 campaign 内取，未相乘。全部数字在构建时由日志算出。</p>'
          % (us0825("type2", 2, 128), us0825("type5", 2, 128),
             us0825("type2", 2, 128) / us0825("type5", 2, 128),
             layer(MAIN, 2, 128), layer(PR89, 2, 128),
             pct(layer(PR89, 2, 128), layer(MAIN, 2, 128)),
             sbase(STACK, 128), slayer(STACK, 128),
             pct(slayer(STACK, 128), sbase(STACK, 128)),
             nlogs, nclaims, n0825),
          "原理 · 安装 · 性能 · 四条待合 PR")

    # ---- 2 findings summary
    slide(None, "结论速览",
          '<ol>'
          '<li><b>后端</b>：decode dispatch %.2f×（2 节点）、%.2f×（4 节点）；prefill %.2f×。'
          '默认配置走 CPU proxy。</li>'
          '<li><b>PR #8+#9</b>：2 节点 decode 一层 %s（%.1f → %.1f µs）。'
          '唯一同时改善 dispatch 与 combine，也是唯一在 prefill 有效的臂（2 节点 %s）。</li>'
          '<li><b>PR #1+#2 须配 <code>EP_NUM_SUB_PARTS=1</code></b>：4 节点一层 %s → %s，'
          '与 PR #8+#9 打平；仅改善 dispatch。</li>'
          '<li><b>四个 PR 可叠加，但是"各取其优"而非求和</b>：decode 一层 %.1f → %.1f µs（%s）；'
          'combine 完全相加，dispatch 残差 %s —— 两条 PR 打的是同一块 dispatch 成本。</li>'
          '<li><b><code>--prefer-overlap-with-compute</code> 把 #8+#9 拆成两半</b>：'
          '两处未被该 flag 门控的改动占 decode dispatch 收益的 %s、prefill combine 收益的 %s。</li>'
          '<li><b>PR #8 的 node 偏斜机制只在 2 节点成立</b>：per-node combine 偏斜 9%% → 1.8–3.8%%；'
          '4 节点三条臂同为 10–11.5%%。</li>'
          '<li><b>补丁收益随规模衰减</b>，衰减倍数取决于测在哪个配置（%.1f× / %.1f×）。</li>'
          '</ol>'
          '<p class="note">均为同硬件、同命令行、臂间交错的 main-vs-patch 比较。方法与明细见%s。</p>'
          % (us0825("type2", 2, 128) / us0825("type5", 2, 128),
             us0825("type2", 4, 128) / us0825("type5", 4, 128),
             us0825("type2", 2, 8192) / us0825("type5", 2, 8192),
             pct(layer(PR89, 2, 128), layer(MAIN, 2, 128)),
             layer(MAIN, 2, 128), layer(PR89, 2, 128),
             pct(layer(PR89, 2, 8192), layer(MAIN, 2, 8192)),
             pct(layer(PR12, 4, 128), layer(MAIN, 4, 128)),
             pct(layer(PR12, 4, 128, SUB1), layer(MAIN, 4, 128)),
             sbase(STACK, 128), slayer(STACK, 128),
             pct(slayer(STACK, 128), sbase(STACK, 128)),
             residual("dispatch"),
             unguarded(128, "dispatch"), unguarded(8192, "reduced combine"),
             decay(DFLT), decay(SUB1), ref("method", "scaling")))

    # ---- 3 what is it
    slide("what", "DeepEP V2 是什么",
          '<ul>'
          '<li><b>MoE 的 GPU all-to-all 内核</b>：dispatch 把 token 路由到 expert，combine 收回</li>'
          '<li>内网后端 <b>NVSHMEM → NCCL GIN</b>：CUDA kernel 直接发 RDMA，不经 CPU proxy</li>'
          '<li>统一 <code>ElasticBuffer</code> API；SM 数解析式给出，不再 auto-tune</li>'
          '<li>显存随 <code>num_max_tokens_per_rank</code> 线性缩放：'
          '128 tok ≈ 1.6 GB/GPU，8192 tok ≈ 100 GB/GPU</li>'
          '</ul>'
          '<p class="note">p5en / p6-b300 均为 EFA 网络，需要专门的 GDA 通路，'
          'IB 上的结论不能直接套用。</p>')

    # ---- 4 dependency chain
    ch = EXTERNAL["chain"]
    slide("chain", "EFA-GDA 依赖链",
          '<p><b>GDA = GPU Direct Async</b>：NIC 的 WQ 映射进 GPU 显存，'
          'kernel 自己写 64 B WQE 并 ring MMIO doorbell，CPU 全程不介入。</p>'
          + table(["层", "版本", "判据"], ch["rows"])
          + '<p class="note">加粗四层由 <b>EFA installer 1.50.0</b> 一次给齐，'
            '这是版本门槛的来源。来源：%s</p>' % esc(ch["src"]))

    # ---- 5 the two env vars
    rows = []
    for n, tok in ((2, 8192), (2, 128), (4, 8192), (4, 128)):
        t2, t5 = us0825("type2", n, tok), us0825("type5", n, tok)
        rows.append(["%dN / %d rank / %d tok" % (n, n * 8, tok),
                     "%.1f µs / %s GB/s" % (t2, so0825("type2", n, tok)),
                     "<b>%.1f µs / %s</b>" % (t5, so0825("type5", n, tok)),
                     "<b>%.2f×</b>" % (t2 / t5)])
    slide("env", "两个必须的环境变量",
          '<p><code>libnccl-net-ofi.so</code> 注册两个 GIN backend：type 2（CPU proxy）'
          '与 type 5（GDAKI），<b>默认 type 2</b>。两者日志都会打 '
          '<code>Loaded gin plugin Libfabric_GDAKI (v14)</code>，'
          '<span class="warn">该行不能当判据</span>。</p>'
          '<pre>export NCCL_GIN_TYPE=5\n'
          'export NCCL_SYM_GIN_KERNELS_ENABLE=0   # 缺此项会 crash：sym-GIN 要强信号，GDAKI 无</pre>'
          + table(["dispatch", "type 2（默认）", "type 5（GDAKI）", "加速比"], rows)
          + '<p class="note">差距集中在 decode 且随规模放大（128 tok：%.2f× → %.2f×）；'
            'prefill 两个规模均 %.2f× / %.2f×，大消息已摊薄 proxy 的每消息开销。<br>'
            '数据由 <code>results/p5en_2n4n_20260825/make_tables.py</code> 算出，'
            '8 个 cell 与其 summary.txt 一致。</p>'
          % (us0825("type2", 2, 128) / us0825("type5", 2, 128),
             us0825("type2", 4, 128) / us0825("type5", 4, 128),
             us0825("type2", 2, 8192) / us0825("type5", 2, 8192),
             us0825("type2", 4, 8192) / us0825("type5", 4, 8192)))

    # ---- 6 why GDA is faster
    slide("path", "GDA 与 CPU proxy 的路径差异",
          table(["栈", "发送路径", "每 rank CPU 代价"],
                [["DeepEP V1 + NVSHMEM (EFA)",
                  "kernel → NVSHMEM proxy 线程 → libfabric → NIC",
                  "CPU 唤醒 + syscall + verbs"],
                 ["UCCL-EP",
                  "kernel → 4 个 Rust proxy 线程/rank → ibverbs → NIC",
                  "<b>4 线程/rank，32 线程/节点</b>"],
                 ["<b>DeepEP V2 + GDAKI (type 5)</b>",
                  "kernel 写 <b>64 B WQE</b> + ring MMIO doorbell（<code>FI_EFA_GDA_OPS</code>）",
                  "<b>0 线程</b>"]])
          + '<p class="note"><b>延迟</b>：省掉 CPU 唤醒 + syscall + verbs 三段。'
            '<b>规模弹性</b>：CPU proxy 的 handoff 按链序列化，GPU 把 N 个 flush 分给 N 个 thread。'
            '这就是上一页的差距在 4 节点（%.2f×）大于 2 节点（%.2f×）的原因。</p>'
          % (us0825("type2", 4, 128) / us0825("type5", 4, 128),
             us0825("type2", 2, 128) / us0825("type5", 2, 128)))

    # ---- 7 host install
    slide("host", "Host 安装（每台机器一次）",
          '<pre>curl -O https://efa-installer.amazonaws.com/aws-efa-installer-1.50.0.tar.gz\n'
          'tar xzf aws-efa-installer-1.50.0.tar.gz\n'
          'sudo ./aws-efa-installer/efa_installer.sh -y --no-verify</pre>'
          '<p><b>验证：</b></p><ul>'
          '<li><code>modinfo efa | grep ^version</code> → 3.3.0</li>'
          '<li><code>ibv_devinfo -l</code> → 16 张 EFA 设备（p5en 每 GPU 2 张）</li>'
          '<li><code>ce_probe.c</code>：<code>ibv_create_comp_cntr</code> 成功 = GDAKI 可用</li>'
          '</ul>'
          '<p class="warn">p6-b300 出厂为 installer 1.47.0（<code>efa.ko</code> 3.0.0，'
          '无 <code>GinPlugin</code> 符号），GDAKI 无法启动；须先升到 1.50.0，'
          '且升在 <b>host</b> 而非容器。</p>')

    # ---- 8 build image
    slide("build", "构建镜像",
          '<pre>git clone https://github.com/whn09/ep-benchmarks-efa.git ~/work/ep-benchmarks-efa\n'
          'cd ~/work/ep-benchmarks-efa/deepep-v2-efa-official\n'
          './build_image.sh sm90     # p5en / H200\n'
          './build_image.sh sm103    # p6-b300 / B300（需 CUDA ≥ 13.3 的 base）</pre>'
          '<ol>'
          '<li>装 EFA installer 1.50.0 <b>用户态</b>（<code>--skip-kmod</code>）</li>'
          '<li><b>构建期 gate</b>：<code>nm libnccl-net-ofi.so | grep ncclGinPlugin_v14</code></li>'
          '<li>删 apt 的 libnccl 2.28.3，装 pip 的 2.31.2 + nvshmem 3.7.2</li>'
          '<li>build DeepEP，镜像 tag 自带 sha</li></ol>'
          + table(["镜像 tag", "对应代码", "这条臂测什么"],
                  [["<code>sm90-54fffef</code>", "<code>main</code>", "基线"],
                   ["<code>sm90-bfbdd15</code>", "PR <b>#1 + #2</b> head",
                    "decode part 几何 + 把 <code>EP_NUM_SUB_PARTS</code> 转发进 JIT"],
                   ["<code>sm90-3c737dc</code>", "PR <b>#8 + #9</b> head",
                    "channels / QPs / forward warp pairs + remote-first combine"]])
          + '<p class="note">SHA 必须钉住并归档：上游会 rewrite history，'
            'Dockerfile 中的浮动 ref 会因 layer cache 而看起来更新过、实际未变。</p>')

    # ---- 9 first cell
    sm_m, qp_m = cfg_of(MAIN, 2, 128)
    sm_p, qp_p = cfg_of(PR89, 2, 128)
    slide("first", "跑第一个 cell",
          '<pre># leader（P5EN-1）\n'
          'MASTER_IP=$(hostname -I | awk \'{print $1}\')   # 用私网 v4，不要用 hostname\n'
          './run_test_ep.sh 0 "$MASTER_IP"\n\n'
          '# worker（P5EN-2，几秒内启动）\n'
          './run_test_ep.sh 1 172.31.14.20</pre>'
          '<p>日志中应出现：</p>'
          '<pre>NCCL INFO GIN/Plugin: Loaded gin plugin Libfabric_GDAKI (v14)   # 必要不充分\n'
          '&gt; #SM: %s, #QPs: %s      # main\n'
          '&gt; #SM: %s, #QPs: %s      # PR #8+#9，kDefaultGinContextCnt 11→13 在此可见</pre>'
          '<p class="warn"><code>ip-172-31-14-20</code> 这类 hostname 在 EC2 上同时解析出 IPv4 和 '
          'link-local IPv6，libuv 尝试 v6 时报 <code>errno 22 (EINVAL)</code>。用私网 IPv4 直连。</p>'
          % (sm_m, qp_m, sm_p, qp_p))

    # ---- 10 campaign / methodology
    slide("method", "Campaign 与验收口径",
          '<pre>NODES="P5EN-1 P5EN-2" ./run_campaign.sh</pre>'
          '<ul>'
          '<li><b>Reps 轮转</b>：一 rep 跑完所有 cell 再进下一 rep，慢漂移不会被误认成 arm 效应</li>'
          '<li>每个 cell 独立 <code>MASTER_PORT</code>；日志名自带全部变量'
          '（arm / 节点数 / SM / tokens / knob / debug / GIN 后端 / rep / node）</li>'
          '<li><b>全 rank 池化</b>：combine 按机器分层，单节点均值最多偏 11%%（%s）</li>'
          '<li><b>离群 rep 显式排除</b>：偏离本 cell <b>中位数</b> 25%% 以上才排除，'
          '日志保留并在 AUDIT 中打印</li>'
          '<li><code>./verify_run.sh logs/*.log</code> → <code>no FAILs</code>；'
          '<code>check_comparison.py</code> 从日志重算全部 <b>%d</b> 条断言</li>'
          '</ul>'
          '<p class="note">口径固定项：<code>--test-first-only</code> ⇒ 全部数字为 '
          '<b>FP8 dispatch @ expert_alignment=128</b>；<code>--ignore-local-traffic</code> '
          '<b>关</b> ⇒ SO GB/s 含机内流量、<b>非线速</b>，故以时间为指标；'
          '<code>EP_BUFFER_DEBUG</code> <b>关</b>（它在 dispatch 计时循环内 printf）。</p>'
          % (ref("layering"), nclaims))

    # ---- 11 three arms, layer total
    rows = []
    base2, base4 = layer(MAIN, 2, 128), layer(MAIN, 4, 128)
    bp2, bp4 = layer(MAIN, 2, 8192), layer(MAIN, 4, 8192)
    for arm, knob, name in ((MAIN, DFLT, "main（基线）"),
                            (PR12, DFLT, "PR #1+#2（默认几何）"),
                            (PR12, SUB1, "PR #1+#2 + <code>EP_NUM_SUB_PARTS=1</code>"),
                            (PR89, DFLT, "PR #8+#9")):
        # `is_base`, not `ref`: ref() is the module-level cross-reference helper and
        # a local of that name shadows it for the whole function body.
        is_base = arm == MAIN
        rows.append([
            name,
            "%.1f" % layer(arm, 2, 128, knob),
            "—" if is_base else "<b>%s</b>" % pct(layer(arm, 2, 128, knob), base2),
            "%.1f" % layer(arm, 4, 128, knob),
            "—" if is_base else "<b>%s</b>" % pct(layer(arm, 4, 128, knob), base4),
            "%.0f" % layer(arm, 2, 8192, knob),
            "—" if is_base else pct(layer(arm, 2, 8192, knob), bp2),
            "%.0f" % layer(arm, 4, 8192, knob),
            "—" if is_base else pct(layer(arm, 4, 8192, knob), bp4)])
    slide("overview", "三条臂总览：一层 MoE 的时间",
          '<p><code>dispatch + reduced combine</code>，即一层 MoE 发起的两次调用之和'
          '（µs，小为优）。Δ 对 <b>main 的默认几何</b>取。</p>'
          + table(["arm", "decode 2N", "Δ", "decode 4N", "Δ",
                   "prefill 2N", "Δ", "prefill 4N", "Δ"], rows)
          + '<p class="note">decode 2 节点 PR #8+#9 最优（%s）。decode 4 节点 PR #1+#2 配工作点 env '
            '与 PR #8+#9 打平（%.1f vs %.1f µs），但两者机制不同（%s）。'
            'prefill 仅 PR #8+#9 在 2 节点显著（%s）；4 节点三条臂均为 −1.5 ~ −2.0%%。</p>'
          % (pct(layer(PR89, 2, 128), base2),
             layer(PR12, 4, 128, SUB1), layer(PR89, 4, 128), ref("decode"),
             pct(layer(PR89, 2, 8192), bp2)))

    # ---- 12 prefill detail
    def detail(tok, nodes):
        rows = []
        for op in g.OPS:
            r = [op if op != "reduced combine" else "<b>reduced combine</b>"]
            base = us(MAIN, nodes, tok, op)
            for arm in (MAIN, PR12, PR89):
                v = us(arm, nodes, tok, op)
                cell = "%.1f" % v
                if arm != MAIN:
                    cell += " <span class='%s'>%s</span>" % (
                        "good" if v < base * 0.99 else "dim", pct(v, base))
                r.append(cell)
            r.append(so_range(MAIN, nodes, tok, op))
            rows.append(r)
        return table(["op", "main µs", "PR #1+#2 µs", "PR #8+#9 µs", "main SO GB/s"], rows)

    slide("prefill", "Prefill 明细（8192 tok, 12 SM, GIN type 5）",
          '<div class="two"><div><h3>2 节点 / 16 rank</h3>%s</div>'
          '<div><h3>4 节点 / 32 rank</h3>%s</div></div>'
          '<p class="note"><b>main 的 <code>cached dispatch</code> 慢于普通 dispatch</b>'
          '（2N %.1f vs %.1f µs，4N %.1f vs %.1f µs）。PR #8+#9 的作用是消除该惩罚 —— '
          '其 cached 与 plain 打平；这也是 4 节点 prefill 唯一超出噪声的一行。'
          'SO 含机内流量，非线速。</p>'
          % (detail(8192, 2), detail(8192, 4),
             us(MAIN, 2, 8192, "cached dispatch"), us(MAIN, 2, 8192, "dispatch"),
             us(MAIN, 4, 8192, "cached dispatch"), us(MAIN, 4, 8192, "dispatch")))

    # ---- 13 decode detail
    slide("decode", "Decode 明细（128 tok, 12 SM, GIN type 5）",
          '<div class="two"><div><h3>2 节点 / 16 rank</h3>%s</div>'
          '<div><h3>4 节点 / 32 rank</h3>%s</div></div>'
          '<p class="note">2 节点上两条臂机制不同：PR #1+#2 只动 dispatch'
          '（%.1f → %.1f µs，combine 持平）；PR #8+#9 两者都动'
          '（%.1f → %.1f、%.1f → %.1f µs）。两者<b>已实测叠加</b>（%s）。</p>'
          % (detail(128, 2), detail(128, 4),
             us(MAIN, 2, 128, "dispatch"), us(PR12, 2, 128, "dispatch"),
             us(MAIN, 2, 128, "dispatch"), us(PR89, 2, 128, "dispatch"),
             us(MAIN, 2, 128, "reduced combine"), us(PR89, 2, 128, "reduced combine"),
             ref("stack")))

    # ---- 14 EP_NUM_SUB_PARTS
    rows = []
    for nodes in (2, 4):
        for op in ("dispatch", "cached dispatch", "reduced combine"):
            a, b = us(PR12, nodes, 128, op), us(PR12, nodes, 128, op, SUB1)
            rows.append(["%dN / 128 tok" % nodes, op, "%.1f" % a, "<b>%.1f</b>" % b,
                         "<span class='%s'>%s</span>" % (
                             "good" if b < a * 0.99 else "dim", pct(b, a)),
                         "<b>%s</b>" % pct(b, us(MAIN, nodes, 128, op))])
    slide("subparts", "<code>EP_NUM_SUB_PARTS=1</code> 是 PR #1+#2 的工作点",
          '<p><code>main</code> 与 <code>#8+#9</code> 只有 <code>.cuh</code> 读这个宏、'
          '无人定义它，因此完全惰性；只有 <code>bfbdd15</code> 的 '
          '<code>csrc/jit/compiler.hpp</code> 把它写入 JIT flags。这是 PR #1 的全部贡献。</p>'
          + table(["cell", "op", "默认几何 µs", "+env µs", "Δ vs 自身默认", "Δ vs main"], rows)
          + '<p class="note"><b>惰性对照</b>：同一 env 加到 <code>main</code> 得 %s、'
            '<code>#8+#9</code> 得 %s，均落在各自默认上。<br>'
            '<b>边际贡献在 4 节点更大</b>（dispatch %s vs 2 节点 %s，对同臂默认取），'
            '使 PR #1+#2 的 4 节点一层收益翻倍：%s → <b>%s</b>（对 main 取）。<br>'
            '<b>代价</b>：2N prefill dispatch %s（+%.1f µs），被 combine 的 −%.1f µs 抵掉，'
            '一层净 +%.1f µs；4 节点该代价消失，一层 −%.1f µs（%s）。<br>'
            '<b>不要用头文件补丁替代</b>：JIT cache key 哈希 flags 但不哈希头文件内容，'
            '纯头文件改动可能被喂到旧 cubin，测起来像 no-op。</p>'
          % (pct(us(MAIN, 2, 128, "dispatch", SUB1), us(MAIN, 2, 128, "dispatch")),
             pct(us(PR89, 2, 128, "dispatch", SUB1), us(PR89, 2, 128, "dispatch")),
             pct(us(PR12, 4, 128, "dispatch", SUB1), us(PR12, 4, 128, "dispatch")),
             pct(us(PR12, 2, 128, "dispatch", SUB1), us(PR12, 2, 128, "dispatch")),
             pct(layer(PR12, 4, 128), layer(MAIN, 4, 128)),
             pct(layer(PR12, 4, 128, SUB1), layer(MAIN, 4, 128)),
             pct(us(PR12, 2, 8192, "dispatch", SUB1), us(PR12, 2, 8192, "dispatch")),
             us(PR12, 2, 8192, "dispatch", SUB1) - us(PR12, 2, 8192, "dispatch"),
             us(PR12, 2, 8192, "reduced combine") - us(PR12, 2, 8192, "reduced combine", SUB1),
             layer(PR12, 2, 8192, SUB1) - layer(PR12, 2, 8192),
             layer(PR12, 4, 8192) - layer(PR12, 4, 8192, SUB1),
             pct(layer(PR12, 4, 8192, SUB1), layer(PR12, 4, 8192))))

    # ---- stack: four PRs at once (own campaign, 2 nodes)
    KNOB = {DFLT: "默认", SUB1: "<code>EP_NUM_SUB_PARTS=1</code>"}
    rows = []
    for arm in (MAIN, PR12, PR89, STACK):
        knob = sknob(arm, 128)
        tot = slayer(arm, 128)
        rows.append([LABEL[arm] if arm != STACK else "<b>四个叠加</b>", KNOB[knob],
                     "%.1f" % sus(arm, 128, "dispatch"),
                     "%.1f" % sus(arm, 128, "reduced combine"),
                     "<b>%.1f</b>" % tot,
                     "" if arm == MAIN else "<b>%s</b>" % pct(tot, sbase(arm, 128))])
    arows = []
    for op, name in (("dispatch", "dispatch"), ("reduced combine", "reduced combine"),
                     ("layer", "<b>一层</b>")):
        m, a, b, exp, s, res = add_row(op)
        arows.append([name, "%.1f" % m, "%.1f" % a, "%.1f" % b, "%.1f" % exp,
                      "<b>%.1f</b>" % s,
                      "<span class='%s'>%+.1f</span>"
                      % ("warn" if abs(res) > 5 else "good", res)])
    _dres = add_row("dispatch")
    slide("stack", "四个 PR 叠加：decode 一层 %s" % pct(slayer(STACK, 128),
                                                 sbase(STACK, 128)),
          '<p>两组 PR 改的是<b>不同文件</b>（dispatch 侧 vs combine 侧），'
          '<code>git merge</code> 无冲突，因此镜像由两个 parent SHA 加一次 merge 完全确定。'
          '各臂取自己会部署的 knob：</p>'
          + table(["arm", "knob", "dispatch µs", "redComb µs", "一层 µs", "Δ vs main"],
                  rows)
          + '<p><b>叠加性</b>（<code>expected = main + (#1+#2 − main) + (#8+#9 − main)</code>，'
            '即两份收益直接相加；<code>residual = 实测 − expected</code>，> 0 表示两者打的是'
            '同一笔开销）：</p>'
          + table(["op", "main", "#1+#2", "#8+#9", "expected", "实测叠加", "residual"],
                  arows)
          + '<p class="note"><b>"各取其优"，不是求和。</b>combine 上 residual %+.1f µs —— '
            'PR #1+#2 在那儿本就为 0，叠加直接拿到 #8+#9 的 combine；dispatch 上 residual '
            '<b>%+.1f µs = 较大单项收益的 %.0f%%</b>，两组打的是同一笔 dispatch 开销。'
            '净收益相对两条单臂各 %s / %s。<br>'
            '<b>prefill 全部来自 #8+#9</b>：一层 %.1f → %.1f（#8+#9）→ %.1f µs（叠加），'
            '叠加不再加分。<br>'
            '<b>口径</b>：独立 campaign（不与前几节混算，叠加性是差之差）；机器提前释放，'
            '30 个 cell 跑了 15 个，跨臂行只取各臂<b>共有的 rep</b>。</p>'
          % (add_row("reduced combine")[5], _dres[5],
             100.0 * _dres[5] / max(abs(_dres[1] - _dres[0]), abs(_dres[2] - _dres[0])),
             pct(slayer(STACK, 128), slayer(PR12, 128)),
             pct(slayer(STACK, 128), slayer(PR89, 128)),
             slayer(MAIN, 8192), slayer(PR89, 8192), slayer(STACK, 8192)),
          "results/p5en_stack_20260831/ · 2 节点 · 12 SM · type 5")

    # ---- ovlp: the flag as an instrument on PR #8+#9
    orows = []
    for tok, op, name in ((128, "dispatch", "decode dispatch"),
                          (128, "reduced combine", "decode redComb"),
                          (8192, "dispatch", "prefill dispatch"),
                          (8192, "reduced combine", "prefill redComb")):
        r = [name]
        for ovlp in (0, 1):
            m, p = ous(MAIN, tok, ovlp, op), ous(PR89, tok, ovlp, op)
            r += ["%.1f" % m, "<b>%.1f</b> <span class='%s'>%s</span>"
                  % (p, "good" if p < m * 0.99 else "dim", pct(p, m))]
        orows.append(r)
    d0 = ous(PR89, 128, 0, "dispatch") - ous(MAIN, 128, 0, "dispatch")
    d1 = ous(PR89, 128, 1, "dispatch") - ous(MAIN, 128, 1, "dispatch")
    c0 = ous(PR89, 8192, 0, "reduced combine") - ous(MAIN, 8192, 0, "reduced combine")
    c1 = ous(PR89, 8192, 1, "reduced combine") - ous(MAIN, 8192, 1, "reduced combine")
    slide("ovlp", "<code>--prefer-overlap-with-compute</code> 把 #8+#9 拆成两半",
          '<p>PR #8+#9 捆了四个改动。其中<b>删 channel clamp</b> 与 <b>forward warp 配对</b>'
          '门控在 <code>not prefer_overlap_with_compute</code> 上，另两个（QP 11→13、'
          'remote-first）不是。所以在 <code>=1</code> 上跑一次 main-vs-#9 就把后两个'
          '单独量出来了 —— <b>不用重新 build</b>。</p>'
          + table(["op", "main ovlp=0", "#8+#9 ovlp=0", "main ovlp=1", "#8+#9 ovlp=1"],
                  orows)
          + '<p class="note"><b>划分按 op 变，这才是结论。</b>decode dispatch：%.1f µs 的收益里'
            '<b>无门控的两个占 %.0f%%</b>（%.1f µs）。prefill redComb：%.1f µs 的收益里'
            '<b>被门控的两个占 %.0f%%</b>（%.1f µs）。prefill dispatch 两个取值都是平的。<br>'
            '<b>这是 bracket，不是归因</b>：<code>=1</code> 同时改了 double-buffering 与 warp 数，'
            '两个取值是<b>两种配置</b>，只能在同一取值内部比臂。<br>'
            '<b>顺带</b>：flag 本身是一根轴 —— <code>main</code> 一层 prefill %s、decode %s。'
            '但 <code>test_ep.py</code> 不发并发 compute，这是纯通信开销，'
            '不能据此推断 overlap 后的端到端吞吐。本文其余数字均为 <code>=0</code>。</p>'
          % (-d0, 100.0 * d1 / d0, -d1, -c0, 100.0 * (c0 - c1) / c0, c1 - c0,
             pct(olayer(MAIN, 8192, 1), olayer(MAIN, 8192, 0)),
             pct(olayer(MAIN, 128, 1), olayer(MAIN, 128, 0))),
          "results/p5en_ovlp_20260831/ · 24 cell 全 rc 0 · 三个 rep")

    # ---- node layering / PR #8
    def spread_row(arm, nodes, tok):
        return " / ".join("%.1f%%" % x
                          for x in node_spread(arm, nodes, tok, "reduced combine"))

    slide("layering", "node 分层：PR #8 的机制在 2 节点成立、4 节点不成立",
          '<p>combine 按<b>机器</b>分层，PR #8 的 remote-first 调度针对的就是这一点。'
          '下表为每个 rep 的 per-node <code>reduced combine</code> 均值的 (max−min)/min。</p>'
          + table(["arm", "2N / 8192 tok（三个 rep）", "4N / 128 tok（三个 rep）"],
                  [[LABEL[a], spread_row(a, 2, 8192), spread_row(a, 4, 128)]
                   for a in (MAIN, PR12, PR89)])
          + '<p class="note"><b>2 节点</b>：<code>main</code> 偏斜 ~9%%，且慢的那台机器在 rep 之间'
            '<b>翻转</b>；PR #8+#9 压到 1.8–3.8%%。以 PR #8 自己引用的统计量（combine SO 的 rank 间 '
            'min–max）看一致：main %s GB/s → PR #8+#9 %s GB/s。<br>'
            '<b>4 节点</b>：三条臂同为 ~10–11.5%%，且<b>按节点序单调、跨 rep 稳定</b>而非翻转，'
            '是另一种系统性偏斜，remote-first 未触及。dispatch 分层在所有臂、两个规模均 ≤2.6%%，'
            '故此现象为 combine 特有。</p>'
          % (so_range(MAIN, 2, 8192, "combine"), so_range(PR89, 2, 8192, "combine")))

    # ---- 16 scaling
    rows = []
    for arm, knob, name in ((MAIN, DFLT, "main"),
                            (PR12, DFLT, "PR #1+#2（默认几何）"),
                            (PR12, SUB1, "PR #1+#2 + <code>EP_NUM_SUB_PARTS=1</code>"),
                            (PR89, DFLT, "PR #8+#9")):
        rows.append([name] + ["%.2f×" % (us(arm, 4, tok, op, knob)
                                         / us(arm, 2, tok, op, knob))
                              for tok, op in ((8192, "dispatch"),
                                              (8192, "reduced combine"),
                                              (128, "dispatch"),
                                              (128, "reduced combine"))])
    slide("scaling", "规模效应",
          '<p>4 节点 / 2 节点的时间比（越接近 1 = 规模弹性越好）：</p>'
          + table(["arm", "prefill dispatch", "prefill redComb",
                   "decode dispatch", "decode redComb"], rows)
          + '<p class="note">decode dispatch 在 <code>main</code> 上只有 %.2f×，'
            'GDAKI 的 fan-out 代价基本是常数；同时也意味着 4 节点基线本身已好，补丁可拿的余量更少。<br>'
            'PR #1+#2 的 dispatch 收益在默认几何下 %s → %s（<b>衰减 %.1f×</b>），'
            '在工作点 <code>EP_NUM_SUB_PARTS=1</code> 下 %s → %s（<b>衰减 %.1f×</b>）。'
            '衰减速度取决于臂测在哪个配置上。</p>'
          % (us(MAIN, 4, 128, "dispatch") / us(MAIN, 2, 128, "dispatch"),
             pct(us(PR12, 2, 128, "dispatch"), us(MAIN, 2, 128, "dispatch")),
             pct(us(PR12, 4, 128, "dispatch"), us(MAIN, 4, 128, "dispatch")),
             decay(DFLT),
             pct(us(PR12, 2, 128, "dispatch", SUB1), us(MAIN, 2, 128, "dispatch")),
             pct(us(PR12, 4, 128, "dispatch", SUB1), us(MAIN, 4, 128, "dispatch")),
             decay(SUB1)))

    # ---- 17 cross-stack pointer
    slide("crossstack", "跨栈对比的口径",
          '<p>V1 / UCCL-EP / V2 的横向数字见 <code>README.md § Side-by-side results</code>。'
          '该节的行组是<b>不同 shape、不同时钟</b>（2026-05/BF16 的 4096 tok 组 vs '
          '2026-08/GDAKI 的 8192 tok FP8 组），<b>不可按列比较</b>，故此处不复制成单表。</p>'
          '<p>跨栈对比前须对齐三项：</p><ol>'
          '<li><b>口径</b>：DeepEP 把每个 op 拆成 <b>2 个 kernel</b>，只引 1 个会系统性偏低</li>'
          '<li><b>字节分母</b>：per-selection 记账把 DeepEP 在 2 节点高估 4×、4 节点 2×，'
          '<b>足以反转 scaling 结论</b></li>'
          '<li><b>时间与带宽并列</b>：仅给 GB/s 会反转结论，须同时给 µs 并说明分母</li>'
          '</ol>'
          '<p class="note">' + ref("overview", "scaling") + '的 main-vs-patch 为同硬件同命令行，不受上述影响；'
          '但不可读作"与 IB 上的官方 DeepEP 打平" —— 官方的 '
          '<code>prefer_overlap_with_compute</code> 取值不同。</p>')

    # ---- 18 gotchas
    slide("gotchas", "四个坑",
          '<ol>'
          '<li><b>版本号不是判据</b>：同名 <code>aws-efa-installer-1.50.0.tar.gz</code> 内的 '
          'aws-ofi-nccl 可能是 1.21.1（有 <code>ncclGinPlugin_v14</code>）或 1.20.0（无）。'
          '以 <code>nm</code> 的符号为准。</li>'
          '<li><b>hostname 会解析出 link-local IPv6</b>，rendezvous 报 <code>errno 22</code>。'
          '用私网 IPv4。</li>'
          '<li><b><code>docker run</code> 须加 <code>--init</code></b>：PID 1 为 python 时不 reap '
          '子进程、signal 转发不可靠，卡在 rendezvous 的 worker 会成为孤儿并<b>持续占用 '
          '<code>MASTER_PORT</code></b>。</li>'
          '<li><b>GDAKI context 不能与另一进程共用同一张 EFA 网卡</b>：常驻的 DeepEP-v2 容器'
          '即使 GPU util 0% 也持续持有 GIN context，第二个进程在 <code>ElasticBuffer</code> '
          '构造时报 <code>fi_enable failed: Cannot allocate memory</code>。'
          '空闲显存与 util 均无法判断此项；<b>第几个 local rank 失败</b>才是有效信号。</li>'
          '</ol>')

    # ---- 19 status / next
    slide("status", "PR 现状与下一步",
          table(["PR", "内容", "实测", "状态"],
                [["<b>#1</b>", "把 4 个 part 几何 env 转发进 JIT",
                  "本身不改性能，但是 <code>EP_NUM_SUB_PARTS</code> 的<b>唯一</b>开关", "open"],
                 ["<b>#2</b>", "decode part 几何 clamp",
                  "decode dispatch 2N <b>%s</b>、4N %s（对 main）" % (
                      pct(us(PR12, 2, 128, "dispatch"), us(MAIN, 2, 128, "dispatch")),
                      pct(us(PR12, 4, 128, "dispatch"), us(MAIN, 4, 128, "dispatch"))),
                  "open"],
                 ["<b>#8</b>", "combine remote-first 调度",
                  "2 节点 node 偏斜 9% → 1.8–3.8%；4 节点无效", "open"],
                 ["<b>#9</b>", "channels / QPs 11→13 / forward warp pairs（含 #8）",
                  "decode 一层 2N <b>%s</b>、4N %s；prefill 一层 2N <b>%s</b>" % (
                      pct(layer(PR89, 2, 128), layer(MAIN, 2, 128)),
                      pct(layer(PR89, 4, 128), layer(MAIN, 4, 128)),
                      pct(layer(PR89, 2, 8192), layer(MAIN, 2, 8192))),
                  "open"],
                 ["<b>四者合并</b>", "<code>git merge</code>，冲突为零（除 README）",
                  "decode 一层 <b>%s</b>、prefill %s（本地 merge sha，%s）" % (
                      pct(slayer(STACK, 128), sbase(STACK, 128)),
                      pct(slayer(STACK, 8192), sbase(STACK, 8192)),
                      ref("stack")),
                  "未提交"]])
          + '<p><b>下一步（按价值排序）：</b></p><ol>'
            '<li>不平衡场景（<code>--unbalanced-ratio</code> / <code>--masked-ratio</code>），'
            'PR #8 的 remote-first 在此应最有效</li>'
            '<li><b>干净归因 PR #9</b>：把两处 gated hunk revert 掉单独 build。'
            '当前的 flag 夹取是<b>区间而非归因</b>（%s）</li>'
            '<li><b>prefill 叠加性</b>仍不可算：#1+#2 缺 8192 tok 的两个 cell'
            '（2 cell × 3 rep，约 20 min）</li>'
            '<li>b300 复跑（sm_103 + CUDA ≥ 13.3），并在 #8+#9 / 叠加臂上加 SM 轴</li></ol>'
            % ref("ovlp"))

    # ---- 20 refs
    slide(None, "参考",
          '<ul>'
          '<li>完整方法与全部数字：<code>results/p5en_3arm_20260831/comparison.md</code>'
          '（%d 条断言由 <code>check_comparison.py</code> 从日志重算）</li>'
          '<li>叠加性与 flag 夹取：<code>results/p5en_stack_20260831/tables.txt</code>、'
          '<code>results/p5en_ovlp_20260831/tables.txt</code>'
          '（各自的 <code>make_*_tables.py</code> 复用同一套池化与离群规则）</li>'
          '<li>后端对比与 SM 轴：<code>results/p5en_2n4n_20260825/summary.txt</code></li>'
          '<li>Runbook：<code>docs/runbook_zh.md</code>，其中的表由 '
          '<code>docs/check_runbook_numbers.py</code> 从日志逐格重算（%d 个值）</li>'
          '<li>复现脚本：<code>run_test_ep.sh</code> / <code>run_campaign.sh</code> / '
          '<code>verify_run.sh</code>；叠加镜像 <code>build_stack_image.sh</code></li>'
          '<li>本 deck 生成器：<code>docs/build_slides_html.py</code>'
          '（改数字请改日志或生成器，不要改 HTML）</li>'
          '<li>PR：<a href="https://github.com/amazon-contributing/DeepEP/pull/1">#1</a> · '
          '<a href="https://github.com/amazon-contributing/DeepEP/pull/2">#2</a> · '
          '<a href="https://github.com/amazon-contributing/DeepEP/pull/8">#8</a> · '
          '<a href="https://github.com/amazon-contributing/DeepEP/pull/9">#9</a></li>'
          '</ul>' % (nclaims, nrunbook))


CSS = """
:root { --ink:#16222f; --accent:#1f4e79; --accent2:#2e75b6; --dim:#6b7a88;
        --good:#1a7f4b; --warn:#a8410a; --line:#dde4ea; --bg:#f4f6f8; }
* { box-sizing:border-box; }
html,body { margin:0; padding:0; background:var(--bg); color:var(--ink);
  font-family:-apple-system,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",
    "Noto Sans CJK SC",Helvetica,Arial,sans-serif; }
#deck { }
.slide { position:relative; width:100vw; height:100vh; display:none;
  padding:3.0vh 4vw 6vh; background:#fff; overflow:hidden; }
.slide.on { display:flex; flex-direction:column; }
h1 { color:var(--accent); font-size:2.5vw; margin:0 0 .4vh; line-height:1.25; }
h1 code { font-size:.92em; }
.sub { color:var(--accent2); font-size:1.5vw; margin:0 0 1.4vh; font-weight:600; }
h3 { color:var(--accent2); font-size:1.15vw; margin:.2vh 0 .5vh; }
.body { flex:1; min-height:0; overflow:auto; font-size:1.18vw; line-height:1.62; }
.cover h1 { font-size:3.6vw; margin-top:14vh; }
.cover .sub { font-size:1.9vw; }
ul,ol { margin:.3em 0 .5em; padding-left:1.35em; }
li { margin:.3em 0; }
p { margin:.45em 0; }
code { background:#eef2f6; padding:1px 5px; border-radius:3px;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:.93em; }
pre { background:#16222f; color:#e6edf3; padding:.85em 1em; border-radius:6px;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:.92em; line-height:1.5; overflow-x:auto; margin:.5em 0; }
pre code { background:none; color:inherit; padding:0; }
table { border-collapse:collapse; width:100%; margin:.5em 0; font-size:.92em; }
th,td { border:1px solid var(--line); padding:.34em .6em; text-align:right; }
th:first-child,td:first-child { text-align:left; }
th { background:#eef2f6; color:var(--accent); font-weight:600; white-space:nowrap; }
tbody tr:nth-child(even) { background:#fafcfd; }
.note { font-size:.86em; color:var(--dim); line-height:1.6;
  border-left:3px solid var(--line); padding-left:.8em; margin-top:.7em; }
.note b { color:var(--ink); }
.dim { color:var(--dim); }
.good { color:var(--good); font-weight:600; }
.warn { color:var(--warn); font-weight:600; }
.two { display:grid; grid-template-columns:1fr 1fr; gap:1.6vw; }
.two table { font-size:.84em; }
.two td,.two th { padding:.26em .45em; }
#bar { position:fixed; bottom:0; left:0; right:0; height:4.2vh;
  display:flex; align-items:center; justify-content:space-between;
  padding:0 4vw; font-size:.85vw; color:var(--dim);
  background:linear-gradient(#fff0,#fff 40%); pointer-events:none; }
#bar b { color:var(--accent); }
@media print {
  @page { size:1600px 900px; margin:0; }
  html,body { background:#fff; }
  .slide { display:flex !important; flex-direction:column;
    width:1600px; height:900px; page-break-after:always; padding:40px 64px 60px; }
  h1{font-size:40px} .sub{font-size:24px} .body{font-size:19px} h3{font-size:19px}
  .cover h1{font-size:58px} .cover .sub{font-size:30px} #bar{display:none}
}
"""

JS = """
var S = [].slice.call(document.querySelectorAll('.slide')), i = 0;
function go(n){ i = Math.max(0, Math.min(S.length-1, n));
  S.forEach(function(s,k){ s.classList.toggle('on', k===i); });
  document.getElementById('pos').textContent = (i+1) + ' / ' + S.length;
  location.hash = 'p' + (i+1); }
document.addEventListener('keydown', function(e){
  if (['ArrowRight','PageDown',' ','Enter','j','l'].indexOf(e.key)>=0){ go(i+1); e.preventDefault(); }
  if (['ArrowLeft','PageUp','Backspace','k','h'].indexOf(e.key)>=0){ go(i-1); e.preventDefault(); }
  if (e.key==='Home') go(0);
  if (e.key==='End') go(S.length-1);
});
document.addEventListener('click', function(e){ if(!e.target.closest('a')) go(i+1); });
var m = /^#p(\\d+)$/.exec(location.hash);
go(m ? parseInt(m[1],10)-1 : 0);
"""


def check_order():
    """Every ORDER key used exactly once, in ORDER's order.

    NUM comes from ORDER, so a key that is declared but never used shifts every later
    slide's number while the deck still renders -- and then `ref()` points one slide off
    with nothing to notice it. Cheapest possible guard against that.
    """
    used = [k for k, _t, _s, _b in SLIDES if k is not None]
    if used != ORDER:
        raise SystemExit("build_slides_html.py: ORDER does not match the slides built.\n"
                         "  ORDER: %s\n  built: %s" % (ORDER, used))


def render(path):
    check_order()
    parts = ['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>DeepEP V2 on AWS EFA</title><style>%s</style></head><body>' % CSS,
             '<div id="deck">']
    for k, (_key, title, sub, body) in enumerate(SLIDES):
        cls = "slide cover" if k == 0 else "slide"
        parts.append('<section class="%s" id="p%d"><h1>%s</h1>%s<div class="body">%s</div></section>'
                     % (cls, k + 1, title,
                        '<p class="sub">%s</p>' % sub if sub else "", body))
    parts.append('</div><div id="bar"><span>DeepEP V2 on AWS EFA · '
                 'p5en.48xlarge × 2 / 4 · 2026-08</span>'
                 '<span><b id="pos"></b> · ← → 翻页 · 打印可存 PDF</span></div>')
    parts.append('<script>%s</script></body></html>' % JS)
    with open(path, "w") as fh:
        fh.write("\n".join(parts))
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=os.path.join(HERE, "slides_zh.html"))
    a = ap.parse_args()
    build()
    p = render(a.out)
    print("wrote %s (%d slides, %.0f KB)" % (p, len(SLIDES), os.path.getsize(p) / 1024.0))
    if g.EXCLUDED:
        print("outliers excluded by the 3-arm aggregation (recorded, not silent):")
        for e in g.EXCLUDED:
            print("  " + e)
    if g.EMPTY:
        print("WARNING empty logs skipped: %s" % "; ".join(sorted(set(g.EMPTY))))
        sys.exit(1)
