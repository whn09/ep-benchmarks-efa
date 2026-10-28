"""Build the 2026-10 deck from the 2026-08 one.

    python3 update_slides.py                       # -> DeepEPv2_on_EFA.pptx (public)
    python3 update_slides.py --extra-slide make_slide.py --out deck.pptx
                                                   # also inserts a slide drawn by another
                                                   # python-pptx script (kept out of this repo)

The base deck `DeepEPv2_on_EFA_2026-08.pptx` is never modified, so the script can be rerun.
The V2.5 table is computed from the raw logs in ../../deepep-v25-efa-port/results, not typed in.
"""
import argparse
import collections
import copy
import glob
import os
import re

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, 'DeepEPv2_on_EFA_2026-08.pptx')
V25_RESULTS = os.path.join(HERE, '..', '..', 'deepep-v25-efa-port', 'results')
DECK = 'DeepEP V2 on AWS EFA'
DATE = '2026-10'

TITLE = RGBColor(0x1F, 0x4E, 0x79)
ACCENT = RGBColor(0x2E, 0x75, 0xB6)
RED = RGBColor(0xC0, 0x50, 0x4D)
INK = RGBColor(0x20, 0x20, 0x20)
GREY = RGBColor(0x90, 0x90, 0x90)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


# ---------------------------------------------------------------------------- results
LINE = re.compile(r'EP:\s+(\d+)/(\d+) \| (dispatch|expanded dispatch|cached dispatch|combine|reduced combine): '
                  r'.*?, ([\d.]+) us, \d+ bytes \| (copy|reduce): \d+ GB/s, ([\d.]+) us')


def mean_of_runs(results_dir: str, arm: str, op: str) -> float:
    """Mean over runs of the all-rank mean kernel time (µs) of `op` for one arm (`v25` / `amz`)."""
    per_run = collections.defaultdict(dict)
    for path in glob.glob(os.path.join(results_dir, '*.log')):
        run = os.path.basename(path).split('.node')[0]
        if f'_{arm}_' not in run:
            continue
        for m in LINE.finditer(open(path).read()):
            if m.group(3) == op:
                per_run[run][int(m.group(1))] = float(m.group(4))
    assert per_run, f'no {arm}/{op} lines in {results_dir}'
    num_ranks = {len(v) for v in per_run.values()}
    assert len(num_ranks) == 1, f'incomplete rank set in {results_dir}: {num_ranks}'
    return sum(sum(v.values()) / len(v) for v in per_run.values()) / len(per_run)


def v25_rows():
    cells = [('8192 tok · 24 SM · 2N', 'n2_matched_20261010'), ('8192 tok · 24 SM · 4N', 'n4_matched_20261010'),
             ('128 tok · 12 SM · 2N', 'decode128_n2_20261010'), ('128 tok · 12 SM · 4N', 'decode128_n4_20261010')]
    rows = []
    for label, d in cells:
        d = os.path.join(V25_RESULTS, d)
        row = [label]
        for op in ('dispatch', 'combine'):
            v25, amz = mean_of_runs(d, 'v25', op), mean_of_runs(d, 'amz', op)
            delta = (v25 / amz - 1) * 100
            row += [f'{amz:.1f}', f'{v25:.1f}', f'{delta:+.2f}%' if abs(delta) < 0.05 else f'{delta:+.1f}%']
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------- drawing
def textbox(slide, x, y, w, h, paras, size=14, color=INK, bold=False):
    """paras: list of str, or of (str, size, color, bold)."""
    tf = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
    tf.word_wrap = True
    for i, para in enumerate(paras):
        t, sz, c, b = para if isinstance(para, tuple) else (para, size, color, bold)
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = t
        r.font.size, r.font.bold = Pt(sz), b
        r.font.color.rgb = c
    return tf


def table(slide, x, y, col_widths, rows, row_h=0.42, size=12, header_size=13):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y),
                                   Inches(sum(col_widths)), Inches(row_h * len(rows)))
    tbl = shape.table
    for j, w in enumerate(col_widths):
        tbl.columns[j].width = Inches(w)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, value in enumerate(row):
            cell = tbl.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = ACCENT if i == 0 else WHITE
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            tf = cell.text_frame
            tf.word_wrap = True
            for k, line in enumerate(str(value).split('\n')):
                p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
                r = p.add_run()
                r.text = line
                r.font.size = Pt(header_size if i == 0 else size)
                r.font.bold = i == 0
                r.font.color.rgb = WHITE if i == 0 else INK
    return tbl


def new_slide(prs, title):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    textbox(s, 0.4, 7.1, 12.5, 0.3, [f'{DECK} · {DATE} · ?/?'], size=10, color=GREY)
    textbox(s, 0.5, 0.35, 12.3, 0.85, [(title, 30, TITLE, True)])
    return s


def move_slide(prs, slide, index):
    ids = prs.slides._sldIdLst
    el = next(e for e in ids if prs.slides.part.related_part(e.rId) is slide.part)
    ids.remove(el)
    ids.insert(index, el)


def replace_text(slide, old, new):
    """Replace inside one run (the deck keeps one run per line) of a text box or a table cell."""
    frames = [sh.text_frame for sh in slide.shapes if sh.has_text_frame]
    for sh in slide.shapes:
        if sh.has_table:
            frames += [c.text_frame for r in sh.table.rows for c in r.cells]
    hits = 0
    for tf in frames:
        for p in tf.paragraphs:
            for r in p.runs:
                if old in r.text:
                    r.text = r.text.replace(old, new)
                    hits += 1
    assert hits, f'{old!r} not found'


# ---------------------------------------------------------------------------- new slides
def slide_prs(prs):
    s = new_slide(prs, '12 · amazon-contributing/DeepEP：8 月以来的 PR')
    textbox(s, 0.5, 1.2, 12.3, 0.45,
            [('main 现在是 51815ea（10-26）；8 月这份 deck 之后合入 10 个 PR，6 个是正确性修复', 15, RED, True)])
    table(s, 0.5, 1.75, [0.75, 0.75, 6.4, 4.4], [
        ['PR', '类型', '内容', '合入 · 我们的实测'],
        ['#15', 'perf', 'token 少时少切 dispatch part（#2 的 clamp；#1 的 JIT env 透传没合）', '09-30 · b300 decode dispatch 277.5 → 129.4 µs'],
        ['#17', 'perf', 'combine forward chunk 48 → 16 槽', '10-19 · b300 prefill dispatch −2.8%，combine 持平'],
        ['#14', 'fix', '8 ch/SM 形状下 combine warp 对的寄存器预算', '09-09'],
        ['#18', 'fix', 'cached dispatch 的数据通道避开 QP 0（Xid 13/31）', '10-09 · 见下页'],
        ['#19', 'fix', 'lane 之间判断 token batch 是否有效的竞态', '10-13'],
        ['#23 #24', 'fix', 'proxy warp 退出前 flush；gin.flush 和等 ring 空槽加超时', '10-19'],
        ['#28', 'fix', 'unordered dispatch 退出前 flush GIN（取代 #20）', '10-24 · 见下页'],
        ['#21', 'fix', 'GIN signal 计数器回绕时 shadow 比较失效', '10-26'],
        ['#32', 'test', 'test_ep 按真实 top-k 计算 buffer 大小', '10-24'],
    ], row_h=0.4, size=11.5, header_size=12.5)
    textbox(s, 0.5, 6.05, 12.3, 1.0, [
        '• 未合：#16 combine 优化（b300 prefill combine +12.1%，待查）、#26 proxy 独占 put、#30/#31 dispatch 超时诊断、'
        '#33/#34 EP_DEBUG_CHECKS、#35 只在跨 NVLink 域时申请 GIN、#36 迁移到 V2.5',
        '• #1/#2 本身是 CLOSED：#2 的逻辑由 #15 进了 main，所以第 10 页的 EP_NUM_SUB_PARTS 在 main 上不生效',
    ], size=12)


def slide_stability(prs):
    s = new_slide(prs, '13 · 两个训练才会遇到的稳定性 bug（都已合入）')
    textbox(s, 0.5, 1.2, 12.3, 0.45,
            [('只在训练里出现：test_ep 压测 60 万次没有触发，Megatron 几分钟就复现', 15, RED, True)])
    table(s, 0.5, 1.75, [1.3, 5.5, 5.5], [
        ['', '#18 cached dispatch 撞 barrier signal', '#28（原 #20）kernel 退出时 put 还在飞'],
        ['现象', 'Xid 13 → Xid 31（MMU fault，野地址 TMA store）；\n约 100 s 后其他 rank 报 barrier timeout；或 grad Inf/NaN',
         '一个 EP 组的所有 rank 同时 grad norm NaN/Inf；\n有时先是单个 rank 的 forward loss NaN'],
        ['根因', 'cached dispatch（combine 的反向）里 channel 0–2 落在 QP 0，\nsignal id 0–2 和 rail barrier 重叠，barrier 的计数被当成数据',
         'unordered dispatch 不等 put 本地完成就退出；下一个 kernel 改写同一块\nsend buffer，SRD 重传会重读源内存，对端收到下一个 kernel 的字节'],
        ['修复', '数据通道从 QP 1 开始（kDataChannelsAboveQP0）', '退出前 flush 所有 GIN context'],
        ['验证', '4 × p6-b200，Megatron EP32 A/B：未修复 6/6 次出事（58 min）；\n修复后 166 min 0 次。b90a617 第 255 步出事，6de427b 1432 步无事',
         'PR 作者在 FP8 形状下验证（combine 的 scale-up 区压在\ndispatch 的 send 区上）；我们没有独立复测'],
    ], row_h=0.78, size=11.5, header_size=12.5)
    textbox(s, 0.5, 6.15, 12.3, 0.8, [
        ('训练用户：直接用当前 main（51815ea），两者都在里面。推理基本不走 cached dispatch，受 #18 影响小。', 14, INK, True)])


def slide_v25(prs):
    s = new_slide(prs, '14 · DeepEP V2.5 on EFA')
    textbox(s, 0.5, 1.2, 12.3, 0.45,
            [('deepseek-ai/DeepEP main（V2.5）原样在 EFA 上跑不起来；移植后与 V2 main 持平', 15, RED, True)])
    textbox(s, 0.5, 1.7, 12.3, 2.0, [
        '• NCCL ≥ 2.32.3 是硬性要求，而 2.32 的 EFA-GDA 要 backendVersion 2：aws-ofi-nccl 1.21.1（installer 1.50.0）只支持 0..1，'
        '目前只有 master（452f302 之后）能用',
        '• JIT 只编 GDAKI 一个设备端后端（NCCL_GIN_EFA_GDA_ENABLE=0）→ 第一次 dispatch 就 Xid 13/31，要打开 EFA-GDA',
        '• context.cpp 写死 GDAKI；Engram 强制 NCCL_GIN_TYPE=3，而且 NCCL 默认要求 EFA-GDA 没有的 strong / VA signal',
        '• 还要 NCCL_CUMEM_HOST_ENABLE=0（2.32.3 下 test_ep 里 torch 的 all_to_all 报 host 内存 OOM）和 libdw-dev（DeepJIT）',
    ], size=12.5)
    rows = [['4 × p6-b200，µs', 'dispatch\nV2 main', 'dispatch\nV2.5', 'Δ',
             'combine\nV2 main', 'combine\nV2.5', 'Δ']] + v25_rows()
    table(s, 0.5, 3.65, [3.0, 1.55, 1.55, 1.0, 1.55, 1.55, 1.0], rows, row_h=0.4, size=12, header_size=12)
    textbox(s, 0.5, 5.95, 12.3, 1.1, [
        '• 同一套环境（NCCL 2.32.3 + aws-ofi-nccl master，GIN type 5），V2 main = 6de427b，全 rank 均值，2–3 次轮转；'
        '4N 全配置 + 20 seed 压测通过。比 V2 时用 --defer-epilogue=0，否则 reduce 看起来慢 50%',
        '• 两条移植：whn09/DeepEP@amazon-v2.5（我们，带性能对比）；amazon-contributing/DeepEP#36（官方，8 节点 p6-b200 / p5en '
        '全矩阵正确性 PASS，待合；已评论 bucket / Engram 的 GIN 布局问题）',
    ], size=11.5)


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'DeepEPv2_on_EFA.pptx'))
    ap.add_argument('--extra-slide', help='python-pptx script that draws one slide on `prs` (inserted before the last slide)')
    args = ap.parse_args()

    prs = Presentation(BASE)
    slides = list(prs.slides)

    # Edits to the 2026-08 slides
    replace_text(slides[0], '原理 · 安装 · 性能', '原理 · 安装 · 性能 · 最新 PR · V2.5')
    replace_text(slides[0], '2026-08 · p5en.48xlarge × 2 / 4', '2026-08 初版 · 2026-10 更新 · p5en / p6-b200 / p6-b300')
    replace_text(slides[6], '• build DeepEP，pin 到 9c1f2511；sm90-bfbdd15 是 PR #1 + #2 那条臂',
                 '• build DeepEP：默认取 main，先解析成 sha 再写进 tag（如 sm100-6de427b），BUILD_REF 记录实际代码')
    replace_text(slides[10], '+PR #1 #2', '+PR #1 #2（#2 已经由 #15 进 main）')
    replace_text(slides[10], '+ EP_NUM_SUB_PARTS=1', '+ EP_NUM_SUB_PARTS=1（需要 #1，未合）')
    replace_text(slides[12], '12 · 三个坑 + 参考', '15 · 三个坑 + 参考')
    replace_text(slides[12], '• 待合 PR：amazon-contributing/DeepEP#1 / #2',
                 '• PR：amazon-contributing/DeepEP（第 12–13 页）；V2.5 迁移 #36；'
                 'V2.5 移植 + 数据：whn09/DeepEP@amazon-v2.5、deepep-v25-efa-port/')

    # New slides, before the last one
    last = len(slides) - 1
    for i, build in enumerate((slide_prs, slide_stability, slide_v25)):
        build(prs)
        move_slide(prs, prs.slides[-1], last + i)

    if args.extra_slide:
        code = open(args.extra_slide).read()
        for old, new in (('prs = Presentation()\n', ''),
                         ('prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)\n', '')):
            assert old in code, f'{args.extra_slide}: {old!r} not found'
            code = code.replace(old, new)
        code = re.sub(r'\nprs\.save\(.*\)\n', '\n', code)
        exec(compile(code, args.extra_slide, 'exec'), {'prs': prs, '__name__': 'extra_slide'})
        move_slide(prs, prs.slides[-1], len(prs.slides) - 2)

    # Page numbers (the extra slide, if any, has no footer)
    numbered = [s for s in prs.slides
                if any(sh.has_text_frame and sh.text_frame.text.startswith(f'{DECK} · ') for sh in s.shapes)]
    for i, s in enumerate(numbered, 1):
        sh = next(sh for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.startswith(f'{DECK} · '))
        sh.text_frame.paragraphs[0].runs[0].text = f'{DECK} · {DATE} · {i}/{len(numbered)}'
        for extra in sh.text_frame.paragraphs[0].runs[1:]:
            extra.text = ''

    prs.save(args.out)
    print(f'saved {args.out}: {len(prs.slides)} slides')


if __name__ == '__main__':
    main()
