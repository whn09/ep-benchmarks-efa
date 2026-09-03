#!/usr/bin/env python3
"""Same additivity question as p5en, on b300: is #1+#2+#8+#9 worth the sum of its parts?

usage: make_stack_tables.py            (logs are read from ./logs, or $EPRUNS_B300)

Deliberately NOT $EPRUNS: docs/check_runbook_numbers.py loads several generators in
one process and each sets EPRUNS for its own campaign, so keying off it here would
silently make this driver read the p5en logs.

2 x p6-b300.48xlarge (B300-1/B300-2, ap-northeast-2), Blackwell Ultra **sm_103**,
16 EFA NICs/node at 100 GB/s per GPU, EFA installer 1.50.0 (efa.ko 3.3.0,
rdma-core 64.0, libfabric 2.6.0, libnccl-ofi 1.21.1), GIN **type 5** (EFA GDA,
`efa-direct`, verified from an NCCL_DEBUG=INFO run: the type-2 plugin is skipped),
`NCCL_IB_HCA=rdmap` auto-injected because the b300 device list also carries two
non-EFA `ibp19*` ports. 12 SM, --prefer-overlap-with-compute=0, 4 arms x 3 rotated
reps, --test-first-only (so FP8 dispatch at expert_alignment=128, not BF16).

  main   54fffeff810723f574c574b1790dff189f3c6ffb
  pr12   bfbdd15ff448783f877cb2210cb3246c8452b05e   PRs #1 + #2   (dispatch side)
  pr89   3c737dcf0da5889ba7efd26e05b4808307cc38af   PRs #8 + #9   (combine side)
  stack  a35285f0af98856625e542df24bd17a985bc05d9   git merge of the two

The stack merge sha is byte-identical to the p5en campaign's because
build_stack_image.sh pins the git author/committer dates; both b300 hosts also
produced it independently and were checked to agree before any cell ran. So a b300
row and a p5en row with the same arm label really are the same tree.

All the aggregation, pooling and table code is the p5en generator -- this file only
points it at the b300 logs and prints the host header, so the two campaigns cannot
drift into different arithmetic. TIME IS THE METRIC; --ignore-local-traffic is OFF,
so the SO column includes intra-node traffic and is not a wire rate.
"""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["EPRUNS_STACK"] = os.environ.get("EPRUNS_B300", os.path.join(HERE, "logs"))
_p5en = os.path.join(os.path.dirname(HERE), "p5en_stack_20260831", "make_stack_tables.py")
_spec = importlib.util.spec_from_file_location("gen_stack", _p5en)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

if __name__ == "__main__":
    print("2 x p6-b300.48xlarge (sm_103), EFA 1.50.0 + efa.ko 3.3.0, GIN type 5")
    print("EPRUNS=%s" % m.g.EPRUNS)
    print()
    m.config_table()
    for tok, what in m.TOKS:
        m.perf_table(tok, what)
    for tok, what in m.TOKS:
        m.additivity(tok, what)
    for tok, what in m.TOKS:
        m.best_of(tok, what)
    m.audit()
