# [experiment, UNSAFE in general] EP_SKIP_START_RAIL_BARRIER=1 skips the cross-node (rail)
# half of the start barrier in hybrid dispatch and combine. The NVLink half and both grid
# syncs stay. Purpose: measure the upper bound of removing it. test_ep's timed loop puts
# sleep + all_reduce before every op, which already orders ranks, so the timed numbers are
# race-free; the correctness phase calls ops back-to-back and is the stress check.
import sys
root = sys.argv[1]
def edit(path, subs):
    s = open(path).read()
    for old, new in subs:
        n = s.count(old); assert n == 1, f'{path}: anchor {n}x: {old[:70]!r}'
        s = s.replace(old, new)
    open(path, 'w').write(s)
GUARD = '\n#ifndef EP_SKIP_START_RAIL_BARRIER\n#define EP_SKIP_START_RAIL_BARRIER 0\n#endif\n'
edit(f'{root}/deep_ep/include/deep_ep/impls/hybrid_dispatch_unordered.cuh', [
 ('#include <deep_ep/common/ptx.cuh>\n', '#include <deep_ep/common/ptx.cuh>\n' + GUARD),
 ('''                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kHybridDispatchTag0, false, true, true>(
        gin, workspace_layout, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);''',
  '''                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kHybridDispatchTag0, false, true, true>(
        gin, workspace_layout, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx,
        /* do_scaleout */ (EP_SKIP_START_RAIL_BARRIER) == 0);'''),
])
edit(f'{root}/deep_ep/include/deep_ep/impls/hybrid_combine_unordered.cuh', [
 ('#include <deep_ep/impls/proxy_ring.cuh>\n', '#include <deep_ep/impls/proxy_ring.cuh>\n' + GUARD),
 ('''                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kHybridCombineTag0, false, true, true>(
        gin, workspace_layout, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);''',
  '''                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kHybridCombineTag0, false, true, true>(
        gin, workspace_layout, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx,
        /* do_scaleout */ (EP_SKIP_START_RAIL_BARRIER) == 0);'''),
])
edit(f'{root}/csrc/jit/compiler.hpp', [
 ('''        // [experiment] scale-up local-reduce unroll cap; in `flags`, so in the JIT cache key
''', '''        // [experiment] skip the start rail barrier (UNSAFE); in `flags`, so in the JIT cache key
        if (get_env("EP_SKIP_START_RAIL_BARRIER", 0))
            flags += " -DEP_SKIP_START_RAIL_BARRIER=1";
        // [experiment] scale-up local-reduce unroll cap; in `flags`, so in the JIT cache key
'''),
])
print('start-rail-barrier knob applied')
