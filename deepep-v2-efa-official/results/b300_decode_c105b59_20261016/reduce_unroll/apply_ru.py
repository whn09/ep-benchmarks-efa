# [experiment] EP_COMBINE_REDUCE_UNROLL: max unroll of the scale-up warps' local top-k
# reduce (default 4 = unchanged). At hidden 7168 with int4 vectors each lane walks 28
# vectors per slot; unroll 4 => 7 serial load rounds per token, each paying a full
# global-load latency (~1.3 us/token measured). 7 => 4 rounds, 14 => 2 rounds.
import sys
root = sys.argv[1]
def edit(path, subs):
    s = open(path).read()
    for old, new in subs:
        n = s.count(old); assert n == 1, f'{path}: anchor {n}x: {old[:70]!r}'
        s = s.replace(old, new)
    open(path, 'w').write(s)
edit(f'{root}/deep_ep/include/deep_ep/impls/hybrid_combine_unordered.cuh', [
 ('#include <deep_ep/impls/proxy_ring.cuh>\n',
  '#include <deep_ep/impls/proxy_ring.cuh>\n\n#ifndef EP_COMBINE_REDUCE_UNROLL\n#define EP_COMBINE_REDUCE_UNROLL 4\n#endif\n'),
 ('                    constexpr int kUnrollFactor = get_max_unroll_factor<kHiddenVec, 4>();',
  '                    constexpr int kUnrollFactor = get_max_unroll_factor<kHiddenVec, EP_COMBINE_REDUCE_UNROLL>();'),
])
edit(f'{root}/csrc/jit/compiler.hpp', [
 ('''        // [experiment] combine scale-up double buffering; in `flags`, so in the JIT cache key
''', '''        // [experiment] scale-up local-reduce unroll cap; in `flags`, so in the JIT cache key
        if (int reduce_unroll = get_env("EP_COMBINE_REDUCE_UNROLL", 0); reduce_unroll > 0)
            flags += fmt::format(" -DEP_COMBINE_REDUCE_UNROLL={}", reduce_unroll);
        // [experiment] combine scale-up double buffering; in `flags`, so in the JIT cache key
'''),
])
print('reduce unroll knob applied')
