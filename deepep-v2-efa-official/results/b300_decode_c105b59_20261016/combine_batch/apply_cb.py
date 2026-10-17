# [experiment] Make the unordered combine's RDMA batch size (tokens per put) a JIT
# flag driven by env EP_COMBINE_BATCH_SIZE (default 12 = unchanged). At decode a
# channel never fills a 12-token batch, so every put waits for the channel loop to
# finish; a smaller batch lets puts start while the scale-up/reduce phase runs.
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
  '#include <deep_ep/impls/proxy_ring.cuh>\n\n#ifndef EP_COMBINE_BATCH_SIZE\n#define EP_COMBINE_BATCH_SIZE 12\n#endif\n'),
 ('          int kBatchSize = 12,', '          int kBatchSize = EP_COMBINE_BATCH_SIZE,'),
])
edit(f'{root}/csrc/jit/compiler.hpp', [
 ('''            flags += fmt::format(" -DEP_NUM_TOPK_IDX_BITS={}", num_topk_idx_bits);
''', '''            flags += fmt::format(" -DEP_NUM_TOPK_IDX_BITS={}", num_topk_idx_bits);
        // [experiment] combine RDMA batch size; part of `flags`, so of the JIT cache key
        if (int combine_batch = get_env("EP_COMBINE_BATCH_SIZE", 0); combine_batch > 0)
            flags += fmt::format(" -DEP_COMBINE_BATCH_SIZE={}", combine_batch);
'''),
])
print('combine batch knob applied')
