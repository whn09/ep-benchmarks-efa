# [experiment] EP_DISPATCH_SCALEOUT_BASE=<bytes>: place hybrid dispatch's scale-out send/recv
# regions at buffer+BASE instead of right after its scale-up region. With BASE >= the end
# of every combine region, a remote peer's NEXT dispatch can no longer land on combine
# memory that this rank is still reading. Needed because #9's remote-first combine issues
# its last remote put BEFORE it finishes reading local-source tokens from the scale-up
# region, which the default dispatch recv region overlaps (rank-dependent silent errors).
# The host asserts 2*BASE <= buffer bytes (the dispatch scale-out footprint is < BASE here).
import sys
root = sys.argv[1]
def edit(path, subs):
    s = open(path).read()
    for old, new in subs:
        n = s.count(old); assert n == 1, f'{path}: anchor {n}x: {old[:70]!r}'
        s = s.replace(old, new)
    open(path, 'w').write(s)
edit(f'{root}/deep_ep/include/deep_ep/impls/hybrid_dispatch_unordered.cuh', [
 ('#include <deep_ep/common/ptx.cuh>\n',
  '#include <deep_ep/common/ptx.cuh>\n\n#ifndef EP_DISPATCH_SCALEOUT_BASE\n#define EP_DISPATCH_SCALEOUT_BASE 0\n#endif\n'),
 ('''        scaleout_token_layout, kNumScaleoutRanks, kNumChannels * kNumSlotsPerChannel, scaleup_buffer.get_buffer_end_ptr());''',
  '''        scaleout_token_layout, kNumScaleoutRanks, kNumChannels * kNumSlotsPerChannel,
        (EP_DISPATCH_SCALEOUT_BASE) > 0 ? math::advance_ptr<void>(buffer, EP_DISPATCH_SCALEOUT_BASE)
                                         : scaleup_buffer.get_buffer_end_ptr());'''),
])
edit(f'{root}/csrc/elastic/buffer.hpp', [
 ('''        // Decide number of channels by shared memory consumption
        // Only for hybrid version''',
  '''        // [experiment] EP_DISPATCH_SCALEOUT_BASE must leave room for the moved regions
        if (const int64_t base = get_env<int>("EP_DISPATCH_SCALEOUT_BASE", 0); base > 0)
            EP_HOST_ASSERT(base % 128 == 0 and 2 * base <= num_buffer_bytes and
                           "EP_DISPATCH_SCALEOUT_BASE does not fit the buffer");
        // Decide number of channels by shared memory consumption
        // Only for hybrid version'''),
])
edit(f'{root}/csrc/jit/compiler.hpp', [
 ('''        // [experiment] combine signal offset + layout print; in `flags`, so in the JIT cache key
''', '''        // [experiment] dispatch scale-out region base; in `flags`, so in the JIT cache key
        if (int64_t so_base = get_env<int>("EP_DISPATCH_SCALEOUT_BASE", 0); so_base > 0)
            flags += fmt::format(" -DEP_DISPATCH_SCALEOUT_BASE={}ll", so_base);
        // [experiment] combine signal offset + layout print; in `flags`, so in the JIT cache key
'''),
])
print('dispatch scale-out base applied')
