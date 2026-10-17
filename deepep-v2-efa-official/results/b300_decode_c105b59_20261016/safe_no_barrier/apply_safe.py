# [experiment] Make EP_SKIP_START_RAIL_BARRIER race-free on GIN signals, and expose the
# buffer layouts so region overlap can be checked.
#  * EP_COMBINE_SIGNAL_OFFSET=N: combine's per-channel indexed signals move to [N, 2N);
#    the host plans dispatch (part count, channel tuner) within [0, N). Without it a
#    remote peer's EARLY combine puts (SRD is unordered) could bump the very signal my
#    still-running dispatch waits on for the same channel. Host asserts 2N <= budget.
#  * EP_PRINT_LAYOUT=1: rank 0 / block 0 prints each region's [start, end) relative to
#    `buffer` once per kernel variant, for an offline overlap check.
import sys
root = sys.argv[1]
def edit(path, subs):
    s = open(path).read()
    for old, new in subs:
        n = s.count(old); assert n == 1, f'{path}: anchor {n}x: {old[:70]!r}'
        s = s.replace(old, new)
    open(path, 'w').write(s)
GUARD = ('\n#ifndef EP_COMBINE_SIGNAL_OFFSET\n#define EP_COMBINE_SIGNAL_OFFSET 0\n#endif\n'
         '#ifndef EP_PRINT_LAYOUT\n#define EP_PRINT_LAYOUT 0\n#endif\n')
C = f'{root}/deep_ep/include/deep_ep/impls/hybrid_combine_unordered.cuh'
s = open(C).read()
old_sig = 'comm::get_qp_signal_id<kNumSMs, kNumQPs, kNumChannelsPerSM, true>('
assert s.count(old_sig) == 3, s.count(old_sig)
s = s.replace(old_sig, '(EP_COMBINE_SIGNAL_OFFSET) + comm::get_qp_signal_id<kNumSMs, kNumQPs, kNumChannelsPerSM, true>(')
open(C, 'w').write(s)
edit(C, [
 ('#include <deep_ep/impls/proxy_ring.cuh>\n', '#include <deep_ep/impls/proxy_ring.cuh>\n' + GUARD),
 ('''    // Init TMA for scale-up and forward warps
    ptx::arrival_phase phase = 0;''', '''    if constexpr (EP_PRINT_LAYOUT) {
        if (sm_idx == 0 and thread_idx == 0 and scaleout_rank_idx == 0 and scaleup_rank_idx == 0)
            printf("EPLAYOUT comb expanded=%d scaleup_end=%lld recv_end=%lld send_end=%lld\\n",
                   static_cast<int>(kUseExpandedLayout),
                   static_cast<long long>(static_cast<int8_t*>(scaleup_buffer.get_buffer_end_ptr()) - static_cast<int8_t*>(buffer)),
                   static_cast<long long>(static_cast<int8_t*>(scaleout_recv_buffer.get_buffer_end_ptr()) - static_cast<int8_t*>(buffer)),
                   static_cast<long long>(static_cast<int8_t*>(scaleout_send_buffer.get_buffer_end_ptr()) - static_cast<int8_t*>(buffer)));
    }
    // Init TMA for scale-up and forward warps
    ptx::arrival_phase phase = 0;'''),
])
Dp = f'{root}/deep_ep/include/deep_ep/impls/hybrid_dispatch_unordered.cuh'
edit(Dp, [
 ('#include <deep_ep/common/ptx.cuh>\n', '#include <deep_ep/common/ptx.cuh>\n' + GUARD),
 ('''    // Init TMA for scale-out and forward warps
    ptx::arrival_phase phase = 0;''', '''    if constexpr (EP_PRINT_LAYOUT) {
        if (sm_idx == 0 and thread_idx == 0 and scaleout_rank_idx == 0 and scaleup_rank_idx == 0)
            printf("EPLAYOUT disp expand=%d cached=%d scaleup_end=%lld send_end=%lld recv_end=%lld\\n",
                   static_cast<int>(kDoExpand), static_cast<int>(kReuseSlotIndices),
                   static_cast<long long>(static_cast<int8_t*>(scaleup_buffer.get_buffer_end_ptr()) - static_cast<int8_t*>(buffer)),
                   static_cast<long long>(static_cast<int8_t*>(scaleout_send_buffer.get_buffer_end_ptr()) - static_cast<int8_t*>(buffer)),
                   static_cast<long long>(static_cast<int8_t*>(scaleout_recv_buffer.get_buffer_end_ptr()) - static_cast<int8_t*>(buffer)));
    }
    // Init TMA for scale-out and forward warps
    ptx::arrival_phase phase = 0;'''),
])
B = f'{root}/csrc/elastic/buffer.hpp'
edit(B, [
 # channel tuner and the dispatch kernel both plan within the lower half of the budget
 ('''                num_channels_per_sm = elastic::gin_alloc::constexpr_channels_per_sm(
                    nccl_context->gin_config.gin_indexed_signals_cnt,''',
  '''                num_channels_per_sm = elastic::gin_alloc::constexpr_channels_per_sm(
                    dispatch_signal_budget(nccl_context->gin_config.gin_indexed_signals_cnt),'''),
 ('''                        nccl_context->gin_config.gin_indexed_signals_cnt,''',
  '''                        dispatch_signal_budget(nccl_context->gin_config.gin_indexed_signals_cnt),'''),
])
s = open(B).read()
anchor = 'namespace deep_ep'
i = s.index(anchor); j = s.index('{', i) + 1
helper = '''

// [experiment] EP_COMBINE_SIGNAL_OFFSET: dispatch plans within [0, N), combine uses [N, 2N).
static inline int dispatch_signal_budget(const int& budget) {
    const int offset = get_env<int>("EP_COMBINE_SIGNAL_OFFSET", 0);
    if (offset <= 0)
        return budget;
    EP_HOST_ASSERT(2 * offset <= budget and "EP_COMBINE_SIGNAL_OFFSET: 2N must fit the GIN signal budget");
    return offset;
}
'''
s = s[:j] + helper + s[j:]
open(B, 'w').write(s)
edit(f'{root}/csrc/jit/compiler.hpp', [
 ('''        // [experiment] skip the start rail barrier (UNSAFE); in `flags`, so in the JIT cache key
''', '''        // [experiment] combine signal offset + layout print; in `flags`, so in the JIT cache key
        if (int sig_off = get_env("EP_COMBINE_SIGNAL_OFFSET", 0); sig_off > 0)
            flags += fmt::format(" -DEP_COMBINE_SIGNAL_OFFSET={}", sig_off);
        if (get_env("EP_PRINT_LAYOUT", 0))
            flags += " -DEP_PRINT_LAYOUT=1";
        // [experiment] skip the start rail barrier (UNSAFE); in `flags`, so in the JIT cache key
'''),
])
print('signal split + layout print applied')
