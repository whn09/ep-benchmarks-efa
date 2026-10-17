# [experiment] Double-buffer the unordered combine's scale-up warps (EP_COMBINE_SCALEUP_DB=1).
# Today each scale-up warp owns ONE TMA buffer, and before refilling it waits for the
# previous token's NVLink store to fully drain (tma_store_wait<0>), so a warp has a
# single token in flight and NVLink push throughput scales only with channel count.
# With two buffers the store of token i overlaps the load/reduce of token i+1; a buffer
# is refilled after waiting only for the store issued two tokens ago (wait_group 1).
# Default 0 = byte-identical behaviour. Expanded-send path is left single-buffered.
import sys
root = sys.argv[1]
def edit(path, subs):
    s = open(path).read()
    for old, new in subs:
        n = s.count(old); assert n == 1, f'{path}: anchor {n}x: {old[:80]!r}'
        s = s.replace(old, new)
    open(path, 'w').write(s)

K = f'{root}/deep_ep/include/deep_ep/impls/hybrid_combine_unordered.cuh'
edit(K, [
 ('#include <deep_ep/impls/proxy_ring.cuh>\n',
  '#include <deep_ep/impls/proxy_ring.cuh>\n\n#ifndef EP_COMBINE_SCALEUP_DB\n#define EP_COMBINE_SCALEUP_DB 0\n#endif\n'),
 # extra buffers, one per scale-up warp, appended after the existing ones
 ('    const auto tma_buffer_layout = layout::BufferLayout<true>(token_layout, kNumTmaWarps, 1, smem);',
  '''    constexpr bool kScaleupDB = (EP_COMBINE_SCALEUP_DB) != 0;
    const auto tma_buffer_layout = layout::BufferLayout<true>(
        token_layout, kNumTmaWarps + (kScaleupDB ? kNumScaleupWarps : 0), 1, smem);'''),
 # alt buffer + its mbarrier, set up at the top of the scale-up role
 ('''        // Shift into the right buffer if using rank layout
        if constexpr (kUseScaleupRankLayout)
            scaleup_buffer = scaleup_buffer.get_rank_buffer(scaleup_rank_idx);
''', '''        // Shift into the right buffer if using rank layout
        if constexpr (kUseScaleupRankLayout)
            scaleup_buffer = scaleup_buffer.get_rank_buffer(scaleup_rank_idx);

        // [experiment] double buffering: buffer 0 = the warp's own, buffer 1 = appended one
        const auto tma_buffer_main_ = tma_buffer;
        const auto tma_buffer_alt_ = tma_buffer_layout
            .get_rank_buffer(kScaleupDB ? kNumTmaWarps + warp_idx : tma_buffer_idx).get_token_buffer(0);
        auto& phase_main_ = phase;
        ptx::arrival_phase phase_alt_ = 0;
        int db_cur = 0;
        if (kScaleupDB and ptx::elect_one_sync())
            ptx::mbarrier_init_with_fence(tma_buffer_alt_.get_mbarrier_ptr(), 1);
        __syncwarp();
'''),
 # per-token buffer selection (shadows the kernel-scope names inside the loop body)
 ('''            while (wip_mask) {
                // Find next active rank after `dst_scaleup_rank_idx` (round-robin)''',
  '''            while (wip_mask) {
                const auto tma_buffer = (kScaleupDB and db_cur) ? tma_buffer_alt_ : tma_buffer_main_;
                const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();
                auto& phase = (kScaleupDB and db_cur) ? phase_alt_ : phase_main_;
                // Find next active rank after `dst_scaleup_rank_idx` (round-robin)'''),
 # direct-load path: wait only for the store that used THIS buffer (two tokens ago)
 ('''                        ptx::tma_store_wait();
                        ptx::tma_load_1d(tma_buffer.get_base_ptr(), load_ptr, mbarrier_ptr, kNumHiddenBytes);
                    }
                    __syncwarp();
                } else if constexpr (kAllowMultipleReduction) {''',
  '''                        ptx::tma_store_wait<kScaleupDB ? 1 : 0>();
                        ptx::tma_load_1d(tma_buffer.get_base_ptr(), load_ptr, mbarrier_ptr, kNumHiddenBytes);
                    }
                    __syncwarp();
                } else if constexpr (kAllowMultipleReduction) {'''),
 # local-reduce path: same relaxed wait before writing the reduced token into smem
 ('''                        /* Wait buffer release */ [=]() {
                            ptx::tma_store_wait();
                            __syncwarp();
                        }
                    );
                    ptx::tma_store_fence();''',
  '''                        /* Wait buffer release */ [=]() {
                            ptx::tma_store_wait<kScaleupDB ? 1 : 0>();
                            __syncwarp();
                        }
                    );
                    ptx::tma_store_fence();'''),
 # flip after this token's store has been committed
 ('''                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                    stored_num_tokens_sent[j] += (j * 32 + lane_idx) == dst_scaleup_rank_idx;
                __syncwarp();
            }''',
  '''                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                    stored_num_tokens_sent[j] += (j * 32 + lane_idx) == dst_scaleup_rank_idx;
                __syncwarp();
                if constexpr (kScaleupDB)
                    db_cur ^= 1;
            }'''),
])
edit(f'{root}/csrc/jit/compiler.hpp', [
 ('''            flags += fmt::format(" -DEP_NUM_TOPK_IDX_BITS={}", num_topk_idx_bits);
''', '''            flags += fmt::format(" -DEP_NUM_TOPK_IDX_BITS={}", num_topk_idx_bits);
        // [experiment] combine scale-up double buffering; in `flags`, so in the JIT cache key
        if (get_env("EP_COMBINE_SCALEUP_DB", 0))
            flags += " -DEP_COMBINE_SCALEUP_DB=1";
'''),
])
edit(f'{root}/csrc/kernels/elastic/combine.hpp', [
 ('            const int64_t tma_smem_bytes = static_cast<int64_t>(num_data_warps) * token_layout.get_num_bytes<true>();',
  '''            const int64_t tma_smem_bytes = static_cast<int64_t>(
                num_data_warps + (get_env<int>("EP_COMBINE_SCALEUP_DB", 0) ? num_scaleup_warps : 0)) *
                token_layout.get_num_bytes<true>();'''),
])
edit(f'{root}/csrc/elastic/buffer.hpp', [
 ('                   static_cast<int64_t>(2 * num_channels_per_sm) *',
  '                   static_cast<int64_t>((get_env<int>("EP_COMBINE_SCALEUP_DB", 0) ? 3 : 2) * num_channels_per_sm) *'),
])
print('scale-up double buffering applied')
