# Inserts EP_TIMELINE marks into the hybrid unordered dispatch/combine kernels.
# Every anchor must match exactly once, or the script fails (no silent half-patch).
import sys, re
root = sys.argv[1]
def edit(path, subs):
    s = open(path).read()
    for old, new in subs:
        n = s.count(old)
        assert n == 1, f'{path}: anchor found {n}x: {old[:80]!r}'
        s = s.replace(old, new)
    open(path, 'w').write(s)

D = f'{root}/deep_ep/include/deep_ep/impls/hybrid_dispatch_unordered.cuh'
edit(D, [
 ('#include <deep_ep/common/ptx.cuh>\n', '#include <deep_ep/common/ptx.cuh>\n#include <deep_ep/common/timeline.cuh>\n'),
 # entry stamp right before the start barrier; event 1 = start barrier exit
 ('''    comm::gpu_barrier<true, kNumScaleoutRanks, kNumScaleupRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kHybridDispatchTag0, false, true, true>(
        gin, workspace_layout, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);
''', '''    EP_TL_ENTRY("disp", (kDoCPUSync ? 4 : 0) + (kReuseSlotIndices ? 2 : 0) + (kDoExpand ? 1 : 0),
                scaleout_rank_idx * kNumScaleupRanks + scaleup_rank_idx);
    comm::gpu_barrier<true, kNumScaleoutRanks, kNumScaleupRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kHybridDispatchTag0, false, true, true>(
        gin, workspace_layout, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);
    EP_TL_MARK(1);
'''),
 # event 7 = notify role done
 ('''                do_psum(expert_count, psum_num_recv_tokens_per_expert, kNumExpertsPerRank, 1);
            }
        }
    } else if (warp_idx < kNumNotifyWarps + kNumScaleoutWarps) {''',
  '''                do_psum(expert_count, psum_num_recv_tokens_per_expert, kNumExpertsPerRank, 1);
            }
        }
        EP_TL_MARK(7);
    } else if (warp_idx < kNumNotifyWarps + kNumScaleoutWarps) {'''),
 # event 2 = each scale-out data put (min = first, max = last); issued by one lane
 ('''            const int num_put_tokens = count > 0 ? count : 1;
            gin.put<ncclTeamTagRail>(''',
  '''            const int num_put_tokens = count > 0 ? count : 1;
            EP_TL_MARK_LANE(2);
            gin.put<ncclTeamTagRail>('''),
 # event 3 = scale-out role done
 ('''        // Flush unflushed tails
        update_scaleout_tail(true);
''', '''        // Flush unflushed tails
        update_scaleout_tail(true);
        EP_TL_MARK(3);
'''),
 # event 4 = forward warp saw data/finish for a source (min = first arrival, max = last)
 ('''                __syncwarp();
                return false;
            });

            // Process one chunk from the current rank''',
  '''                __syncwarp();
                return false;
            });
            EP_TL_MARK(4);

            // Process one chunk from the current rank'''),
 # event 5 = forward loop done
 ('''        // Assign the source token index part of the metadata into `-1` as an ending mark
        if (not kReuseSlotIndices and ptx::elect_one_sync())''',
  '''        EP_TL_MARK(5);
        // Assign the source token index part of the metadata into `-1` as an ending mark
        if (not kReuseSlotIndices and ptx::elect_one_sync())'''),
 # event 6 = forward shadow drain done (forward role end)
 ('''        if (lane_idx < kNumScaleoutRanks)
            *workspace_layout.get_scaleout_channel_signaled_tail_ptr(channel_idx, lane_idx) = 0;
        __syncwarp();
    }
''', '''        if (lane_idx < kNumScaleoutRanks)
            *workspace_layout.get_scaleout_channel_signaled_tail_ptr(channel_idx, lane_idx) = 0;
        __syncwarp();
        EP_TL_MARK(6);
    }
'''),
 # event 8 = end (scale-up) barrier exit
 ('''    // Trigger the copy epilogue kernel
    cudaTriggerProgrammaticLaunchCompletion();''',
  '''    EP_TL_MARK(8);
    // Trigger the copy epilogue kernel
    cudaTriggerProgrammaticLaunchCompletion();'''),
])

C = f'{root}/deep_ep/include/deep_ep/impls/hybrid_combine_unordered.cuh'
edit(C, [
 ('#include <deep_ep/common/ptx.cuh>\n', '#include <deep_ep/common/ptx.cuh>\n#include <deep_ep/common/timeline.cuh>\n'),
 ('''    comm::gpu_barrier<true, kNumScaleoutRanks, kNumScaleupRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kHybridCombineTag0, false, true, true>(
        gin, workspace_layout, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);
''', '''    EP_TL_ENTRY("comb", kUseExpandedLayout ? 1 : 0, scaleout_rank_idx * kNumScaleupRanks + scaleup_rank_idx);
    comm::gpu_barrier<true, kNumScaleoutRanks, kNumScaleupRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kHybridCombineTag0, false, true, true>(
        gin, workspace_layout, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);
    EP_TL_MARK(1);
'''),
 # event 2 = scale-up role done
 ('''        // Update for the unissued ones
        update_tails(true);
        }
    } else if (warp_idx < kNumDataWarps) {''',
  '''        // Update for the unissued ones
        update_tails(true);
        }
        EP_TL_MARK(2);
    } else if (warp_idx < kNumDataWarps) {'''),
 # event 3 = forward staging done, about to wait for remote arrivals
 ('''        int num_expected_arrivals = 0;
        #pragma unroll
        for (int r = 0; r < kNumScaleoutRanks; ++ r) {''',
  '''        EP_TL_MARK(3);
        int num_expected_arrivals = 0;
        #pragma unroll
        for (int r = 0; r < kNumScaleoutRanks; ++ r) {'''),
 # event 4 = remote arrivals complete (forward role end)
 ('''                return false;
            });
        }
        __syncwarp();
    } else {
        // Proxy warp loop shape''',
  '''                return false;
            });
        }
        __syncwarp();
        EP_TL_MARK(4);
    } else {
        // Proxy warp loop shape'''),
 # event 5 = each proxy RDMA put (single-lane sweeper; lane-level mark)
 ('''                        gin_ctx[forward_warp_idx].put<ncclTeamTagRail>(''',
  '''                        EP_TL_MARK_LANE(5);
                        gin_ctx[forward_warp_idx].put<ncclTeamTagRail>('''),
 # event 6 = proxy done
 ('''                    if (gin_put_ops == 0)
                        __nanosleep(1000);
                }
            }
            __syncwarp();''',
  '''                    if (gin_put_ops == 0)
                        __nanosleep(1000);
                }
            }
            __syncwarp();
            EP_TL_MARK(6);'''),
 # event 7 = forward warp hands a batch to the proxy ring (lane-level, under elect)
 ('''        const auto issue_batched_rdma = [&](const int& dst) {
            if (batch_count[dst] == 0)
                return;
''', '''        const auto issue_batched_rdma = [&](const int& dst) {
            if (batch_count[dst] == 0)
                return;
            EP_TL_MARK_LANE(7);
'''),
])
print('timeline marks applied')
