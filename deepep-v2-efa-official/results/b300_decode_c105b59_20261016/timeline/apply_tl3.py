# Timeline marks for the dispatch copy epilogue (PDL-launched, full GPU):
#   entry = block start, 1 = cudaGridDependencySynchronize() returned, 2 = block end.
# Same GPU clock as the dispatch kernel, so entry/1 can be aligned with dispatch's end.
import sys
root = sys.argv[1]
K = f'{root}/deep_ep/include/deep_ep/impls/dispatch_copy_epilogue.cuh'
s = open(K).read()
subs = [
 ('#include <deep_ep/common/ptx.cuh>\n', '#include <deep_ep/common/ptx.cuh>\n#include <deep_ep/common/timeline.cuh>\n'),
 ('''    // Will block until the main dispatch kernel has finished and all data are visible
    // NOTES: PDL is used, please do not use `__ldg`
    cudaGridDependencySynchronize();
''', '''    EP_TL_ENTRY("copy", (kCachedMode ? 2 : 0) + (kDoExpand ? 1 : 0), scaleout_rank_idx * kNumScaleupRanks + scaleup_rank_idx);
    // Will block until the main dispatch kernel has finished and all data are visible
    // NOTES: PDL is used, please do not use `__ldg`
    cudaGridDependencySynchronize();
    EP_TL_MARK(1);
'''),
 ('''    }
}

}  // namespace deep_ep::elastic''', '''    }
    EP_TL_MARK(2);
}

}  // namespace deep_ep::elastic'''),
]
for old, new in subs:
    n = s.count(old); assert n == 1, f'anchor {n}x: {old[:70]!r}'
    s = s.replace(old, new)
open(K, 'w').write(s)
print('epilogue marks applied')
