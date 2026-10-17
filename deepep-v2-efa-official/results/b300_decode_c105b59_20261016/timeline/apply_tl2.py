# Per-token phase accumulators inside the combine scale-up loop (needs apply_db.py first).
#  9: metadata/routing (loop-body start -> before load/reduce)
# 10: direct path = TMA load ISSUE only; local-reduce path = the whole register reduce
# 11: direct path = wait for the load + issue store; reduce path = issue store
import sys
root = sys.argv[1]
K = f'{root}/deep_ep/include/deep_ep/impls/hybrid_combine_unordered.cuh'
s = open(K).read()
subs = [
 ('''                auto& phase = (kScaleupDB and db_cur) ? phase_alt_ : phase_main_;
''', '''                auto& phase = (kScaleupDB and db_cur) ? phase_alt_ : phase_main_;
                const long long tl_c0 = clock64();
'''),
 ('''                auto no_local_reduce = not kUseExpandedLayout or (kAllowMultipleReduction and __popc(reduce_valid_mask) == 1);
                if (no_local_reduce) {''', '''                auto no_local_reduce = not kUseExpandedLayout or (kAllowMultipleReduction and __popc(reduce_valid_mask) == 1);
                const long long tl_c1 = clock64();
                EP_TL_ACC(9, tl_c1 - tl_c0);
                if (no_local_reduce) {'''),
 ('''                // Write top-k weights (expanded send handled inside the loop above)''',
  '''                const long long tl_c2 = clock64();
                EP_TL_ACC(10, tl_c2 - tl_c1);
                // Write top-k weights (expanded send handled inside the loop above)'''),
 ('''                if constexpr (kScaleupDB)
                    db_cur ^= 1;''', '''                EP_TL_ACC(11, clock64() - tl_c2);
                if constexpr (kScaleupDB)
                    db_cur ^= 1;'''),
]
for old, new in subs:
    n = s.count(old); assert n == 1, f'anchor {n}x: {old[:70]!r}'
    s = s.replace(old, new)
open(K, 'w').write(s)
print('scale-up accumulators applied')
