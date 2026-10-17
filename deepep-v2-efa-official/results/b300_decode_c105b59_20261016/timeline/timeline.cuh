#pragma once
// [experiment] Per-block phase timeline for the hybrid unordered kernels.
// Compiled in only with -DEP_TIMELINE. Each block keeps min/max %globaltimer stamps
// per event in a __device__ array (no shared memory, so the channel budget is
// unchanged). Call N's stamps are printed at the ENTRY of call N+1 by thread 0, when
// call N has fully retired, so no end-of-kernel sync is needed. Only calls in the
// sampled set are printed; all others pay a handful of global atomics.
#ifdef EP_TIMELINE
namespace deep_ep::timeline {
constexpr int kMaxBlocks = 192;
constexpr int kMaxEvents = 12;
__device__ unsigned long long g_tl[kMaxBlocks][2 * kMaxEvents];
__device__ unsigned int g_tl_calls[kMaxBlocks];

__device__ __forceinline__ unsigned long long now() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ void mark(const int e) {
    const auto t = now();
    atomicMin(&g_tl[blockIdx.x][2 * e], t);
    atomicMax(&g_tl[blockIdx.x][2 * e + 1], t);
}
// Accumulator events (kFirstAccEvent..): slot 2e sums a clock64() delta, 2e+1 counts.
constexpr int kFirstAccEvent = 9;
__device__ __forceinline__ void acc(const int e, const long long dt) {
    atomicAdd(&g_tl[blockIdx.x][2 * e], static_cast<unsigned long long>(dt));
    atomicAdd(&g_tl[blockIdx.x][2 * e + 1], 1ull);
}
__device__ __forceinline__ bool sampled(const unsigned c) {
    return c == 30 or c == 40 or c == 50;
}
// Thread 0 only, at kernel entry.
__device__ __forceinline__ void entry(const char* kname, const int variant, const int rank) {
    const int b = blockIdx.x;
    const unsigned c = g_tl_calls[b]++;
    auto* s = g_tl[b];
    if (c > 0 and sampled(c - 1)) {
        // Build the whole line locally and emit it with a single %s: device printf
        // writes each conversion separately, so 8 processes sharing a pipe would
        // otherwise interleave mid-line.
        char line[512];
        int n = 0;
        const auto put_str = [&](const char* str) { while (*str and n < 500) line[n ++] = *str ++; };
        const auto put_int = [&](long long v) {
            if (v < 0) { line[n ++] = '-'; v = -v; }
            char tmp[24]; int k = 0;
            do { tmp[k ++] = static_cast<char>('0' + v % 10); v /= 10; } while (v > 0 and k < 23);
            while (k > 0 and n < 500) line[n ++] = tmp[-- k];
        };
        const auto t0 = s[0];
        put_str("EPTL "); put_str(kname);
        put_str(" v"); put_int(variant); put_str(" r"); put_int(rank);
        put_str(" b"); put_int(b); put_str(" c"); put_int(static_cast<long long>(c - 1));
        put_str(" t0 "); put_int(static_cast<long long>(t0 % 1000000000000ull));
        for (int i = 2; i < 2 * kMaxEvents; ++ i) {
            put_str(" ");
            if (i >= 2 * kFirstAccEvent)
                put_int(static_cast<long long>(s[i]));
            else
                put_int((s[i] == ~0ull or s[i] == 0ull) ? -1ll : static_cast<long long>(s[i] - t0));
        }
        line[n ++] = '\n'; line[n] = 0;
        printf("%s", line);
    }
    #pragma unroll
    for (int i = 0; i < 2 * kMaxEvents; ++ i)
        s[i] = (i >= 2 * kFirstAccEvent or (i & 1)) ? 0ull : ~0ull;
    const auto t = now();
    s[0] = t; s[1] = t;
    __threadfence();
}
}  // namespace deep_ep::timeline
#define EP_TL_ENTRY(kname, variant, rank) do { if (threadIdx.x == 0) deep_ep::timeline::entry(kname, variant, rank); } while (0)
#define EP_TL_MARK(e) do { if ((threadIdx.x & 31) == 0) deep_ep::timeline::mark(e); } while (0)
#define EP_TL_MARK_LANE(e) deep_ep::timeline::mark(e)
#define EP_TL_ACC(e, dt) do { if ((threadIdx.x & 31) == 0) deep_ep::timeline::acc(e, dt); } while (0)
#else
#define EP_TL_ACC(e, dt) do {} while (0)
#define EP_TL_ENTRY(kname, variant, rank) do {} while (0)
#define EP_TL_MARK(e) do {} while (0)
#define EP_TL_MARK_LANE(e) do {} while (0)
#endif
