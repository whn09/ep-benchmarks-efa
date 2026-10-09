#!/usr/bin/env python3
"""1F1B-shaped DeepEP v2 stress for the Xid31-in-cached-hybrid-dispatch fault.

What it adds over tests/elastic/test_ep.py --do-pressure-test:
  - several handles IN FLIGHT at once and the training call order: forward dispatch +
    combine of microbatch k, then the BACKWARD of microbatch k-D: cached dispatch
    (combine's backward, the path that puts data channels on GIN context 0) + combine;
  - a per-rank, per-call random GPU spin before every collective, so the ranks of the EP
    group enter each dispatch skewed (the barrier/data signal overlap needs skew);
  - routing skewed onto one EP rank's experts;
  - an exact content check on every forward AND cached dispatch: each row carries
    (source token id, column, call salt) encoded as small integers, so a stale row,
    a half-landed row or a row in the wrong slot fails bit-exactly.

Run through run_test_ep.sh with TEST_SCRIPT=<this file>; it accepts (and ignores) the
test_ep.py arguments that script always passes, so the same launcher works.
"""
import argparse
import os
import random
import time
from collections import deque

import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import init_dist, dist_print


def pattern(src_global: torch.Tensor, hidden: int, salt: int) -> torch.Tensor:
    """[n, hidden] bf16 rows that are exact small integers in [-125, 125]"""
    j = torch.arange(hidden, device='cuda', dtype=torch.int64)
    v = (src_global.view(-1, 1) * 131 + j.view(1, -1) * 7 + salt * 17) % 251 - 125
    return v.to(torch.bfloat16)


def make_routing(num_tokens, num_experts, num_topk, num_ranks, hot_rank, hot_frac, masked_frac, gen):
    """top-k with `hot_frac` of tokens forced onto `hot_rank`'s experts and `masked_frac` of
    the selections set to -1 (dropped), the two routing shapes a synthetic uniform gate never makes"""
    scores = torch.rand((num_tokens, num_experts), device='cuda', generator=gen)
    if hot_rank >= 0 and hot_frac > 0:
        per_rank = num_experts // num_ranks
        hot = torch.rand((num_tokens,), device='cuda', generator=gen) < hot_frac
        scores[hot, hot_rank * per_rank:(hot_rank + 1) * per_rank] += 1.0
    w, idx = torch.topk(scores, num_topk, dim=-1, sorted=False)
    idx, w = idx.to(deep_ep.topk_idx_t), w.float()
    if masked_frac > 0:
        drop = torch.rand(idx.shape, device='cuda', generator=gen) < masked_frac
        idx.masked_fill_(drop, -1)
        w.masked_fill_(drop, 0)
    return idx, w


def check(tag, recv_x, handle, hidden, salt, num_max_tokens_per_rank, num_local_experts, recv_topk_idx):
    n = int(handle.psum_num_recv_tokens_per_scaleup_rank[-1].item())
    rx = recv_x[:n]
    src = handle.recv_src_metadata[:n, 0].to(torch.int64)
    bad_src = (src < 0) | (src >= dist.get_world_size() * num_max_tokens_per_rank)
    if bad_src.any():
        raise RuntimeError(f'{tag}: {int(bad_src.sum())} recv_src_metadata out of range, e.g. {src[bad_src][:4].tolist()}')
    exp = pattern(src, hidden, salt)
    diff = (rx != exp).any(dim=1)
    if diff.any():
        rows = diff.nonzero().flatten()
        r = int(rows[0])
        raise RuntimeError(f'{tag}: {int(diff.sum())}/{n} rows wrong; first row {r} src={int(src[r])} '
                           f'got[:4]={rx[r, :4].tolist()} exp[:4]={exp[r, :4].tolist()}')
    if recv_topk_idx is not None:
        t = recv_topk_idx[:n]
        if ((t < -1) | (t >= num_local_experts)).any():
            raise RuntimeError(f'{tag}: recv_topk_idx out of [-1, {num_local_experts})')
    return n


def spin(max_us, rng):
    if max_us > 0:
        torch.cuda._sleep(int(rng.uniform(0, max_us) * 1900))   # ~1.9 GHz SM clock


def worker(local_rank, num_local_ranks, args):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks, seed=args.seed)
    buffer = deep_ep.ElasticBuffer(group, num_max_tokens_per_rank=args.num_tokens, hidden=args.hidden,
                                   deterministic=args.deterministic, allow_hybrid_mode=True,
                                   num_gpu_timeout_secs=args.num_gpu_timeout_secs, explicitly_destroy=True)
    T, H, E, K = args.num_tokens, args.hidden, args.num_experts, args.num_topk
    L = E // num_ranks
    dist_print(f'stress_1f1b: ranks={num_ranks} logical={buffer.get_logical_domain_size()} T={T} H={H} '
               f'E={E} topk={K} sms={args.num_sms} inflight={args.inflight} skew_us={args.skew_us} '
               f'hot_rank={args.hot_rank} hot_frac={args.hot_frac} masked_frac={args.masked_frac} deterministic={args.deterministic} '
               f'qps={buffer.num_allocated_qps}', once_in_node=True)
    gen = torch.Generator(device='cuda')
    rng = random.Random(args.seed * 1000003 + rank)
    my_src = rank * T + torch.arange(T, device='cuda', dtype=torch.int64)
    inflight = deque()
    t0, calls = time.time(), 0
    for it in range(args.iterations):
        gen.manual_seed(args.seed * 7919 + it)          # same routing seed on every rank, different tokens
        topk_idx, topk_w = make_routing(T, E, K, num_ranks, args.hot_rank, args.hot_frac, args.masked_frac, gen)
        topk_idx = topk_idx.roll(rank, 0)               # decorrelate ranks

        # forward of microbatch `it`
        salt = 2 * it
        spin(args.skew_us, rng)
        recv_x, recv_idx, recv_w, handle, _ = buffer.dispatch(
            pattern(my_src, H, salt), topk_idx=topk_idx, topk_weights=topk_w,
            num_experts=E, num_max_tokens_per_rank=T, num_sms=args.num_sms)
        check(f'it{it} fwd', recv_x, handle, H, salt, T, L, recv_idx)
        spin(args.skew_us, rng)
        buffer.combine(recv_x, handle, topk_weights=recv_w, num_sms=args.num_sms)
        inflight.append((it, handle))
        calls += 2

        # backward of microbatch `it - inflight`: combine-backward = CACHED dispatch
        while len(inflight) > args.inflight:
            bit, bh = inflight.popleft()
            bsalt = 2 * bit + 1
            spin(args.skew_us, rng)
            g_x, _, _, _, _ = buffer.dispatch(pattern(my_src, H, bsalt), handle=bh, num_sms=args.num_sms)
            check(f'it{it} bwd(mb{bit}) cached', g_x, bh, H, bsalt, T, L, None)
            spin(args.skew_us, rng)
            buffer.combine(g_x, bh, num_sms=args.num_sms)
            calls += 2
            del g_x, bh
        del recv_x, recv_idx, recv_w, handle
        if (it + 1) % args.log_every == 0:
            torch.cuda.synchronize()
            dist_print(f'  it {it + 1}/{args.iterations}  {calls} collectives  {time.time() - t0:.0f}s  OK',
                       once_in_node=True)
    torch.cuda.synchronize()
    dist_print(f'stress_1f1b PASS: {args.iterations} iterations, {calls} collectives per rank', once_in_node=True)
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num-processes', type=int, default=8)
    p.add_argument('--num-tokens', type=int, default=8192)
    p.add_argument('--hidden', type=int, default=7168)
    p.add_argument('--num-topk', type=int, default=8)
    p.add_argument('--num-experts', type=int, default=256)
    p.add_argument('--num-sms', type=int, default=28)
    p.add_argument('--num-gpu-timeout-secs', type=int, default=100)
    p.add_argument('--iterations', type=int, default=2000)
    p.add_argument('--inflight', type=int, default=2, help='microbatches whose handle is alive before their backward')
    p.add_argument('--skew-us', type=float, default=200.0, help='max per-rank random spin before each collective')
    p.add_argument('--hot-rank', type=int, default=6)
    p.add_argument('--hot-frac', type=float, default=0.3)
    p.add_argument('--masked-frac', type=float, default=0.0, help='fraction of top-k selections set to -1')
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--log-every', type=int, default=50)
    args, _unused = p.parse_known_args()   # run_test_ep.sh always passes test_ep.py flags
    torch.multiprocessing.spawn(worker, args=(args.num_processes, args), nprocs=args.num_processes)
