"""Bitwise determinism check for EP dispatch/combine: DeepEP v2 ElasticBuffer vs an NCCL all-to-all reference.

Every rank builds fixed inputs from a fixed seed, then loops for --duration seconds over the same
call pattern Megatron uses for one MoE layer:
  forward : recv = dispatch(x)            -> y = expert(recv) -> out = combine(y)
  backward: g    = cached dispatch(dy)    -> gy = expert(g)   -> dx  = combine(gy)
and compares every iteration bit for bit against iteration 0:
  recv_raw    recv_x in the order it was received (arrival order unless deterministic=True)
  recv_sorted recv_x / top-k idx / weights sorted by source token id (content, order-free)
  out, dx     the combined forward output and the backward gradient
It counts mismatches and non-finite values per tensor, logs the UTC time of each mismatch (to line
up with background EFA traffic), and prints a SHA256 per tensor at the end for cross-run comparison.

Launched like tests/elastic/test_ep.py (RANK = node index, WORLD_SIZE = node count, mp.spawn of
--num-processes local ranks); it accepts and ignores test_ep.py's other arguments, so
run_test_ep.sh can start it through TEST_SCRIPT.
  --mode deepep|nccl   --deterministic 0|1   --duration SECONDS   --seed N
"""
import argparse
import datetime
import hashlib
import time

import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import init_dist, dist_print
from deep_ep.utils.gate import get_unbalanced_scores


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S.%f')[:-3]


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def expert(t: torch.Tensor) -> torch.Tensor:
    # Row-wise and order-free, so any difference downstream comes from communication, not compute.
    return (t.float() * 1.5 + 0.25).to(torch.bfloat16)


class NcclEP:
    """Dispatch/combine with all_to_all_single: one copy per (token, k), fixed send and reduce order."""

    def __init__(self, num_experts: int, group):
        self.group, self.world = group, dist.get_world_size(group)
        self.experts_per_rank = num_experts // self.world

    def _a2a(self, send: torch.Tensor, send_counts: torch.Tensor, recv_counts: torch.Tensor) -> torch.Tensor:
        recv = send.new_empty((int(recv_counts.sum()), ) + tuple(send.shape[1:]))
        dist.all_to_all_single(recv, send, recv_counts.tolist(), send_counts.tolist(), group=self.group)
        return recv

    def dispatch(self, x, topk_idx, topk_weights):
        num_tokens, num_topk = topk_idx.shape
        flat = topk_idx.reshape(-1)
        valid = flat >= 0
        token_of = torch.arange(num_tokens, device=x.device).repeat_interleave(num_topk)
        dst = torch.where(valid, flat // self.experts_per_rank, torch.full_like(flat, self.world))
        order = torch.sort(dst * (num_tokens * num_topk) + torch.arange(flat.numel(), device=x.device), stable=True).indices
        order = order[valid[order]]
        send_counts = torch.bincount(dst[order], minlength=self.world)[:self.world].contiguous()
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.group)  # NCCL: counts exchanged on GPU
        send_counts, recv_counts = send_counts.cpu(), recv_counts.cpu()
        self.state = (order, token_of, send_counts, recv_counts, num_tokens)
        recv_x = self._a2a(x[token_of[order]], send_counts, recv_counts)
        recv_w = self._a2a(topk_weights.reshape(-1)[order].contiguous(), send_counts, recv_counts)
        return recv_x, recv_w

    def combine(self, y):
        order, token_of, send_counts, recv_counts, num_tokens = self.state
        back = self._a2a(y, recv_counts, send_counts)                     # rows in `order` = flat (token, k) slots
        num_topk = token_of.numel() // num_tokens
        buf = torch.zeros((num_tokens * num_topk, y.shape[1]), dtype=torch.float, device=y.device)
        buf[order] = back.float()                                         # unique slots: plain scatter, no atomics
        return buf.view(num_tokens, num_topk, -1).sum(1).to(torch.bfloat16)  # fixed k order


@torch.inference_mode()
def worker(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks, seed=args.seed)
    num_tokens, hidden, num_topk, num_experts = args.num_tokens, args.hidden, args.num_topk, args.num_experts

    gen = torch.Generator(device='cuda').manual_seed(args.seed * 100003 + rank)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda', generator=gen)
    dy = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda', generator=gen)
    torch.manual_seed(args.seed * 100003 + rank)
    scores = get_unbalanced_scores(num_tokens, num_experts, num_ranks, num_topk, args.unbalanced_ratio, False)
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)
    topk_weights = topk_weights.float()

    if args.mode == 'deepep':
        buffer = deep_ep.ElasticBuffer(group, num_max_tokens_per_rank=num_tokens, hidden=hidden,
                                       deterministic=bool(args.deterministic), allow_hybrid_mode=True,
                                       allow_multiple_reduction=True, prefer_overlap_with_compute=False,
                                       explicitly_destroy=True)
        num_sms = args.num_sms or buffer.get_theoretical_num_sms(num_experts, num_topk)

        def step():
            recv_x, recv_idx, recv_w, handle, _ = buffer.dispatch(
                x=x, topk_idx=topk_idx, topk_weights=topk_weights, num_sms=num_sms,
                num_max_tokens_per_rank=num_tokens, num_experts=num_experts, expert_alignment=1)
            n = handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()
            src = handle.recv_src_metadata[:n, 0]
            perm = torch.argsort(src, stable=True)
            y = torch.zeros_like(recv_x)
            y[:n] = expert(recv_x[:n])
            out, _, _ = buffer.combine(y, handle=handle, topk_weights=recv_w, num_sms=num_sms)
            g, _, _, _, _ = buffer.dispatch(x=dy, handle=handle, num_sms=num_sms)
            gy = torch.zeros_like(g)
            gy[:n] = expert(g[:n])
            dx, _, _ = buffer.combine(gy, handle=handle, num_sms=num_sms)
            return dict(recv_raw=recv_x[:n].clone(), recv_sorted=recv_x[:n][perm].clone(),
                        recv_idx_sorted=recv_idx[:n][perm].clone(), recv_w_sorted=recv_w[:n][perm].clone(),
                        out=out.clone(), dx=dx.clone())
    else:
        ep = NcclEP(num_experts, group)

        def step():
            recv_x, recv_w = ep.dispatch(x, topk_idx, topk_weights)
            out = ep.combine(expert(recv_x))
            g, _ = ep.dispatch(dy, topk_idx, topk_weights)
            dx = ep.combine(expert(g))
            return dict(recv_raw=recv_x.clone(), recv_w=recv_w.clone(), out=out.clone(), dx=dx.clone())

    ref = step()
    torch.cuda.synchronize()
    mismatch = {k: 0 for k in ref}
    nonfinite = {k: 0 for k in ref}
    events = []
    iters, t_end = 0, time.time() + args.duration
    while True:
        cur = step()
        iters += 1
        for k, v in cur.items():
            if v.shape != ref[k].shape or not torch.equal(v, ref[k]):
                mismatch[k] += 1
                if len(events) < 20:
                    events.append(f'{utc()} iter={iters} {k}')
            if v.is_floating_point() and not torch.isfinite(v).all():
                nonfinite[k] += 1
        # Every rank must agree on when to stop, or a straggler would hang in the next collective.
        stop = torch.tensor([1 if time.time() >= t_end else 0], device='cuda')
        dist.all_reduce(stop, group=group)
        if stop.item() > 0:
            break

    tag = f'mode={args.mode} det={args.deterministic}'
    print(f'EP_DET rank={rank} {tag} iters={iters} '
          + ' '.join(f'{k}:mis={mismatch[k]},nan={nonfinite[k]}' for k in ref)
          + ' sha ' + ' '.join(f'{k}={sha(ref[k])}' for k in ref), flush=True)
    for e in events:
        print(f'EP_DET_EVENT rank={rank} {tag} {e}', flush=True)
    if args.mode == 'deepep':
        buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num-processes', type=int, default=8)
    p.add_argument('--num-tokens', type=int, default=8192)
    p.add_argument('--hidden', type=int, default=7168)
    p.add_argument('--num-topk', type=int, default=6)
    p.add_argument('--num-experts', type=int, default=256)
    p.add_argument('--num-sms', type=int, default=0)
    p.add_argument('--unbalanced-ratio', type=float, default=1.0)
    p.add_argument('--mode', choices=('deepep', 'nccl'), default='deepep')
    p.add_argument('--deterministic', type=int, default=1)
    p.add_argument('--duration', type=float, default=300)
    p.add_argument('--seed', type=int, default=0)
    args, _ = p.parse_known_args()
    torch.multiprocessing.spawn(worker, args=(args.num_processes, args), nprocs=args.num_processes)
