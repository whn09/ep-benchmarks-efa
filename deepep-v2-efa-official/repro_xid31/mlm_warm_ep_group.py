"""Patch Megatron dev's get_elastic_buffer(): one 4-byte all_reduce on the EP group first.

ElasticBuffer.get_buffer_size_hint() reads the group's NCCL comm via backend._comm_ptr(), which
is NULL until that group has run its first collective. Under --overlap-moe-expert-parallel-comm
the first MoE dispatch is the first thing to touch the EP group, so every rank SIGSEGVs in
elastic.py:441. Warming the group changes nothing about dispatch/combine themselves.
No-op on Megatron-LM main >= 30b9ffe (#5153), which does the same all_reduce upstream.
"""
import re
import sys

path = sys.argv[1]
src = open(path).read()
old = "    global _elastic_buffer\n\n    num_bytes = ElasticBuffer.get_buffer_size_hint("
new = ("    global _elastic_buffer\n\n"
       "    if _elastic_buffer is None:  # repro_xid31: create the EP group's NCCL comm first\n"
       "        torch.distributed.all_reduce(torch.zeros(1, device='cuda'), group=group)\n\n"
       "    num_bytes = ElasticBuffer.get_buffer_size_hint(")
fn = src[src.index('def get_elastic_buffer('):]
fn = fn[:fn.index('\nclass ')]
if 'all_reduce(' in fn:
    # Megatron-LM main since #5153 (30b9ffe) warms the group itself.
    print('upstream get_elastic_buffer() already warms the EP group; nothing to patch', path)
    sys.exit(0)
assert src.count(old) == 1, 'get_elastic_buffer() no longer matches; re-check the patch'
open(path, 'w').write(src.replace(old, new))
print('patched', path)
