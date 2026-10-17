#!/usr/bin/env bash
# Regenerates tables.txt from logs/ and timeline/logs/. Do not hand-edit tables.txt.
set -euo pipefail
cd "$(dirname "$0")"
T=tools; L=logs; TL=timeline/logs
{
echo "b300 decode campaign, amazon-contributing/DeepEP main c105b59 (+ PR merges / experiment layers)"
echo "2 x p6-b300.48xlarge, 2N/16 ranks unless noted, GIN type 5, ovlp=0, FP8 dispatch (--test-first-only)"
echo "Every number = mean over reps of the all-16-rank mean, us. Layer = dispatch + reduced combine;"
echo "'+epi' adds the copy and reduce epilogue kernels."
echo
echo "########## A. PR arms, 12 SM (batch 1, 3 reps)"
for t in 128 8192; do python3 -I $T/dec.py $L $t; echo; done
echo "########## B. token floor, 12 SM (batch 2, 3 reps)"
for t in 1 32; do python3 -I $T/dec.py $L $t; echo; done
echo "########## C. 1-node control, 8 ranks, NVLink only (3 reps)"
for t in 1 128; do python3 -I $T/dec1n.py $L $t; echo; done
echo "########## D. SM sweep at 128 tok (batch 3, 2 reps)"
python3 -I $T/sm.py $L; echo
echo "########## E. channels-per-SM cap at 12 SM, 128 tok (EP_MAX_CHANNELS_PER_SM, 2 reps)"
python3 -I $T/ch.py $L; echo
echo "########## F. combine RDMA batch size, main+#8+#9 8 SM (EP_COMBINE_BATCH_SIZE, 2 reps)"
python3 -I $T/gen.py $L pr89cb 8 128 cb12,cb8,cb6,cb4,cb3,cb2,cb1; echo
python3 -I $T/gen.py $L pr89cb 8 8192 cb12,cb4; echo
echo "########## G. scale-up local-reduce unroll cap, main+#8+#9 8 SM (EP_COMBINE_REDUCE_UNROLL, 2 reps)"
python3 -I $T/gen.py $L pr89ru 8 128 ru4,ru7,ru14; echo
echo "########## H. epilogue kernels per config (kineto, us)"
python3 -I $T/epi.py $L main_1N_12sm_128tok_default_nodbg_gin5_ovlp0 main_2N_12sm_128tok_default_nodbg_gin5_ovlp0 pr89_2N_12sm_128tok_default_nodbg_gin5_ovlp0 main_2N_8sm_128tok_lowsm_nodbg_gin5_ovlp0 pr89_2N_8sm_128tok_lowsm_nodbg_gin5_ovlp0 pr89_2N_12sm_128tok_ch3_nodbg_gin5_ovlp0 main_2N_12sm_1tok_default_nodbg_gin5_ovlp0
echo
echo "########## I. phase timeline, main+#8+#9 128 tok (EP_TIMELINE images; us since block entry)"
for c in pr89tl_2N_8sm_128tok_tl_ pr89tl_2N_12sm_128tok_tl_ pr89tl_2N_12sm_128tok_tlch2_; do
  echo "--- $c"; python3 -I timeline/parse_tl.py $TL/${c}*.log; echo; done
echo "########## J. per-token scale-up cost (accumulators; DB = double buffering, RU = reduce unroll)"
for c in pr89tl2_2N_8sm_128tok_tl2db0_ pr89tl2_2N_8sm_128tok_tl2db1_; do echo "--- $c"; python3 -I timeline/parse_acc.py 1.92 $TL/${c}*.log; done
for u in 4 7 14; do echo "--- unroll $u"; python3 -I timeline/parse_acc.py 1.92 $L/pr89tlru_2N_8sm_128tok_tlru${u}_*.log; done
echo
echo "########## K. dispatch -> copy epilogue alignment (same GPU clock)"
for s in 12 4; do echo "--- $s SM"; python3 -I timeline/parse_copy.py $TL/pr89tl3_2N_${s}sm_128tok_tl3_*.log; done
} > tables.txt
echo "wrote tables.txt ($(wc -l < tables.txt) lines)"
