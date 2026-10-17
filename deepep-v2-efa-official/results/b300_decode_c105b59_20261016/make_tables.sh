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
echo
echo "########## L. start rail barrier skipped (UNSAFE experiment, EP_SKIP_START_RAIL_BARRIER, 3 reps)"
python3 -I $T/gen.py $L pr89nb 8 128 nb0,nb1 3; echo
python3 -I $T/gen.py $L pr89nb 8 8192 nb0,nb1 3; echo
for k in tlnb0 tlnb1; do echo "--- timeline $k"; python3 -I timeline/parse_tl.py $L/pr89tlnb_2N_8sm_128tok_${k}_*.log | awk '/^== (dispatch|reduced combine) /{p=1} /^== (cached|expanded|combine) /{p=0} p'; done
echo
echo "########## M. SAFE version: skip + EP_COMBINE_SIGNAL_OFFSET=8 + EP_DISPATCH_SCALEOUT_BASE=38707200 (3 reps)"
echo "   (BASE is the 128-tok layout's combine end; at 8192 tok it overlaps -> illegal address, so no 8192 safe row)"
python3 -I $T/gen.py $L sepf 8 128 base,safe,saferu7 3; echo
python3 -I $T/gen.py $L sepf 8 8192 base 3; echo
echo "########## N. back-to-back stress (stress/ep_determinism.py, 128 tok, 8 SM, no barrier between the 4 ops of a step)"
echo "   nb0 = barrier kept | nb1 = skipped | safe = skipped + signal split | sep, sep3 = skipped + signal split + region separation"
python3 -I $T/stress_summary.py stress nb0 nb1 safe sep sep3
echo
echo "########## O. buffer layouts at 128 tok (rank 0; region ends in bytes from buffer start)"
for f in layout/*.log; do echo "--- $(basename $f | cut -d_ -f1) (safe = default dispatch placement, sep = EP_DISPATCH_SCALEOUT_BASE)"; grep -ho 'EPLAYOUT [a-z][^E]*' $f | sed 's/ *$//' | sort -u; done
} > tables.txt
echo "wrote tables.txt ($(wc -l < tables.txt) lines)"
