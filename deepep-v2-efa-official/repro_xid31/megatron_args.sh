# Megatron-LM argument list for the repro, shared by run_megatron.sh (laptop driver) and
# run_node.sh (on-host launcher) so both launch byte-identical trainings.
# Inputs: N (nodes, 8 GPUs each), NUM_LAYERS, TRAIN_ITERS, OVERLAP (1|0), HERE (this dir).
# Output: MLM_ARGS. The MoE shape is shape.env (public placeholders) then shape.local.env (gitignored).
# shellcheck disable=SC1091
. "$HERE/shape.env"; [ ! -f "$HERE/shape.local.env" ] || . "$HERE/shape.local.env"
NDENSE=$NUM_DENSE_LAYERS; NMOE=$((NUM_LAYERS - NDENSE))
GBS=$((N * 8 * 2))   # DP = all GPUs (TP1/PP1), MBS 1, GA 2 -- the target GA
# OVERLAP=0 drops --overlap-moe-expert-parallel-comm (combined-1F1B EP A2A overlap): the A/B that
# separates "the overlap schedule / comm-stream lifetimes" from "DeepEP on its own".
OVERLAP="${OVERLAP:-1}"
OVERLAP_ARG=""; [ "$OVERLAP" = 0 ] || OVERLAP_ARG="--overlap-moe-expert-parallel-comm"

MLM_ARGS="--num-layers $NUM_LAYERS --hidden-size $HIDDEN --ffn-hidden-size $FFN_HIDDEN --num-attention-heads $NUM_HEADS \
--seq-length $SEQ_LEN --max-position-embeddings $SEQ_LEN --micro-batch-size 1 --global-batch-size $GBS \
--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --expert-model-parallel-size $((N * 8)) \
--moe-layer-freq '([0]*$NDENSE+[1]*$NMOE)' --num-experts $NUM_EXPERTS --moe-router-topk $TOPK --moe-ffn-hidden-size $MOE_FFN_HIDDEN \
--moe-shared-expert-intermediate-size $SHARED_EXPERT_FFN --moe-router-score-function sigmoid \
--moe-router-topk-scaling-factor $TOPK_SCALING --moe-router-enable-expert-bias --moe-router-bias-update-rate 0.001 \
--moe-router-load-balancing-type global_aux_loss --moe-aux-loss-coeff 0.0001 --moe-router-dtype fp32 \
--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend deepepv2 --moe-deepep-num-sms 28 \
--moe-grouped-gemm --moe-permute-fusion --moe-router-fusion $OVERLAP_ARG \
--mtp-num-layers 1 \
--swiglu --normalization RMSNorm --position-embedding-type rope --untie-embeddings-and-output-weights \
--disable-bias-linear --hidden-dropout 0.0 --attention-dropout 0.0 --transformer-impl transformer_engine \
--bf16 --no-gradient-accumulation-fusion --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
--lr 1e-4 --min-lr 1e-5 --lr-decay-style cosine --lr-warmup-iters 10 --weight-decay 0.1 --clip-grad 1.0 \
--train-iters $TRAIN_ITERS --mock-data --tokenizer-type NullTokenizer --vocab-size $VOCAB \
--log-interval 1 --eval-iters 0 --eval-interval 100000000 --seed 1234"

