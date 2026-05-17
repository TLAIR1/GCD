#!/bin/bash
# scripts/graph_coop/main.sh
#
# PLACEMENT: Save as  CoOp/scripts/graph_coop/main.sh
# Make executable:    chmod +x scripts/graph_coop/main.sh
#
# Runs Graph-CoOp for all shot counts and seeds on a single dataset,
# mirroring the protocol in the CoOp paper (Table 3 of Graph-CoOp paper).
#
# Usage:
#   bash scripts/graph_coop/main.sh caltech101 end 16 False
#
# Arguments:
#   $1  DATASET       e.g. caltech101 | oxford_pets | stanford_cars |
#                          oxford_flowers | food101 | fgvc_aircraft |
#                          sun397 | dtd | eurosat | ucf101 | imagenet
#   $2  CTX_POSITION  end | middle
#   $3  N_CTX         number of context tokens (16 recommended)
#   $4  CSC           True | False  (class-specific context)
#
# Ablation flags (edit here to change the configuration):
USE_GLS=True
USE_LAP=True
USE_GCN=True
USE_GCD=False        # set True to test Modification 4
ALPHA=0.1
LAMBDA_LAP=0.01
THRESHOLD=0.3
N_PROTO=8
GAMMA_DIFF=1.0
PREC=fp16            # fp16 | fp32 | amp

# ── argument parsing ──────────────────────────────────────────────────────
DATASET=$1
CTX_POSITION=${2:-end}
N_CTX=${3:-16}
CSC=${4:-False}

if [ -z "$DATASET" ]; then
    echo "Usage: bash scripts/graph_coop/main.sh <dataset> [ctx_pos] [n_ctx] [csc]"
    exit 1
fi

# ── path setup ────────────────────────────────────────────────────────────
DATA="${DATA:-/path/to/your/datasets}"   # override with: export DATA=/your/path
OUTPUT_DIR="output/${DATASET}/GraphCoOp/rn50_${N_CTX}shots"
CFG="configs/trainers/GraphCoOp/rn50_ep200.yaml"
DATASET_CFG="configs/datasets/${DATASET}.yaml"

# ── sweep ─────────────────────────────────────────────────────────────────
for SHOTS in 1 2 4 8 16; do
    for SEED in 1 2 3; do
        # Each shot-count AND seed gets its own directory — never overwrite
        DIR="${OUTPUT_DIR}/shots${SHOTS}/seed${SEED}"
        echo "===== Dataset: ${DATASET} | Shots: ${SHOTS} | Seed: ${SEED} ====="

        python train.py \
            --root "${DATA}" \
            --trainer GraphCoOp \
            --dataset-config-file "${DATASET_CFG}" \
            --config-file "${CFG}" \
            --output-dir "${DIR}" \
            DATASET.NUM_SHOTS "${SHOTS}" \
            SEED "${SEED}" \
            TRAINER.GRAPHCOOP.N_CTX "${N_CTX}" \
            TRAINER.GRAPHCOOP.CLASS_TOKEN_POSITION "${CTX_POSITION}" \
            TRAINER.GRAPHCOOP.CSC "${CSC}" \
            TRAINER.GRAPHCOOP.PREC "${PREC}" \
            TRAINER.GRAPHCOOP.USE_GLS "${USE_GLS}" \
            TRAINER.GRAPHCOOP.USE_LAP "${USE_LAP}" \
            TRAINER.GRAPHCOOP.USE_GCN "${USE_GCN}" \
            TRAINER.GRAPHCOOP.USE_GCD "${USE_GCD}" \
            TRAINER.GRAPHCOOP.ALPHA "${ALPHA}" \
            TRAINER.GRAPHCOOP.LAMBDA_LAP "${LAMBDA_LAP}" \
            TRAINER.GRAPHCOOP.THRESHOLD "${THRESHOLD}" \
            TRAINER.GRAPHCOOP.N_PROTO "${N_PROTO}" \
            TRAINER.GRAPHCOOP.GAMMA_DIFF "${GAMMA_DIFF}"
    done
done

echo "Done. Results saved to ${OUTPUT_DIR}/"
