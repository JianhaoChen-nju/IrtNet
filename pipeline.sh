#!/bin/bash
set -e

MODEL_EMBED_DIM=232
NUM_EXPERTS=39 
TOP_K_EXPERTS=39
EXPERT_HIDDEN_DIM=512
SHARED_EXPERT_HIDDEN_DIM=512
EXPERT_OUTPUT_DIM=256
EPOCHS=30
BATCH_SIZE=2048
LR=1e-4
WEIGHT_DECAY=1e-4
DROPOUT_RATE=0.5
EMBEDDING_NOISE=0.05      
EARLY_STOPPING_PATIENCE=5
BIAS_UPDATE_SPEED=0.01


TRAIN_DATA_PATH="../data/train.csv"
VAL_DATA_PATH="../data/val.csv"
TEST_DATA_PATH="../data/test.csv"
PROMPT_EMBED_PATH="../data/question_embeddings.pth"
MODEL_SAVE_PATH="../models/best_irt.pth"

echo "==================================================================="
echo "     Starting Training: IRT Model                  "
echo "     (Comprehensive Evaluation Mode)                             "
echo "==================================================================="
echo


python train_and_eval.py \
    --train_data_path ${TRAIN_DATA_PATH} \
    --val_data_path ${VAL_DATA_PATH} \
    --test_data_path ${TEST_DATA_PATH} \
    --prompt_embedding_path ${PROMPT_EMBED_PATH} \
    --model_save_path ${MODEL_SAVE_PATH} \
    --model_embed_dim ${MODEL_EMBED_DIM} \
    --num_experts ${NUM_EXPERTS} \
    --top_k_experts ${TOP_K_EXPERTS} \
    --bias_update_speed ${BIAS_UPDATE_SPEED} \
    --expert_hidden_dim ${EXPERT_HIDDEN_DIM} \
    --shared_expert_hidden_dim ${SHARED_EXPERT_HIDDEN_DIM} \
    --expert_output_dim ${EXPERT_OUTPUT_DIM} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --dropout_rate ${DROPOUT_RATE} \
    --embedding_noise ${EMBEDDING_NOISE} \
    --early_stopping_patience ${EARLY_STOPPING_PATIENCE}

echo
echo "=========================================================="
echo "  Pipeline Finished Successfully!                         "
echo "  All evaluations (Correctness & Routing) are complete.   "
echo "  Best IRT model saved to: ${MODEL_SAVE_PATH} "
echo "=========================================================="