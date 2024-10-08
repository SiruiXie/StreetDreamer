DEVICES=0,1
PORT=2345
DATASET_NAME=./data/processed_waymo
BATCH_SIZE=8
TRAINABLE_PARAMS=controlnet+unet_decoder
OUTPUT_PATH=../finetuned_checkpoint_output
CHECKPOINT_STEPS=100
CHECKPOINTS_TOTAL_LIMIT=5
GRADIANT_ACCUMULSTION_STEPS=1
REPORT_TO=wandb

cd ../utils

CUDA_VISIBLE_DEVICES=$DEVICES accelerate launch \
    --main_process_port=$PORT \
    --multi_gpu \
    --num_processes=2 \
    tune_controlnet.py \
    --dataset_name $DATASET_NAME \
    --train_batch_size $BATCH_SIZE \
    --parameters_to_optimize $TRAINABLE_PARAMS \
    --output_dir $OUTPUT_PATH \
    --checkpointing_steps $CHECKPOINT_STEPS \
    --checkpoints_total_limit $CHECKPOINTS_TOTAL_LIMIT \
    --gradient_accumulation_steps $GRADIANT_ACCUMULSTION_STEPS \
    --report_to $REPORT_TO \
    --training_mode baseline

cd ../scripts
