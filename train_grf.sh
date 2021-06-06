#!/bin/bash
export OMP_NUM_THREADS=1

python -u main.py \
  --training_mode meta-train \
  --env_name grf \
  --scenarios 3_vs_2_with_keeper \
  --num_controlled_agents 3 \
  --max_num_lplayers 5 \
  --max_num_rplayers 4 \
  --reward_type scoring \
  --nprocesses 16 \
  --num_epochs 500 \
  --epoch_size 5 \
  --batch_size 1000 \
  --hid_size 128 \
  --detach_gap 10 \
  --lrate 0.001 \
  --value_coeff 0.01 \
  --max_steps 80 \
  --gnn_type gat \
  --directed \
  --gat_num_heads 1 \
  --gat_hid_size 128 \
  --gat_num_heads_out 1 \
  --ge_num_heads 8 \
  --gat_dropout 0.2 \
  --use_gconv_encoder \
  --gconv_encoder_out_size 32 \
  --self_loop_type1 2 \
  --self_loop_type2 2 \
  --first_gat_normalize \
  --second_gat_normalize \
  --message_encoder \
  --message_decoder \
  --recurrent \
  --save \
  --seed 5678 \
  --plot \
  --plot_env multi_task_vanilla_magic_grf_33_max_44_dropout_0.2_seed_5678 \
  --plot_port 8097 \
  | tee train_grf.log
