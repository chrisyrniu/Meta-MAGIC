#!/bin/bash
export OMP_NUM_THREADS=1

python -u main.py \
  --mode meta-train \
  --env_name grf \
  --scenarios academy_3_vs_1_with_keeper 3_vs_2_with_keeper 2_vs_1_with_keeper \
  --num_controlled_agents 3 3 2 \
  --max_num_lplayers 5 \
  --max_num_rplayers 4 \
  --reward_type scoring \
  --nprocesses 16 \
  --num_epochs 500 \
  --epoch_size 15 \
  --batch_size 1000 \
  --hid_size 128 \
  --detach_gap 100000 \
  --lrate 0.001 \
  --value_coeff 0.01 \
  --max_steps 80 \
  --gnn_type gat \
  --directed \
  --gat_num_heads 1 \
  --gat_hid_size 128 \
  --gat_num_heads_out 1 \
  --ge_num_heads 8 \
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
  --save_every 50 \
  --seed 700 \
  --plot \
  --plot_env meta_magic_3v2_3v3_2v2 \
  --plot_port 8097 \
  | tee train_grf1.log
