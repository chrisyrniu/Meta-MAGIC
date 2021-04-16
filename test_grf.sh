#!/bin/bash
export OMP_NUM_THREADS=1

python -u test.py \
  --env_name grf \
  --scenario academy_3_vs_1_with_keeper \
  --num_controlled_agents 3 \
  --max_num_lplayers 4 \
  --max_num_rplayers 3 \
  --reward_type scoring \
  --nprocesses 1 \
  --num_epochs 10 \
  --epoch_size 50 \
  --batch_size 10 \
  --hid_size 128 \
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
  --message_encoder \
  --message_decoder \
  --recurrent \
  --run_num 1 \
  --ep_num 290 \
  | tee test_grf.log
  
# Should revise according to the tested trained model
