export OMP_NUM_THREADS=1

python -u test.py \
  --env_name grf \
  --scenarios 4_vs_2_with_keeper \
  --num_controlled_agents 4 \
  --max_num_lplayers 5 \
  --max_num_rplayers 4 \
  --reward_type scoring \
  --nprocesses 16 \
  --num_epochs 10 \
  --test_episode_num 10 \
  --epoch_size 10 \
  --batch_size 1000 \
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
  --seed 5 \
  --load model.pt \
  | tee test.log
