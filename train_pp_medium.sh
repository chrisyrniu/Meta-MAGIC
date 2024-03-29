export OMP_NUM_THREADS=1

python -u main.py \
  --run_mode train \
  --env_name predator_prey \
  --num_controlled_agents 4 5 6 \
  --dim 10 \
  --max_steps 40 \
  --vision 1 \
  --nprocesses 16 \
  --num_epochs 500 \
  --save_epochs 150 \
  --epoch_size 15 \
  --batch_size 1000 \
  --hid_size 128 \
  --value_coeff 0.015 \
  --detach_gap 1000 \
  --lrate 0.001 \
  --gnn_type gat \
  --directed \
  --gat_num_heads 4 \
  --gat_hid_size 32 \
  --gat_num_heads_out 1 \
  --ge_num_heads 8 \
  --use_gconv_encoder \
  --gconv_encoder_out_size 32 \
  --self_loop_type1 2 \
  --self_loop_type2 2 \
  --learn_second_graph \
  --first_gat_normalize \
  --second_gat_normalize \
  --recurrent \
  --save \
  --seed 0 \
  --plot \
  --plot_env meta_magic_pp_medium_4_5_6_seed0 \
  --plot_port 8097 \
  | tee train_pp_medium.log


#   --first_gat_normalize \
#   --second_gat_normalize \

#   --plot \
#   --plot_env pp_medium_gcomm_gat_hid_128_seed0_run11 \
#   --plot_port 8097 \

#   --save \
#   --save_every 200 \

  ## easy
  # --nagents 3 \
  # --dim 5 \
  # --max_steps 20 \
  # --vision 0 \

  ## medium
  # --nagents 5 \
  # --dim 10 \
  # --max_steps 40 \
  # --vision 1 \

  ## hard
  # --nagents 10 \
  # --dim 20 \
  # --max_steps 80 \
  # --vision 1 \
