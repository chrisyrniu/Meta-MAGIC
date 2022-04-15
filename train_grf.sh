export OMP_NUM_THREADS=1

python -u main.py \
  --run_mode train \
  --tarcomm \
  --hard_attn \
  --env_name grf \
  --scenarios 2_vs_1_with_keeper academy_3_vs_1_with_keeper 3_vs_2_with_keeper \
  --num_controlled_agents 2 3 3 \
  --max_num_lplayers 5 \
  --max_num_rplayers 4 \
  --reward_type scoring \
  --nprocesses 16 \
  --num_epochs 500 \
  --epoch_size 15 \
  --batch_size 1000 \
  --hid_size 128 \
  --detach_gap 10 \
  --lrate 0.001 \
  --value_coeff 0.01 \
  --max_steps 80 \
  --recurrent \
  --save \
  --save_epochs 100 \
  --seed 0 \
  --plot \
  --plot_env tar_ic3net_grf_22_32_33_max_44_seed_0 \
  --plot_port 8097 \
  | tee train_grf.log
