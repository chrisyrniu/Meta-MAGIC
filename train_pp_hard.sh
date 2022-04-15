export OMP_NUM_THREADS=1

python -u main.py \
  --run_mode train \
  --tarcomm \
  --hard_attn \
  --env_name predator_prey \
  --num_controlled_agents 8 10 12 \
  --dim 20 \
  --max_steps 80 \
  --vision 1 \
  --nprocesses 16 \
  --num_epochs 1000 \
  --save_epochs 300 \
  --epoch_size 15 \
  --batch_size 1000 \
  --hid_size 128 \
  --value_coeff 0.01 \
  --detach_gap 10 \
  --lrate 0.001 \
  --recurrent \
  --save \
  --seed 0 \
  --plot \
  --plot_env tar_ic3net_pp_hard_8_10_12_seed0 \
  --plot_port 8097 \
  | tee train_pp_hard.log


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
