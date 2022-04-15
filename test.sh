export OMP_NUM_THREADS=1

python -u main.py \
  --run_mode train \
  --tarcomm \
  --hard_attn \
  --env_name grf \
  --scenarios 2_vs_1_with_keeper 3_vs_2_with_keeper \
  --num_controlled_agents 2 3 \
  --max_num_lplayers 5 \
  --max_num_rplayers 4 \
  --reward_type scoring \
  --nprocesses 2 \
  --num_epochs 500 \
  --epoch_size 10 \
  --batch_size 1000 \
  --hid_size 128 \
  --detach_gap 10 \
  --lrate 0.001 \
  --value_coeff 0.01 \
  --max_steps 80 \
  --recurrent \
  --seed 0 \
  | tee test.log
