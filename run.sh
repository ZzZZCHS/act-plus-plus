export MUJOCO_GL=egl
export WANDB_MODE=offline

task_name=sim_transfer_cube_scripted
tag=kl100
data_dir=dataset/$task_name
ckpt_dir=checkpoints/${task_name}_${tag}

# generate 50 episodes of scripted data
# srun --partition=mozi-S1 --gres=gpu:0 python3 record_sim_episodes.py --task_name $task_name --dataset_dir $data_dir --num_episodes 50

# visualize the simulated episodes after it is collected
# srun --partition=mozi-S1 --gres=gpu:0 python3 visualize_episodes.py --dataset_dir $data_dir --episode_idx 1


# train ACT
srun --partition=mozi-S1 --gres=gpu:0 \
    python3 imitate_episodes.py \
    --task_name $task_name \
    --ckpt_dir $ckpt_dir \
    --policy_class ACT \
    --kl_weight 100 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --batch_size 8 \
    --dim_feedforward 3200 \
    --lr 1e-5 \
    --seed 0 \
    --num_steps 10000
