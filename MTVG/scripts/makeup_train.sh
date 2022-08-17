# find all configs in configs/
config=pool_makeup_i3d
# set your gpu id
gpus=0
# number of gpus
gpun=1
# please modify it with different value (e.g., 127.0.0.2, 29502) when you run multi mmn task on the same machine
master_addr=127.0.0.1
master_port=8211

# ------------------------ need not change -----------------------------------
config_file=/apdcephfs/private_chewu/Youmakeup_Challenge2022/MTVG/configs/$config\.yaml
output_dir=/apdcephfs/share_1351585/Youmakeup_Challenge2022/MTVG/outputs/${config}_20220809

CUDA_VISIBLE_DEVICES=$gpus python3 -m torch.distributed.launch \
--nproc_per_node=$gpun --master_addr $master_addr --master_port $master_port train_net.py --config-file $config_file OUTPUT_DIR $output_dir \