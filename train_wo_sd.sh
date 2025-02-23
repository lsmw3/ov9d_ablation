export OPENCV_IO_ENABLE_OPENEXR=1
export NPROC_PER_NODE=1
export WORLD_SIZE=1
export RANK=0

mkdir -p logs

torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$RANK \
         train_wo_sd.py --batch_size 16 --dataset oo3d9dsingle --data_path /home/q672126/project/ov9d/ov9d --data_name oo3d9dsingle --data_train train --data_val test/all \
         --num_filters 32 32 32 --deconv_kernels 2 2 2\
         --layer_decay 0.9 --log_dir logs \
         --scale_size 480 --epochs 25 --auto_resume --dino 