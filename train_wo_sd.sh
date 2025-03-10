# export OPENCV_IO_ENABLE_OPENEXR=1
# export NPROC_PER_NODE=1
# export WORLD_SIZE=1
# export RANK=0

mkdir -p logs

torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$RANK \
         train_wo_sd.py --batch_size 2 --dataset oo3d9dmulti --data_path /home/q672126/project/ov9d/ov9d --data_name oo3d9dmulti --data_train train_multi --data_val test_multi --data_3d_feat ov9d_dataset_test_3d_feature\
         --num_filters 32 32 32 --deconv_kernels 2 2 2\
         --lr 1e-4 --log_dir logs \
         --scale_size 480 --epochs 25 --auto_resume --dino --dino_type large --attn_depth 4 --rot_dim 6