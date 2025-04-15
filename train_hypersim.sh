# export OPENCV_IO_ENABLE_OPENEXR=1
# export NPROC_PER_NODE=1
# export WORLD_SIZE=1
# export RANK=0

mkdir -p logs

python   train_wo_sd.py --batch_size 16 --dataset hypersim --data_path /workspace/di38wiq/datasets/Omninocs/Hypersim --data_name hypersim --data_train train --data_val test --data_3d_feat ov9d_dataset_test_3d_feature\
         --num_filters 32 32 32 32 --deconv_kernels 2 2 2\
         --lr 1e-4 --min_lr 1e-4 --log_dir logs --nocs_bin 64\
         --scale_size 490 --epochs 1000 --auto_resume --dino --dino_type small --attn_depth 12 --rot_dim 6 --num_gpus 2 --with_attn