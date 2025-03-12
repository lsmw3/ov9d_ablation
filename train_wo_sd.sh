# export OPENCV_IO_ENABLE_OPENEXR=1
# export NPROC_PER_NODE=1
# export WORLD_SIZE=1
# export RANK=0

mkdir -p logs

python   train_wo_sd.py --batch_size 3 --dataset oo3d9dmulti --data_path /home/q672126/project/ov9d/ov9d --data_name oo3d9dmulti --data_train train_multi --data_val test_multi --data_3d_feat ov9d_dataset_test_3d_feature\
         --num_filters 32 32 32 --deconv_kernels 2 2 2\
         --lr 5e-4 --log_dir logs --nocs_type CE --nocs_bin 64\
         --scale_size 480 --epochs 1000 --auto_resume --dino --dino_type small --attn_depth 4 --rot_dim 6