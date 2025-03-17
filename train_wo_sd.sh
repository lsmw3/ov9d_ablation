# export OPENCV_IO_ENABLE_OPENEXR=1
# export NPROC_PER_NODE=1
# export WORLD_SIZE=1
# export RANK=0

mkdir -p logs

python   train_wo_sd.py --batch_size 64 --dataset oo3d9dmulti --data_path ov9d --data_name oo3d9dmulti --data_train train_multi --data_val test_multi --data_3d_feat ov9d_dataset_test_3d_feature\
         --raw_w 640 --raw_h 480 --num_filters 32 32 32 32 --deconv_kernels 2 2 2\
         --lr 5 e-4 --log_dir logs --nocs_type L1 --nocs_bin 64\
         --scale_size 490 --epochs 1000 --auto_resume --dino --dino_type small --attn_depth 5 --rot_dim 6