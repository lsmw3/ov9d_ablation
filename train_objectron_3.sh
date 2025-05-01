python   train_3.py --batch_size 16 --dataset objectron_3 --data_path /workspace/di38wiq/datasets/Omninocs/Objectron --data_name objectron --data_train train --data_val test --data_3d_feat ov9d_dataset_test_3d_feature\
         --num_filters 32 32 32 32 --deconv_kernels 2 2 2 \
         --lr 1e-3 --min_lr 1e-4 --log_dir logs --embed_dim 256 \
         --epochs 1000 --auto_resume --dino --dino_type small --attn_depth 12 --rot_dim 6 --num_gpus 2 --decode_rt \
         # --ckpt_path logs/objectron/epoch=4.ckpt