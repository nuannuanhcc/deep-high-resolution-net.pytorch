python tools/eva.py \
    --cfg experiments/coco/hrnet/w48_256x128_adam_lr1e-3.yaml \
    --extract_data_path /data/hanchuchu/datasets/market1501/bounding_box_train \
    --keypoint_save_path /home/yckj2206/code/deep-high-resolution-net.pytorch/256x128_train_coors.pkl \
    TEST.MODEL_FILE models/pytorch/pose_coco/pose_hrnet_w48_256x128.pth

# 384x288
# 256x128
# 384x128
# path = '/data/hanchuchu/datasets/market1501/bounding_box_train'
# path = '/data/hanchuchu/datasets/Partial_iLIDS/Probe'
# path = '/data/hanchuchu/datasets/PartialREID/partial_body_images'