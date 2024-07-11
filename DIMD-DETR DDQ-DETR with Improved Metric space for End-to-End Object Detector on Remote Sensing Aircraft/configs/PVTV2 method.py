_base_ = './ddq-detr-5scale_r50_8xb2-12e_coco.py'

# b3
# # model = dict(
#     # backbone=dict(
#     #     embed_dims=128,
#     #     num_layers=[3, 4, 18, 3],
#     #     init_cfg=dict(checkpoint='https://github.com/whai362/PVT/'
#     #                   'releases/download/v2/pvt_v2_b3.pth')),
#     # neck=dict(in_channels=[64, 128, 320, 512]))
#  b4
# #    model = dict(
# #     backbone=dict(
# #         embed_dims=64,
# #         num_layers=[3, 8, 27, 3],
# #         init_cfg=dict(checkpoint='https://github.com/whai362/PVT/'
# #                       'releases/download/v2/pvt_v2_b4.pth')),
# #     neck=dict(in_channels=[64, 128, 320, 512]))
# b5   
# # model = dict(
# #     backbone=dict(
# #         embed_dims=64,
# #         num_layers=[3, 6, 40, 3],
# #         mlp_ratios=(4, 4, 4, 4),
# #         init_cfg=dict(checkpoint='https://github.com/whai362/PVT/'
# #                       'releases/download/v2/pvt_v2_b5.pth')),
# #     neck=dict(in_channels=[64, 128, 320, 512]))

# b1
#     # backbone=dict(
#     #     _delete_=True,
#     #     type='PyramidVisionTransformerV2',
#     #     embed_dims=64,
#     #     num_layers=[2, 2, 2, 2],
#     #     init_cfg=dict(checkpoint='https://github.com/whai362/PVT/'
#     #                   'releases/download/v2/pvt_v2_b1.pth')),
#     # neck=dict(in_channels=[64, 128, 320, 512]))


# b2
# # model = dict(
# #     backbone=dict(
# #         embed_dims=64,
# #         num_layers=[3, 4, 6, 3],
# #         init_cfg=dict(checkpoint='https://github.com/whai362/PVT/'
# #                       'releases/download/v2/pvt_v2_b2.pth')),
# #     neck=dict(in_channels=[64, 128, 320, 512]))
# # # 








 




# model = dict(
#     type='DDQDETR',
#     num_queries=900,  # num_matching_queries
#     # ratio of num_dense queries to num_queries
#     dense_topk_ratio=1.5,
#     with_box_refine=True,
#     as_two_stage=True,
#     data_preprocessor=dict(
#         type='DetDataPreprocessor',
#         mean=[123.675, 116.28, 103.53],
#         std=[58.395, 57.12, 57.375],
#         bgr_to_rgb=True,
#         pad_size_divisor=1),
#         backbone=dict(
#         _delete_=True,
#         type='PyramidVisionTransformerV2',
#         embed_dims=32,
#         num_layers=[2, 2, 2, 2],
#         init_cfg=dict(checkpoint='/home/userroot/LH/2024/mmdetection-main/pvt_v2_b0.pth')),
#         neck=dict(
#         type='ChannelMapper',
#         in_channels=[32, 64, 160, 256],
#         # in_channels=[64, 128, 320],
#         kernel_size=1,
#         out_channels=256,
#         act_cfg=None,
#         norm_cfg=dict(type='GN', num_groups=32),
#         num_outs=4))

