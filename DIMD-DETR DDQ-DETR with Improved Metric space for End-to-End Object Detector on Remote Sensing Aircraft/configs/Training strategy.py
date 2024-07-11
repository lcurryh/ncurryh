_base_ = './ddq-detr-4scale_r50_8xb2-12e_coco.py'

max_epochs = 200
stage2_num_epochs = 10
base_lr = 0.0002
interval = 10



train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=10)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 100 to 200 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]





     
