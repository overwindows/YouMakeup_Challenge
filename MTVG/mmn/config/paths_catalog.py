"""Centralized catalog of paths."""
import os


class DatasetCatalog(object):
    DATA_DIR = ""

    DATASETS = {
        "makeup_train": {
            "ann_file": "./dataset/makeup/makeup_train.json",
            "feat_mae": "/youtu/xlab-team1/shuxiujun/contest/person_in_context/makeup_videomae_ssv2_rgb_stride_1s.hdf5",
            "feat_swin": "/youtu/xlab-team1/shuxiujun/contest/person_in_context/makeup_videoswin_rgb_stride_1s_k400.hdf5",
            "feat_i3d": "/youtu/xlab-team1/shuxiujun/contest/person_in_context/makeup_i3d_rgb_stride_1s.hdf5",
        },
        "makeup_val": {
            "ann_file": "./dataset/makeup/makeup_val.json",
            "feat_mae": "/youtu/xlab-team1/shuxiujun/contest/person_in_context/makeup_videomae_ssv2_rgb_stride_1s.hdf5",
            "feat_swin":  "/youtu/xlab-team1/shuxiujun/contest/person_in_context/makeup_videoswin_rgb_stride_1s_k400.hdf5",
            "feat_i3d": "/youtu/xlab-team1/shuxiujun/contest/person_in_context/makeup_i3d_rgb_stride_1s.hdf5",
        },
        "makeup_test": {
            "ann_file": "./dataset/makeup/makeup_test.json",
            "feat_mae": "/youtu/xlab-team1/shuxiujun/contest/person_in_context/makeup_videomae_ssv2_rgb_stride_1s.hdf5",
            "feat_swin":  "/youtu/xlab-team1/shuxiujun/contest/person_in_context/makeup_videoswin_rgb_stride_1s_k400.hdf5",
            "feat_i3d": "/youtu/xlab-team1/shuxiujun/contest/person_in_context/makeup_i3d_rgb_stride_1s.hdf5",
        }
    }

    @staticmethod
    def get(name):
        data_dir = DatasetCatalog.DATA_DIR
        attrs = DatasetCatalog.DATASETS[name]
        args = dict(
            feat_swin=os.path.join(data_dir, attrs["feat_swin"]),
            ann_file=os.path.join(data_dir, attrs["ann_file"]),
            feat_mae=os.path.join(data_dir, attrs["feat_mae"]),
            feat_i3d=os.path.join(data_dir, attrs["feat_i3d"]),
            # feat_c3d=os.path.join(data_dir, attrs["feat_c3d"]),
        )
        if "tacos" in name:
            return dict(
                factory="TACoSDataset",
                args=args,
            )
        elif "activitynet" in name:
            return dict(
                factory="ActivityNetDataset",
                args=args
            )
        elif "charades" in name:
            return dict(
                factory="CharadesDataset",
                args=args
            )
        elif "makeup" in name:
            return dict(
                factory="MakeupDataset",
                args=args
            )
        raise RuntimeError("Dataset not available: {}".format(name))
