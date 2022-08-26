import torch
from torch.nn.utils.rnn import pad_sequence
from mmn.structures import TLGBatch


class BatchCollator(object):
    """
    Collect batch for dataloader
    """

    def __init__(self, ):
        pass

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        # [xxx, xxx, xxx], [xxx, xxx, xxx] ......
        feats_mae, feats_swin, feats_i3d, queries, wordlens, ious2d, moments, num_sentence, idxs = transposed_batch
        # print(feats_c3d.size())
        return TLGBatch(
            feats_mae=torch.stack(feats_mae).float(),
            feats_swin=torch.stack(feats_swin).float(),
            feats_i3d=torch.stack(feats_i3d).float(),
            queries=queries,
            wordlens=wordlens,
            all_iou2d=ious2d,
            moments=moments,
            num_sentence=num_sentence,
        ), idxs
