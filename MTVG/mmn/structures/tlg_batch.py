from dataclasses import dataclass
import torch

# temporal localization grounding


@dataclass
class TLGBatch(object):
    # frames: list # [ImageList]
    feats_mae: torch.tensor
    feats_swin: torch.tensor
    feats_i3d: torch.tensor
    feats_c3d: torch.tensor
    queries: list
    wordlens: list
    all_iou2d: list
    moments: list
    num_sentence: list

    def to(self, device):
        self.feats_swin = self.feats_swin.to(device)
        self.feats_mae = self.feats_mae.to(device)
        self.feats_i3d = self.feats_i3d.to(device)
        self.feats_c3d = self.feats_i3d.to(device)
        self.queries = [query.to(device) for query in self.queries]
        self.wordlens = [word_len.to(device) for word_len in self.wordlens]
        self.all_iou2d = [iou2d.to(device) for iou2d in self.all_iou2d]
        self.moments = [moment.to(device) for moment in self.moments]

        return self
