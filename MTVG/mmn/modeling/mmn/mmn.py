import torch
from torch import nn
from torch.functional import F
from .featpool import build_featpool, build_featpool_c3d, build_featpool_i3d  # downsample 1d temporal features to desired length
from .feat2d import build_feat2d  # use MaxPool1d/Conv1d to generate 2d proposal-level feature map from 1d temporal features
from .loss import build_contrastive_loss
from .loss import build_bce_loss
from .text_encoder import build_text_encoder
from .proposal_conv import build_proposal_conv


class MMN(nn.Module):
    def __init__(self, cfg):
        super(MMN, self).__init__()
        self.only_iou_loss_epoch = cfg.SOLVER.ONLY_IOU
        self.featpool = build_featpool(cfg) 
        self.featpool_c3d = build_featpool_c3d(cfg)
        self.featpool_i3d = build_featpool_i3d(cfg)
        self.feat2d = build_feat2d(cfg)
        self.contrastive_loss = build_contrastive_loss(cfg, self.feat2d.mask2d)
        self.iou_score_loss = build_bce_loss(cfg, self.feat2d.mask2d)
        self.text_encoder = build_text_encoder(cfg)
        self.proposal_conv = build_proposal_conv(cfg, self.feat2d.mask2d)
        self.joint_space_size = cfg.MODEL.MMN.JOINT_SPACE_SIZE
        self.encoder_name = cfg.MODEL.MMN.TEXT_ENCODER.NAME

        W = torch.zeros(3,)
        self.ELMo_W = nn.Parameter(W)
        Gamma = torch.ones(1,)
        self.ELMo_Gamma = nn.Parameter(Gamma)

    def forward(self, batches, cur_epoch=1):
        """
        Arguments:
            batches.all_iou2d: list(B) num_sent x T x T
            feat2ds: B x C x T x T
            sent_feats: list(B) num_sent x C
        """
        # backbone
        ious2d = batches.all_iou2d
        assert len(ious2d) == batches.feats.size(0)
        for idx, (iou, sent) in enumerate(zip(ious2d, batches.queries)):
            assert iou.size(0) == sent.size(0)
            assert iou.size(0) == batches.num_sentence[idx]
        # print(batches.feats.size(),batches.feats_c3d.size())
        feats = self.featpool(batches.feats)  # from pre_num_clip to num_clip with overlapped average pooling, e.g., 256 -> 128
        feats_c3d = self.featpool_c3d(batches.feats_c3d)
        feats_i3d = self.featpool_i3d(batches.feats_i3d)

        # print(feats.size(), feats_c3d.size())
        # feats = torch.cat((feats,feats_c3d), 1)
        fusion_feats = [feats, feats_c3d, feats_i3d]
        normed_weights = torch.chunk(
            F.softmax(self.ELMo_W + 1.0 / 3, dim=-1), 3)
        pieces = []

        for w, t in zip(normed_weights, fusion_feats):
            pieces.append(w * t)

        sum_pieces = sum(pieces)
        feats=sum_pieces * self.ELMo_Gamma
        # print(feats.size())
        map2d = self.feat2d(feats)  # use MaxPool1d to generate 2d proposal-level feature map from 1d temporal features
        map2d, map2d_iou = self.proposal_conv(map2d)
        sent_feat, sent_feat_iou = self.text_encoder(batches.queries, batches.wordlens)

        # inference
        contrastive_scores = []
        iou_scores = []
        _, T, _ = map2d[0].size()
        for i, sf_iou in enumerate(sent_feat_iou):  # sent_feat_iou: [num_sent x C] (len=B)
            # iou part
            vid_feat_iou = map2d_iou[i]  # C x T x T
            vid_feat_iou_norm = F.normalize(vid_feat_iou, dim=0)
            sf_iou_norm = F.normalize(sf_iou, dim=1)
            iou_score = torch.mm(sf_iou_norm, vid_feat_iou_norm.reshape(vid_feat_iou_norm.size(0), -1)).reshape(-1, T, T)  # num_sent x T x T
            iou_scores.append((iou_score*10).sigmoid() * self.feat2d.mask2d)

        # loss
        if self.training:
            loss_iou = self.iou_score_loss(torch.cat(iou_scores, dim=0), torch.cat(ious2d, dim=0), cur_epoch)
            loss_vid, loss_sent = self.contrastive_loss(map2d, sent_feat, ious2d, batches.moments)
            return loss_vid, loss_sent, loss_iou
        else:
            for i, sf in enumerate(sent_feat):
                # contrastive part
                vid_feat = map2d[i, ...]  # C x T x T
                vid_feat_norm = F.normalize(vid_feat, dim=0)
                sf_norm = F.normalize(sf, dim=1)  # num_sent x C
                _, T, _ = vid_feat.size()
                contrastive_score = torch.mm(sf_norm, vid_feat_norm.reshape(vid_feat.size(0), -1)).reshape(-1, T, T) * self.feat2d.mask2d  # num_sent x T x T
                contrastive_scores.append(contrastive_score)
            return map2d_iou, sent_feat_iou, contrastive_scores, iou_scores  # first two maps for visualization
