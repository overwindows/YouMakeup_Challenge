import torch
from torch import nn
from torch.functional import F
# downsample 1d temporal features to desired length
from .featpool import build_featpool, build_featpool_c3d, build_featpool_i3d
# use MaxPool1d/Conv1d to generate 2d proposal-level feature map from 1d temporal features
from .feat2d import build_feat2d
from .loss import build_contrastive_loss
from .loss import build_bce_loss
from .text_encoder import build_text_encoder
from .proposal_conv import build_proposal_conv


class MMN(nn.Module):
    def __init__(self, cfg):
        super(MMN, self).__init__()
        self.only_iou_loss_epoch = cfg.SOLVER.ONLY_IOU
        self.featpool = build_featpool(cfg)
        self.featpool_swin = build_featpool_c3d(cfg)
        # self.featpool_c3d = build_featpool_c3d(cfg)
        self.featpool_i3d = build_featpool_i3d(cfg)
        self.feat2d = build_feat2d(cfg)
        self.contrastive_loss = build_contrastive_loss(cfg, self.feat2d.mask2d)
        self.iou_score_loss = build_bce_loss(cfg, self.feat2d.mask2d)
        self.text_encoder = build_text_encoder(cfg)
        self.proposal_conv = build_proposal_conv(cfg, self.feat2d.mask2d)
        self.joint_space_size = cfg.MODEL.MMN.JOINT_SPACE_SIZE
        self.encoder_name = cfg.MODEL.MMN.TEXT_ENCODER.NAME

        # Baseline
        # W = torch.zeros(4,)
        # self.ELMo_W = nn.Parameter(W)

        # MoE is determined by text input 2022/08/08
        # self.num_experts = 3
        # self.MoE = nn.Linear(256*1024, self.num_experts, bias=False)

        # Gamma = torch.ones(1,)
        # self.ELMo_Gamma = nn.Parameter(Gamma)

        self.linear = nn.Linear(512*3, 512)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, batches, cur_epoch=1):
        """
        Arguments:
            batches.all_iou2d: list(B) num_sent x T x T
            feat2ds: B x C x T x T
            sent_feats: list(B) num_sent x C
        """
        # backbone
        ious2d = batches.all_iou2d
        assert len(ious2d) == batches.feats_mae.size(0)
        for idx, (iou, sent) in enumerate(zip(ious2d, batches.queries)):
            assert iou.size(0) == sent.size(0)
            assert iou.size(0) == batches.num_sentence[idx]

        sent_feat, sent_feat_iou = self.text_encoder(
            batches.queries, batches.wordlens)

        # from pre_num_clip to num_clip with overlapped average pooling, e.g., 256 -> 128
        feats_mae = self.featpool(batches.feats_mae)
        feats_swin = self.featpool_swin(batches.feats_swin)
        feats_i3d = self.featpool_i3d(batches.feats_i3d)
        # feats_c3d = self.featpool_c3d(batches.feats_c3d)
        fusion_feats = torch.cat([feats_mae, feats_swin, feats_i3d], dim=1)
        fusion_feats = self.dropout(fusion_feats)
        feats = self.linear(fusion_feats.permute(0, 2, 1)).permute(0, 2, 1)
        # Baseline
        # normed_weights = torch.chunk(
        #     F.softmax(self.ELMo_W + 1.0 / 4, dim=-1), 4)

        # 2022/08/08
        # print(batches.feats_c3d.size())
        # for sf in sent_feat:
        #     doc_feats.append(torch.mean(sf, dim=0))
        # doc_feat = torch.stack(doc_feats)
        # weights = self.MoE(batches.feats_c3d.view(-1, 256*1024))
        # normed_weights = torch.chunk(F.softmax(weights, dim=-1), 3, dim=1)
        # print(normed_weights[-1].size())

        # pieces = []
        # for w, t in zip(normed_weights, fusion_feats):
        #     # c = t * w.view((-1, 1, 1))
        #     # pieces.append(c)
        #     pieces.append(w * t)

        # sum_pieces = sum(pieces)
        # feats = sum_pieces * self.ELMo_Gamma
        # feats = sum_pieces
        # use MaxPool1d to generate 2d proposal-level feature map from 1d temporal features
        map2d = self.feat2d(feats)
        map2d, map2d_iou = self.proposal_conv(map2d)

        # inference
        contrastive_scores = []
        iou_scores = []
        _, T, _ = map2d[0].size()
        # sent_feat_iou: [num_sent x C] (len=B)
        for i, sf_iou in enumerate(sent_feat_iou):
            # iou part
            vid_feat_iou = map2d_iou[i]  # C x T x T
            vid_feat_iou_norm = F.normalize(vid_feat_iou, dim=0)
            sf_iou_norm = F.normalize(sf_iou, dim=1)
            iou_score = torch.mm(sf_iou_norm, vid_feat_iou_norm.reshape(
                vid_feat_iou_norm.size(0), -1)).reshape(-1, T, T)  # num_sent x T x T
            iou_scores.append((iou_score*10).sigmoid() * self.feat2d.mask2d)

        # loss
        if self.training:
            loss_iou = self.iou_score_loss(
                torch.cat(iou_scores, dim=0), torch.cat(ious2d, dim=0), cur_epoch)
            loss_vid, loss_sent = self.contrastive_loss(
                map2d, sent_feat, ious2d, batches.moments)
            return loss_vid, loss_sent, loss_iou
        else:
            for i, sf in enumerate(sent_feat):
                # contrastive part
                vid_feat = map2d[i, ...]  # C x T x T
                vid_feat_norm = F.normalize(vid_feat, dim=0)
                sf_norm = F.normalize(sf, dim=1)  # num_sent x C
                _, T, _ = vid_feat.size()
                contrastive_score = torch.mm(sf_norm, vid_feat_norm.reshape(vid_feat.size(
                    0), -1)).reshape(-1, T, T) * self.feat2d.mask2d  # num_sent x T x T
                contrastive_scores.append(contrastive_score)
            # first two maps for visualization
            return map2d_iou, sent_feat_iou, contrastive_scores, iou_scores
