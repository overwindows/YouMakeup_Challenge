import json
import logging
import torch
from .utils import moment_to_iou2d, bert_embedding, get_vid_feat
from transformers import RobertaTokenizer


class MakeupDataset(torch.utils.data.Dataset):

    def __init__(self, ann_file, feat_mae, feat_swin, feat_i3d, feat_c3d, num_pre_clips, num_clips):
        super(MakeupDataset, self).__init__()
        print("*********")
        print(feat_mae)
        print(feat_swin)
        print(feat_i3d)
        print(feat_c3d)
        print("*********")
        self.feat_mae = feat_mae
        self.feat_swin = feat_swin
        self.feat_i3d = feat_i3d
        self.feat_c3d = feat_c3d

        self.num_pre_clips = num_pre_clips
        with open(ann_file, 'r') as f:
            annos = json.load(f)

        self.annos = []
        logger = logging.getLogger("mmn.trainer")
        logger.info("Preparing data, please wait...")
        tokenizer = RobertaTokenizer.from_pretrained(
            '/apdcephfs/private_chewu/pretrained_models/ft_local/roberta-base')

        for vid, anno in annos.items():
            duration = float(anno['duration'])  # duration of the video
            # Produce annotations
            moments = []
            all_iou2d = []
            sentences = []
            for timestamp, sentence in zip(anno['timestamps'], anno['sentences']):
                if int(timestamp[0]) < int(timestamp[1]):
                    moment = torch.Tensor(
                        [max(int(timestamp[0]), 0), min(int(timestamp[1]), duration)])
                    moments.append(moment)
                    iou2d = moment_to_iou2d(moment, num_clips, duration)
                    all_iou2d.append(iou2d)
                    sentences.append(sentence)

            moments = torch.stack(moments)
            all_iou2d = torch.stack(all_iou2d)
            # padded query of N*word_len, tensor of size = N
            queries, word_lens = bert_embedding(sentences, tokenizer)

            assert moments.size(0) == all_iou2d.size(0)
            assert moments.size(0) == queries.size(0)
            assert moments.size(0) == word_lens.size(0)
            self.annos.append(
                {
                    'vid': vid,
                    'moment': moments,  # N * 2
                    'iou2d': all_iou2d,  # N * 128*128
                    'sentence': sentences,   # list, len=N
                    'query': queries,  # padded query, N*word_len*C for LSTM and N*word_len for BERT
                    'wordlen': word_lens,  # size = N
                    'duration': duration
                }
            )

        #self.feats = video2feats(feat_file, annos.keys(), num_pre_clips, dataset_name="tacos")

    def __getitem__(self, idx):
        feat_swin = get_vid_feat(
            self.feat_swin, self.annos[idx]['vid'], self.num_pre_clips, dataset_name="i3d")
        feat_mae = get_vid_feat(
            self.feat_mae, self.annos[idx]['vid'], self.num_pre_clips, dataset_name="i3d")
        feat_i3d = get_vid_feat(
            self.feat_i3d, self.annos[idx]['vid'], self.num_pre_clips, dataset_name="i3d")
        query = self.annos[idx]['query']
        wordlen = self.annos[idx]['wordlen']
        iou2d = self.annos[idx]['iou2d']
        moment = self.annos[idx]['moment']
        feat_c3d = get_vid_feat(
            self.feat_c3d, self.annos[idx]['vid'], self.num_pre_clips, dataset_name="c3d")
        return feat_mae, feat_swin, feat_i3d, query, wordlen, iou2d, moment, len(self.annos[idx]['sentence']), idx, feat_c3d

    def __len__(self):
        return len(self.annos)

    def get_duration(self, idx):
        return self.annos[idx]['duration']

    def get_sentence(self, idx):
        return self.annos[idx]['sentence']

    def get_moment(self, idx):
        return self.annos[idx]['moment']

    def get_vid(self, idx):
        return self.annos[idx]['vid']
