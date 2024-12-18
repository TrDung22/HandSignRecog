import torch
from torch import nn

from .crossVTN_ultis import FeatureExtractor, LinearClassifier, StreamCrossAttention, ViewsCrossAttention
import torch.nn.functional as F
from pytorch_lightning.utilities.migration import pl_legacy_patch

class TwoStreamCrossVTN(nn.Module):
    def __init__(self, num_classes=199, num_heads=4, num_layers_stream=2, embed_size=512, sequence_length=16, cnn='rn34',
                 freeze_layers=0, dropout=0, **kwargs):
        super().__init__()
        print("Model: 2s-CrossVTN")
        self.sequence_length = sequence_length
        self.embed_size = embed_size
        self.num_classes = num_classes

        self.feature_extractor_heatmap = FeatureExtractor(cnn, embed_size, freeze_layers)
        self.feature_extractor_rgb = FeatureExtractor(cnn, embed_size, freeze_layers)

        num_attn_features = embed_size
        self.cross_attention= StreamCrossAttention(num_attn_features, num_attn_features,
                                                    [num_heads] * num_layers_stream,
                                                    self.sequence_length, layer_norm=True, dropout=dropout)
        self.classifier = LinearClassifier(num_attn_features*2, num_classes, dropout)
        self.num_attn_features = num_attn_features
        self.dropout = dropout
        self.num_classes = num_classes
        self.relu = F.relu

    def reset_head(self, num_classes):
        self.classifier = LinearClassifier(self.num_attn_features*2, num_classes, self.dropout)
        print("Reset to ", num_classes)

    def forward_features(self):
        return None

    def forward(self, heatmap=None, rgb=None, **kwargs):
        """Extract the image feature vectors."""

        b, t, c, h, w = rgb.size()
        rgb_feature = self.feature_extractor_rgb(rgb.view(b, t, c, h, w)).view(b, t, -1)
        heatmap_feature = self.feature_extractor_heatmap(heatmap.view(b, t, c, h, w)).view(b, t, -1)

        heatmap_feature, rgb_feature = self.cross_attention(heatmap_feature, rgb_feature)
        output_feature = torch.cat((heatmap_feature, rgb_feature), dim=-1)

        y = self.classifier(output_feature)

        return {'logits': y}  # train

class TwoStreamCrossViewVTN(nn.Module):
    def __init__(self, num_classes=199, num_heads=4, num_layers_stream=2, num_layers_view=2, embed_size=512, sequence_length=16, cnn='rn34', freeze_layers=0, dropout=0, **kwargs):
        super().__init__()
        print("Model: 2s-CrossViewVTN")
        self.sequence_length = sequence_length
        self.embed_size = embed_size
        self.num_classes = num_classes

        self.feature_extractor_heatmap = FeatureExtractor(cnn, embed_size, freeze_layers)
        self.feature_extractor_rgb = FeatureExtractor(cnn, embed_size, freeze_layers)

        num_attn_features = embed_size
        self.hmap_view_cross_attn = ViewsCrossAttention(num_attn_features, num_attn_features,
                                                    [num_heads] * num_layers_view,
                                                    self.sequence_length, layer_norm=True, dropout=dropout)
        self.rgb_view_cross_attn = ViewsCrossAttention(num_attn_features, num_attn_features,
                                                        [num_heads] * num_layers_view,
                                                        self.sequence_length, layer_norm=True, dropout=dropout)

        self.stream_cross_attention = StreamCrossAttention(num_attn_features, num_attn_features,
                                                    [num_heads] * num_layers_stream,
                                                    self.sequence_length, layer_norm=True, dropout=dropout)
        self.hmap_projector = nn.Linear(embed_size*3,embed_size)
        self.rgb_projector = nn.Linear(embed_size*3,embed_size)

        self.classifier = LinearClassifier(num_attn_features * 2, num_classes, dropout)
        self.num_attn_features = num_attn_features
        self.dropout = dropout
        self.num_classes = num_classes

    def forward_features(self, rgb_left=None, rgb_center=None, rgb_right=None, hmap_left=None, hmap_right=None, hmap_center=None):
        b, t, x, c, h, w = rgb_center.size()

        rgb_left_feature = self.feature_extractor_rgb(rgb_left.view(b, t * x, c, h, w)).view(b, t, -1)
        rgb_right_feature = self.feature_extractor_rgb(rgb_right.view(b, t * x, c, h, w)).view(b, t, -1)
        rgb_center_feature = self.feature_extractor_rgb(rgb_center.view(b, t * x, c, h, w)).view(b, t, -1)

        hmap_left_feature = self.feature_extractor_rgb(hmap_left.view(b, t * x, c, h, w)).view(b, t, -1)
        hmap_right_feature = self.feature_extractor_rgb(hmap_right.view(b, t * x, c, h, w)).view(b, t, -1)
        hmap_center_feature = self.feature_extractor_rgb(hmap_center.view(b, t * x, c, h, w)).view(b, t, -1)

        hmap_center_feature, hmap_right_feature, hmap_left_feature = self.hmap_view_cross_attn(hmap_center_feature,
                                                                                               hmap_right_feature,
                                                                                               hmap_left_feature)
        rgb_center_feature, rgb_right_feature, rgb_left_feature = self.rgb_view_cross_attn(rgb_center_feature,
                                                                                           rgb_right_feature,
                                                                                           rgb_left_feature)

        hmap_ft = torch.cat([hmap_left_feature, hmap_center_feature, hmap_right_feature], dim=-1)
        rgb_ft = torch.cat([rgb_left_feature, rgb_center_feature, rgb_right_feature], dim=-1)

        hmap_ft = self.hmap_projector(hmap_ft)
        rgb_ft = self.rgb_projector(rgb_ft)

        hmap_ft, rgb_ft = self.stream_cross_attention(hmap_ft, rgb_ft)

        output_features = torch.cat([hmap_ft, rgb_ft], dim=-1)

        return output_features

    def forward(self, rgb_left=None, rgb_center=None, rgb_right=None, hmap_left=None, hmap_right=None, hmap_center=None):
        b, t, x, c, h, w = rgb_center.size()

        rgb_left_feature = self.feature_extractor_rgb(rgb_left.view(b, t * x, c, h, w)).view(b, t, -1)
        rgb_right_feature = self.feature_extractor_rgb(rgb_right.view(b, t * x, c, h, w)).view(b, t, -1)
        rgb_center_feature = self.feature_extractor_rgb(rgb_center.view(b, t * x, c, h, w)).view(b, t, -1)

        hmap_left_feature = self.feature_extractor_rgb(hmap_left.view(b, t * x, c, h, w)).view(b, t, -1)
        hmap_right_feature = self.feature_extractor_rgb(hmap_right.view(b, t * x, c, h, w)).view(b, t, -1)
        hmap_center_feature = self.feature_extractor_rgb(hmap_center.view(b, t * x, c, h, w)).view(b, t, -1)

        hmap_center_feature, hmap_right_feature, hmap_left_feature = self.hmap_view_cross_attn(hmap_center_feature, hmap_right_feature, hmap_left_feature)
        rgb_center_feature, rgb_right_feature, rgb_left_feature = self.rgb_view_cross_attn(rgb_center_feature, rgb_right_feature, rgb_left_feature)

        hmap_ft = torch.cat([hmap_left_feature, hmap_center_feature, hmap_right_feature], dim=-1)
        rgb_ft = torch.cat([rgb_left_feature, rgb_center_feature, rgb_right_feature], dim=-1)

        hmap_ft = self.hmap_projector(hmap_ft)
        rgb_ft = self.rgb_projector(rgb_ft)

        hmap_ft, rgb_ft = self.stream_cross_attention(hmap_ft, rgb_ft)

        output_features = torch.cat([hmap_ft, rgb_ft], dim=-1)

        y = self.classifier(output_features)

        return {
            'logits': y
        }
