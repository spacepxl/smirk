import torch
import torch.nn.functional as F
from torch import nn
import timm
from collections import namedtuple


def create_backbone(backbone_name, pretrained=True):
    backbone = timm.create_model(backbone_name, 
                        pretrained=pretrained,
                        features_only=True)
    feature_dim = backbone.feature_info[-1]['num_chs']
    return backbone, feature_dim

class PoseEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
              
        self.encoder, feature_dim = create_backbone('tf_mobilenetv3_small_minimal_100')
        
        self.pose_cam_layers = nn.Sequential(
            nn.Linear(feature_dim, 6)
        )

        self.init_weights()

    def init_weights(self):
        self.pose_cam_layers[-1].weight.data *= 0.001
        self.pose_cam_layers[-1].bias.data *= 0.001

        self.pose_cam_layers[-1].weight.data[3] = 0
        self.pose_cam_layers[-1].bias.data[3] = 7


    def forward(self, img):
        features = self.encoder(img)[-1]
        features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)
        pose_cam = self.pose_cam_layers(features).reshape(img.size(0), -1)
        
        pose_params = pose_cam[...,:3]
        cam = pose_cam[...,3:]
        return pose_params, cam


class ShapeEncoder(nn.Module):
    def __init__(self, n_shape=300) -> None:
        super().__init__()

        self.encoder, feature_dim = create_backbone('tf_mobilenetv3_large_minimal_100')

        self.shape_layers = nn.Sequential(
            nn.Linear(feature_dim, n_shape)
        )

        self.init_weights()


    def init_weights(self):
        self.shape_layers[-1].weight.data *= 0
        self.shape_layers[-1].bias.data *= 0


    def forward(self, img):
        features = self.encoder(img)[-1]
        features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)
        
        shape_params = self.shape_layers(features).reshape(img.size(0), -1)
        return shape_params


class ExpressionEncoder(nn.Module):
    def __init__(self, n_exp=50) -> None:
        super().__init__()

        self.encoder, feature_dim = create_backbone('tf_mobilenetv3_large_minimal_100')
        
        self.expression_layers = nn.Sequential( 
            nn.Linear(feature_dim, n_exp+2+3) # num expressions + jaw + eyelid
        )

        self.n_exp = n_exp
        self.init_weights()


    def init_weights(self):
        self.expression_layers[-1].weight.data *= 0.1
        self.expression_layers[-1].bias.data *= 0.1


    def forward(self, img):
        features = self.encoder(img)[-1]
        features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)
        parameters = self.expression_layers(features).reshape(img.size(0), -1)
        
        expression_params = parameters[...,:self.n_exp]
        eyelid_params = torch.clamp(parameters[...,self.n_exp:self.n_exp+2], 0, 1)
        jaw_params = torch.cat([F.relu(parameters[...,self.n_exp+2].unsqueeze(-1)), 
            torch.clamp(parameters[...,self.n_exp+3:self.n_exp+5], -.2, .2)], dim=-1)
        return expression_params, eyelid_params, jaw_params


class SmirkEncoder(nn.Module):
    def __init__(self, n_exp=50, n_shape=300) -> None:
        super().__init__()

        self.pose_encoder = PoseEncoder()

        self.shape_encoder = ShapeEncoder(n_shape=n_shape)

        self.expression_encoder = ExpressionEncoder(n_exp=n_exp) 

    def forward(self, img):
        pose_params, cam = self.pose_encoder(img)
        shape_params = self.shape_encoder(img)
        expression_params, eyelid_params, jaw_params = self.expression_encoder(img)
        
        # SmirkOutput = namedtuple("SmirkOutput", ["pose_params", "cam", "shape_params", "expression_params", "eyelid_params", "jaw_params"])
        # return SmirkOutput(pose_params, cam, shape_params, expression_params, eyelid_params, jaw_params)
        return pose_params, cam, shape_params, expression_params, eyelid_params, jaw_params
