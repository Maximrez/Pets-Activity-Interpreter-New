import torch
import torchvision
from torchvision.models.detection import KeypointRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator


def keypointrcnn_mobilenet(backbone_name, path, device):
    if backbone_name == "mobilenet_v3_large":
        backbone = torchvision.models.mobilenet_v3_large(pretrained=True).features
        backbone.out_channels = 960
    elif backbone_name == "mobilenet_v3_small":
        backbone = torchvision.models.mobilenet_v3_small(pretrained=True).features
        backbone.out_channels = 576
    elif backbone_name == "mobilenet_v2":
        backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        backbone.out_channels = 1280
    else:
        raise Exception('Bad backbone name')

    anchor_generator = AnchorGenerator(sizes=((16, 32, 64, 128, 256),), aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    keypoint_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=14, sampling_ratio=2)
    model_keypoints = KeypointRCNN(backbone, num_classes=6, num_keypoints=20,
                                   rpn_anchor_generator=anchor_generator,
                                   box_roi_pool=roi_pooler,
                                   keypoint_roi_pool=keypoint_roi_pooler)

    model_keypoints = model_keypoints.to(device)

    model_keypoints.load_state_dict(torch.load(path, map_location=device))
    model_keypoints.eval()

    return model_keypoints
