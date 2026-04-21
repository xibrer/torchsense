from models.DeepLabV3.backbone import mobilenet


def build_backbone(backbone, in_channels):

    if backbone == 'mobilenet':
        return mobilenet.MobileNetV1(in_channels)
    else:
        raise
