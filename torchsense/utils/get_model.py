from torchsense.models import ConvTasNet,ConvTasNetLite,UNet,UNetLSTM
from torchsense.models.network_rrdbnet import RRDBNet, AttentionRRDBNet
from torchsense.models.rrdb_cbam import RRDBNetCBAM

def get_model(model_name:str):
    model_name = model_name.lower()
    # print(model_name)
    if model_name == "ConvTasNet" or model_name == "tasnet":
        return ConvTasNet
    elif model_name == "ConvTasNetLite" or model_name == "tasnetlite":
        return ConvTasNetLite
    elif model_name == "unet":
        return UNet
    elif model_name == "rrdb":
        return RRDBNet
    elif model_name == "rrdb_cbam":
        return RRDBNetCBAM
    elif model_name == "attrrdb":
        return AttentionRRDBNet
    elif model_name == "unet_lstm":
        return UNetLSTM
    else:
        raise ValueError("Model not found")