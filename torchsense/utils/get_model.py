from torchsense.models import ConvTasNet,ConvTasNetLite,UNet,UNetLSTM



def get_model(model_name:str):
    model_name = model_name.lower()
    # print(model_name)
    if model_name == "ConvTasNet" or model_name == "tasnet":
        return ConvTasNet
    elif model_name == "ConvTasNetLite" or model_name == "tasnetlite":
        return ConvTasNetLite
    elif model_name == "unet":
        return UNet
    elif model_name == "unet_lstm":
        return UNetLSTM
    else:
        raise ValueError("Model not found")