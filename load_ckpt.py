import torch

cpt = torch.load("/home/node25_tmpdata/xcli/percepnet/c_aec/8.pt.tar")
state_dict = cpt["model_state_dict"]

print(state_dict["module.mic_encoders.0.conv2d.depth_conv.weight"].shape)