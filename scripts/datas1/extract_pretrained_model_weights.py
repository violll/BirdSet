from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import torch

safetensors_path = hf_hub_download(repo_id="DBD-research-group/EfficientNet-B1-BirdSet-XCL", filename="model.safetensors")
state_dict = load_file(safetensors_path)
checkpoint = {"state_dict": state_dict}
torch.save(checkpoint, "BirdSet/resources/models/EfficientNet-B1-BirdSet-XCL.ckpt")