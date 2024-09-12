import open_clip
import torch
from typing import List, Tuple

class ChimeraCLIP():
    def __init__(
        self,
        models: List[Tuple[str]],
        device = "cpu"
    ):
        self.clip_models = []
        self.preprocessors = []
        self.tokenizers = []
        for arch, pretrained in models:
            model, _, transform = open_clip.create_model_and_transforms(
                arch, pretrained=pretrained
            )
            model = model.to(device)
            model.eval()
            tokenizer = open_clip.get_tokenizer(arch)
            
            self.clip_models.append(model)
            self.preprocessors.append(transform)
            self.tokenizers.append(tokenizer)
        
    
    def encode_image(self, images, normalize: bool = True):
        latents = []
        for model in self.clip_models:
            latent = model.encode_image(images)
            latents.append(latent)
        
        return torch.cat(latents, dim=-1)

    def encode_text(self, text, normalize: bool = True):
        latents = []
        for model in self.clip_models:
            latent = model.encode_text(text)
            latents.append(latent)
        
        return torch.cat(latents, dim=-1)
    