import open_clip
import torch
import torch.nn.functional as F
from typing import List, Tuple

class ChimeraCLIP():
    def __init__(
        self,
        models: List[Tuple[str]],
        device = "cpu"
    ):
        self.device = device
        self.clip_models = []
        self.preprocessors = []
        self.tokenizers = []
        for arch, pretrained in models:
            model, _, transform = open_clip.create_model_and_transforms(
                arch, pretrained=pretrained
            )
            model = model.to(self.device)
            model.eval()
            tokenizer = open_clip.get_tokenizer(arch)
            
            self.clip_models.append(model)
            self.preprocessors.append(transform)
            self.tokenizers.append(tokenizer)
        
    
    def encode_image(self, images, normalize: bool = True):
        latents = []
        for model in self.clip_models:
            latent = model.encode_image(images, normalize=normalize)
            latents.append(latent)
        
        concat = torch.cat(latents, dim=-1)

        if normalize:
            concat = F.normalize(concat, dim=-1)

        return concat

    def encode_text(self, text, normalize: bool = True):
        latents = []
        for model in self.clip_models:
            latent = model.encode_text(text, normalize=normalize)
            latents.append(latent)
        
        concat = torch.cat(latents, dim=-1)

        if normalize:
            concat = F.normalize(concat, dim=-1)

        return concat

    def e2e_encode_text(self, texts: List[str], normalize: bool = True):
        latents = []
        for model, tokenizer in zip(self.clip_models, self.tokenizers):
            tokens = tokenizer(texts).to(self.device)
            latent = model.encode_text(tokens, normalize=normalize)
            latents.append(latent)
        
        concat = torch.cat(latents, dim=-1)

        if normalize:
            concat = F.normalize(concat, dim=-1)

        return concat

    def e2e_encode_image(self, images, normalize: bool = True):
        latents = []
        for model, transform in zip(self.clip_models, self.preprocessors):
            latent = model.encode_image(transform(images).to(self.device), normalize=normalize)
            latents.append(latent)
        
        concat = torch.cat(latents, dim=-1)

        if normalize:
            concat = F.normalize(concat, dim=-1)

        return concat
    