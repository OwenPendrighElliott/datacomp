import open_clip
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoModel
from typing import List, Tuple


class E2ECLIP():
    def __init__(self):
        pass

    def encode_image(self, images, normalize: bool = True):
        pass

    def encode_text(self, text, normalize: bool = True):
        pass

    def e2e_encode_text(self, texts: List[str], normalize: bool = True):
        pass

    def e2e_encode_image(self, images, normalize: bool = True):
        pass

class ChimeraCLIP(E2ECLIP):
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
            transformed_images = torch.stack([transform(img) for img in images]).to(self.device)
            latent = model.encode_image(transformed_images, normalize=normalize)
            latents.append(latent)
        
        concat = torch.cat(latents, dim=-1)

        if normalize:
            concat = F.normalize(concat, dim=-1)

        return concat

class TransformersCLIP(E2ECLIP):
    def __init__(self, model: str):
        self.clip_model = AutoModel.from_pretrained(model, trust_remote_code=True)

    def encode_image(self, images, normalize: bool = True):
        embedding = self.clip_model.encode_image(images) # np.array

        if normalize:
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

        return embedding

    
    def encode_text(self, text, normalize: bool = True):
        embedding = self.clip_model.encode_text(text) # np.array

        if normalize:
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

        return embedding
    
    def encode_text_e2e(self, text: str, normalize: bool = True):
        return self.encode_text(text, normalize=normalize)
    
    def encode_image_e2e(self, images, normalize: bool = True):
        # if the image is a torch tensor then place it on cpu and convert to pil image
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
            # if 0-1 range then convert to 0-255, some safegruard to check it isn't just black
            if images.max() <= 1 and images.min() >= 0 and images.mean() > 0.01:
                images = (images * 255).astype(np.uint8)

            images = [Image.fromarray(np.uint8(img)) for img in images]
        elif isinstance(images, np.ndarray):
            # if 0-1 range then convert to 0-255, some safegruard to check it isn't just black
            if images.max() <= 1 and images.min() >= 0 and images.mean() > 0.01:
                images = (images * 255).astype(np.uint8)

            images = [Image.fromarray(np.uint8(img)) for img in images]

        return self.encode_image(images, normalize=normalize)