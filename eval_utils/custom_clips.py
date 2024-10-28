import open_clip
import torch
import numpy as np
from transformers import AutoModel, AutoImageProcessor
from typing import List, Tuple

class CLIP():
    def __init__(
        self,
        model: str,
        pretrained: str,
        device = "cpu"
    ):
        self.clip_model, _, self.preprocessor = open_clip.create_model_and_transforms(
            model, pretrained=pretrained
        )
        self.clip_model = self.clip_model.to(device)
        self.clip_model.eval()
        self.tokenizer = open_clip.get_tokenizer(model)
    
    def encode_image(self, images, normalize: bool = True):
        return self.clip_model.encode_image(images, normalize=normalize)
    
    def encode_text(self, text, normalize: bool = True):
        return self.clip_model.encode_text(text, normalize=normalize)
    
    def encode_text_e2e(self, text: str, normalize: bool = True):
        text = self.tokenizer(text, return_tensors="pt")
        return self.clip_model.encode_text(text, normalize=normalize)
    
    def encode_image_e2e(self, images, normalize: bool = True):
        images = self.preprocessor(images)
        return self.clip_model.encode_image(images, normalize=normalize)

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
            latent = model.encode_image(images, normalize=normalize)
            latents.append(latent)
        
        return torch.cat(latents, dim=-1)

    def encode_text(self, text, normalize: bool = True):
        latents = []
        for model in self.clip_models:
            latent = model.encode_text(text, normalize=normalize)
            latents.append(latent)
        
        return torch.cat(latents, dim=-1)
    
    def encode_text_e2e(self, text: str, normalize: bool = True):
        latents = []
        for model, tokenizer in zip(self.clip_models, self.tokenizers):
            text = tokenizer(text, return_tensors="pt")
            latent = model.encode_text(text, normalize=normalize)
            latents.append(latent)
        
        return torch.cat(latents, dim=-1)
    
    def encode_image_e2e(self, images, normalize: bool = True):
        latents = []
        for model, transform in zip(self.clip_models, self.preprocessors):
            images = transform(images)
            latent = model.encode_image(images, normalize=normalize)
            latents.append(latent)
        
        return torch.cat(latents, dim=-1)
    

class TransformersCLIP():
    def __init__(self, model: str):
        self.clip_model = AutoModel.from_pretrained(model)

        self.preprocessor = AutoImageProcessor.from_pretrained(model)

    def encode_image(self, images, normalize: bool = True):
        processed_images = self.preprocessor(images)
        embedding = self.clip_model.get_image_features(processed_images) # np.array

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
        return self.encode_image(images, normalize=normalize)