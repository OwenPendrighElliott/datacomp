import open_clip
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import time
import boto3
import json
import cohere
import vertexai
from vertexai.vision_models import Image, MultiModalEmbeddingModel
import os
from PIL import Image
from transformers import AutoModel
from io import BytesIO
import base64
from typing import List, Tuple

class ResizeIfLarger:
    """Resize an image if it is larger than a certain size"""
    def __init__(self, max_size):
        self.max_size = max_size

    def __call__(self, img):
        width, height = img.size
    
        if max(width, height) > self.max_size:
            img = transforms.Resize(self.max_size)(img)
        
        return img

DEFAULT_TRANSFORM = transforms.Compose([
    ResizeIfLarger(512),
])

class E2ECLIP():
    def __init__(self):
        self.transform = DEFAULT_TRANSFORM

    def encode_image(self, images, normalize: bool = True):
        raise NotImplementedError()

    def encode_text(self, text, normalize: bool = True):
        raise NotImplementedError()

    def e2e_encode_text(self, texts: List[str], normalize: bool = True) -> torch.Tensor:
        raise NotImplementedError()

    def e2e_encode_image(self, images, normalize: bool = True) -> torch.Tensor:
        raise NotImplementedError()

    def _image_to_base64_data_url(self, image: str):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            image = Image.fromarray(image)

        with BytesIO() as output:
            image.save(output, format="PNG")
            enc_img = base64.b64encode(output.getvalue()).decode('utf-8')
            enc_img = f"data:image/png;base64,{enc_img}"
        return enc_img

class ChimeraCLIP(E2ECLIP):
    def __init__(
        self,
        models: List[Tuple[str]],
        device = "cpu"
    ):
        super().__init__()
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

    def e2e_encode_text(self, texts: List[str], normalize: bool = True) -> torch.Tensor:
        latents = []
        for model, tokenizer in zip(self.clip_models, self.tokenizers):
            tokens = tokenizer(texts).to(self.device)
            latent = model.encode_text(tokens, normalize=normalize)
            latents.append(latent)
        
        concat = torch.cat(latents, dim=-1)

        if normalize:
            concat = F.normalize(concat, dim=-1)

        return concat

    def e2e_encode_image(self, images, normalize: bool = True) -> torch.Tensor:
        latents = []
        for model, transform in zip(self.clip_models, self.preprocessors):
            transformed_images = torch.stack([transform(img) for img in images]).to(self.device)
            latent = model.encode_image(transformed_images, normalize=normalize)
            latents.append(latent)
        
        concat = torch.cat(latents, dim=-1)

        if normalize:
            concat = F.normalize(concat, dim=-1)

        return concat

class JinaCLIP(E2ECLIP):
    def __init__(self, model: str, device = "cpu"):
        super().__init__()
        self.clip_model = AutoModel.from_pretrained(model, trust_remote_code=True)
        self.device = device
        self.clip_model.to(self.device)

        if not hasattr(self.clip_model, "encode_image") or not hasattr(self.clip_model, "encode_text"):
            raise ValueError("Model does not have encode_image or encode_text methods")

    def encode_image(self, images, normalize: bool = True):
        embedding = self.clip_model.encode_image(images, device=self.device) # np.array

        if normalize:
            embedding = embedding / np.linalg.norm(embedding, axis=-1, keepdims=True)

        return embedding

    
    def encode_text(self, text, normalize: bool = True):
        embedding = self.clip_model.encode_text(text, device=self.device) # np.array

        if normalize:
            embedding = embedding / np.linalg.norm(embedding, axis=-1, keepdims=True)

        return embedding
    
    def e2e_encode_text(self, text: str, normalize: bool = True) -> torch.Tensor:
        embedding = self.encode_text(text, normalize=normalize)

        return torch.tensor(embedding, dtype=torch.float32, device=self.device)
    
    def e2e_encode_image(self, images, normalize: bool = True) -> torch.Tensor:
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

        embedding = self.encode_image(images, normalize=normalize)

        return torch.tensor(embedding, dtype=torch.float32, device=self.device)
    

class CohereCLIP(E2ECLIP):
    def __init__(self, model: str, device: str = "cpu"):
        super().__init__()
        self.model = model
        self.device = device

        per_second_limit_text_buffer = 100
        self.per_second_rate_limit_text = (2000-per_second_limit_text_buffer)/60
        per_second_limit_image_buffer = 3
        self.per_second_rate_limit_image = (40-per_second_limit_image_buffer)/60

        self.retry_limit = 12

        self.retry_delay = 5

        self.last_text_time = 0
        self.last_image_time = 0

        if not os.getenv("COHERE_API_KEY"):
            raise ValueError("COHERE_API_KEY environment variable not set")

        self.co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))

    def encode_image(self, images, normalize: bool = True):
        processed_images = [self._image_to_base64_data_url(img) for img in images]

        embeddings = []
        
        for im in processed_images:
            if time.time() - self.last_image_time < self.per_second_rate_limit_image:
                time.sleep(1/self.per_second_rate_limit_image)

            for i in range(self.retry_limit):
                try:
                    resp = self.co.embed(
                        model=self.model,
                        images=[im],
                        input_type='image'
                    )
                    self.last_image_time = time.time()
                    break
                except Exception as e:
                    time.sleep(self.retry_delay)
                    print(e)
                    continue

            embedding = resp.embeddings[0]
            embeddings.append(embedding)

        tensor_embeddings = torch.tensor(embeddings, dtype=torch.float32, device=self.device)

        if normalize:
            tensor_embeddings = F.normalize(tensor_embeddings, dim=-1)

        return tensor_embeddings
    
    def encode_text(self, text, normalize: bool = True):
        
        if time.time() - self.last_text_time < self.per_second_rate_limit_text:
            time.sleep(1/self.per_second_rate_limit_text)

        for i in range(self.retry_limit):
            try:
                resp = self.co.embed(
                    model=self.model,
                    texts=text,
                    input_type='search_query'
                )
                self.last_text_time = time.time()
                break
            except Exception as e:
                time.sleep(self.retry_delay)
                print(e)
                continue

        embeddings = resp.embeddings
        tensor_embeddings = torch.tensor(embeddings, dtype=torch.float32, device=self.device)

        if normalize:
            tensor_embeddings = F.normalize(tensor_embeddings, dim=-1)

        return tensor_embeddings
    
    def e2e_encode_text(self, text: str, normalize: bool = True) -> torch.Tensor:
        embedding = self.encode_text(text, normalize=normalize)
        return embedding
    
    def e2e_encode_image(self, images, normalize: bool = True) -> torch.Tensor:
        embedding = self.encode_image(images, normalize=normalize)
        return embedding


class AmazonTitanEmbedV1(E2ECLIP):
    def __init__(self, model: str, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.model = model
        self.bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=os.getenv("AWS_REGION"),
        )

        self.retry_limit = 12
        self.retry_delay = 5
        self.last_text_time = 0
        self.last_image_time = 0
        per_second_limit_text_buffer = 100
        self.per_second_rate_limit_text = (20000-per_second_limit_text_buffer)/60
        per_second_limit_image_buffer = 100
        self.per_second_rate_limit_image = (20000-per_second_limit_image_buffer)/60
    

    def encode_image(self, images, normalize: bool = True):
        processed_images = [self._image_to_base64_data_url(img) for img in images]

        embeddings = []
        for image in processed_images:
            if time.time() - self.last_image_time < self.per_second_rate_limit_image:
                time.sleep(1/self.per_second_rate_limit_image)

            for i in range(self.retry_limit):
                try:
                    body = {
                        "inputImage": image
                    }

                    response = self.bedrock_runtime.invoke_model(
                        body=body,
                        modelId=self.model,
                        accept="application/json",
                        contentType="application/json"
                    )

                    response_body = json.loads(response.get("body").read())

                    embedding = response_body["embedding"]
                    embeddings.append(embedding)
                    self.last_image_time = time.time()
                    break
                except Exception as e:
                    time.sleep(self.retry_delay)
                    print(e)
                    continue

        tensor_embeddings = torch.tensor(embeddings, dtype=torch.float32, device=self.device)

        if normalize:
            tensor_embeddings = F.normalize(tensor_embeddings, dim=-1)

        return tensor_embeddings

    def encode_text(self, text, normalize: bool = True):
        if time.time() - self.last_text_time < self.per_second_rate_limit_text:
            time.sleep(1/self.per_second_rate_limit_text)

        embeddings = []
        for text_example in text:
            for i in range(self.retry_limit):
                try:
                    body = {
                        "inputText": text_example
                    }

                    response = self.bedrock_runtime.invoke_model(
                        body=body,
                        modelId=self.model,
                        accept="application/json",
                        contentType="application/json"
                    )

                    response_body = json.loads(response.get("body").read())

                    embedding = response_body["embedding"]
                    embeddings.append(embedding)
                    self.last_text_time = time.time()
                    break
                except Exception as e:
                    time.sleep(self.retry_delay)
                    print(e)
                    continue

        tensor_embeddings = torch.tensor(embeddings, dtype=torch.float32, device=self.device)

        if normalize:
            tensor_embedding = F.normalize(tensor_embedding, dim=-1)

        return tensor_embedding

    def e2e_encode_text(self, text: str, normalize: bool = True) -> torch.Tensor:
        embedding = self.encode_text(text, normalize=normalize)
        return embedding

    def e2e_encode_image(self, images, normalize: bool = True) -> torch.Tensor:
        embedding = self.encode_image(images, normalize=normalize)
        return embedding


class GoogleMultimodalEmbed(E2ECLIP):
    def __init__(self, model: str, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.model = model

        vertexai.init(project=os.getenv("GCP_PROJECT"), location=os.getenv("GCP_LOCATION"))
        self.embedding_dimension = 1408

        self.google_model = MultiModalEmbeddingModel.from_pretrained(model)

        self.retry_limit = 12
        self.retry_delay = 5
        self.last_text_time = 0
        self.last_image_time = 0
        per_second_limit_text_buffer = 2
        self.per_second_rate_limit_text = (120-per_second_limit_text_buffer)/60
        per_second_limit_image_buffer = 2
        self.per_second_rate_limit_image = (120-per_second_limit_image_buffer)/60

    def encode_image(self, images, normalize: bool = True):
        processed_images = [self._image_to_base64_data_url(img) for img in images]

        embeddings = []
        
        for im in processed_images:
            if time.time() - self.last_image_time < self.per_second_rate_limit_image:
                time.sleep(1/self.per_second_rate_limit_image)

            for i in range(self.retry_limit):
                try:
                    resp = self.google_model.get_embeddings(
                        image=im,
                        dimension=self.embedding_dimension,
                    )

                    image_embedding = resp.image_embedding
                    embeddings.append(image_embedding)
                    self.last_image_time = time.time()
                    break
                except Exception as e:
                    time.sleep(self.retry_delay)
                    print(e)
                    continue

        tensor_embeddings = torch.tensor(embeddings, dtype=torch.float32, device=self.device)

        if normalize:
            tensor_embeddings = F.normalize(tensor_embeddings, dim=-1)

        return tensor_embeddings

    def encode_text(self, text, normalize: bool = True):
        if time.time() - self.last_text_time < self.per_second_rate_limit_text:
            time.sleep(1/self.per_second_rate_limit_text)

        embeddings = []
        for txt in text:
            for i in range(self.retry_limit):
                try:
                    resp = self.google_model.get_embeddings(
                        contextual_text=txt,
                        dimension=self.embedding_dimension,
                    )

                    text_embedding = resp.text_embedding
                    embeddings.append(text_embedding)
                    self.last_text_time = time.time()
                    break
                except Exception as e:
                    time.sleep(self.retry_delay)
                    print(e)
                    continue
        
        tensor_embeddings = torch.tensor(embeddings, dtype=torch.float32, device=self.device)

        if normalize:
            tensor_embeddings = F.normalize(tensor_embeddings, dim=-1)

        return tensor_embeddings

    def e2e_encode_text(self, text: str, normalize: bool = True) -> torch.Tensor:
        embedding = self.encode_text(text, normalize=normalize)
        return embedding

    def e2e_encode_image(self, images, normalize: bool = True) -> torch.Tensor:
        embedding = self.encode_image(images, normalize=normalize)
        return embedding

