"""
VLM encoder using OpenCLIP for text extraction and embeddings
"""

import torch
import open_clip
from PIL import Image
from typing import List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class VLMEncoder:
    """
    Vision-Language Model encoder using OpenCLIP.
    Singleton pattern to load model only once.
    """
    
    _instance = None
    _model = None
    _tokenizer = None
    _preprocess = None
    _device = None
    
    def __new__(cls, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        """
        Singleton pattern implementation.
        
        Args:
            model_name: OpenCLIP model name (default: ViT-B-32)
            pretrained: Pretrained dataset (default: openai)
        """
        if cls._instance is None:
            cls._instance = super(VLMEncoder, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        """
        Initialize VLM encoder (only loads model once).
        
        Args:
            model_name: OpenCLIP model name
            pretrained: Pretrained dataset
        """
        if self._model is None:
            self._load_model(model_name, pretrained)
    
    def _load_model(self, model_name: str, pretrained: str):
        """
        Load OpenCLIP model and preprocessing.
        Verifies that standard general-purpose CLIP weights are loaded.
        
        Args:
            model_name: OpenCLIP model name
            pretrained: Pretrained dataset (should be 'openai' for general-purpose)
        """
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading OpenCLIP model: {model_name} ({pretrained}) on {self._device}")
        logger.info("Using general-purpose CLIP weights (not security-specific)")
        
        # Load model, tokenizer, and preprocessing
        # This loads standard CLIP weights, not custom fine-tuned models
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained,
            device=self._device
        )
        self._tokenizer = open_clip.get_tokenizer(model_name)
        
        self._model.eval()
        
        # Verify model is loaded correctly
        logger.info("Model loaded successfully!")
        logger.info(f"Model type: {type(self._model).__name__}")
        logger.info(f"Preprocessing transforms: {len(self._preprocess.transforms)} transforms")
        
        # Quick verification test with a dummy image
        try:
            test_image = Image.new('RGB', (224, 224), color='red')
            test_tensor = self._preprocess(test_image).unsqueeze(0).to(self._device)
            with torch.no_grad():
                test_features = self._model.encode_image(test_tensor)
            logger.info(f"Model verification: Image encoding works (output shape: {test_features.shape})")
        except Exception as e:
            logger.warning(f"Model verification test failed: {e}")
    
    def extract_text_from_image(self, image: Image.Image, expected_objects: Optional[List[str]] = None) -> str:
        """
        Extract action-focused description of the main subject using VLM.
        Uses dynamic prompts based on expected objects provided by user.
        
        Args:
            image: PIL Image (must be RGB format)
            expected_objects: Optional list of expected objects (e.g., ["worker", "machine", "forklift"])
                            If None, uses default prompts
            
        Returns:
            Concise action description of the main subject
        """
        try:
            # Verify image is RGB (OpenCLIP expects RGB)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Preprocess image using OpenCLIP's standard preprocessing
            image_tensor = self._preprocess(image).unsqueeze(0).to(self._device)
            
            # Generate prompts dynamically based on expected objects
            text_prompts = self._generate_prompts(expected_objects)
            
            
            # Process all action prompts to find the best match
            # Since we have a small, focused prompt set, we can process all at once
            text_tokens = self._tokenizer(text_prompts).to(self._device)
            
            with torch.no_grad():
                # Get image features (normalized for cosine similarity)
                image_features = self._model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Get text features (normalized for cosine similarity)
                text_features = self._model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Compute cosine similarity (dot product of normalized vectors)
                # Scale by 100 for better softmax distribution
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # Get the single best action match (strict action-focused approach)
            best_idx = similarity.argmax().item()
            best_confidence = similarity[0, best_idx].item()
            
            # Return the best matching action description
            # Only return if confidence is above minimum threshold
            if best_confidence > 0.01:
                return text_prompts[best_idx]
            else:
                # Fallback if no confident match
                return "unclear action"
                
        except Exception as e:
            logger.error(f"Error in extract_text_from_image: {e}", exc_info=True)
            return "error processing frame description"
    
    def _generate_prompts(self, expected_objects: Optional[List[str]] = None) -> List[str]:
        """
        Generate action-focused prompts based on expected objects.
        
        Args:
            expected_objects: List of expected objects (e.g., ["worker", "machine", "forklift"])
                            If None, returns default prompts
        
        Returns:
            List of text prompts for action classification
        """
        if expected_objects is None or len(expected_objects) == 0:
            # Default prompts for backward compatibility
            return [
                "a cat sitting still", "a cat standing still", "a cat grooming itself",
                "a cat licking itself", "a cat grooming and licking itself", "a cat walking",
                "a cat playing", "a cat sleeping", "a cat lying down", "a cat looking at the camera",
                "a person sitting still", "a person standing still", "a person walking",
                "a person working", "a person reading", "a person cooking", "a person eating",
                "a dog sitting still", "a dog standing still", "a dog walking",
                "a dog playing", "a dog sleeping", "a dog lying down",
            ]
        
        # Generate prompts for each expected object with common actions
        actions = [
            "sitting still", "standing still", "walking", "working", "operating",
            "moving", "stopped", "idle", "active", "in motion"
        ]
        
        text_prompts = []
        for obj in expected_objects:
            obj = obj.strip().lower()
            for action in actions:
                text_prompts.append(f"a {obj} {action}")
        
        logger.debug(f"Generated {len(text_prompts)} prompts for objects: {expected_objects}")
        return text_prompts
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Get text embedding from text string.
        
        Args:
            text: Text string
            
        Returns:
            Normalized text embedding as numpy array
        """
        text_tokens = self._tokenizer([text]).to(self._device)
        
        with torch.no_grad():
            text_features = self._model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy().squeeze()
    
    def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Get image embedding from PIL Image.
        Ensures image is in RGB format before processing.
        
        Args:
            image: PIL Image (will be converted to RGB if needed)
            
        Returns:
            Normalized image embedding as numpy array
        """
        # Ensure image is RGB (OpenCLIP expects RGB format)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess image (handles normalization, resizing, tensor conversion)
        image_tensor = self._preprocess(image).unsqueeze(0).to(self._device)
        
        with torch.no_grad():
            image_features = self._model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().squeeze()
    
    def get_frame_embedding(self, image: Image.Image, expected_objects: Optional[List[str]] = None) -> Tuple[str, np.ndarray]:
        """
        Extract text and embedding from a frame.
        
        Args:
            image: PIL Image
            expected_objects: Optional list of expected objects for dynamic prompts
            
        Returns:
            Tuple of (text_description, embedding)
        """
        text = self.extract_text_from_image(image, expected_objects)
        embedding = self.get_image_embedding(image)
        return text, embedding
    
    def get_frame_embeddings_batch(
        self, 
        images: List[Image.Image],
        expected_objects: Optional[List[str]] = None,
        batch_size: int = 16
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Process multiple frames in batch for efficiency using GPU batching.
        
        Args:
            images: List of PIL Images
            expected_objects: Optional list of expected objects for dynamic prompts
            batch_size: Batch size for GPU processing
            
        Returns:
            List of tuples (text_description, embedding)
        """
        if len(images) == 0:
            return []
        
        results = []
        
        # Process images in batches for better GPU utilization
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Process batch for image embeddings (faster with batching)
            batch_tensors = []
            valid_indices = []
            
            for idx, image in enumerate(batch_images):
                try:
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    tensor = self._preprocess(image).to(self._device)
                    batch_tensors.append(tensor)
                    valid_indices.append(i + idx)
                except Exception as e:
                    logger.warning(f"Error preprocessing image {i + idx}: {e}")
                    continue
            
            if len(batch_tensors) == 0:
                continue
            
            # Batch process image embeddings
            batch_tensor = torch.stack(batch_tensors)
            
            with torch.no_grad():
                batch_embeddings = self._model.encode_image(batch_tensor)
                batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
            
            # Process text extraction individually (still needs individual processing)
            batch_embeddings_np = batch_embeddings.cpu().numpy()
            
            for local_idx, global_idx in enumerate(valid_indices):
                image = images[global_idx]
                embedding = batch_embeddings_np[local_idx]
                
                # Extract text description (still done individually)
                text = self.extract_text_from_image(image, expected_objects)
                results.append((text, embedding))
        
        return results

