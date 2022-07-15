from typing import Callable, List, Union
import clip
import numpy as np
import torch
from PIL import Image


class CLIPClassifier:
    def __init__(
        self,
        classes: List[str],
        sentence: Callable[[str], str],
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.device = device
        self.classes = classes
        self.sentence = sentence

        self.texts = clip.tokenize([self.sentence(x) for x in self.classes]).to(
            self.device
        )

        model, preprocess = clip.load("ViT-B/32", device=self.device)
        self.model = model
        self.preprocess = preprocess

    def classify(self, image: Image) -> np.ndarray:
        image_preprocessed = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits_per_image, _ = self.model(image_preprocessed, self.texts)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        return probs
