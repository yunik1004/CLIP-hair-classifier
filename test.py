import argparse
import numpy as np
from PIL import Image
from classifier import CLIPClassifier
from database.classes import WOMAN_HAIR_CLASSES, MAN_HAIR_CLASSES, SEX_CLASSES
from database.sentences import SENTENCE, HAIRSTYLE_SENTENCE


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP zero-shot hair classifier")
    parser.add_argument(
        "--img", type=str, required=True, help="Path of the facial image file"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Path of the facial image file"
    )
    args = parser.parse_args()

    DEVICE = args.device
    IMAGE_PATH = args.img

    sex_classifier = CLIPClassifier(SEX_CLASSES, SENTENCE, DEVICE)

    image = Image.open(IMAGE_PATH)

    sex_probs = sex_classifier.classify(image)
    sex_idx = np.argmax(sex_probs)
    print(f"Sex: {SEX_CLASSES[sex_idx]} (prob: {sex_probs[sex_idx]})")

    if sex_idx == 0:
        hair_classes = MAN_HAIR_CLASSES
    else:
        hair_classes = WOMAN_HAIR_CLASSES

    hair_classifier = CLIPClassifier(hair_classes, HAIRSTYLE_SENTENCE, DEVICE)

    hair_probs = hair_classifier.classify(image)
    hair_idx = np.argmax(hair_probs)

    print(f"Hairstyle: {hair_classes[hair_idx]} (prob: {hair_probs[hair_idx]})")
