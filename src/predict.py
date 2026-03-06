"""
predict.py — Run inference on a single Indian currency note image.

Usage:
    python src/predict.py --image path/to/note.jpg --model cnn
    python src/predict.py --image path/to/note.jpg --model resnet
"""

import argparse
import numpy as np
from PIL import Image

LABELS = {
    0: "₹10 — Ten Rupees",
    1: "₹20 — Twenty Rupees",
    2: "₹50 — Fifty Rupees",
    3: "₹100 — One Hundred Rupees",
    4: "₹200 — Two Hundred Rupees",
    5: "₹500 — Five Hundred Rupees",
    6: "₹2000 — Two Thousand Rupees",
}


def predict_cnn(image_path: str):
    """Predict using the custom CNN (TensorFlow/Keras)."""
    import tensorflow as tf

    model = tf.keras.models.load_model("models/cnn_currency.h5")
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    x = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    probs = model.predict(x)[0]
    pred_class = np.argmax(probs)
    confidence = probs[pred_class] * 100

    print(f"\n🔍 Image     : {image_path}")
    print(f"🏷️  Prediction: {LABELS[pred_class]}")
    print(f"📊 Confidence: {confidence:.2f}%\n")

    print("All class probabilities:")
    for i, p in enumerate(probs):
        bar = "█" * int(p * 30)
        print(f"  {LABELS[i]:<35} {p*100:5.1f}% {bar}")


def predict_resnet(image_path: str):
    """Predict using the FastAI ResNet34 model."""
    from fastai.vision.all import load_learner, PILImage

    learn = load_learner("models/resnet34_currency.pkl")
    img = PILImage.create(image_path)
    pred, pred_idx, probs = learn.predict(img)

    print(f"\n🔍 Image     : {image_path}")
    print(f"🏷️  Prediction: {pred}")
    print(f"📊 Confidence: {probs[pred_idx]*100:.2f}%\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indian Currency Note Classifier")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument(
        "--model",
        choices=["cnn", "resnet"],
        default="resnet",
        help="Which model to use for prediction (default: resnet)",
    )
    args = parser.parse_args()

    if args.model == "cnn":
        predict_cnn(args.image)
    else:
        predict_resnet(args.image)
