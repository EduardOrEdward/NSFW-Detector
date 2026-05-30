import sys, time
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from vit_detector import VitClassifier

detector = VitClassifier("data/models/vit/quantized_model.onnx")

# Подставьте путь к любому изображению
with open("app/xv_p.jpg", "rb") as f:
    result = detector.predict(f.read(), threshold=0.5)

print("✅ ViT result:")
print(f"  score: {result['score']} | label: {result['label']}")
print(f"  latency: {result['latency_ms']} ms")
print(f"  probs: {result['meta']['probabilities']}")
print(f' meta : {result['meta']}')