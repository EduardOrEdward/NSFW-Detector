import asyncio
from vit_detector import VitClassifier
from nudenet_detector import NudeNetDetector
from hybrid import HybridDetector

async def main():
    nudenet = NudeNetDetector(model_path="data/models/nudenet/nudenet.onnx")
    vit = VitClassifier(model_path="data/models/vit/quantized_model.onnx")
    
    hybrid = HybridDetector(
        detectors=[nudenet, vit],
        strategy="voting",  # или "weighted", "voting"
        weights={"nudenet_detector": 0.4, "vit_classifier": 0.6}
    )
    
    with open("test_images/xv_p.jpg", "rb") as f:
        result = await hybrid.predict(f.read(), threshold=0.5)
        
    print(f"✅ Hybrid → score: {result['score']}, label: {result['label']}")
    print(f"   latency: {result['latency_ms']} ms")
    print(f"   sources: {result['meta']['sources']}")
    print(f"   zones: {len(result['meta']['detected_zones'])} found")
    print(f'Errors: {result['meta']['errors']}')

if __name__ == "__main__":
    asyncio.run(main())