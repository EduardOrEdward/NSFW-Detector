import asyncio
from opennsfw2_detector import OpenNSFW2Detector
from nudenet_detector import NudeNetDetector
from hybrid import HybridDetector

async def main():
    nudenet = NudeNetDetector(model_path="data/models/nudenet/nudenet.onnx")
    opennsfw2 = OpenNSFW2Detector()
    
    hybrid = HybridDetector(
        detectors=[nudenet, opennsfw2],
        strategy="max",  # или "weighted", "voting"
        weights={"nudenet_detector": 0.4, "opennsfw2_detector": 0.6}
    )
    
    with open("test_images/porn.jpg", "rb") as f:
        result = await hybrid.predict(f.read(), threshold=0.5)
        
    print(f"✅ Hybrid → score: {result['score']}, label: {result['label']}")
    print(f"   latency: {result['latency_ms']} ms")
    print(f"   sources: {result['meta']['sources']}")
    print(f"   zones: {len(result['meta']['detected_zones'])} found")
    print(f'Detected zones: {result['meta']['detected_zones']}')
    print(f'Errors: {result['meta']['errors']}')

if __name__ == "__main__":
    asyncio.run(main())