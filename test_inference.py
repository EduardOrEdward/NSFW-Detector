from pathlib import Path
import sys,time
sys.path.append(str(Path(__file__).parent))
from inference import NudeNetDetector

MODEL_PATH = 'data/models/nudenet.onnx'

detector = NudeNetDetector(model_path=MODEL_PATH)

TEST_FILES = {
    "sfw.jpg": "expected_sfw",
    "xv_p.jpg": "expected_nsfw",
    "edge_art.jpg": "expected_edge"
}

for fname,excepted in TEST_FILES.items():
    with open(fname,'rb') as f:
        res = detector.predict(f.read())
    print(f"📄 {fname}")
    print(f"   → label: {res['label']}, score: {res['nsfw_score']}")
    print(f"   → zones: {[z['zone'] for z in res['detected_zones']]}")
    print(f"   → latency: {res['latency_ms']} ms\n")