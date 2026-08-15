import torch
from PIL import Image
from multimodal.vision_adapter import VisionPatchExtractor, preprocess_image, VisionLanguageAdapter
from quantization.quantize import get_model_size_mb, quantize_model_to_int4
from evaluation.safety import SafetyEvaluator

def test_real_multimodal_patch_extraction():
    dummy_img = Image.new('RGB', (224, 224), color = 'red')
    tensor = preprocess_image(dummy_img, size=224)
    extractor = VisionPatchExtractor(patch_size=16, in_channels=3, vision_dim=512)
    adapter = VisionLanguageAdapter(vision_dim=512, llm_dim=768)
    
    patches = extractor(tensor)
    projected = adapter(patches)
    
    assert projected.shape == (1, 196, 768), "Vision-Language projection failed spatial mapping!"

def test_int4_quantization_simulation():
    layer = torch.nn.Linear(100, 100)
    fp32_size = get_model_size_mb(layer)
    int4_model = quantize_model_to_int4(layer)
    int4_size = get_model_size_mb(int4_model)
    
    assert int4_size < fp32_size
    assert round(fp32_size / int4_size) == 8 

def test_safety_hallucination_detection():
    evaluator = SafetyEvaluator()
    context = "The capital of France is Paris."
    good_ans = "Paris is the capital."
    bad_ans = "Tokyo is a great city."
    
    assert evaluator.evaluate_hallucination(context, good_ans)["hallucination_detected"] is False
    assert evaluator.evaluate_hallucination(context, bad_ans)["hallucination_detected"] is True
