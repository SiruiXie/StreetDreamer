import torch
from PIL import Image

path = '/data/xiesr/lucidsim/LucidSimNana/LucidSim/LucidDreamer/teaser_images/240915_005755_gsplat_rgb.png'

d_model = torch.hub.load('/data/xiesr/lucidsim/LucidSimNana/LucidSim/LucidDreamer/ZoeDepth', 'ZoeD_K', source='local', pretrained=True, trust_repo=True, skip_validation=True,).to('cuda')

image = Image.open(path)
depth = d_model.infer_pil(image)

depth = depth


