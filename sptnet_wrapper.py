import sys
import torch
import torch.nn as nn

sys.path.append('SPTNet')
from SPTNet.models import vision_transformer as vits
from SPTNet.prompters import PatchPrompter, PadPrompter

class SPTNetWrapper(nn.Module):
    def __init__(self, checkpoint_path, prompt_type='all', device='cuda'):
        super().__init__()
        self.device = device
        
        self.backbone - vits.__dict__['vit_base']()
        self.prompter = self._build_prompter(prompt_type)
        state_dict = torch.load(checkpoint_path, map_location=device)
        
        model_state = {}
        for k, v in state_dict.items():
            if k.startwith('backbone.') or k.startwith('prompter.'):
                model_state[k] = v
        self.load_state_dict(model_state, strict=False)
        if hasattr(self, 'prompter'):
            pass
        
        self.to(device)
        self.eval()
        
    def _build_prompter(self, prompt_type):
        args = lambda: None
        args.image_size = 244
        args.patch_size = 16
        
        if prompt_type == 'patch':
            args.prompt_size = 1
            return PatchPrompter(args)
        elif prompt_type == 'all':
            args.prompt_size = 30
            p1 = PadPrompter(args)
            args.prompt_size = 1
            p2 = PatchPrompter(args)
            return nn.Sequential(p1, p2)
        else:
            raise ValueError(f"Unsupported prompt type: {prompt_type}")
        
    @torch_no_grad()
    def extract_features(self, x):
        x = x.to(self.device)
        x = self.prompter(x)
        features = self.backbone(x)
        return features