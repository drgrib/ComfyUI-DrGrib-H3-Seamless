import os
import sys
import torch
import comfy.nested_tensor
from comfy_extras.nodes_minimax_h3 import align_frame_count, video_latent_t

# Safely route to the upstream H3 Inpaint Tools folder so we can borrow their tensor loaders
h3_tools_path = os.path.join(os.path.dirname(__file__), "..", "ComfyUI-MiniMax-H3-Inpaint-Tools")
if h3_tools_path not in sys.path:
    sys.path.append(h3_tools_path)
    
import _comfy_bridge as bridge
from _core_import import core

load_tensors = core.load_tensors
FPS = 24
AUDIO_LATENT_FPS = 40

def _as_nested(samples, label):
    if not getattr(samples, "is_nested", False):
        raise ValueError(f"{label} is not a nested tensor.")
    tensors = samples.unbind()
    return tensors[0], tensors[1]

class DrGrib_H3LoadLatentAbsolute:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'latent': ('STRING', {'default': ''})}}

    RETURN_TYPES = ('LATENT', 'STRING')
    RETURN_NAMES = ('samples', 'prompt')
    FUNCTION = 'run'
    CATEGORY = 'DrGrib/Video'

    @classmethod
    def IS_CHANGED(cls, latent):
        return latent

    @staticmethod
    def _resolve(name):
        if os.path.isabs(name):
            return name
        root = os.path.realpath(bridge.output_directory())
        return os.path.realpath(os.path.join(root, name))

    def run(self, latent):
        path = self._resolve(latent)
        tensors, nested, prompt, _meta = load_tensors(path)
        if nested:
            return ({'samples': bridge.make_nested(tensors)}, prompt)
        return ({'samples': tensors[0]}, prompt)

class DrGrib_H3SeamlessContinuation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "overlap_frames": ("INT", {"default": 39, "min": 5, "max": 362, "step": 1}),
                "feather_frames": ("INT", {"default": 8, "min": 0, "max": 39, "step": 1}),
                "lock_audio": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "prev_latent": ("LATENT", {}),
                "target_latent": ("LATENT", {}),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "report")
    FUNCTION = "continue_latent"
    CATEGORY = "DrGrib/Video"

    def continue_latent(self, overlap_frames=39, feather_frames=8, lock_audio=False, prev_latent=None, target_latent=None):
        prev_video, prev_audio = _as_nested(prev_latent.get("samples"), "prev_latent")
        tgt_video, tgt_audio = _as_nested(target_latent.get("samples"), "target_latent")

        overlap_aligned = align_frame_count(max(5, int(overlap_frames)))
        k_v = min(video_latent_t(overlap_aligned), prev_video.shape[2], tgt_video.shape[2])
        k_a = min(round(overlap_aligned / FPS * AUDIO_LATENT_FPS), prev_audio.shape[-1], tgt_audio.shape[-1])

        new_video = tgt_video.clone()
        new_video[:, :, :k_v] = prev_video[:, :, -k_v:]
        new_audio = tgt_audio.clone()
        if k_a > 0:
            new_audio[:, :, :, :k_a] = prev_audio[:, :, :, -k_a:]

        new_samples = comfy.nested_tensor.NestedTensor((new_video, new_audio))

        video_mask = torch.ones_like(new_video[:, :1], dtype=torch.float32)
        video_mask[:, :, :k_v] = 0.0
        
        if feather_frames > 0 and k_v > feather_frames:
            for i in range(feather_frames):
                val = (i + 1) / (feather_frames + 1)
                video_mask[:, :, k_v - feather_frames + i] = val

        if lock_audio:
            audio_mask = torch.zeros_like(new_audio[:, :1], dtype=torch.float32)
        else:
            audio_mask = torch.ones_like(new_audio[:, :1], dtype=torch.float32)
            if k_a > 0:
                audio_mask[:, :, :, :k_a] = 0.0

        out = {k: v for k, v in target_latent.items()}
        out["samples"] = new_samples
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))

        return (out, "DrGrib Seamless Continuation Applied")

NODE_CLASS_MAPPINGS = {
    "DrGrib_H3LoadLatentAbsolute": DrGrib_H3LoadLatentAbsolute,
    "DrGrib_H3SeamlessContinuation": DrGrib_H3SeamlessContinuation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DrGrib_H3LoadLatentAbsolute": "DrGrib H3 Load Latent (Absolute Path)",
    "DrGrib_H3SeamlessContinuation": "DrGrib H3 Seamless Continuation (Feathered)",
}