import collections
import logging
import os
import copy
import comfy

from safetensors.torch import load_file
from comfy.model_patcher import ModelPatcher


class FluxFillModelPatcher(ModelPatcher):
    FILL_PATCHES = None

    def process_key(self, key):
        # logging.info(f'self.fill_keys: {self.fill_keys}')
        return any(k in key for k in self.fill_keys)

    def apply_patch(self):
        for k, fill_weight in FluxFillModelPatcher.FILL_PATCHES.items():
            if not self.process_key(k):
                continue

            w = comfy.utils.get_attr(self.model, k)

            if k not in self.backup:
                self.backup[k] = collections.namedtuple(
                    "Dimension", ["weight", "inplace_update"]
                )(w.to(device=self.offload_device, copy=False), False)

            comfy.utils.set_attr_param(self.model, k, fill_weight)

        self.fill_patched = True
        logging.info(f"[flux fill inpaint lora] weights patched")

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        if not self.fill_patched:
            self.apply_patch()

        super().patch_weight_to_device(key, device_to, inplace_update)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        self.fill_patched = False
        logging.info(f"[flux fill inpaint lora] unpatch weights")

        super().unpatch_model(device_to, unpatch_weights)

    def clone(self, *args, **kwargs):
        n = FluxFillModelPatcher(
            self.model,
            self.load_device,
            self.offload_device,
            self.size,
            weight_inplace_update=self.weight_inplace_update,
        )
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]
        n.patches_uuid = self.patches_uuid

        n.object_patches = self.object_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        n.backup = self.backup
        n.object_patches_backup = self.object_patches_backup

        n.fill_patched = False
        n.fill_keys = getattr(self, "fill_keys", [])

        if not FluxFillModelPatcher.FILL_PATCHES:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            state = load_file(os.path.join(current_dir, "fill_state.safetensors"))

            FluxFillModelPatcher.FILL_PATCHES = {}
            for k in state:
                w = state[k]
                FluxFillModelPatcher.FILL_PATCHES[k] = w

        return n


class ApplyFluxFillPatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "img_in": ("BOOLEAN", {"default": True}),
                "txt_in": ("BOOLEAN", {"default": True}),
                "vector_in.in_layer": ("BOOLEAN", {"default": True}),
                "vector_in.out_layer": ("BOOLEAN", {"default": True}),
                "time_in.in_layer": ("BOOLEAN", {"default": True}),
                "time_in.out_layer": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    CATEGORY = "inpaint"
    FUNCTION = "patch"

    def patch(self, model, **kwargs):
        m = FluxFillModelPatcher.clone(model)
        m.fill_keys = [k for k, v in kwargs.items() if v]
        return (m,)


NODE_CLASS_MAPPINGS = {
    "ApplyFluxFillPatch": ApplyFluxFillPatch,
}
