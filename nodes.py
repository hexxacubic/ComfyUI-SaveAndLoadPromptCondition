import os
import random

import folder_paths
from PIL import Image
import numpy as np
import torch
import hashlib


class SaveConditioning:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"conditionings": ("CONDITIONING", ),},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_conditioning"
    OUTPUT_NODE = True
    CATEGORY = "endman100"

    def save_conditioning(self, conditionings): # conditionings : [[text, {"pooled_output"}]...]
        results = list()
        for (batch_number, conditioning) in enumerate(conditionings):
            save_path = os.path.join(self.output_dir, f"{batch_number:05}_conditionings.bin")
            print(conditioning[0].shape, conditioning[1]["pooled_output"].shape, save_path)
            torch.save(conditioning, save_path)

        return { "ui": { "conditionings": results } }

class LoadContditioning():
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".bin")]
        return {"required": {"conditioning": [sorted(files), ]}, }

    CATEGORY = "endman100"
    RETURN_TYPES = ("CONDITIONING", )
    FUNCTION = "load_conditioning"

    def load_conditioning(self, conditioning):
        conditioning_path = folder_paths.get_annotated_filepath(conditioning)
        conditioning_list = torch.load(conditioning_path)
        conditioning_list[0] = conditioning_list[0].cpu()
        conditioning_list[1]["pooled_output"] = conditioning_list[1]["pooled_output"].cpu()
        print(conditioning_list)
        return ([conditioning_list], )

    @classmethod
    def IS_CHANGED(s, conditioning):
        image_path = folder_paths.get_annotated_filepath(conditioning)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, conditioning):
        if not folder_paths.exists_annotated_filepath(conditioning):
            return "Invalid latent file: {}".format(conditioning)
        return True
    

NODE_CLASS_MAPPINGS = {
    "SaveConditioning": SaveConditioning,
    "LoadContditioning": LoadContditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Save Conditioning": "SaveConditioning",
    "Load Contditioning": "LoadContditioning"
}