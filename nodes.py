import os
import random

import folder_paths
from PIL import Image
import numpy as np
import torch
import hashlib
conditioning_dir = os.path.join(folder_paths.models_dir, "conditioning")
if not os.path.exists(conditioning_dir):
    os.makedirs(conditioning_dir)

class SaveConditioning:
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
            save_path = os.path.join(conditioning_dir, f"{batch_number:05}_conditionings.bin")
            print(conditioning[0].shape, conditioning[1]["pooled_output"].shape, save_path)
            torch.save(conditioning, save_path)

        return { "ui": { "conditionings": results } }

class LoadContditioning():
    @classmethod
    def INPUT_TYPES(s):
        files = []
        for root, dirs, files in os.walk(conditioning_dir):
            for file in files:
                if file.endswith(".bin"):
                    file_path = os.path.join(root, file)
                    files.append(file_path)
        return {"required": {"conditioning": [sorted(files), ]}, }

    CATEGORY = "endman100"
    RETURN_TYPES = ("CONDITIONING", )
    FUNCTION = "load_conditioning"

    def load_conditioning(self, conditioning):
        conditioning_path = os.path.join(conditioning_dir, conditioning)
        conditioning_list = torch.load(conditioning_path)
        conditioning_list[0] = conditioning_list[0].cpu()
        conditioning_list[1]["pooled_output"] = conditioning_list[1]["pooled_output"].cpu()
        return ([conditioning_list], )

    @classmethod
    def IS_CHANGED(s, conditioning):
        conditioning_path = os.path.join(conditioning_dir, conditioning)
        m = hashlib.sha256()
        with open(conditioning_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, conditioning):
        conditioning_path = os.path.join(conditioning_dir, conditioning)
        if not os.path.exists(conditioning_path):
            return "Invalid conditioning file: {}".format(conditioning_path)
        return True
    

NODE_CLASS_MAPPINGS = {
    "SaveConditioning": SaveConditioning,
    "LoadContditioning": LoadContditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Save Conditioning": "SaveConditioning",
    "Load Contditioning": "LoadContditioning"
}