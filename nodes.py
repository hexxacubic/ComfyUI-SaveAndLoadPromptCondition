import os
import random

import folder_paths
from PIL import Image
import numpy as np
import torch
import hashlib

# set the models directory
if "conditionings" not in folder_paths.folder_names_and_paths:
    current_paths = [os.path.join(folder_paths.models_dir, "conditionings")]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["conditionings"]
folder_paths.folder_names_and_paths["conditionings"] = (current_paths, ".bin")

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
        print(folder_paths.get_filename_list("conditionings"))
        return {"required": { "conditionings": (folder_paths.get_filename_list("conditionings"), )}}

    CATEGORY = "endman100"
    RETURN_TYPES = ("CONDITIONING", )
    FUNCTION = "load_conditioning"

    def load_conditioning(self, conditioning):
        conditioning_path = folder_paths.get_full_path("ipadapter", conditioning)
        conditioning_list = torch.load(conditioning_path)
        conditioning_list[0] = conditioning_list[0].cpu()
        conditioning_list[1]["pooled_output"] = conditioning_list[1]["pooled_output"].cpu()
        return ([conditioning_list], )

    @classmethod
    def IS_CHANGED(s, conditioning):
        conditioning_path = folder_paths.get_full_path("ipadapter", conditioning)
        m = hashlib.sha256()
        with open(conditioning_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, conditioning):
        conditioning_path = folder_paths.get_full_path("ipadapter", conditioning)
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