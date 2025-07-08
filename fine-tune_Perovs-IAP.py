from __future__ import annotations

import os
import dgl
import shutil
import warnings
import logging
import numpy as np
import pandas as pd
import ast
from pymatgen.core.structure import Structure
import torch
import pytorch_lightning as pl
from functools import partial
from pytorch_lightning.loggers import CSVLogger

# Import updated MGLDataset, MGLDataLoader, and collate_fn_pes
from mgl_data_utils_1 import MGLDataset, MGLDataLoader, collate_fn_pes

import matgl
from matgl.ext.pymatgen import Structure2Graph
from matgl.models import M3GNet
from matgl.utils.training import PotentialLightningModule
from matgl.config import DEFAULT_ELEMENTS


# Display DGL version
print(dgl.__version__)

# Set up logging to file and console
log_file = "training.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler(log_file),
    logging.StreamHandler()
])
logger = logging.getLogger(__name__)

logger.info("Script started...")

# To use GPU
torch.set_default_device("cuda")
logger.info(f"Is CUDA available: {torch.cuda.is_available()}")

# Set CUDA as the default tensor type
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('medium')

# To suppress warnings for clearer output
warnings.simplefilter("ignore")

# Load dataset
df = pd.read_csv('./path_to_your_csv.csv')
df['Forces_array'] = df['Forces'].apply(lambda x: np.array(ast.literal_eval(x))) # Read force matrix
df['Stresses_array'] = df['Stress'].apply(lambda x: np.array(ast.literal_eval(x)) * -0.1) # Read stress matrix

structures = [Structure.from_dict(ast.literal_eval(s)) for s in df['Structure']] #Read structure
structure_names = df['Structure Name'].tolist()
logger.info(f"Total number of structures: {len(structures)}")
logger.info(f"Number of structure names: {len(structure_names)}")

energies = df['Energy'].tolist()
forces = [f.tolist() for f in df['Forces_array']]
stresses = [s.tolist() for s in df['Stresses_array']]

# Define labels excluding structure names
labels = {
    "energies": energies,
    "forces": forces,
    "stresses": stresses
}

# Set up structure converter
element_types = DEFAULT_ELEMENTS
converter = Structure2Graph(element_types=element_types, cutoff=5.0)

# Define directory for dataset
save_dir = './mgl_dataset'
os.makedirs(save_dir, exist_ok=True)

# Create dataset with structure names
dataset = MGLDataset(
    threebody_cutoff=4.0,
    structures=structures,
    structure_names=structure_names,
    converter=converter,
    labels=labels,
    include_line_graph=True,
    save_dir=save_dir
)

# Split dataset into train, validation, and test
train_data, val_data, test_data = dgl.data.utils.split_dataset(
    dataset,
    frac_list=[0.8, 0.1, 0.1],
    shuffle=True,
    random_state=42,
)

## Get indices and structure names for each split (Optional to print the train, test and val entries)
#train_names = [structure_names[i] for i in train_data.indices]
#val_names = [structure_names[i] for i in val_data.indices]
#test_names = [structure_names[i] for i in test_data.indices]

## Save the structure names to separate CSV files (Optional to print the train, test and val entries)
#pd.DataFrame(train_names, columns=["Structure Name"]).to_csv(os.path.join(save_dir, "train_names.csv"), index=False)
#pd.DataFrame(val_names, columns=["Structure Name"]).to_csv(os.path.join(save_dir, "val_names.csv"), index=False)
#pd.DataFrame(test_names, columns=["Structure Name"]).to_csv(os.path.join(save_dir, "test_names.csv"), index=False)

## Log the structure names save completion
#logger.info("Structure names saved to train_names.csv, val_names.csv, and test_names.csv")

# Set up collate function
my_collate_fn = partial(collate_fn_pes, include_line_graph=True, include_stress=True)

# Create CUDA generator
generator = torch.Generator(device="cuda")
generator.manual_seed(42)
logger.info(f"Generator device: {generator.device}")

# Initialize data loaders
train_loader, val_loader, test_loader = MGLDataLoader(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    collate_fn=my_collate_fn,
    batch_size=16,
    num_workers=0,
    generator=generator,
)

# Load pre-trained M3GNet model
m3gnet_nnp = matgl.load_model("M3GNet-MP-2021.2.8-PES")
model_pretrained = m3gnet_nnp.model
property_offset = m3gnet_nnp.element_refs.property_offset

# Set up lightning module for fine-tuning
lit_module_finetune = PotentialLightningModule(
    model=model_pretrained,
    element_refs=property_offset,
    lr=5e-4,
    include_line_graph=True,
    stress_weight=0.01
)

# Set up trainer
logger.info("Setting up trainer...")
logger_csv = CSVLogger("logs", name="M3GNet_finetuning")
trainer = pl.Trainer(max_epochs=550, accelerator="gpu", devices=1, logger=logger_csv, inference_mode=False)
logger.info("Starting training...")
trainer.fit(model=lit_module_finetune, train_dataloaders=train_loader, val_dataloaders=val_loader)

# Save trained model
model_save_path = "./Perovs-IAP_2025_PES/"
lit_module_finetune.model.save(model_save_path)
logger.info(f"Model saved to {model_save_path}")




