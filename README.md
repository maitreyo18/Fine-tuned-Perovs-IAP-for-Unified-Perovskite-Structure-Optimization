# Project Description
This repository contains a fine-tuned M3GNet IAP for halide perovskites (HaPs), which is generalizable across bulk alloys, defects, impurities, and surfaces. The model is fine-tuned on a large-scale in-house DFT dataset from Mannodi group, with energies, forces, and stresses, enabling accurate predictions and geometry optimization across diverse structural motifs. It enables efficient exploration of the perovskite potential energy surface for energy prediction, defect screening, and surface relaxation. Pretrained model files and datasets are included for easy deployment and fine-tuning.

 
# Data availability
Intermediate structures and train-val-test data: https://zenodo.org/records/15832377  <br>
Saved model checkpoint: ./Perovs-IAP_finetuned_bulk+defect+surface_2025_PES

# Installation and usage
• Follow example_usage.ipynb for installation and test usage. For further details, visit the original MatGL documentation: https://matgl.ai. Github: https://github.com/materialsvirtuallab/matgl
• For GPU-enabled fine-tuning use the ./fine-tune_Perovs-IAP.py script. Use the corresponding data collected from Zenodo. 
• The relaxed structures used in the three test cases have been included inside ./Structures_test_cases directory.

