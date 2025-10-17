# Project Description
This repository contains a fine-tuned M3GNet-IAP for halide perovskites (HaPs), which is generalizable across bulk alloys, defects, impurities, and surfaces. The model is fine-tuned on a large-scale in-house DFT dataset from Mannodi group, with energies, forces, and stresses, enabling accurate predictions and geometry optimization across diverse structural motifs. It enables efficient exploration of the perovskite potential energy surface for energy prediction, bulk alloy screening, defect screening, and surface reconstructions. Pretrained model checkpoint and datasets are included for easy deployment and fine-tuning.

 
# Data availability
Intermediate structures and train-val-test data:  https://zenodo.org/records/17363611  <br>
Saved model checkpoints: 
(i) ./Perovs-IAP_finetuned_bulk+defect+surface_2025_PES (fine-tuned on bulk+defects+surfaces ~12,0000 structures, RECOMMENDED for downstream tasks)
(ii)./Additional_fine-tuned_IAPs/Perovskites_2025_PES_bulk+defects (trained on bulk+defect ~10,000 structures)
(iii) ./Additional_fine-tuned_IAPs/Perovskites_2025_PES_unsampled (trained on the whole dataset without any sampling ~37,200 structures)


# Installation and usage
• Follow example_usage.ipynb for installation and test usage. For further details, visit the original MatGL documentation: https://matgl.ai. Github: https://github.com/materialsvirtuallab/matgl  <br>
• For GPU-enabled fine-tuning use the ./fine-tune_Perovs-IAP.py script. Use the corresponding data collected from Zenodo.  <br> 
• The relaxed structures used in the three test cases have been included inside ./Structures_test_cases directory.

