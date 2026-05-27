# CofCED_running codes

## 1. Installing requirement packages
```
conda create -n fact22 python=3.8
source activate fact22
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install transformers pandas==1.1.2 tqdm==4.50.0 nltk==3.5 rouge-score==0.0.4 sklearn
pip install sentence_transformers   # for evaluation
pip install torch>=1.8
```
Tips: - Adding a `logs` dir to the path of `datasets`. 

## 2. Follow the guide to download the datasets and put them in the correct location. 

## 3. Run the code
It is recommended to run on linux servers with the following script: 
`python train_exp_fc5_xxx.py`

## 4. Build source-weighted competitive evidence features
This project also includes an incremental evidence feature pipeline under
`Codes/competitive_evidence/`. It keeps the original CofCED model untouched and
exports source-weighted support/refute evidence pools for later ablation or model
integration.

LIAR-RAW example:
```
python Codes/competitive_evidence/build_features.py --input Codes/dataset/LIAR-RAW/train.json --output Codes/dataset/features/liar_raw_train_competitive.json
```

RAWFC example:
```
python Codes/competitive_evidence/build_features.py --input Codes/dataset/RAWFC/test --output Codes/dataset/features/rawfc_test_competitive.json
```

Quick smoke test:
```
python Codes/competitive_evidence/build_features.py --input Codes/dataset/LIAR-RAW/train.json --output Codes/dataset/features/liar_raw_smoke_competitive.json --limit 3
```

Build all LIAR-RAW feature splits used by the competitive CofCED pipeline:
```
python Codes/competitive_evidence/build_features.py --input Codes/dataset/LIAR-RAW/train.json --output Codes/dataset/features/liar_raw_train_competitive.json
python Codes/competitive_evidence/build_features.py --input Codes/dataset/LIAR-RAW/val.json --output Codes/dataset/features/liar_raw_val_competitive.json
python Codes/competitive_evidence/build_features.py --input Codes/dataset/LIAR-RAW/test.json --output Codes/dataset/features/liar_raw_test_competitive.json
```

After training `train_exp_fc5_LIAR_RAW2.py`, export six-class test predictions:
```
python Codes/predict_liar_raw_competitive.py --model path/to/best_model.pt --data Codes/dataset/LIAR-RAW/test.json --features Codes/dataset/features/liar_raw_test_competitive.json --output Codes/dataset/features/predictions_liar_raw_competitive.jsonl
```

## 5. Please cite this paper as follows （BibTeX）: 
```
@inproceedings{yang2022cofced,
  title={A Coarse-to-fine Cascaded Evidence-Distillation Neural Network for Explainable Fake News Detection},
  author={Yang, Zhiwei and Ma, Jing and Chen, Hechang and Lin, Hongzhan and Luo, Ziyang and Chang Yi},
  booktitle={Proceedings of the 29th International Conference on Computational Linguistics (COLING)},
  pages={2608--2621},
  month={oct},
  year={2022},
  url={https://aclanthology.org/2022.coling-1.230},
}
```

PDF: https://aclanthology.org/2022.coling-1.230.pdf
