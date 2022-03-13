# OOTD: Object-Oriented Text Dynamics

Python implementation for the paper [Learning Object-Oriented Dynamics for Planning from Text](https://openreview.net/pdf?id=B6EIcyp-Rb7). This code is based on the code of [GATA](https://github.com/xingdi-eric-yuan/GATA-public).

## Build the environment
```
# Dependencies
conda create -n oorl-public-venv python=3.7
conda activate oorl-public-venv
pip install --upgrade pip
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
python -m spacy download en

# Download FastText Word Embeddings
curl -L -o crawl-300d-2M.vec.h5 "https://bit.ly/2U3Mde2"
mv crawl-300d-2M.vec.h5 ./source/embeddings
```

## Prepare the dataset
- First TextWorld Problem competition.
- Download the pretraining datasets from https://aka.ms/twkg/sp.0.2.zip. Once downloaded, extract its contents into ```/source/dataset/sp.0.2/general```.


## Object-Supervised OOTD 
This code will use the games at difficulty level 3 in Textworld  as an example (here the difficulty level 3 correspond to the **difficulty level 1** in our paper).

### Graph Auto-Encoder
For your convenience, we provide an well-trained object extractor model in the ```./saved_models/graph_auto_encoder_pretrain/saved_model_ComplEx_seenGraph_Mar-04-2021.pt```. We shoe how to obtain this model in the following.
```
cd ./interface
python pretrain_graph_autoencoder.py ../configs/pretrain_graphs_random_encoder_seen.yaml
```

### Object extractor
For your convenience, we provide an well-trained object extractor model in the ```./saved_models/graph_dynamics_ims/saved_model_dynamic_linear_ComplEx_all-independent_label_df-3_sample-100_weight-0_May-12-2021.pt```. We will explain how to obtain this model in the following.
- Pretrain the deterministic object extractor with a general dataset.
```
cd ./interface
python predict_graph_dynamics_ims ../configs/predict_graphs_dynamics_linear_seen.yaml
```
- Fine-tune the model with dataset for the game in difficulty-level-3
```
cd ./interface
python predict_graph_dynamics_ims ../configs/predict_graphs_dynamics_linear_seen_fineTune_df3.yaml
```

