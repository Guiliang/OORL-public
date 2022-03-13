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
**Important Reminder:**
- This code will use the games at difficulty level 3 in Textworld  as an example (here the difficulty level 3 correspond to the **difficulty level 1** in our paper).
- For the rest of the code, you can always add ```-d 1``` after the command line (e.g., ```python run.py -d 1```). It will start the program under a debugging model, allowing debugging on the local machine with a limited GPU power. 

### Graph Auto-Encoder
We provide a well-trained object extractor model in the ```./saved_models/graph_auto_encoder_pretrain/saved_model_ComplEx_seenGraph_Mar-04-2021.pt```. This model is trained by running:
```
cd ./interface
python pretrain_graph_autoencoder.py ../configs/pretrain_graphs_random_encoder_seen.yaml
```

### Object extractor
We provide a well-trained object extractor model in the ```./saved_models/graph_dynamics_ims/saved_model_dynamic_linear_ComplEx_all-independent_label_df-3_sample-100_weight-0_May-12-2021.pt```. We will explain how to obtain this model in the following.
- Pretrain the deterministic object extractor with a general dataset.
```
cd ./interface
python predict_graph_dynamics_ims ../configs/predict_graphs_dynamics_linear_seen.yaml
```
- Fine-tune the model for the games in difficulty-level-3
```
cd ./interface
python predict_graph_dynamics_ims ../configs/predict_graphs_dynamics_linear_seen_fineTune_df3.yaml
```

### OOTD Transition and reward Model

We provide a well-trained OOTD Transition model in the ```./saved_models/graph_dynamics_ims/saved_model_dynamic_linear_ComplEx_all-independent_latent_df-3_sample-100_weight-1_May-10-2021.pt``` and a well-trained reward predictor in ```../reward_predictor/difficulty_level_3/saved_model_Dynamic_Reward_Predictor_Goal_Linear_Jun-13-2021_real_goal.pt```. There are two options of training these models. 
1. A faster version by using a pre-collected dataset. The dataset is based on [First TextWorld Problem (FTWP)](https://competitions.codalab.org/competitions/21557) competition.
   - Pre-train the OOTD model with a general dataset.
    ```
    cd ./interface
    python 
    python predict_graph_dynamics_ims.py ../configs/predict_graphs_dynamics_linear_latent_loss_seen.yaml
    ```
   - Fine-tune the OOTD model for the games in the difficulty-level-3.
   ```
   cd ./interface
   python predict_graph_dynamics_ims.py ../configs/predict_graphs_dynamics_linear_latent_loss_seen_fineTune_df3.yaml
   ```
   - Train the reward model for the games in the difficulty-level-3.
   ```
   cd ./interface
   python pretrain_reward_predictor_dynamic_goal.py ../configs/pretrain_reward_predictor_dynamic_goal_df3.yaml
   ```
2. A slower version by collecting samples from the environment with Dyna-Q. This is used to show the training plots (e.g., Figure 3 in our paper).  
**Warning: This method will update the transition model, the reward model and a Q function together. It requires a large GPU memory.**
```
cd ./interface
python train_rl_with_supervised_planning.py ../configs/train_rl_with_planning_df3.yaml
```
