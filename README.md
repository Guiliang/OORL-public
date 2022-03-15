# OOTD: Object-Oriented Text Dynamics

Python implementation for the paper [Learning Object-Oriented Dynamics for Planning from Text](https://openreview.net/pdf?id=B6EIcyp-Rb7). This code is based on [GATA](https://github.com/xingdi-eric-yuan/GATA-public).

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

## Important Reminder:
- This code will use the games at **difficulty level 3 in Textworld** as an example (here the **difficulty level 3** corresponds to the **difficulty level 1 in our paper**).
- For the rest of the code, you can always add ```-d 1``` after the command line (e.g., ```python run.py -d 1```). It will start the program under a debugging model, allowing debugging on the local machine with a limited GPU power. 
- If you prefer checking the planning results, please directly go to **Part III**. We have provide some pre-trained models in this repo.

## Prepare the dataset
- Download the pretraining datasets from [link](https://drive.google.com/file/d/1AKAn33_e5ndOQqs9bPbx_IXSAMHTDymE/view?usp=sharing). Once downloaded, extract its contents into ```/source/dataset/```.
- This dataset applies the games at difficulty level 3 in Textworld  as an example. The dataset is based on The dataset is based on [First TextWorld Problem (FTWP)](https://competitions.codalab.org/competitions/21557) competition. The dataset for other games can be generated from FTWP dataset (```wget https://bit.ly/2Mb4CBR -O rl.0.2.zip```) (we will revise and release the data generation code in the future.)

## Part I: Object-Supervised (OS)-OOTD

### Graph Auto-Encoder
We provide a well-trained graph auto-encoder in the ```./saved_models/graph_auto_encoder_pretrain/saved_model_ComplEx_seenGraph_Mar-04-2021.pt```. This model is trained by running:
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

### OOTD Transition and Reward Model

We provide a well-trained OOTD Transition model in the ```./saved_models/graph_dynamics_ims/saved_model_dynamic_linear_ComplEx_all-independent_latent_df-3_sample-100_weight-1_May-10-2021.pt``` and a well-trained reward predictor in ```../reward_predictor/difficulty_level_3/saved_model_Dynamic_Reward_Predictor_Goal_Linear_Jun-13-2021_real_goal.pt```. There are two options of training these models. 
1. A faster version by using a pre-collected dataset.
   - Pre-train the OOTD transition model with a general dataset.
    ```
    cd ./interface
    python 
    python predict_graph_dynamics_ims.py ../configs/predict_graphs_dynamics_linear_latent_loss_seen.yaml
    ```
   - Fine-tune the OOTD transition model for the games in the difficulty-level-3.
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

## Part II: Self-Supervised (SS)-OOTD
**Important Reminder:**
- This code will use the games at difficulty level 3 in Textworld  as an example (here the difficulty level 3 correspond to the **difficulty level 1** in our paper).
- For the rest of the code, you can always add ```-d 1``` after the command line (e.g., ```python run.py -d 1```). It will start the program under a debugging model, allowing debugging on the local machine with a limited GPU power. 

### Graph Auto-Encoder
We provide a well-trained graph auto-encoder in the ```./saved_models/graph_auto_encoder_pretrain/saved_model_ComplEx_seenGraph_Mar-04-2021.pt```. This model is trained by running:
```
cd ./interface
python pretrain_graph_autoencoder.py ../configs/pretrain_graphs_random_encoder_seen.yaml
```

### OOTD Transition and Reward models
We provide a well-trained , There are two options of training these models. 
1. A faster version by using a pre-collected dataset. The dataset is based on [First TextWorld Problem (FTWP)](https://competitions.codalab.org/competitions/21557) competition.
```
cd ./interface
python predict_graph_dynamics_unspervised_ims_goal.py ../configs/predict_unsupervised_graphs_dynamics_linear_latent_loss_seen_semi_goal_df3.yaml
```
2. A slower version by collecting samples from the environment with Dyna-Q. This is used to show the training plots (e.g., Figure 3 in our paper).  
**Warning: This method will update the transition model, the reward model and a Q function together. It requires a large GPU memory.**
```
cd ./interface
python train_rl_with_unsupervised_planning.py ../configs/train_rl_with_planning_unsupervised_semi_goal_dynamics_df3.yaml
```

## Part III: Planning

### Planning with MCTS and OS-OOTD
```
python test_rl_planning.py ../configs/test_rl_with_planning_df3.yaml -d 1 -t 0 --test_mode mcts
```

### Planning with MCTS and SS-OOTD
```
python test_rl_planning.py ../configs/test_rl_with_planning_unsupervised_semi_goal_dynamics_df3.yaml -d 1 -t 0 --test_mode mcts
```

### Planning with MCTS + DQN
```
python test_rl_planning.py ../configs/train_rl_with_planning_df3.yaml -d 1 --test_mode mcts -t 0 -f df-3-mem-300000-epi-20000-maxstep-200-anneal-0.3-cstr_scratch_neg_reward-me-0.3-seed-123_Sept-26-2021_best
```

If you can get some help from this code, please use the following bibtex:
```
@inproceedings{
liu2022oorl,
title={Learning Object-Oriented Dynamics for Planning from Text},
author={Guiliang Liu and Ashutosh Adhikari and Amir-massoud Farahmand and Pascal Poupart},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=B6EIcyp-Rb7}
}
```