# OOTD: Object-Oriented Text Dynamics

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