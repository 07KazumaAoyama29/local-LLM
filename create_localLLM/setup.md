local LLM を構築するときのセットアップ方法です。

# Step 1: 環境構築 (WSL2上のターミナルで実行)

### 1. Minicondaのインストール（もし入っていなければ）
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc

### 2. 学習用仮想環境の作成（Python 3.10推奨）
conda create --name unsloth_env python=3.10 pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers -y
conda activate unsloth_env

### 3. Unslothのインストール
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes

# Step 2: 学習スクリプトの作成

ローカルで動くPythonファイル train.py を作成します。 

作業ディレクトリに data.jsonl（先ほどの10件データ）がある状態で実行してください。



