local LLM を構築するときのセットアップ方法です。

# Step 1: 環境構築 (WSL2上のターミナルで実行)

### 1. Minicondaのインストール（もし入っていなければ）
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### 2. 学習用仮想環境の作成（Python 3.10推奨）
```bash
conda create --name unsloth_env python=3.10 pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers -y
conda activate unsloth_env
```

### 3. Unslothのインストール
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes
```

### 4. torchaoのアンインストール
```bash
pip uninstall torchao -y
```

### 5. パッケージリストを更新
```bash
sudo apt update
```

### 6. ビルドツール一式（gccなど）をインストール
```bash
sudo apt install build-essential -y
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install rich
```

### 7. GPUが認識されているかの確認
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); import unsloth; print('Unsloth loaded successfully')"
```


# Step 2: 学習スクリプトの作成

### 1. 作業ディレクトリの作成
```bash
mkdir localllm
cd localllm
```

ローカルで動くPythonファイル train.py を作成します。 

作業ディレクトリに data.jsonl（先ほどの10件データ）がある状態で実行してください。

powershell.exe -Command "cd 'C:\\Users\\akazu\\llm\\local-LLM'; ollama create gemma-2-2b-it.Q4_K_M.gguf -f Modelfile"



