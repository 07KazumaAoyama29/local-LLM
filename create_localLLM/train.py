from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import json

# データの行数をカウント
data_count = 0
with open("data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data_count += 1

print(f">>> データ数: {data_count} 件")

# ロジック: データ数が多いほどエポック数を減らす
if data_count < 500:
    target_epochs = 5   # 中くらいなら標準的に
else:
    target_epochs = 3   # 多いならサラッと

print(f">>> 自動設定されたエポック数: {target_epochs}")

# === 設定 ===
# VRAM節約のため、まずは軽量なGemma-2-2Bを使用します
# 4060 Tiなら余裕で動きます
model_name = "unsloth/gemma-2-2b-it-bnb-4bit"
max_seq_length = 2048
output_dir = "outputs" # 結果の保存先

print(">>> モデルをロード中...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

# LoRAアダプターの設定
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
)

# データのロードとフォーマット
print(">>> データをロード中...")
dataset = load_dataset("json", data_files="data.jsonl", split="train")

prompt_style = """<start_of_turn>user
{}
{}<end_of_turn>
<start_of_turn>model
{}<end_of_turn>"""

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = prompt_style.format(instruction, input, output) + tokenizer.eos_token
        texts.append(text)
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

# 学習設定
print(">>> 学習開始...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = target_epochs,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = output_dir,
        optim = "adamw_8bit",
        report_to = "none", # WandBなどへの送信を無効化
    ),
)

trainer.train()

# GGUFへの変換と保存
print(">>> GGUF形式で保存中...")
# q4_k_m はバランスの良い量子化形式です
model.save_pretrained_gguf("my_model_gguf", tokenizer, quantization_method = "q4_k_m")

print(">>> 完了しました！ 'my_model_gguf' フォルダを確認してください。")