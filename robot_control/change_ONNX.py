import torch
from stable_baselines3 import PPO
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "..", "ppo_inverted_pendulum.zip")


# 1. 今まで学習したモデルを読み込む
model = PPO.load(model_path)

# 2. ダミーの入力データを作る（観測値が5つなので [1, 5] の形）
# AIに「とりあえず5つの数字が入ってくるよ」と教えるためのものです
dummy_input = torch.randn(1, 5)

# 3. ONNX形式で書き出す
torch.onnx.export(
    model.policy,                  # 脳みそ本体
    dummy_input,                   # 入力の形
    "robot_brain.onnx",            # 保存ファイル名
    verbose=True,
    input_names=['input'],         # 入力に名前をつける
    output_names=['output'],        # 出力に名前をつける
    opset_version=11               # 互換性のためのバージョン
)

print("ONNXモデルの書き出しが完了しました！")