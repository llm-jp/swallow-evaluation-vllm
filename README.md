# swallow-eval-vllm

このリポジトリは、[swallow-evaluation](https://github.com/swallow-llm/swallow-evaluation/tree/04948a0e81075cc461b80e98ba2ce483d4edb0bc)のvLLM版を非公式で提供するものです。
vLLMを使用することで、より高速な評価を実現します。

## 評価スクリプトの実行方法

### 準備：環境構築

`swallow-evaluation-vllm/`にて

```bash
python -m venv .venv_harness_en
```

`swallow-evaluation/`にて

```bash
source .venv_harness_en/bin/activate
cd lm-evaluation-harness-en
pip install -e .
pip install lm-eval[vllm]
```

### 評価結果算出

```bash
python scripts/aggregate_result.py --model <checkpoint path>
python scripts/show_result.py --model-list <model_list path>   # 1 行 1 モデル名を並べたテキスト
```

## 実装の検証

`llm-jp/llm-jp-3-13b`モデルを使用して、オリジナルの実装とvLLM実装で同等の結果が得られることを確認しています。

### 英語タスクでの比較

| タスク名 | パラメータ | オリジナル実装 | vLLM実装 |
|---------|-----------|--------------|----------|
| TriviaQA | num_fewshot=4 | 0.6020 | 0.6043 |
| GSM8K | num_fewshot=4 | 0.1577 | 0.1668 |
| OpenBookQA | num_fewshot=4 | 0.3320 | 0.3240 |
| HellaSWAG | num_fewshot=4 | 0.5701 | 0.5684 |
| XWinograd | num_fewshot=4 | 0.9011 | 0.9062 |
| MMLU (平均) | num_fewshot=5 | 0.4625 | 0.4597 |
| BBH (CoT) | num_fewshot=3 | 0.4022 | 0.4052 |
