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
