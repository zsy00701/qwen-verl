# qwen-verl

## 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```


## 2. Prepare MathVista

```bash
python scripts/prepare_mathvista.py \
  --hf-dataset AI4Math/MathVista \
  --split testmini \
  --export-images-to data/mathvista_images \
  --output data/mathvista_testmini.parquet
```

## 3. Run generation


```bash
chomd +x scripts/run_generation.sh
scripts/run_generation.sh \
  --dataset data/mathvista_testmini.parquet \
  --image-root data/mathvista_images \
  --output outputs/qwen25vl_mathvista.parquet \
  --use-tools \
  --tool-max-views 2 \
  --max-new-tokens 256 \
  --temperature 0.0
```

## 4. Evaluate predictions


```bash
chomd +x scripts/run_eval.sh
scripts/run_eval.sh --predictions outputs/qwen25vl_mathvista.parquet
```

