# MTCIR Codebase

This repository contains the code for the paper *Distilling Slow Reasoning into Fast Retrieval: A Benchmark and Framework for User-Guided Conversational Image Retrieval*.

## 1. Repository Structure

```text
repo/
├── Gallery_images/
├── data/
│   ├── gallery/
│   ├── metadata_release_10k/
│   ├── test/
│   └── train/
├── evaluation/
│   ├── retrieval/
│   └── simuser/
├── patches/
│   └── ms_swift/
├── src/
│   └── baselines/
│       ├── gme/
│       └── qwen3_vl_embedding/
└── templates/
    ├── multi_target/
    └── single_target/
```

## 2. Environment and Dependencies

Dependency file:

- [requirements.txt](requirements.txt)

Recommended installation:

```bash
pip install -r requirements.txt
```

Some packages are sensitive to CUDA, PyTorch, and hardware settings, including `torch`, `torchvision`, `torchaudio`, `flash_attn`, `deepspeed`, and `bitsandbytes`. Adjust versions if required by your local CUDA environment.

SimUser-based retrieval uses Gemini 2.5 Pro as the simulated user. Configure an OpenAI-compatible Gemini endpoint with:

```bash
export GEMINI_API_KEY=your_gemini_api_key
export GEMINI_BASE_URL=your_openai_compatible_gemini_base_url
```

The GPT-5-based automatic SimUser metric scripts use a separate endpoint:

```bash
export GPT5_API_KEY=your_gpt5_api_key
export GPT5_BASE_URL=your_openai_compatible_gpt5_base_url
```

## 3. Data

This repository includes the code and a lightweight metadata-only data release. The included metadata is intended for data-format inspection, source tracing, and qualitative examples. It does not redistribute image binaries, and it is not a replacement for the full MTCIR image gallery.

Metadata-only Open Images subset:

- [data/metadata_release_10k](data/metadata_release_10k)
- [data/metadata_release_10k/README.md](data/metadata_release_10k/README.md)
- [data/metadata_release_10k/metadata/openimages_licenses_10k.jsonl](data/metadata_release_10k/metadata/openimages_licenses_10k.jsonl)
- [data/metadata_release_10k/metadata/openimages_licenses_10k.csv](data/metadata_release_10k/metadata/openimages_licenses_10k.csv)
- [data/metadata_release_10k/metadata/subset_manifest.json](data/metadata_release_10k/metadata/subset_manifest.json)

The metadata-only subset contains 10,000 Open Images entries that can be traced by filename ImageID to official Open Images metadata. Each record includes fields such as `relative_path`, `image_id`, `category`, `license`, `author`, `title`, `original_url`, `original_landing_url`, and `original_md5`. Image files are not included; use the source URLs and license fields for provenance inspection. See [data/metadata_release_10k/README.md](data/metadata_release_10k/README.md) for the exact file format.

Showcase train/test files in the metadata release:

- [data/metadata_release_10k/data/train/train_showcase.jsonl](data/metadata_release_10k/data/train/train_showcase.jsonl)
- [data/metadata_release_10k/data/test/test_targets_showcase.json](data/metadata_release_10k/data/test/test_targets_showcase.json)
- [data/metadata_release_10k/data/gallery/gallery_paths.txt](data/metadata_release_10k/data/gallery/gallery_paths.txt)

These files are retained to show the MTCIR training and testing data formats. Since this is a metadata-only release, image paths referenced by the train/test JSON are not guaranteed to be available in this repository. They should be treated as format examples unless the corresponding image gallery is prepared separately.

Verify the metadata release with:

```bash
cd data/metadata_release_10k
sha256sum -c metadata/checksums.sha256 --quiet
```

Full-pipeline RIDE training data:

- [data/train/train_infonce_hardneg_10k.json](data/train/train_infonce_hardneg_10k.json)

This file is used by the training scripts in the full pipeline. It contains multi-turn history, target images, hard negatives, and other fields required by RIDE training. Running the full training pipeline requires the corresponding image files to be prepared under [Gallery_images](Gallery_images).

Full-pipeline test tasks:

- [data/test/test_targets.json](data/test/test_targets.json)

This file defines the target images, target descriptions, template IDs, and related fields used for multi-turn retrieval evaluation. Running the full evaluation pipeline requires gallery embeddings built from the corresponding image files.

Gallery data expected by the full pipeline:

- [Gallery_images](Gallery_images)
- [data/gallery/gallery_paths.txt](data/gallery/gallery_paths.txt)
- [data/gallery/README.md](data/gallery/README.md)

For full training and evaluation, place or symlink gallery images under [Gallery_images](Gallery_images). [data/gallery/gallery_paths.txt](data/gallery/gallery_paths.txt) records one gallery image path per line and is read by the index-building scripts. See [data/gallery/README.md](data/gallery/README.md) for the gallery directory layout, path-file format, and recommended index output locations.

## 4. Building Gallery Embedding Indexes

In the full pipeline, build gallery embedding indexes before RIDE training and multi-turn retrieval evaluation.

Index-building scripts:

- [src/baselines/gme/buildindex_gme.py](src/baselines/gme/buildindex_gme.py)
- [src/baselines/qwen3_vl_embedding/buildindex_qwen.py](src/baselines/qwen3_vl_embedding/buildindex_qwen.py)

### 4.1 GME Gallery Index

```bash
BEGIN=0
END=1000

cd src/baselines/gme
python buildindex_gme.py \
  --begin "$BEGIN" \
  --end "$END" \
  --input_file ../../../data/gallery/gallery_paths.txt \
  --output_dir ../../../data/gallery/embeddings_gme \
  --model_path Alibaba-NLP/gme-Qwen2-VL-2B-Instruct
```

### 4.2 Qwen3-VL-Embedding Gallery Index

```bash
BEGIN=0
END=1000

cd src/baselines/qwen3_vl_embedding
python buildindex_qwen.py \
  --begin "$BEGIN" \
  --end "$END" \
  --input_file ../../../data/gallery/gallery_paths.txt \
  --output_dir ../../../data/gallery/embeddings_qwen \
  --model_name_or_path Qwen/Qwen3-VL-Embedding-2B
```

Index output directories:

- [data/gallery/embeddings_gme](data/gallery/embeddings_gme)
- [data/gallery/embeddings_qwen](data/gallery/embeddings_qwen)

If multiple index shards are generated, they can be merged into:

- `data/gallery/gallery_embeddings_gme.npz`
- `data/gallery/gallery_embeddings_qwen.npz`

## 5. RIDE Training

RIDE training depends on a patched `ms-swift` checkout. Prepare the training environment following [patches/ms_swift/README.md](patches/ms_swift/README.md) before running the scripts below.

Training entry points:

- [src/baselines/gme/train_gme_lora.sh](src/baselines/gme/train_gme_lora.sh)
- [src/baselines/qwen3_vl_embedding/train_qwen_lora.sh](src/baselines/qwen3_vl_embedding/train_qwen_lora.sh)

### 5.1 GME + RIDE Training

```bash
cd src/baselines/gme
bash train_gme_lora.sh
```

### 5.2 Qwen3-VL-Embedding + RIDE Training

```bash
cd src/baselines/qwen3_vl_embedding
bash train_qwen_lora.sh
```

The default training data is:

- [data/train/train_infonce_hardneg_10k.json](data/train/train_infonce_hardneg_10k.json)

## 6. Multi-turn Retrieval Evaluation

Evaluation entry points:

- [src/baselines/gme/run_gme_retrieval.py](src/baselines/gme/run_gme_retrieval.py)
- [src/baselines/qwen3_vl_embedding/run_qwen_retrieval.py](src/baselines/qwen3_vl_embedding/run_qwen_retrieval.py)

### 6.1 GME Multi-turn Retrieval Evaluation

```bash
cd src/baselines/gme
python run_gme_retrieval.py \
  --gallery_embedding_path ../../../data/gallery/gallery_embeddings_gme.npz \
  --target_items ../../../data/test/test_targets.json \
  --output_file ../../../evaluation/retrieval/gme_results.json \
  --user_model gemini-2.5-pro
```

### 6.2 Qwen3-VL-Embedding Multi-turn Retrieval Evaluation

```bash
cd src/baselines/qwen3_vl_embedding
python run_qwen_retrieval.py \
  --gallery_embedding_path ../../../data/gallery/gallery_embeddings_qwen.npz \
  --target_items ../../../data/test/test_targets.json \
  --output_file ../../../evaluation/retrieval/qwen_results.json \
  --user_model gemini-2.5-pro
```

Test tasks used by the evaluation scripts:

- [data/test/test_targets.json](data/test/test_targets.json)

## 7. Metric Computation

Retrieval metric script:

- [evaluation/retrieval/evaluate_metrics.py](evaluation/retrieval/evaluate_metrics.py)

Run:

```bash
python evaluation/retrieval/evaluate_metrics.py \
  --input_file evaluation/retrieval/gme_results.json

python evaluation/retrieval/evaluate_metrics.py \
  --input_file evaluation/retrieval/qwen_results.json
```

This script computes the main multi-turn image retrieval metrics, such as success rate, average number of turns, Recall@1, and Success Rate at Turns (SR@T).

SimUser metrics are computed from HTML retrieval reports. Generate the HTML reports first:

```bash
python evaluation/simuser/html.py \
  --input_file evaluation/retrieval/gme_results.json \
  --output_dir evaluation/simuser/html
```

SimUser metric scripts:

- [evaluation/simuser/html.py](evaluation/simuser/html.py)
- [evaluation/simuser/eval_GNS.py](evaluation/simuser/eval_GNS.py)
- [evaluation/simuser/eval_IRR.py](evaluation/simuser/eval_IRR.py)
- [evaluation/simuser/eval_PPL.py](evaluation/simuser/eval_PPL.py)
- [evaluation/simuser/eval_SIC.py](evaluation/simuser/eval_SIC.py)
- [evaluation/simuser/eval_rankgain.py](evaluation/simuser/eval_rankgain.py)

These scripts compute SimUser-related metrics, including linguistic characteristics, behavioral consistency, information release, and target ranking improvement. The terminology follows the paper: General Naturalness Score (GNS), Persona Stylistic Alignment (PPL<100), Semantic-Intent Consistency (SIC), Information Release Rate (IRR), and Rank Gain.
