# MTCIR Metadata-Only Data Release

This package is a lightweight metadata-only release for the MTCIR project. It contains 10,000 Open Images entries that can be traced by filename ImageID to official Open Images metadata. Image binaries are not redistributed in this package.

## Contents

```text
data_metadata_release_10k/
├── data/
│   ├── gallery/gallery_paths.txt
│   ├── train/train_showcase.jsonl
│   └── test/test_targets_showcase.json
└── metadata/
    ├── openimages_licenses_10k.jsonl
    ├── openimages_licenses_10k.csv
    ├── subset_manifest.json
    └── checksums.sha256
```

## Metadata

`metadata/openimages_licenses_10k.jsonl` and `metadata/openimages_licenses_10k.csv` contain one record per metadata entry. Each record includes:

- `relative_path`: the path used by the MTCIR gallery layout; the image file itself is not included.
- `image_id`: the Open Images ImageID.
- `category`: the MTCIR category directory.
- `license`: the source image license reported by Open Images.
- `author`, `author_profile_url`, `title`: attribution fields when available.
- `original_url`, `original_landing_url`: source URLs reported by Open Images.
- `original_md5`: Open Images MD5 for the original source file.

## Train And Test JSON

`data/train/train_showcase.jsonl` and `data/test/test_targets_showcase.json` are retained to show the training and testing data formats used by MTCIR. Since this is a metadata-only release, referenced image files are not guaranteed to be present in this package.

## Checksums

Run the following command from this directory:

```bash
sha256sum -c metadata/checksums.sha256
```
