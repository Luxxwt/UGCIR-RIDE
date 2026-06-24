# UGCIR Metadata-Only Data Release

This package is a lightweight metadata-only release for the UGCIR project. It contains 10,000 Open Images entries that can be traced by filename ImageID to official Open Images metadata. Image binaries are not redistributed in this package.

## Contents

```text
data_metadata_release_10k/
├── gallery/
│   └── gallery_paths.txt
└── metadata/
    ├── openimages_licenses_10k.jsonl
    ├── openimages_licenses_10k.csv
    ├── subset_manifest.json
    └── checksums.sha256
```

## Metadata

`metadata/openimages_licenses_10k.jsonl` and `metadata/openimages_licenses_10k.csv` contain one record per metadata entry. Each record includes:

- `relative_path`: the path used by the UGCIR gallery layout; the image file itself is not included.
- `image_id`: the Open Images ImageID.
- `category`: the UGCIR category directory.
- `license`: the source image license reported by Open Images.
- `author`, `author_profile_url`, `title`: attribution fields when available.
- `original_url`, `original_landing_url`: source URLs reported by Open Images.
- `original_md5`: Open Images MD5 for the original source file.

## Gallery Paths

`gallery/gallery_paths.txt` lists the 10,000 metadata entries using the same relative path format as the full UGCIR gallery. Image files are not included in this metadata-only release.

## Checksums

Run the following command from this directory:

```bash
sha256sum -c metadata/checksums.sha256
```
