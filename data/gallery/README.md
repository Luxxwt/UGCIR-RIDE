# Gallery Images

`Gallery_images/` is the root directory for the image gallery used by retrieval, training, and evaluation scripts. Each image listed in [gallery_paths.txt](gallery_paths.txt) should be reachable from the repository root.

## Directory Layout

Use the following structure:

```text
Gallery_images/
├── TargetImage/
│   └── <category_or_task>/<image_file>
├── OpenImage/
│   └── <category>/<image_file>
└── Others/
    └── <image_id_or_task>/<image_file>
```

The three folders serve different roles:

- `TargetImage/`: target images and image-text pairs used by retrieval tasks.
- `OpenImage/`: gallery images collected from OpenImages.
- `Others/`: additional gallery images, including images from CC3M and extra images.

## Gallery Path File

[gallery_paths.txt](gallery_paths.txt) stores the gallery image list. The file is plain text, with one image path per line. Paths should be relative to the repository root.

Example:

```text
Gallery_images/OpenImage/Bottle/ac35c968b2509ad0.jpg
Gallery_images/Others/001070174/raw_image.jpg
Gallery_images/Others/T12/example_modified.jpg
```

Index-building scripts read this file line by line and generate embeddings for the corresponding images.

## Index Outputs

Recommended output directories for gallery embeddings:

```text
data/gallery/embeddings_gme/
data/gallery/embeddings_qwen/
```

Recommended merged index filenames:

```text
data/gallery/gallery_embeddings_gme.npz
data/gallery/gallery_embeddings_qwen.npz
```
