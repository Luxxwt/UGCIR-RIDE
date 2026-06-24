import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
Image.MAX_IMAGE_PIXELS = None

from gme_inference import GmeQwen2VL


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..'))
DEFAULT_GALLERY_LIST = os.path.join(REPO_ROOT, 'data', 'gallery', 'gallery_paths.txt')
DEFAULT_GALLERY_EMBEDDINGS_DIR = os.path.join(REPO_ROOT, 'data', 'gallery', 'embeddings_gme')


def main():
    parser = argparse.ArgumentParser(description='GmeQwen2VL image embedding generation')
    parser.add_argument('--begin', type=int, default=0, help='begin index')
    parser.add_argument('--end', type=int, default=1000, help='end index')
    parser.add_argument(
        '--output_dir',
        type=str,
        default=DEFAULT_GALLERY_EMBEDDINGS_DIR,
        help='directory used to save gallery embedding shards',
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='Alibaba-NLP/gme-Qwen2-VL-2B-Instruct',
        help='public model name or local checkpoint path',
    )
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument(
        '--input_file',
        type=str,
        default=DEFAULT_GALLERY_LIST,
        help='text file containing one gallery image path per line',
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Read the gallery path list
    print(f"Reading input file: {args.input_file}")
    with open(args.input_file, "r") as f:
        all_data = [line.strip() for line in f if line.strip()]
    print(f"Total number of images: {len(all_data)}")

    # Guard against out-of-range indices
    end_idx = min(args.end, len(all_data))
    current_data = all_data[args.begin: end_idx]

    print(f"Number of images to process: {len(current_data)}")
    print(f"Index range: {args.begin} - {end_idx}")

    if len(current_data) == 0:
        print("No data to process. Exiting.")
        return

    # 2. Load the embedding model
    print(f"Loading model: {args.model_path} ...")
    model = GmeQwen2VL(
        model_name=args.model_path,
        device="cuda",
    )

    output_file = os.path.join(args.output_dir, f'embeddings_{args.begin}_{end_idx}.npz')
    try:
        embeddings_tensor = model.get_image_embeddings(
            images=current_data,
            batch_size=args.batch_size,
            instruction='',
            show_progress_bar=True,
        )

        # Convert and save the embeddings
        embeddings_array = embeddings_tensor.cpu().float().numpy()
        paths_array = np.array(current_data, dtype=object)
        print(f"Saving embeddings to: {output_file}, Shape: {embeddings_array.shape}")
        print(f"Saving paths to: {output_file}, Shape: {paths_array.shape}")
        np.savez(output_file, embeddings=embeddings_array, paths=paths_array)
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
