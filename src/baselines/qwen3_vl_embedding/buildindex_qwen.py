import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from qwen3_vl_embedding import Qwen3VLEmbedder


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..'))
DEFAULT_GALLERY_LIST = os.path.join(REPO_ROOT, 'data', 'gallery', 'gallery_paths.txt')
DEFAULT_GALLERY_EMBEDDINGS_DIR = os.path.join(REPO_ROOT, 'data', 'gallery', 'embeddings_qwen')


def main():
    parser = argparse.ArgumentParser(description='Image Embedding')
    parser.add_argument('--begin', type=int, required=True, help='begin index')
    parser.add_argument('--end', type=int, required=True, help='end index')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_GALLERY_EMBEDDINGS_DIR, help='output dir')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--input_file', type=str, default=DEFAULT_GALLERY_LIST, help='input file')
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default='Qwen/Qwen3-VL-Embedding-2B',
        help='public model name or local checkpoint path',
    )
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(args.input_file, "r") as f:
        all_data = f.readlines()
    all_data = [item.rstrip() for item in all_data]
    current_data = all_data[args.begin: args.end]
    
    print(f"Number of images to process: {len(current_data)}")
    print(f"Index range: {args.begin} - {args.end}")
    
    model = Qwen3VLEmbedder(
        model_name_or_path=args.model_name_or_path,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
    )
    
    output_file = os.path.join(args.output_dir, f'embeddings_{args.begin}_{args.end}.npz')
    all_embeddings = []
    
    for i in tqdm(range(0, len(current_data), args.batch_size)):
        batch_paths = current_data[i:i + args.batch_size]
        batch_inputs = []
        for img_path in batch_paths:
            batch_inputs.append({
                'image': img_path,
            })
        
        batch_embeddings = model.process(batch_inputs)
        
        batch_embeddings_np = batch_embeddings.cpu().numpy()
        all_embeddings.append(batch_embeddings_np)      
    
    embeddings_array = np.concatenate(all_embeddings, axis=0)
    paths_array = np.array(current_data, dtype=object)
    np.savez(output_file, embeddings=embeddings_array, paths=paths_array)
        

if __name__ == "__main__":
    main()
    
