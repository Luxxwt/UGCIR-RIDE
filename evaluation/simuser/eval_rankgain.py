import os
import sys
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
ms_swift_root = os.environ.get("MS_SWIFT_ROOT")
if not ms_swift_root:
    raise EnvironmentError("Please set MS_SWIFT_ROOT to your patched ms-swift checkout.")
if ms_swift_root not in sys.path:
    sys.path.insert(0, ms_swift_root)

from swift.infer_engine import InferRequest, TransformersEngine

DEFAULT_GALLERY_EMBEDDING_PATH = os.path.join(repo_root, 'data', 'gallery', 'gallery_embeddings_qwen.npz')

class SwiftConsistencyEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"⚙️ Loading Gallery Embeddings: {args.gallery_embedding_path} ...")
        if not os.path.exists(args.gallery_embedding_path):
            raise FileNotFoundError(f"Gallery file not found: {args.gallery_embedding_path}")
            
        data = np.load(args.gallery_embedding_path, allow_pickle=True)
        if 'paths' in data:
            self.gallery_names = data['paths']
        else:
            self.gallery_names = data['filenames']
        self.basename_to_idx = {os.path.basename(name): i for i, name in enumerate(self.gallery_names)}
        self.gallery_embeddings = torch.from_numpy(data['embeddings'])
        if self.device == "cuda":
            self.gallery_embeddings = self.gallery_embeddings.cuda().to(torch.float16)
        
        print(f"✅ Gallery Loaded. Count: {len(self.gallery_names)}")

        print(f"⚙️ Initializing TransformersEngine: {args.model_name_or_path} ...")
        adapters = [args.adapter_file] if args.adapter_file else []
        
        self.model = TransformersEngine(
            args.model_name_or_path,
            model_type='qwen3_vl_emb',
            task_type='embedding',
            attn_impl='flash_attention_2',
            adapters=adapters
        )
        print("✅ SWIFT Engine Ready.")

    def get_rank(self, query_payload, target_basename):
        """Compute the target rank for a query payload with the SWIFT engine."""
        try:
            item = query_payload[0]
            raw_text = item.get('text', "")
            if isinstance(raw_text, list):
                combined_text = "\n".join(raw_text)
            else:
                combined_text = raw_text

            images = item.get('image', [])
            
            if images:
                placeholders = "<image>" * len(images)
                content_str = f"{placeholders}{combined_text}"
                
                request = InferRequest(
                    messages=[{'role': 'user', 'content': content_str}],
                    images=images
                )
            else:
                request = InferRequest(
                    messages=[{'role': 'user', 'content': combined_text}]
                )

            resp_list = self.model.infer([request])
            emb_list = resp_list[0].data[0].embedding
            input_embeddings = torch.tensor(emb_list).cuda().reshape(1, -1).to(self.gallery_embeddings.dtype)

            similarity_scores = input_embeddings @ self.gallery_embeddings.T
            scores = similarity_scores.cpu().float().numpy()[0]
            sorted_indices = np.argsort(scores)[::-1]
            target_idx = self.basename_to_idx.get(target_basename)
            if target_idx is None:
                return None 
                
            rank_location = np.where(sorted_indices == target_idx)[0]
            
            if len(rank_location) > 0:
                return int(rank_location[0]) + 1
            else:
                return None

        except Exception as e:
            print(f"⚠️ Inference Error: {e}")
            return None

    def evaluate(self, json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            tasks = json.load(f)

        all_consistency_scores = []
        valid_tasks_count = 0
        
        print(f"🚀 Starting Evaluation on {len(tasks)} tasks...")
        
        for task in tqdm(tasks, desc="Evaluating"):
            target_path = task['target_image_path']
            target_basename = os.path.basename(target_path)
            history = task['history']
            
            valid_ranks = []
            for turn in history:
                if 'query' not in turn or not turn['query']:
                    continue
                
                query_payload = turn['query']
                rank = self.get_rank(query_payload, target_basename)
                
                if rank is not None:
                    valid_ranks.append(rank)

            if len(valid_ranks) == 0:
                continue
            
            if len(valid_ranks) == 1:
                all_consistency_scores.append(1.0)
                valid_tasks_count += 1
                continue
            
            valid_tasks_count += 1
            score_points = 0
            comparisons = len(valid_ranks) - 1
            print(f"Task {task['task_id']} Rank Trajectory: {valid_ranks}")
            
            for i in range(1, len(valid_ranks)):
                prev = valid_ranks[i-1]
                curr = valid_ranks[i]
                if curr < prev:
                    score_points += 1
                elif curr == 1 and prev == 1:
                    score_points += 1
            
            consistency = score_points / comparisons
            all_consistency_scores.append(consistency)

        if not all_consistency_scores:
            print("⚠️ No valid tasks found.")
            return

        avg_consistency = np.mean(all_consistency_scores)
        
        print("\n" + "="*50)
        print(f"📊 GOAL-ORIENTED CONSISTENCY REPORT (SWIFT Engine)")
        print("="*50)
        print(f"Tasks in File        : {len(tasks)}")
        print(f"Valid Scored Tasks   : {valid_tasks_count}")
        print("-" * 50)
        print(f"📈 Avg Consistency      : {avg_consistency:.4f}")
        print("-" * 50)
        print("Scoring Rules:")
        print(" - 1 Valid Turn          : Score = 1.0")
        print(" - >1 Valid Turns        : Score = (Improved Moves / Total Moves)")
        print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, default=os.path.join(current_dir, "sample_results.json"), help="Path to retrieval result JSON")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--adapter_file", type=str, default=None, help="LoRA Adapter path if any")
    parser.add_argument("--gallery_embedding_path", type=str, default=DEFAULT_GALLERY_EMBEDDING_PATH)
    
    args = parser.parse_args()

    if not os.path.exists(args.results_file):
        print(f"Error: Result file {args.results_file} not found.")
        sys.exit(1)

    evaluator = SwiftConsistencyEvaluator(args)
    evaluator.evaluate(args.results_file)
