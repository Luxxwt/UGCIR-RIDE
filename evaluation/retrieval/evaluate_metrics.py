import os
import json
import glob
import argparse
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DATA_DIR = os.path.join(CURRENT_DIR, "sample_results.json")
CHECKPOINTS = [4, 7, 10, 11]

def load_all_tasks(data_dir):
    """Load all task results from a directory."""
    all_tasks = []
    if not os.path.exists(data_dir):
        print(f"❌ Directory not found: {data_dir}")
        return []

    json_files = sorted([f for f in os.listdir(data_dir) 
                         if f.endswith(".json") and "global" not in f])

    print(f"📂 Loading {len(json_files)} result files...")
    
    for filename in tqdm(json_files):
        path = os.path.join(data_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_tasks.extend(data)
                else:
                    all_tasks.append(data)
        except Exception as e:
            print(f"⚠️ Failed to read {filename}: {e}")
            
    return all_tasks

def load_all_tasks_json(data_dir):
    """Load all task results from a single JSON file."""
    all_tasks = []
    with open(data_dir, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            all_tasks.extend(data)
        else:
            all_tasks.append(data)            
    return all_tasks

def calculate_metrics(tasks):
    """Compute the core retrieval metrics."""
    total_tasks = len(tasks)
    if total_tasks == 0:
        return None

    successful_tasks = [t for t in tasks if t.get('is_success', False)]
    num_success = len(successful_tasks)

    if num_success > 0:
        total_turns_success = sum(t.get('turns_used', 0) for t in successful_tasks)
        ats = total_turns_success / num_success
    else:
        ats = 0.0

    if num_success > 0:
        rank1_count = sum(1 for t in successful_tasks if t.get('final_rank') == 1)
        recall_at_1 = (rank1_count / num_success) * 100
    else:
        recall_at_1 = 0.0

    sr_at_t = {}
    for t_limit in CHECKPOINTS:
        count_in_limit = sum(1 for t in successful_tasks if t.get('turns_used', 999) <= t_limit)
        sr_at_t[t_limit] = (count_in_limit / total_tasks) * 100

    return {
        "total_attempts": total_tasks,
        "total_success": num_success,
        "overall_success_rate": (num_success / total_tasks) * 100,
        "ATS": ats,
        "Recall_at_1": recall_at_1,
        "SR_at_T": sr_at_t
    }

def main():
    parser = argparse.ArgumentParser(description="Compute multi-turn retrieval metrics.")
    parser.add_argument(
        "--input_file",
        type=str,
        default=DATA_DIR,
        help="Path to a retrieval result JSON file.",
    )
    args = parser.parse_args()

    tasks = load_all_tasks_json(args.input_file)
    metrics = calculate_metrics(tasks)

    if not metrics:
        print("❌ No valid data loaded.")
        return

    print("\n" + "="*50)
    print("📊 EVALUATION REPORT")
    print("="*50)
    
    print(f"Total Tasks Evaluated: {metrics['total_attempts']}")
    print(f"Total Successful Tasks: {metrics['total_success']}")
    print(f"Overall Success Rate:   {metrics['overall_success_rate']:.2f}%")
    
    print("\n" + "-"*30)
    print("1. Search Efficiency")
    print("-"*30)
    print(f"Average Turns to Success (ATS): {metrics['ATS']:.2f}")
    print("(Lower is better. Indicates rapid grasp of intent.)")

    print("\n" + "-"*30)
    print("2. Retrieval Precision")
    print("-"*30)
    print(f"Recall@1 (Final Turn): {metrics['Recall_at_1']:.2f}%")
    print("(Higher is better. Percentage of successes where Target was Rank #1.)")

    print("\n" + "-"*30)
    print("3. Success Dynamics (Cumulative)")
    print("-"*30)
    for t in CHECKPOINTS:
        rate = metrics['SR_at_T'][t]
        print(f"SR @ Turn {t:<2}: {rate:.2f}%")
    print("(Higher is better. Shows capability to handle incremental info.)")
    
    print("="*50)

if __name__ == "__main__":
    main()
