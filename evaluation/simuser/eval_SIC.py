import os
import glob
import json
import re
from bs4 import BeautifulSoup
from typing import List, Dict
from tqdm import tqdm
from openai import OpenAI

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_DIR = os.path.join(CURRENT_DIR, "html")
API_KEY_ENV = "GPT5_API_KEY"
BASE_URL_ENV = "GPT5_BASE_URL"
JUDGE_MODEL = "gpt-5" 

class AutoSemanticJudge:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ.get(API_KEY_ENV),
            base_url=os.environ.get(BASE_URL_ENV),
        )

    def evaluate_turn(self, stage_name: str, user_text: str) -> Dict:
        """Judge whether the user utterance matches the current stage intent."""
        prompt = f"""
        [ROLE]
        You are a strict Logic Evaluator for a User Simulation System.
        
        [TASK]
        Determine if the "User's Spoken Text" semantically aligns with the intent implied by the "Stage Name".
        You must infer the definition of the Stage Name using common sense.

        [INPUT DATA]
        1. Current Stage Name: "{stage_name}"
        2. User's Spoken Text: "{user_text}"

        [JUDGMENT LOGIC]
        1. First, interpret what a user SHOULD do in a stage named "{stage_name}".
           - Example: If stage is "Broad Category", user should describe general types, NOT specific tiny details.
           - Example: If stage is "Lock on Target", user should declare "The [x]th image in the retrieved results".
        2. Compare the Spoken Text against this interpretation.
        3. If the user creates a contradiction (e.g., saying "I found it" while in "Initial Search" stage), mark as FALSE.

        [OUTPUT FORMAT]
        Return ONLY a JSON object:
        {{
            "stage_interpretation": "Briefly, what this stage means",
            "is_match": true/false,
            "reason": "Explanation of the verdict"
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"⚠️ API Error: {e}")
            return {"is_match": False, "reason": f"API Error: {str(e)}"}

def extract_dialogue_pairs(html_path):
    """Extract (stage, text) pairs from the HTML report."""
    with open(html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
    
    pairs = []
    user_containers = soup.select(".user-container")
    
    for container in user_containers:
        thought = container.select_one(".thought-bubble")
        if not thought: continue
        
        thought_text = thought.get_text(separator=" ", strip=True)
        stage_match = re.search(r"Stage:\s*(.*?)(?=\s*\||$|\n)", thought_text, re.IGNORECASE)
        if not stage_match: continue
        stage_name = stage_match.group(1).strip()
        
        speech = container.select_one(".speech-bubble")
        if not speech: continue
        user_text = speech.get_text().strip()
        
        pairs.append({
            "stage": stage_name,
            "text": user_text
        })
    
    return pairs

def main():
    judge = AutoSemanticJudge()
    html_files = glob.glob(os.path.join(HTML_DIR, "*.html"))
    
    if not html_files:
        print("❌ No HTML files found.")
        return

    print(f"🚀 Starting semantic intent accuracy evaluation ({len(html_files)} files)...")
    
    results_log = []
    stage_stats = {}
    
    total_turns = 0
    passed_turns = 0

    for html_file in tqdm(html_files):
        task_id = os.path.basename(html_file).replace(".html", "")
        pairs = extract_dialogue_pairs(html_file)
        
        for i, pair in enumerate(pairs):
            stage = pair['stage']
            text = pair['text']
            eval_result = judge.evaluate_turn(stage, text)
            
            is_match = eval_result.get("is_match", False)
            reason = eval_result.get("reason", "Unknown")
            interpretation = eval_result.get("stage_interpretation", "N/A")
            total_turns += 1
            if is_match: passed_turns += 1
            
            if not is_match:
                results_log.append({
                    "task": task_id,
                    "turn": i + 1,
                    "stage": stage,
                    "text": text,
                    "interpretation": interpretation,
                    "reason": reason
                })
            
            stage_key = stage.lower()
            if stage_key not in stage_stats:
                stage_stats[stage_key] = {"total": 0, "pass": 0}
            stage_stats[stage_key]["total"] += 1
            if is_match:
                stage_stats[stage_key]["pass"] += 1

    print("\n" + "="*60)
    print("🧠 Semantic Intent Accuracy Report (Dynamic Stage Inference)")
    print("="*60)
    
    if total_turns > 0:
        acc = (passed_turns / total_turns) * 100
        print(f"Overall Accuracy: {acc:.2f}% ({passed_turns}/{total_turns})")
    else:
        print("No dialogue turns found.")
        
    print("\n[Performance by Stage Name]")
    print(f"{'Stage Name':<30} | {'Acc':<8} | {'Count':<10}")
    print("-" * 60)
    sorted_stages = sorted(stage_stats.items(), key=lambda x: x[1]['total'], reverse=True)
    for stage, stats in sorted_stages[:15]:
        s_acc = (stats['pass'] / stats['total']) * 100
        print(f"{stage:<30} | {s_acc:.1f}%   | {stats['pass']}/{stats['total']}")

    print("\n[Bad Case Analysis (Top 3)]")
    for case in results_log[:3]:
        print(f"❌ Task: {case['task']} (Turn {case['turn']})")
        print(f"   Stage Name    : {case['stage']}")
        print(f"   LLM Interpretation: {case['interpretation']}")
        print(f"   User Text     : {case['text']}")
        print(f"   Fail Reason   : {case['reason']}")
        print("-" * 40)

    out_file = os.path.join(CURRENT_DIR, "intent_accuracy_failures.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_log, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Full failure log saved to {out_file}")

if __name__ == "__main__":
    main()
