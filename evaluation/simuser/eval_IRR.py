EVALUATOR_SYSTEM_PROMPT = """
You are an expert Information Extraction Auditor for an image retrieval simulation.
Your task is to identify NEW visual attributes from the User's input, comparing them against the Existing Knowledge Base.

[INPUT DATA]
1. Existing Knowledge Base: A list of attributes already established in previous turns.
2. Current User Input: The user's speech in the current turn.

[RULES]
1. **PREFER COMPOUND PHRASES**: Do NOT split an object from its modifiers. Keep them as a single semantic unit.
   - BAD: "red car" -> ["color: red", "object: car"]
   - GOOD: "red car" -> ["red car"].
2. **De-duplication**: Compare with the 'Existing Knowledge Base'. If an attribute is semantically identical or a subset of an existing one, IGNORE it.
   - Example: If Knowledge Base has "red sports car", and User says "show me the car", ignore "car".
   - Example: If Knowledge Base has "dog", and User says "brown dog", extract "brown" (new detail).
3. **NO BACK-FILLING**: Do NOT add adjectives or context from the 'Existing Knowledge Base' into the current extraction.
4. **Output Format**: Return a valid JSON object with a list of NEW attributes.

[OUTPUT FORMAT]
{
    "new_attributes": ["attr1", "attr2"],
    "count": 2,
    "reasoning": "Explanation of why these are new..."
}
"""


import os
import json
import glob
from bs4 import BeautifulSoup
from openai import OpenAI
from typing import List, Dict
from tqdm import tqdm
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_DIR = os.path.join(CURRENT_DIR, "html")
API_KEY_ENV = "GPT5_API_KEY"
BASE_URL_ENV = "GPT5_BASE_URL"
EVAL_MODEL = "gpt-5"
MAX_INFO_PER_TURN = 2 

class ThrottleEvaluator:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ.get(API_KEY_ENV),
            base_url=os.environ.get(BASE_URL_ENV),
        )

    def extract_new_info(self, current_text: str, known_attributes: List[str]) -> Dict:
        """Extract new attributes while accounting for prior context."""
        user_content = f"""
        [Existing Knowledge Base]
        {json.dumps(known_attributes, indent=2)}

        [Current User Input]
        "{current_text}"
        
        Please identify the NEW visual attributes.
        """

        try:
            response = self.client.chat.completions.create(
                model=EVAL_MODEL,
                messages=[
                    {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"⚠️ LLM Eval Error: {e}")
            return {"new_attributes": [], "count": 0, "reasoning": "Error"}

def parse_user_turns(html_path):
    """Parse all user turns from the HTML report."""
    with open(html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
    turns_data = []
    
    user_containers = soup.select(".user-container")
    
    for container in user_containers:
        thought_el = container.select_one(".thought-bubble")
        stage_text = ""
        if thought_el:
            raw_thought = thought_el.get_text()
            if "Current Stage" in raw_thought:
                stage_text = raw_thought
        
        speech_el = container.select_one(".speech-bubble")
        speech_text = speech_el.get_text().strip() if speech_el else ""
        
        turns_data.append({
            "stage_raw": stage_text,
            "text": speech_text
        })
        
    return turns_data

def evaluate_throttling():
    evaluator = ThrottleEvaluator()
    html_files = glob.glob(os.path.join(HTML_DIR, "*.html"))
    
    if not html_files:
        print("❌ No HTML files found.")
        return

    print(f"🚀 Starting information throttling evaluation (files: {len(html_files)})...")
    
    global_analyzed_count = 0 
    global_passed_count = 0
    task_reports = []

    for file_path in tqdm(html_files, desc="Processing Files", unit="task"):
        task_id = os.path.basename(file_path).replace(".html", "")
        tqdm.write(f"\n📄 Analyzing Task: {task_id}")
        turn_items = parse_user_turns(file_path)
        
        known_attributes = []
        task_violation_count = 0
        task_analyzed_count = 0
        turn_logs = []

        for i, item in enumerate(turn_items):
            text = item['text']
            stage_info = item['stage_raw']
            if "Lock on Target" in stage_info or "lock on target" in stage_info.lower():
                tqdm.write(f"   ⏩ Turn {i+1} Skipped (Lock on Target Phase)")
                turn_logs.append({
                    "turn": i + 1,
                    "text": text,
                    "status": "skipped_final_stage"
                })
                continue
            result = evaluator.extract_new_info(text, known_attributes)
            new_attrs = result.get("new_attributes", [])
            count = len(new_attrs)
            is_pass = count <= MAX_INFO_PER_TURN
            known_attributes.extend(new_attrs)
            task_analyzed_count += 1
            global_analyzed_count += 1
            if is_pass:
                global_passed_count += 1
            else:
                task_violation_count += 1
                tqdm.write(f"   ⚠️ Turn {i+1} Violation! (+{count}): {new_attrs}")

            turn_logs.append({
                "turn": i + 1,
                "text": text,
                "new_extracted": new_attrs,
                "is_pass": is_pass
            })
        
        task_reports.append({
            "task_id": task_id,
            "total_turns": len(turn_items),
            "analyzed_turns": task_analyzed_count,
            "violations": task_violation_count,
            "final_knowledge_base": known_attributes,
            "logs": turn_logs
        })

    throttle_rate = (global_passed_count / global_analyzed_count) * 100 if global_analyzed_count  > 0 else 0
    
    print("\n" + "="*40)
    print("📊 Information Throttling Report")
    print("="*40)
    print(f"Total Valid Turns     : {global_analyzed_count}")
    print(f"Passed Turns          : {global_passed_count}")
    print(f"Throttling Rate       : {throttle_rate:.2f}%")
    print("="*40)

    output_path = os.path.join(CURRENT_DIR, "throttling_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(task_reports, f, indent=2, ensure_ascii=False)
    print(f"Detailed report saved to {output_path}")

if __name__ == "__main__":
    evaluate_throttling()
