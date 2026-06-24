import os
from bs4 import BeautifulSoup
from openai import OpenAI
import json
import numpy as np
from tqdm import tqdm
import tiktoken
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_FOLDER = os.path.join(CURRENT_DIR, "html")
OUTPUT_FILE = os.path.join(CURRENT_DIR, "naturalness_scores.json")
API_KEY_ENV = "GPT5_API_KEY"
BASE_URL_ENV = "GPT5_BASE_URL"
EVAL_MODEL = "gpt-5"

try:
    ENC = tiktoken.get_encoding("cl100k_base")
except:
    print("⚠️ tiktoken not found. Falling back to a rough character/4 estimate.")
    ENC = None

GEVAL_PROMPT_TEMPLATE = """
You will be given a dialogue between a User (Simulated) and an Image Search System.
The User is trying to find a specific target image by describing it progressively.

[Dialogue History]
{dialogue_content}

[Evaluation Task]
Please evaluate the **Naturalness** of the User's responses in the dialogue above.
NOTE: Do not give perfect scores easily. A score of 5 should be indistinguishable from a casual real human.

[Evaluation Criteria]
1. Fluency (1-5): 
   - Is the User's language grammatically correct and idiomatic? Does it sound like a native speaker?
   - **PENALTY RULE:** If the User speaks in dense, descriptive noun phrases without verbs, this is Caption Style, you MUST give a lower score.
2. Interactive Coherence (1-5) : 
   - Does the User **acknowledge** the images shown by the System? (e.g., "Not this one," "Close, but...", "Remove the background").Is the flow of information natural?
   - **PENALTY RULE:** If the User acts like the System does not exist and simply continues reciting a pre-written description segment by segment (Soliloquy/Monologue), you MUST give a lower score.
3. Human-likeness (1-5): 
   - Does the User sound like a human searching for an image?
   - Humans are reactive, emotional, and sometimes imperfect.
   - **PENALTY RULE:** If the User sounds like a text-to-speech engine reading a paragraph sentence by sentence, give a lower score.

[Evaluation Steps]
1. Read the dialogue carefully.
2. Analyze the User's turns specifically. Check if they ignore the System's previous images or repeat themselves.
3. Check if the User simulates "thinking" or "observing" naturally (e.g., acknowledging mistakes, refining descriptions).
4. Assign a score from 1 to 5 for each criterion.

[Output Format]
Return the result in valid JSON format:
{{
  "reasoning": "Your step-by-step analysis here...",
  "scores": {{
    "fluency": <int>,
    "coherence": <int>,
    "human_likeness": <int>,
    "overall": <float_average>
  }}
}}
"""

client = OpenAI(
    api_key=os.environ.get(API_KEY_ENV),
    base_url=os.environ.get(BASE_URL_ENV),
)

def extract_dialogue_from_html(html_path):
    """Extract plain-text dialogue from the HTML report."""
    if not os.path.exists(html_path):
        return ""

    with open(html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
    
    is_success = False
    task_info = soup.select_one(".task-info")
    if task_info:
        text = task_info.get_text()
        if "SUCCESS" in text:
            is_success = True

    dialogue_text = ""
    user_token_counts = []
    turns = soup.find_all(class_="message-group")
    user_turn_count = 0
    
    for group in turns:
        user_div = group.find(class_="user-container")
        if user_div:
            speech = user_div.find(class_="speech-bubble")
            if speech:
                user_turn_count += 1
                text = speech.get_text(strip=True)
                dialogue_text += f"User (Turn {user_turn_count}): {text}\n"
                
                if ENC:
                    tokens = len(ENC.encode(text))
                else:
                    tokens = len(text) // 4
                user_token_counts.append(tokens)
                dialogue_text += f"User (Turn {user_turn_count}): {text}\n"
        sys_div = group.find(class_="system-container")
        if sys_div:
            imgs = sys_div.find_all("img")
            img_count = len(imgs)
            query_content = "N/A"
            query_tag = sys_div.find(class_="tag-summary")
            if query_tag:
                query_content = query_tag.get_text(strip=True).replace("📝 Updated Query:", "").strip()
            if img_count > 0 or query_content != "N/A":
                dialogue_text += f"System: [Showed {img_count} images] (Internal Query: {query_content})\n"
            
    return dialogue_text, user_token_counts, is_success

def evaluate_naturalness(dialogue_text):
    """Run G-Eval on the extracted dialogue."""
    prompt = GEVAL_PROMPT_TEMPLATE.format(dialogue_content=dialogue_text)
    
    try:
        response = client.chat.completions.create(
            model=EVAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0, 
            response_format={"type": "json_object"} 
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None

def main():
    print("🚀 Starting G-Eval naturalness evaluation (SimUser only)...")
    successful_task_turns = []
    all_results = []
    output_file = OUTPUT_FILE
    
    if not os.path.exists(HTML_FOLDER):
        print(f"❌ Folder not found: {HTML_FOLDER}")
        return

    html_files = [f for f in os.listdir(HTML_FOLDER) if f.endswith(".html")]
    for filename in tqdm(html_files, desc="Evaluating"):
        file_path = os.path.join(HTML_FOLDER, filename)
        dialogue, token_counts, is_success = extract_dialogue_from_html(file_path)
        if not dialogue: 
            continue
        num_turns = len(token_counts)
        avg_token_len = np.mean(token_counts) if token_counts else 0.0
        if is_success:
            successful_task_turns.append(num_turns)
        eval_json_str = evaluate_naturalness(dialogue)
                       

        if eval_json_str:
            try:
                data = json.loads(eval_json_str)
                if 'scores' not in data or 'fluency' not in data['scores']:
                    print(f"\n⚠️ Unexpected response format, skipping: {filename}")
                    print(f"   Response preview: {str(data)[:100]}...")
                    continue
                data['file_id'] = filename
                data['stats'] = {
                    "is_success": is_success,
                    "total_turns": len(token_counts),
                    "avg_user_tokens": round(avg_token_len, 2),
                    "raw_token_counts": token_counts
                }
                data['dialogue_preview'] = dialogue[:] 
                all_results.append(data)
            except json.JSONDecodeError:
                print(f"⚠️ Failed to parse JSON: {filename}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    if all_results:
        fluency_avg = np.mean([r['scores']['fluency'] for r in all_results])
        coherence_avg = np.mean([r['scores']['coherence'] for r in all_results])
        human_avg = np.mean([r['scores']['human_likeness'] for r in all_results])
        overall_avg = np.mean([r['scores']['overall'] for r in all_results])
        
        global_avg_tokens = np.mean([r['stats']['avg_user_tokens'] for r in all_results])
        all_turns_tokens = [t for r in all_results for t in r['stats']['raw_token_counts']]
        micro_avg_tokens = np.mean(all_turns_tokens) if all_turns_tokens else 0
        
        success_count = len(successful_task_turns)
        total_count = len(all_results)
        avg_turns_success = np.mean(successful_task_turns) if successful_task_turns else 0.0
        success_rate = (success_count / total_count * 100) if total_count > 0 else 0
        print("\n" + "="*50)
        print("📊 FINAL REPORT")
        print("="*50)
        print(f"Total Tasks Processed:  {total_count}")
        print(f"Success Rate:           {success_rate:.2f}% ({success_count}/{total_count})")
        print("-" * 50)
        print(f"Naturalness Metrics (G-Eval):")
        print(f"Total Evaluated: {len(all_results)}")
        print(f"Fluency:       {fluency_avg:.2f} / 5.0")
        print(f"Coherence:     {coherence_avg:.2f} / 5.0")
        print(f"Human-likeness:{human_avg:.2f} / 5.0")
        print(f"OVERALL SCORE: {overall_avg:.2f} / 5.0")
        print("-" * 50)
        print(f"User Behavior Stats:")
        print(f"Avg User Tokens: {global_avg_tokens:.2f} tokens/turn (Macro)")
        print(f"                 {micro_avg_tokens:.2f} tokens/turn (Micro)")
        print(f"Avg Turns (Success):  {avg_turns_success:.2f} turns (Successful tasks only)")
        print("="*50)
        print(f"Report saved to: {output_file}")
        

if __name__ == "__main__":
    main()
