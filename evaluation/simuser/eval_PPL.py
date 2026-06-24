import os
import torch
import math
import numpy as np
from bs4 import BeautifulSoup
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_FOLDER = os.path.join(CURRENT_DIR, "html")
MODEL_ID = "Qwen/Qwen3-8B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class PPLEvaluator:
    def __init__(self, model_id, device):
        print(f"🔄 Loading PPL Evaluation Model: {model_id} ...")
        self.device = device
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.float16, 
                device_map="auto",
                trust_remote_code=True
            )
            self.model.eval()
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise e

    def calculate_sentence_ppl(self, text):
        """Compute sentence-level perplexity."""
        if not text or len(text.strip()) == 0:
            return None

        encodings = self.tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(self.device)
        
        if input_ids.shape[1] > 8192:
            input_ids = input_ids[:, :8192]

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss
            
            if torch.isnan(loss):
                return None

        ppl = torch.exp(loss).item()
        return ppl

def extract_dialogues_from_html(html_path):
    """Extract all valid SimUser utterances from the HTML report."""
    if not os.path.exists(html_path):
        return []

    with open(html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    valid_texts = []
    user_containers = soup.select(".user-container")
    
    for container in user_containers:
        thought_bubble = container.select_one(".thought-bubble")
        if thought_bubble:
            thought_text = thought_bubble.get_text().lower()
            if "Lock on Target" in thought_text:
                continue

        speech_bubble = container.select_one(".speech-bubble")
        if not speech_bubble:
             speech_bubble = container.select_one(".speech_bubble")

        if speech_bubble:
            text = speech_bubble.get_text(strip=True)
            text_lower = text.lower()
            if "image in the retrieved results" in text_lower:
                continue
            if len(text) > 1:
                valid_texts.append(text)
            
    return valid_texts

def main():
    try:
        evaluator = PPLEvaluator(MODEL_ID, DEVICE)
    except:
        return

    html_files = [f for f in os.listdir(HTML_FOLDER) if f.endswith(".html")]
    print(f"📂 Found {len(html_files)} HTML files for evaluation.")

    all_ppl_scores = []
    file_ppl_results = {}

    for filename in tqdm(html_files, desc="Calculating PPL"):
        file_path = os.path.join(HTML_FOLDER, filename)
        dialogues = extract_dialogues_from_html(file_path)
        
        if not dialogues:
            continue
            
        file_scores = []
        for text in dialogues:
            ppl = evaluator.calculate_sentence_ppl(text)
            if ppl is not None:
                file_scores.append(ppl)
                all_ppl_scores.append(ppl)
        
        if file_scores:
            avg_file_ppl = sum(file_scores) / len(file_scores)
            file_ppl_results[filename] = avg_file_ppl

    if all_ppl_scores:
        global_avg_ppl = sum(all_ppl_scores) / len(all_ppl_scores)
        median_ppl = np.median(all_ppl_scores)
        
        print("\n" + "="*40)
        print("📊 PPL EVALUATION REPORT")
        print("="*40)
        print(f"Model: {MODEL_ID}")
        print(f"Total Utterances Evaluated: {len(all_ppl_scores)}")
        print(f"Global Average PPL: {global_avg_ppl:.4f} (Lower is Better)")
        print(f"Global Median PPL:  {median_ppl:.4f}")
        print("-" * 40)
        
        sorted_files = sorted(file_ppl_results.items(), key=lambda x: x[1])
        minx = 0
        miny = 0
        minz = 0
        zong = 0
        for name, score in sorted_files:
            zong +=1
            if score <=50: minx +=1
            if score <=70: miny +=1
            if score <=100: minz +=1 
        print(f"PPL <= 50 ratio:  {minx}/{zong}")
        print(f"PPL <= 70 ratio:  {miny}/{zong}")
        print(f"PPL <= 100 ratio: {minz}/{zong}")
        print("\n🏆 Top 5 Most Natural Dialogues (Lowest PPL):")
        for name, score in sorted_files[:5]:
            print(f"   {name}: {score:.2f}")
            
        print("\n⚠️ Top 20 Least Natural Dialogues (Highest PPL):")
        for name, score in sorted_files[-20:]:
            print(f"   {name}: {score:.2f}")

    else:
        print("❌ No valid dialogues found to evaluate.")

if __name__ == "__main__":
    main()
