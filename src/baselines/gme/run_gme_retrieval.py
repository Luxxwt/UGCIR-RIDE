import os
import re
import sys
import json
import time
import copy
import glob
import yaml
import torch
import base64
import hashlib
import mimetypes
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm
from typing import List, Dict
from openai import OpenAI
from gme_inference import GmeQwen2VL
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
ms_swift_root = os.environ.get("MS_SWIFT_ROOT")
if not ms_swift_root:
    raise EnvironmentError("Please set MS_SWIFT_ROOT to your patched ms-swift checkout.")
if ms_swift_root not in sys.path:
    sys.path.insert(0, ms_swift_root)
from swift.infer_engine import InferRequest, TransformersEngine
os.environ["TOKENIZERS_PARALLELISM"] = "false"


USER_MODEL = "gemini-2.5-pro"
DEFAULT_API_KEY_ENV = "GEMINI_API_KEY"
DEFAULT_BASE_URL_ENV = "GEMINI_BASE_URL"
DEFAULT_SINGLE_TEMPLATE_DIR = os.path.join(repo_root, "templates", "single_target")
DEFAULT_MULTI_TEMPLATE_DIR = os.path.join(repo_root, "templates", "multi_target")
DEFAULT_TARGET_ITEMS = os.path.join(repo_root, "data", "test", "test_targets.json")
DEFAULT_OUTPUT_FILE = os.path.join(repo_root, "evaluation", "retrieval", "gme_results.json")
DEFAULT_GALLERY_EMBEDDING_PATH = os.path.join(repo_root, "data", "gallery", "gallery_embeddings_gme.npz")

USER_SYSTEM_PROMPT_TEMPLATE = """
[ROLE]
Act as a User in an image retrieval simulation.You must guide the Search System to find this specific image by revealing details progressively using natural speech. 
You are a regular person with NO knowledge of photography or technical terms. You have a vague memory of the image.
[TARGET DEFINITIONS]
{{target_definitions}} 

[LOGIC & RULES]
 
1. Strict Structure (CRITICAL): 
   - EVERY response must have TWO blocks:
    (1) [Reflection] ... 
    (2) [Output] ... 
   - If you do not include [Output] with specific "Requirements:", the system will fail.
 
2. Pacing & Stage Logic (DYNAMIC & STRICT):
   - Point Constraint: Each [Output] MUST reveal MAX 2 new visual attributes. 
   - Self-Check: In [Reflection], you must count how many new attributes you are providing.
   - NO Summary.
   - Refer to the 'dialogue_pattern' in [CURRENT TASK CONFIG] below.
   - Stage Sequence: You MUST move FORWARD through the stages defined in `dialogue_pattern.description`.
   - Stage Rules: You MUST strictly obey the `important_notes` defined in the config.
   - Stage Looping Policy (CRITICAL):
     (a) First Stage (Broad Category): Execute ONCE.
     (b) ALL Intermediate Stages: REPEATABLE/LOOPABLE. 
         - You MAY stay in the same intermediate stage for multiple turns.
         - Do NOT rush to the next stage if there are still details to cover in the current category.
   - CONTINUOUS FEEDBACK: Do NOT stop the conversation. Do NOT say "found". Even if the image looks correct, assume there might be small differences and verify details or reinforce the description. 
   - The system will automatically decide when to stop. Your job is ONLY to describe and refine.
   - Natural Transition: Do NOT say "Next stage is...". Just change the specific attribute you are describing based on the current stage definition.

3. Match Check Strategy (CRITICAL for Multi-Target Tasks):
   - IF the task involves a "Baseline" (Target 1) and a "Modified" version (Target 2):
     (1) Before you declare "Baseline Match Found": Compare retrieved images against Target 1.
     (2) After you declare "Baseline Match Found": Compare retrieved images against Target 2.
   - ELSE (Single Target Task):
     - Always compare against the single Target Description.

[OUTPUT FORMATS] 

[FIRST TURN FORMAT]: 
[Reflection]  
Current Stage: [Copy the Stage Name explicitly from 'dialogue_pattern'] | Progress: 0 percent 
Target Focus: [Target 1 / Target 2 / Single Target]
[Output]
Requirements: [Your broad, initial request in natural speech]

[INTERMEDIATE TURN FORMAT]: 
[Reflection] 
Match Check: [No/Yes, Baseline Found(for Multi-Target Tasks)]
Current Stage: [Copy the Stage Name explicitly from 'dialogue_pattern'] | Progress: [X]percent revealed | Gap: [Key missing attributes]
Attributes Count: [X] (Must be 1 or 2) 
Target Focus: [Which target are you comparing against now?]
Plan: [Continue to add more details / Refine existing details] 
[Output] 
Requirements: [Natural speech: ONLY New specific detail(s) for this turn] Role Simulation: This mimics how a REAL USER would speak (fragmented, incremental). 

[CURRENT TASK CONFIG]
{task_template}
"""



class OpenAICompatibleClient:
    def __init__(self, api_key, base_url, model_name):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def encode_image(self, image_path):
        if isinstance(image_path, bytes): image_path = image_path.decode('utf-8')
        if not os.path.exists(image_path): return None
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type: mime_type = 'image/jpeg'
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return f"data:{mime_type};base64,{encoded_string}"
        except: return None

    def chat_with_history(self, messages, max_retries=3, temperature=0.7):
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=2048,
                    timeout=180
                )
                content = response.choices[0].message.content
                if not content: raise ValueError("Empty response")
                return content
            except Exception as e:
                error_str = str(e)
                wait_time = 2 ** attempt
                if "429" in error_str:
                    wait_time = 10 + (attempt * 5)
                    print(f"🛑 API Rate Limit (429). Cooling down for {wait_time}s...")
                print(f"⚠️ API Error (Attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
        return "Error: API Request Failed Max Retries"

    def chat_completion(self, system_prompt, user_text, image_paths=None, max_retries=3, temperature=0.7):
        messages = [{"role": "system", "content": system_prompt}]
        user_content = [{"type": "text", "text": user_text}]
        if image_paths:
            for path in image_paths:
                base64_img = self.encode_image(path)
                if base64_img:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": base64_img, "detail": "low"}
                    })
        messages.append({"role": "user", "content": user_content})
        return self.chat_with_history(messages, max_retries, temperature=temperature)


def get_env_setting(env_name, default=None):
    if not env_name:
        return default
    return os.environ.get(env_name, default)


def build_user_client(args):
    api_key = get_env_setting(args.api_key_env)
    if not api_key:
        raise ValueError(
            f"Missing API key. Set the {args.api_key_env} environment variable before running this script."
        )
    base_url = get_env_setting(args.base_url_env, args.base_url)
    return OpenAICompatibleClient(api_key=api_key, base_url=base_url, model_name=args.user_model)


def resolve_template_paths(template_dir):
    if os.path.isdir(template_dir):
        return sorted(glob.glob(os.path.join(template_dir, "*.yaml")))

    candidate_dir = os.path.abspath(template_dir)
    if os.path.isdir(candidate_dir):
        return sorted(glob.glob(os.path.join(candidate_dir, "*.yaml")))

    raise FileNotFoundError(f"Template directory not found: {template_dir}")


class UserAgent:
    def __init__(self, client: OpenAICompatibleClient, task_config: Dict):
        self.client = client
        self.config = task_config
        base_prompt = USER_SYSTEM_PROMPT_TEMPLATE.replace(
            "{{target_definitions}}", task_config['target_definitions']
        )
        display_config = copy.deepcopy(task_config)
        display_config.pop('min_turns', None)
        display_config.pop('max_turns', None)
        final_prompt = base_prompt.replace("{task_template}", json.dumps(display_config, indent=2))
        self.system_prompt = final_prompt
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def act(self, retrieved_images: List[str], turn_idx: int):
        current_content = []
        if turn_idx == 0:
            txt = "START THE SIMULATION. Please provide the first requirements."
        else:
            txt = (
                f"=== CURRENT TURN: {turn_idx + 1} ===\n"
                f"I have retrieved {len(retrieved_images)} images.\n"
                f"TASK: Compare the retrieved images with the Target.\n"
                f"Identify the differences and output the next specific requirement to refine the search.\n"
            )
        current_content.append({"type": "text", "text": txt})
        if retrieved_images:
            for path in retrieved_images:
                base64_img = self.client.encode_image(path)
                if base64_img:
                    current_content.append({
                        "type": "image_url",
                        "image_url": {"url": base64_img, "detail": "low"}
                    })

        messages_to_send = copy.deepcopy(self.messages)
        messages_to_send.append({"role": "user", "content": current_content})

        for attempt in range(3):
            raw_response = self.client.chat_with_history(messages_to_send, temperature=1)
            parsed_result = self._parse(raw_response)
            if parsed_result['is_valid']:
                self._update_memory(current_content, raw_response)
                parsed_result['raw_response'] = raw_response
                return parsed_result
            else:
                print(f"⚠️ [Format Error] Retrying...")
                messages_to_send.append({"role": "assistant", "content": raw_response})
                messages_to_send.append({
                    "role": "user",
                    "content": "SYSTEM ERROR: Missing [Output] block. Please Re-generate."
                })
        return {"status": "searching", "requirements": "Error", "is_valid": False}

    def _update_memory(self, user_content, assistant_response):
        for msg in self.messages:
            if msg['role'] == 'user' and isinstance(msg['content'], list):
                for item in msg['content']:
                    if item['type'] == 'image_url':
                        item['type'] = 'text'
                        item['text'] = "[Image from previous turn removed to save memory]"
                        item.pop('image_url', None)
        self.messages.append({"role": "user", "content": user_content})
        self.messages.append({"role": "assistant", "content": assistant_response})

    def _parse(self, text):
        result = {"status": "searching", "reflection": "", "requirements": "", "is_valid": False}
        reflect_match = re.search(r"\[Reflection\](.*?)(?=\[Output\]|$)", text, re.IGNORECASE | re.DOTALL)
        if reflect_match:
            result["reflection"] = reflect_match.group(1).strip()
        req_match = re.search(r"Requirements:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
        if req_match:
            req_content = req_match.group(1).strip()
            if "[" in req_content:
                req_content = req_content.split("[")[0].strip()
            result["requirements"] = req_content
            result["is_valid"] = True
            return result
        if "[Output]" in text:
            output_content = text.split("[Output]")[1].strip()
            if len(output_content) > 5:
                result["requirements"] = output_content
                result["is_valid"] = True
                return result
        return result

def calculate_md5(file_path):
    if not os.path.exists(file_path): return None
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_ordinal(n):
    suffix = 'th' if 10 <= n % 100 <= 20 else {1:'st', 2:'nd', 3:'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"



QUERY_INSTRUCTION = "Find an image that matches the given text."


def get_query_embedding(embedding_model: GmeQwen2VL, new_query: list, gallery_embeddings: torch.Tensor):
    """
    Replace the original Qwen3VL embedding logic with the GME interface.

    new_query format (kept consistent with the original script):
      Turn 0 : [{"text": "user_req_string"}]
      Turn 1+: [{"text": ["req1", "req2", ...], "image": ["path1", "path2", "path3"]}]

    Returns: torch.Tensor with shape (1, D) and the same dtype as gallery_embeddings
    """
    item = new_query[0]

    # ---------- Merge text content ----------
    if isinstance(item["text"], str):
        combined_text = item["text"]
    else:
        combined_text = "\n".join(item["text"])

    # ---------- Turn 0: text-only query ----------
    if "image" not in item or not item["image"]:
        emb = embedding_model.get_text_embeddings(
            texts=[combined_text],
            instruction=QUERY_INSTRUCTION,
            batch_size=1,
            show_progress_bar=False,
        )  # shape: (1, D)

    # ---------- Turn 1+: text + top-k images, pairwise embedding then mean pool ----------
    else:
        image_paths = item["image"]
        pil_images = []
        for p in image_paths:
            try:
                pil_images.append(Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"⚠️ Failed to open image {p}: {e}")

        if not pil_images:
            # Fall back to text-only retrieval if all images fail to load.
            emb = embedding_model.get_text_embeddings(
                texts=[combined_text],
                instruction=QUERY_INSTRUCTION,
                batch_size=1,
                show_progress_bar=False,
            )
        else:
            # Pair the same text with each image, then average the fused embeddings.
            texts_batch = [combined_text] * len(pil_images)
            emb_all = embedding_model.get_fused_embeddings(
                texts=texts_batch,
                images=pil_images,
                instruction=QUERY_INSTRUCTION,
                is_query=True,
                batch_size=len(pil_images),
                show_progress_bar=False,
            )  # shape: (k, D)
            emb = emb_all.mean(dim=0, keepdim=True)  # shape: (1, D)

    emb = emb.to(dtype=gallery_embeddings.dtype, device=gallery_embeddings.device)
    return emb



def main_simulation(args, template_config, target_data, user_client,
                    gallery_embeddings, gallery_names, is_single_target, top_k=3):
    target_abs_path = target_data['image_path']
    target_filename = os.path.basename(target_abs_path)
    target_md5 = calculate_md5(target_abs_path)

    if is_single_target:
        task_config = {
            "template_id": template_config['template_id'],
            "task_id": target_data['id'],
            "target_definitions": f"Target Description: {target_data['description']}",
            "dialogue_pattern": template_config['dialogue_pattern'],
            "special_instructions": template_config.get('special_instructions', {}),
            "max_turns": template_config['settings']['max_turns']
        }
    else:
        target_def_str = (
            f"target_image_1_description: {target_data['first_target_description']}\n"
            f"target_image_2_description: {target_data['description']}\n"
        )
        task_config = {
            "template_id": template_config['template_id'],
            "task_id": target_data['id'],
            "target_definitions": target_def_str.strip(),
            "dialogue_pattern": template_config['dialogue_pattern'],
            "special_instructions": template_config.get('special_instructions', {}),
            "max_turns": template_config['settings']['max_turns']
        }

    user_agent = UserAgent(user_client, task_config)
    retrieved_paths = []
    history_log = []
    is_success = False
    fail_reason = "Max turns reached"
    final_rank = -1

    print(f"\n==========================================")
    print(f"🚀 TASK STARTED: {target_data['id']}")
    print(f"Target: {target_abs_path}")
    print(f"==========================================")

    for turn in range(task_config['max_turns']):
        rename_turn = get_ordinal(turn + 1)
        print(f"--- The {rename_turn} Turn ---")

        user_result = user_agent.act(retrieved_paths, turn)
        print(f"User Raw Output:\n{user_result.get('raw_response', '').strip()}")
        print(f"\n>> User Requirements: {user_result['requirements']}")

        turn_data = {
            "turn": turn + 1,
            "user_reflection": user_result.get("reflection", ""),
            "user_req": user_result['requirements']
        }

        if turn == 0:
            new_query = [{"text": turn_data["user_req"]}]
        else:
            new_query = [{
                "text": [item["user_req"] for item in history_log] + [turn_data["user_req"]],
                "image": history_log[-1]["retrieved_top3"]
            }]
        turn_data["query"] = new_query
        print(f"{new_query}")

        
        if args.adapter_file is None:
            input_embeddings = get_query_embedding(embedding_model, new_query, gallery_embeddings)
        else:
            
            if isinstance(new_query[0]['text'], str):
                combined_text = new_query[0]['text']
            else:
                combined_text = "\n".join(new_query[0]['text'])
            if 'image' in new_query[0]:
                content_str = f"<image><image><image>{combined_text}"
                infer_requests = [InferRequest(
                    messages = [
                        {
                            'role': 'user',
                            'content': content_str
                        }
                    ],
                    images = new_query[0]['image']
                )]
            else:    
                content_str = f"{combined_text}"
                infer_requests = [InferRequest(
                    messages = [
                        {
                            'role': 'user',
                            'content': content_str
                        }
                    ],
                )]
            resp_list = embedding_model.infer(infer_requests)
            input_embeddings = torch.tensor(resp_list[0].data[0].embedding).cuda().reshape(1, -1)
            input_embeddings = input_embeddings.to(dtype=gallery_embeddings.dtype)

        # Similarity computation
        input_embeddings = input_embeddings.cpu().float()#new fix
        similarity_scores = input_embeddings @ gallery_embeddings.float().T
        scores = similarity_scores.numpy()[0]

        # similarity_scores = input_embeddings @ gallery_embeddings.T
        # scores = similarity_scores.cpu().float().numpy()[0]

        top_indices = np.argsort(scores)[::-1][:top_k]
        new_retrieved_paths = [gallery_names[index] for index in top_indices]
        sims = [scores[index] for index in top_indices]

        print(f">> Retrieved Top {top_k} Images:")
        for i, (sim, path) in enumerate(zip(sims, new_retrieved_paths)):
            print(f"   [{i+1}] Path: {os.path.join(os.getcwd(), path)} (Sim: {sim:.4f})")

        turn_data["retrieved_top3"] = new_retrieved_paths
        retrieved_paths = new_retrieved_paths
        history_log.append(turn_data)

        # Target hit detection
        match_rank = -1
        for rank, db_image_path in enumerate(retrieved_paths):
            db_filename = os.path.basename(db_image_path)
            if db_filename == target_filename:
                match_rank = rank + 1
            elif target_md5 and os.path.exists(db_image_path):
                if calculate_md5(db_image_path) == target_md5:
                    match_rank = rank + 1
            if match_rank != -1:
                break

        if match_rank != -1:
            print(f"\n✅ SYSTEM DETECTED MATCH at Rank {match_rank}!")
            is_success = True
            final_rank = match_rank
            fail_reason = None
            ordinal_rank = get_ordinal(match_rank)
            history_log.append({
                "turn": turn + 2,
                "user_reflection": "Match Check: Yes \nCurrent Stage: Lock on Target | Progress: 100 percent match",
                "user_req": f"The {ordinal_rank} image in the retrieved results",
                "status": "found",
                "retrieved_top3": retrieved_paths
            })
            break

        if turn == task_config['max_turns'] - 1:
            print(f"\n⌛ Max turn reached!")
            history_log[-1]["status"] = "max_turns_reached"
            break

        time.sleep(0.5)

    base_result = {
        "task_id": target_data['id'],
        "template_id": template_config.get('template_id', 'Unknown'),
        "target_description": target_data['description'],
        "target_image_path": target_data['image_path'],
        "is_success": is_success,
        "final_rank": final_rank,
        "turns_used": len(history_log),
        "fail_reason": fail_reason,
        "history": history_log
    }
    if not is_single_target:
        base_result["first_target_description"] = target_data['first_target_description']
        base_result["first_target_image_path"] = target_data['first_target_image_path']
    return base_result



def run_adaptive_generation(args):
    global embedding_model  

    current_results = []
    finished_task_ids = set()
    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    current_results = json.loads(content)
                    finished_task_ids = {str(item.get('target_image_path')) for item in current_results}
            print(f"✅ Resuming... {len(finished_task_ids)} tasks already done.")
        except: pass

    print(f"⚙️  Initializing Clients...")
    user_client = build_user_client(args)

    # Initialize the GME model.
    if args.adapter_file is None:
        embedding_model = GmeQwen2VL(model_name=args.model_name_or_path,device="cuda" if torch.cuda.is_available() else "cpu",)
    else:
        embedding_model = TransformersEngine(
                    args.model_name_or_path,
                    torch_dtype='bfloat16',
                    task_type='embedding',
                    attn_impl='flash_attention_2',
                    adapters=[args.adapter_file])#
    
    print(f"✅ GME Embedding Model Loaded: {args.model_name_or_path}")

    # Load the gallery embedding index.
    if not os.path.exists(args.gallery_embedding_path):
        print("❌ Index files not found.")
        return
    try:
        data = np.load(args.gallery_embedding_path, allow_pickle=True)
        gallery_names = data['paths']
        gallery_embeddings = torch.from_numpy(data['embeddings'])#.cuda()
        print(f"✅ Gallery Loaded: {gallery_embeddings.shape}")
    except Exception as e:
        print(f"❌ Retriever Init Error: {e}")
        return

    # Load template files.
    all_single_templates = resolve_template_paths(args.template_single_target)
    all_double_templates = resolve_template_paths(args.template_double_target)
    all_template_names = (
        [os.path.basename(t).replace("_1", "") for t in all_single_templates] +
        [os.path.basename(t) for t in all_double_templates]
    )
    print(f"🔍 Found {len(all_template_names)} Template Files.")

    with open(args.target_items, "r") as f:
        test_pairs = json.load(f)
    pending_tasks = [t for t in test_pairs if str(t['target_image_path']) not in finished_task_ids]
    print(f"⏭️  Tasks remaining: {len(pending_tasks)}")

    for instance in tqdm(pending_tasks):
        description = instance["target_description"]
        task_id = instance["task_id"]
        img_path = instance["target_image_path"]
        template_id_name = instance["template_id"].split("_")[0] + ".yaml"

        path_index = all_template_names.index(template_id_name)
        is_single_target = path_index < len(all_single_templates)
        template_path = (
            all_single_templates[path_index]
            if is_single_target
            else all_double_templates[path_index - len(all_single_templates)]
        )

        with open(template_path, "r", encoding="utf-8") as f:
            template_config = yaml.safe_load(f)

        if is_single_target:
            target_data = {
                "id": task_id,
                "first_target_image_path": '',
                "first_target_description": '',
                "image_path": img_path,
                "description": description
            }
        else:
            target_data = {
                "id": task_id,
                "first_target_image_path": instance["first_target_image_path"],
                "first_target_description": instance["first_target_description"],
                "image_path": img_path,
                "description": description
            }

        try:
            result = main_simulation(
                args, template_config, target_data, user_client,
                gallery_embeddings, gallery_names, is_single_target
            )
            current_results.append(result)
            with open(args.output_file, "w", encoding='utf-8') as f:
                json.dump(current_results, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Error processing {task_id}: {e}")
            time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_single_target", type=str, default=DEFAULT_SINGLE_TEMPLATE_DIR)
    parser.add_argument("--template_double_target", type=str, default=DEFAULT_MULTI_TEMPLATE_DIR)
    parser.add_argument("--target_items", type=str, default=DEFAULT_TARGET_ITEMS)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--gallery_embedding_path",type=str,default=DEFAULT_GALLERY_EMBEDDING_PATH,help="path to the prebuilt gallery embedding index",)
    parser.add_argument("--model_name_or_path",type=str,default="Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",help="public model name or local checkpoint path",)
    parser.add_argument("--output_file", type=str, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--adapter_file", type=str, default=None, help="optional adapter checkpoint path")
    parser.add_argument("--user_model", type=str, default=USER_MODEL)
    parser.add_argument("--base_url", type=str, default=None, help="fallback OpenAI-compatible base URL")
    parser.add_argument("--api_key_env", type=str, default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--base_url_env", type=str, default=DEFAULT_BASE_URL_ENV)

    args = parser.parse_args()

    run_adaptive_generation(args)
