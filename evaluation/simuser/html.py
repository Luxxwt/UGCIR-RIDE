import argparse
import base64
import json
import os
from io import BytesIO

from PIL import Image
from tqdm import tqdm


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DEFAULT_INPUT_FILE = os.path.join(REPO_ROOT, "evaluation", "retrieval", "gme_results.json")
DEFAULT_OUTPUT_DIR = os.path.join(CURRENT_DIR, "html")
DEFAULT_MAX_IMG_SIZE = (500, 500)


def image_to_base64(image_path, max_img_size=DEFAULT_MAX_IMG_SIZE):
    """Convert a local image to a base64 data URI."""
    try:
        if not image_path or not os.path.exists(image_path):
            return None

        img = Image.open(image_path)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        if max_img_size:
            img.thumbnail(max_img_size)

        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return None


def generate_html_content(task_data, max_img_size=DEFAULT_MAX_IMG_SIZE):
    """Generate one HTML report for a retrieval task."""
    task_id = task_data.get("task_id", "unknown")
    template_id = task_data.get("template_id", "N/A")
    target_desc = task_data.get("target_description", "N/A")
    target_path = task_data.get("target_image_path", "not recorded")
    is_success = task_data.get("is_success", False)
    status_color = "#d4edda" if is_success else "#f8d7da"
    status_text = "SUCCESS" if is_success else "FAILED"
    status_font_color = "#155724" if is_success else "#721c24"
    history = task_data.get("history", [])

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{task_id}</title>
  <style>
    body {{ font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; background-color: #f2f2f7; margin: 0; padding: 20px; }}
    .chat-container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
    .task-info {{ background: {status_color}; color: {status_font_color}; padding: 15px; border-radius: 8px; margin-bottom: 25px; border: 1px solid rgba(0,0,0,0.1); }}
    .task-info h2 {{ margin: 0 0 10px 0; display: flex; justify-content: space-between; align-items: center; }}
    .task-meta {{ font-size: 14px; margin-top: 5px; opacity: 0.9; }}
    .target-box {{ background: rgba(255,255,255,0.6); padding: 10px; border-radius: 6px; margin-top: 10px; font-style: italic; font-size: 14px; }}
    .path-text {{ font-family: Consolas, "Courier New", monospace; font-size: 10px; color: #555; background: #f1f1f1; padding: 4px; border-radius: 4px; word-break: break-all; margin-top: 4px; line-height: 1.2; border: 1px solid #ddd; }}
    .target-path-box {{ margin-top: 5px; font-weight: bold; font-size: 12px; }}
    .message-group {{ margin-bottom: 30px; clear: both; overflow: hidden; }}
    .user-container {{ float: right; max-width: 85%; display: flex; flex-direction: column; align-items: flex-end; }}
    .thought-bubble {{ background-color: #f9f9fc; color: #555; border: 1px dashed #ccc; padding: 10px 15px; border-radius: 12px; font-size: 13px; font-family: Consolas, "Courier New", monospace; margin-bottom: 8px; max-width: 100%; text-align: left; white-space: pre-wrap; }}
    .speech-bubble {{ background-color: #007aff; color: white; padding: 12px 18px; border-radius: 18px; border-bottom-right-radius: 4px; font-size: 15px; line-height: 1.4; box-shadow: 0 1px 2px rgba(0,0,0,0.1); text-align: left; }}
    .meta {{ font-size: 12px; color: #999; margin-bottom: 4px; text-align: right; margin-right: 5px; }}
    .system-container {{ float: left; max-width: 90%; margin-top: 10px; }}
    .system-bubble {{ background-color: #e5e5ea; color: black; padding: 15px; border-radius: 18px; border-bottom-left-radius: 4px; }}
    .system-meta {{ font-size: 12px; color: #999; margin-bottom: 4px; text-align: left; margin-left: 5px; }}
    .image-grid {{ display: flex; flex-wrap: wrap; gap: 15px; margin-top: 12px; align-items: flex-start; }}
    .img-card {{ display: flex; flex-direction: column; align-items: center; width: 180px; }}
    .img-wrapper {{ width: 100%; height: 180px; border-radius: 6px; border: 2px solid white; box-shadow: 0 2px 5px rgba(0,0,0,0.1); background-color: #f0f0f0; display: flex; justify-content: center; align-items: center; overflow: hidden; }}
    .img-wrapper:hover {{ transform: scale(1.05); transition: transform 0.2s; }}
    .img-wrapper img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
    .img-caption {{ font-size: 11px; color: #333; margin-top: 4px; font-weight: bold; text-align: center; }}
    .tag {{ display: inline-block; padding: 3px 8px; border-radius: 6px; font-size: 12px; font-weight: bold; margin-bottom: 8px; }}
    .tag-summary {{ background-color: #ffcc00; color: #333; }}
    .found-banner, .fail-banner {{ clear: both; text-align: center; margin: 40px 0; padding: 10px; border-radius: 8px; font-weight: bold; }}
    .found-banner {{ background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }}
    .fail-banner {{ background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }}
  </style>
</head>
<body>
  <div class="chat-container">
    <div class="task-info">
      <h2><span>Task: {task_id}</span> <span>[{status_text}]</span></h2>
      <div class="task-meta">Template: <strong>{template_id}</strong> | Total Turns: {len(history)}</div>
      <div class="target-box">
        <div><strong>Description:</strong> {target_desc}</div>
        <div class="target-path-box"><strong>Target Path:</strong> <span class="path-text" style="background:none; border:none; padding:0;">{target_path}</span></div>
      </div>
    </div>
"""

    for turn_data in history:
        turn_num = turn_data.get("turn", "?")
        user_reflection = turn_data.get("user_reflection", "")
        user_req = turn_data.get("user_req", "")
        query = turn_data.get("query", "")
        images = turn_data.get("retrieved_top3", [])
        status = turn_data.get("status", "searching")

        html += f"""
    <div class="message-group">
      <div class="user-container">
        <div class="meta">User (Turn {turn_num})</div>
"""
        if user_reflection:
            html += f"""        <div class="thought-bubble"><strong>Reflection:</strong> {user_reflection}</div>\n"""
        html += f"""        <div class="speech-bubble">{user_req}</div>\n      </div>\n    </div>\n"""

        if status == "found":
            html += f"""    <div class="found-banner">TARGET FOUND at Turn {turn_num}</div>\n"""
        elif status == "max_turns_reached":
            html += """    <div class="fail-banner">Max Turns Reached</div>\n"""

        if images:
            image_cards = []
            for idx, image_path in enumerate(images):
                b64_src = image_to_base64(image_path, max_img_size=max_img_size)
                if b64_src:
                    image_markup = f'<img src="{b64_src}" alt="Rank {idx + 1}" loading="lazy">'
                else:
                    image_markup = '<div style="padding:20px; text-align:center; font-size:12px;">File Not Found</div>'

                image_cards.append(f"""
          <div class="img-card">
            <div class="img-wrapper">{image_markup}</div>
            <div class="img-caption">Rank {idx + 1}</div>
            <div class="path-text" title="{image_path}">{image_path}</div>
          </div>""")

            query_html = f'<div class="tag tag-summary">Updated Query: {query}</div>' if query else ""
            html += f"""
    <div class="message-group">
      <div class="system-container">
        <div class="system-meta">System</div>
        <div class="system-bubble">
          {query_html}
          <div class="image-grid">{''.join(image_cards)}
          </div>
        </div>
      </div>
    </div>
"""

    html += """  </div>\n</body>\n</html>\n"""
    return html


def sanitize_filename(name):
    return "".join(c for c in name if c.isalnum() or c in ("-", "_", ".")).strip()


def load_tasks(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def main():
    parser = argparse.ArgumentParser(description="Generate HTML reports from retrieval results.")
    parser.add_argument(
        "--input_file",
        type=str,
        default=DEFAULT_INPUT_FILE,
        help="Path to a retrieval result JSON file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used to save generated HTML files.",
    )
    parser.add_argument(
        "--max_image_size",
        type=int,
        default=500,
        help="Maximum width/height used when embedding images in HTML.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input result file not found: {args.input_file}")

    os.makedirs(args.output_dir, exist_ok=True)
    tasks = load_tasks(args.input_file)
    max_img_size = (args.max_image_size, args.max_image_size) if args.max_image_size > 0 else None

    for task in tqdm(tasks, desc="Generating HTML"):
        task_id = task.get("task_id", "unknown")
        template_id = task.get("template_id", "no_template")
        safe_name = f"{sanitize_filename(template_id)[:3]}_{sanitize_filename(task_id)}"
        output_path = os.path.join(args.output_dir, f"{safe_name}.html")
        html_content = generate_html_content(task, max_img_size=max_img_size)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    print(f"Generated {len(tasks)} HTML report(s) in: {args.output_dir}")


if __name__ == "__main__":
    main()
