from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
import threading
import pandas as pd
import json
from datetime import datetime, timezone
from pfc_backend.api_v3 import factcheck, process_file, process_rss, trusted_df  
from flask import send_file
import shutil
import time

app = Flask(__name__)
CORS(app)
# CORS(app, origins=["http://localhost:5173"])

OUTPUTS_FOLDER = "outputs"
os.makedirs(OUTPUTS_FOLDER, exist_ok=True)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Track the task threads
tasks = {}

def analyze_task(task_id, file_path, rss_url, openai_key, perplexity_key):
    result = {}
    try:
        print(f"[{task_id}] Task started.")

        if file_path:
            print(f"[{task_id}] Processing local file: {file_path}")
            transcript = process_file(file_path)
        elif rss_url:
            print(f"[{task_id}] Processing RSS feed: {rss_url}")
            transcript = process_rss(rss_url)
        else:
            print(f"[{task_id}] No file or RSS provided.")
            transcript = ""

        print(f"[{task_id}] Transcription done. Length: {len(transcript)} characters.")

        # factcheck
        print(f"[{task_id}] Starting factcheck...")
        df = factcheck(transcript, openai_key, perplexity_key)
        print(f"[{task_id}] Factcheck done. {len(df)} claims extracted.")

        finished_time = datetime.now(timezone.utc).isoformat()
        file_name = "RSS Link" if rss_url else os.path.basename(file_path).split("_", 1)[1]
        data = df.to_dict(orient="records")
        result = {
            "task_id": task_id,
            "metadata": {
                "finished_time": finished_time,
                "file_name": file_name,
                "openai_model": "gpt-5-mini",
                "temprature": "1",
                "perplexity_model": "Sonar",
                "source": "local" if file_path else "rss",
                "record_count": len(data)
            },
            "data": data,
            "status": "done"
        }

    except Exception as e:
        print(f"[{task_id}] ERROR: {e}")
        result = {
            "task_id": task_id,
            "metadata": {},
            "data": [],
            "status": "error",
            "error": str(e)
        }

    finally:
        output_file = os.path.join(OUTPUTS_FOLDER, f"{task_id}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[{task_id}] Task finished. Result saved to {output_file}")
    return result

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("file")
    rss_url = request.form.get("rss_url")
    openai_key = request.form.get("api_key_openai")
    perplexity_key = request.form.get("api_key_perplexity")

    if not openai_key or not perplexity_key:
        return jsonify({
            "error": "Both OpenAI and Perplexity API keys are required"
        }), 400
    
    task_id = str(uuid.uuid4())
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, f"{task_id}_{file.filename}")
        file.save(file_path)
    else:
        file_path = None

    thread = threading.Thread(target=analyze_task, args=(task_id, file_path, rss_url, openai_key, perplexity_key))
    thread.start()
    tasks[task_id] = thread

    return jsonify({
        "task_id": task_id,
        "message": "Task started. Wait a few seconds and fetch result by task_id."
    })

@app.route("/result/<task_id>", methods=["GET"])
def get_result(task_id):
    output_file = os.path.join(OUTPUTS_FOLDER, f"{task_id}.json")
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    else:
        return jsonify({"task_id": task_id, "status": "processing", "metadata": {}, "data": []})

@app.route("/download/<task_id>", methods=["GET"])
def download(task_id):
    file_path = os.path.join(OUTPUTS_FOLDER, f"{task_id}.json")
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    return send_file(
        file_path,
        as_attachment=True,
        download_name=f"factcheck_{task_id}.json",
        mimetype="application/json"
    )

@app.route("/test-analyze", methods=["POST"])
def test_analyze():
    task_id = str(uuid.uuid4())

    time.sleep(5)

    template_file = os.path.join(OUTPUTS_FOLDER, "DO_NOT_DELETE", "DO_NOT_DELETE_FOR_TESTING.json")
    new_file_name = f"{task_id}.json"
    new_file_path = os.path.join(OUTPUTS_FOLDER, new_file_name)

    shutil.copy(template_file, new_file_path)

    with open(new_file_path, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data["task_id"] = task_id
        f.seek(0)
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.truncate()

    return send_file(
        new_file_path,
        as_attachment=True,
        download_name=f"factcheck_{task_id}.json",
        mimetype="application/json"
    )

@app.route("/cancel/<task_id>", methods=["POST"])
def cancel_task(task_id):
    cancel_file = f"/tmp/cancel_{task_id}.flag"
    with open(cancel_file, "w") as f:
        f.write("cancel")
    return jsonify({"task_id": task_id, "message": "Cancel requested"})

if __name__ == "__main__":
    app.run(debug=True, port=8000)
