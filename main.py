import os
import json
import whisper
import openai
from flask import Flask, request, jsonify, render_template_string
import subprocess

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

memory_file = "memory.json"
if not os.path.exists(memory_file):
    with open(memory_file, "w") as f:
        json.dump([], f)

def extract_audio(video_path, audio_path):
    command = f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 44100 -ac 2 {audio_path} -y"
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def transcribe_video(file_path):
    audio_path = "temp_audio.wav"
    extract_audio(file_path, audio_path)
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    os.remove(audio_path)
    return result["text"]

def learn_from_text(text):
    with open(memory_file, "r") as f:
        memory = json.load(f)
    memory.append({"source": "video", "content": text})
    with open(memory_file, "w") as f:
        json.dump(memory, f)

@app.route("/", methods=["GET"])
def home():
    return render_template_string(open("frontend.html").read())

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("message")
    with open(memory_file, "r") as f:
        memory = json.load(f)
    context = "\n".join([m["content"] for m in memory])
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Eres un asistente que ha aprendido de estos videos: " + context},
            {"role": "user", "content": question}
        ]
    )
    return jsonify({"reply": response.choices[0].message.content})

@app.route("/learn", methods=["POST"])
def learn():
    file = request.files["video"]
    path = "temp_video.mp4"
    file.save(path)
    text = transcribe_video(path)
    os.remove(path)
    learn_from_text(text)
    return jsonify({"message": "Aprendido correctamente del video."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
