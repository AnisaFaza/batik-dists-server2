from flask import Flask, request, jsonify
from flask_cors import CORS
from DISTS_pt import DISTS
import torch
from PIL import Image
import requests
import io

app = Flask(__name__)
CORS(app)

# Setup DISTS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DISTS().to(device)
model.eval()

# Transform image
import torchvision.transforms as transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def compute_dists_from_url(ref_url, user_url):
    # Download images
    r1 = requests.get(ref_url)
    r2 = requests.get(user_url)
    img1 = Image.open(io.BytesIO(r1.content)).convert('RGB')
    img2 = Image.open(io.BytesIO(r2.content)).convert('RGB')
    
    # Transform
    t1 = transform(img1).unsqueeze(0).to(device)
    t2 = transform(img2).unsqueeze(0).to(device)
    
    # Hitung DISTS
    with torch.no_grad():
        score = model(t1, t2).item()
    
    # Normalisasi supaya 0–1 (mirip dengan script percobaanmu)
    score = 1 - score
    score = max(0, min(score, 1))
    
    return score

@app.route("/compare", methods=["POST"])
def compare():
    data = request.json
    ref_url = data.get("refImageUrl")
    user_url = data.get("userImageUrl")
    if not ref_url or not user_url:
        return jsonify({"error": "URL gambar tidak lengkap"}), 400

    try:
        score = compute_dists_from_url(ref_url, user_url)
        return jsonify({"score": score, "metric": "DISTS"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
