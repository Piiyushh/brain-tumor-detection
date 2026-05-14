from flask import Flask, render_template, request, redirect, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

from datetime import datetime
import pytz
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import gc

# ================= PERFORMANCE =================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

torch.set_num_threads(1)

device = torch.device("cpu")

# ================= APP =================

app = Flask(__name__)
app.secret_key = "secret123"

# ================= DATABASE =================

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

IST = pytz.timezone('Asia/Kolkata')

# ================= DATABASE MODELS =================

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(300))


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    result = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(IST))


# ================= MODEL =================

classes = [
    'glioma_tumor',
    'meningioma_tumor',
    'no_tumor',
    'pituitary_tumor'
]

print("========== LOADING MODEL ==========")

model = EfficientNet.from_name('efficientnet-b0')

model._fc = nn.Linear(
    model._fc.in_features,
    4
)

model.load_state_dict(
    torch.load(
        "model/brain_tumor_model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

print("========== MODEL LOADED ==========")

# ================= IMAGE TRANSFORM =================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

UPLOAD_FOLDER = "static/images"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= HOME =================

@app.route("/")
def home():
    return render_template("home.html")


# ================= REGISTER =================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        existing_user = User.query.filter_by(
            username=username
        ).first()

        if existing_user:
            return "User already exists"

        hashed_password = generate_password_hash(password)

        user = User(
            username=username,
            password=hashed_password
        )

        db.session.add(user)
        db.session.commit()

        return redirect("/login")

    return render_template("register.html")


# ================= LOGIN =================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        user = User.query.filter_by(
            username=username
        ).first()

        if user and check_password_hash(user.password, password):

            session["user"] = username

            return redirect("/detect")

        return "Invalid credentials"

    return render_template("login.html")


# ================= LOGOUT =================

@app.route("/logout")
def logout():

    session.pop("user", None)

    return redirect("/login")


# ================= DETECT =================

@app.route("/detect", methods=["GET", "POST"])
def detect():

    if "user" not in session:
        return redirect("/login")

    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":

        try:

            file = request.files.get("image")

            if not file:
                return "No image uploaded"

            img_pil = Image.open(file).convert("RGB")

            image_path = os.path.join(
                UPLOAD_FOLDER,
                "uploaded.jpg"
            )

            img_pil.save(image_path)

            img_tensor = transform(img_pil).unsqueeze(0).to(device)

            with torch.no_grad():

                outputs = model(img_tensor)

                probs = F.softmax(outputs, dim=1)

                conf, pred = torch.max(probs, 1)

            prediction = classes[pred.item()]
            confidence = round(conf.item() * 100, 2)

            # ================= SAVE HISTORY =================

            entry = Prediction(
                username=session["user"],
                result=prediction,
                confidence=confidence
            )

            db.session.add(entry)
            db.session.commit()

            # cleanup memory
            del img_tensor
            gc.collect()

        except Exception as e:

            print("========== DETECTION ERROR ==========")
            print(str(e))

            return f"Detection Error: {str(e)}"

    return render_template(
        "detect.html",
        prediction=prediction,
        confidence=confidence,
        image="/" + image_path if image_path else None
    )


# ================= HISTORY =================

@app.route("/history")
def history():

    if "user" not in session:
        return redirect("/login")

    data = Prediction.query.filter_by(
        username=session["user"]
    ).order_by(
        Prediction.created_at.desc()
    ).all()

    return render_template(
        "history.html",
        history=data
    )


# ================= DASHBOARD =================

@app.route("/dashboard")
def dashboard():

    if "user" not in session:
        return redirect("/login")

    user = session["user"]

    records = Prediction.query.filter_by(
        username=user
    ).all()

    total = len(records)

    tumor = sum(
        1 for r in records
        if r.result != "no_tumor"
    )

    no_tumor = sum(
        1 for r in records
        if r.result == "no_tumor"
    )

    return render_template(
        "dashboard.html",
        total=total,
        tumor=tumor,
        no_tumor=no_tumor
    )


# ================= MODELS PAGE =================

@app.route("/models")
def models_page():
    return render_template("models.html")


# ================= ABOUT PAGE =================

@app.route("/about")
def about():
    return render_template("about.html")


# ================= CHATBOT =================

@app.route("/chat", methods=["POST"])
def chat():

    user_msg = request.json.get("message")

    if "tumor" in user_msg.lower():
        reply = "A tumor is an abnormal growth of cells in the brain."

    elif "glioma" in user_msg.lower():
        reply = "Glioma is a type of tumor that occurs in the brain and spinal cord."

    elif "meningioma" in user_msg.lower():
        reply = "Meningioma arises from the membranes surrounding the brain."

    elif "pituitary" in user_msg.lower():
        reply = "Pituitary tumors occur in the pituitary gland."

    else:
        reply = "I'm here to help with brain tumor related questions."

    return {"reply": reply}


# ================= DATABASE INIT =================

with app.app_context():
    db.create_all()


# ================= RUN =================

if __name__ == "__main__":
    app.run(debug=True)