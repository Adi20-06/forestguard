# 🌲 ForestGuard

### **Smarter Detection. Safer Forests.**

ForestGuard is an **AI-powered forest fire detection and alert system** built to help identify possible fire or smoke from forest images before a small incident becomes a major threat.

It combines a **Convolutional Neural Network (CNN)** with a simple, clean **Streamlit dashboard** so users can upload an image or capture one using a camera, run an AI analysis, and immediately see the result with a confidence score.

> 🌍 **The idea is simple:** use AI to turn an ordinary forest image into an early warning signal.

---

## 🚨 Why ForestGuard?

Forest fires can spread rapidly, especially in remote areas where continuous human monitoring is difficult.

Traditional monitoring can be:

* 👁️ Dependent on manual observation
* 🕐 Slow to respond
* 🌲 Difficult to scale across large forest regions
* 📡 Dependent on expensive monitoring infrastructure

**ForestGuard explores a lightweight alternative:**

```text
📷 Image → 🤖 AI Model → 🔥 Risk Level → 🚨 Alert
```

The goal is not to replace forest rangers, but to provide them with an additional **early-screening and decision-support tool**.

---

## ✨ Features

### 📸 1. Forest Image Analysis

Upload a `.jpg`, `.jpeg`, or `.png` image and let the trained CNN estimate the probability of fire.

### 📷 2. Camera Input

Capture an image directly through the browser and send it for analysis.

### 🤖 3. AI-Based Fire Detection

The application loads a trained TensorFlow/Keras CNN model and preprocesses the image before prediction.

### 📊 4. Confidence Score

Every analysis can display information such as:

* Fire probability
* Confidence score
* Detection level
* Timestamp
* Optional location/GPS
* Optional camera ID

### 🚦 5. Three-Level Detection System

| Level          | Meaning               |
| -------------- | --------------------- |
| 🟢 **SAFE**    | No fire detected      |
| 🟡 **WARNING** | Possible fire / smoke |
| 🔴 **ALERT**   | Fire detected         |

The detection threshold is configurable through `config.json`.

### 🕘 6. Detection History

ForestGuard maintains previous detection information locally so that results can be reviewed later.

Users can:

* View previous detections
* Filter detections
* Review timestamps
* Clear stored history when required

### 📍 7. Location Support

An optional GPS/location field can be attached to a detection.

Alert results can also provide a **View on Map** action when location information is available.

---

# 🧠 How ForestGuard Works

```text
                    ┌──────────────────────┐
                    │   Forest Image 📷    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Image Preprocessing  │
                    │   RGB + Resize       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   CNN Fire Model 🤖  │
                    │  TensorFlow / Keras  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Fire Probability 🔥  │
                    └──────────┬───────────┘
                               │
                  ┌────────────┼────────────┐
                  ▼            ▼            ▼
              🟢 SAFE      🟡 WARNING    🔴 ALERT
                  │            │            │
                  └────────────┼────────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Result + History 🕘  │
                    └──────────────────────┘
```

---

# 🧪 AI Prediction Pipeline

ForestGuard uses the trained model:

```text
fire_cnn_final.keras
```

When an image is submitted:

1. The image is converted to RGB.
2. The image is resized according to the project configuration.
3. The CNN processes the image.
4. The model generates a prediction score.
5. The score is compared with the configured threshold.
6. ForestGuard determines the detection level.
7. The result can be stored in the local detection history.

Current configuration:

```json
{
  "img_size": 128,
  "threshold": 0.3
}
```

> ⚠️ **Important:** A detection result should be treated as an AI-assisted indication, not as definitive proof of a fire. Small, distant, obscured, or unusual fires/smoke may be missed.

---

# 🖥️ Application Workflow

## 🌅 Home

The dashboard provides an overview of recent activity, including information such as:

* Number of images analyzed
* Recent alerts
* Recent detection activity
* Latest scan status

---

## 🔍 Detect Fire

This is the main AI analysis page.

Users can:

* Upload an image
* Use the camera
* Preview the image
* Add optional GPS/location information
* Add an optional camera ID
* Run the AI analysis

---

## 📋 Analysis Result

After prediction, ForestGuard presents the detection result.

Possible information includes:

```text
Detection Status
      ↓
Fire Probability
      ↓
Confidence Score
      ↓
Timestamp
      ↓
Location
      ↓
Camera ID
```

---

## 🕘 Detection History

Previous detections can be reviewed through the history interface.

Results can be organized by status:

```text
ALL
 │
 ├── 🔴 ALERT
 │
 ├── 🟡 WARNING
 │
 └── 🟢 SAFE
```

This makes it easier to review previous monitoring activity.

---

# 🛠️ Tech Stack

| Technology        | Purpose                         |
| ----------------- | ------------------------------- |
| 🐍 **Python**     | Core application                |
| 🎈 **Streamlit**  | Interactive web interface       |
| 🧠 **TensorFlow** | Machine learning framework      |
| 🤖 **Keras**      | CNN model loading and inference |
| 🖼️ **Pillow**    | Image processing                |
| 🔢 **NumPy**      | Numerical operations            |
| 🗂️ **JSON**      | Configuration and local data    |

### Architecture

```text
                 FORESTGUARD
                      │
        ┌─────────────┴─────────────┐
        │                           │
   Streamlit UI                AI Model
        │                           │
        │                    TensorFlow/Keras
        │                           │
        ▼                           ▼
  Image Input  ───────────────► CNN Prediction
        │                           │
        └─────────────┬─────────────┘
                      │
                      ▼
                Risk Analysis
                      │
             ┌────────┼────────┐
             ▼        ▼        ▼
           SAFE    WARNING    ALERT
                      │
                      ▼
                Detection History
```

---

# 📁 Project Structure

```text
forestguard/
│
├── .streamlit/
│   └── ...
│
├── app.py
│
├── config.json
│
├── fire_cnn_final.keras
│
└── requirements.txt
```

### Main Files

| File                   | Description                |
| ---------------------- | -------------------------- |
| `app.py`               | Main Streamlit application |
| `config.json`          | Model/image configuration  |
| `fire_cnn_final.keras` | Trained CNN model          |
| `requirements.txt`     | Python dependencies        |
| `.streamlit/`          | Streamlit configuration    |

The current GitHub repository contains these core project files.

---

# 🚀 Getting Started

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/Adi20-06/forestguard.git
```

Move into the project directory:

```bash
cd forestguard
```

---

## 2️⃣ Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

---

# 📦 3️⃣ Install Dependencies

Run:

```bash
pip install -r requirements.txt
```

The project uses packages including:

```text
tensorflow
keras
streamlit
pillow
numpy
tzdata
```

---

# ▶️ 4️⃣ Run ForestGuard

Start the Streamlit application:

```bash
streamlit run app.py
```

Streamlit will provide a local address similar to:

```text
http://localhost:8501
```

Open the address in your browser.

---

# ⚙️ Configuration

ForestGuard keeps important model settings inside:

```text
config.json
```

Example:

```json
{
  "img_size": 128,
  "threshold": 0.3
}
```

---

## 🖼️ Image Size

```json
"img_size": 128
```

This determines the image dimensions used during preprocessing before the image is passed to the CNN.

---

## 🔥 Detection Threshold

```json
"threshold": 0.3
```

The threshold controls how the prediction score is interpreted by the application.

The system can then classify the result into different levels:

```text
                 Prediction
                     │
                     ▼
              ┌─────────────┐
              │ AI Score    │
              └──────┬──────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
     🟢 SAFE      🟡 WARNING    🔴 ALERT
```

> **Note:** Thresholds should ideally be selected using validation data and evaluated using metrics such as precision, recall, F1-score, and false-positive/false-negative rates.

---

# 📊 Example Workflow

Imagine a forest monitoring camera captures an image.

### Step 1 — Capture

📷 A forest image is captured.

### Step 2 — Upload

The image is provided to ForestGuard.

### Step 3 — Preprocessing

The image is converted and resized.

### Step 4 — AI Analysis

🤖 The CNN analyzes the image.

### Step 5 — Prediction

The model generates a fire probability.

### Step 6 — Classification

ForestGuard determines whether the result should be treated as:

```text
🟢 SAFE
🟡 WARNING
🔴 ALERT
```

### Step 7 — Record

The detection can be added to the local history.

### Step 8 — Response

If the result is concerning, the user can investigate the image and, when available, its associated location.

---

# 🎯 Project Goals

ForestGuard is built around four major goals.

## 1. 🔥 Early Detection

Use computer vision to identify possible signs of forest fire from images.

## 2. 🌲 Accessibility

Provide a simple interface so users do not need to interact directly with machine-learning code.

## 3. 📊 Traceability

Maintain a history of previous detection events for review.

## 4. 🚀 Extensibility

Create a foundation that can later integrate with:

* Forest monitoring cameras
* IoT sensors
* GPS
* Cloud platforms
* Notification systems
* Real-time monitoring

---

# 🔮 Future Improvements

ForestGuard can evolve from an image-analysis prototype into a complete forest monitoring platform.

## 🔥 Real-Time Monitoring

Connect fixed forest cameras and automatically analyze incoming frames.

```text
Camera
   ↓
Continuous Frames
   ↓
AI Detection
   ↓
Risk Analysis
   ↓
Alert
```

---

## 📡 IoT Integration

Combine computer vision with environmental sensors such as:

* 🌡️ Temperature
* 💨 Smoke
* 💧 Humidity
* 🌬️ Air quality

Possible hardware:

* ESP32
* Raspberry Pi
* LoRa/LoRaWAN devices
* Edge AI devices

---

## 🗺️ Live Monitoring Map

Create a centralized map showing:

```text
🟢 Normal monitoring point
🟡 Warning
🔴 Active alert
```

This would allow multiple forest locations to be monitored from one dashboard.

---

## 🔔 Notification System

Future versions could automatically notify responsible personnel through:

* 📧 Email
* 📱 SMS
* 🔔 Push notifications
* 💬 Messaging platforms

---

## ☁️ Cloud Deployment

The current application can be extended with cloud infrastructure for:

* Centralized detection history
* Multi-user access
* Remote monitoring
* Large-scale image storage
* Model serving

---

## 🧠 Explainable AI

A major future improvement would be adding **Explainable AI (XAI)**.

Techniques such as **Grad-CAM** could highlight the regions of an image that influenced the CNN prediction.

For example:

```text
Original Image
      ↓
    CNN
      ↓
Prediction: FIRE
      ↓
  Grad-CAM
      ↓
Highlighted Fire Region
```

This would make the system more transparent and useful for human verification.

---

# 📈 Model Evaluation

For a production-level version, the AI model should be evaluated using more than accuracy.

Important metrics include:

### Precision

How many predicted fire cases were actually fire?

### Recall

How many actual fire cases were successfully detected?

### F1-Score

Balances precision and recall.

### Confusion Matrix

Shows:

```text
                 Actual
              Fire    No Fire
           ┌────────┬────────┐
Pred Fire  │   TP   │   FP   │
           ├────────┼────────┤
No Fire    │   FN   │   TN   │
           └────────┴────────┘
```

### Other useful metrics

* ROC-AUC
* False Positive Rate
* False Negative Rate
* Inference time
* Model size
* Robustness to image quality

For a fire-detection system, **false negatives are especially important**, because failing to detect a real fire can have serious consequences.

---

# ⚠️ Limitations

ForestGuard is currently an **AI-assisted prototype**.

It should not be considered a standalone fire alarm or a replacement for professional forest-fire detection infrastructure.

Important limitations include:

### 📷 Image Quality

Predictions depend on the quality of the input image.

Poor lighting, blur, obstruction, or distance may affect performance.

### 🔥 Small Fires

Very small or distant fires may not be detected reliably.

### 🌫️ Smoke Ambiguity

Smoke can sometimes resemble:

* Fog
* Clouds
* Dust
* Atmospheric haze

This can lead to false predictions.

### 🌲 Environmental Variation

Forest environments can vary significantly depending on:

* Season
* Weather
* Lighting
* Vegetation
* Camera position

A model trained on limited conditions may not generalize perfectly to every environment.

### 🧠 AI Uncertainty

A model prediction is not absolute proof.

A result such as:

```text
SAFE
```

does **not** guarantee that no fire exists.

Similarly:

```text
ALERT
```

should ideally be verified before emergency action.

### 💾 Local Storage

The current project is designed around local application data rather than a centralized production database.

---

# 🔐 Responsible Use

ForestGuard is intended as an **early-screening and decision-support system**.

For real-world deployment, the system should be combined with:

* Human verification
* Multiple sensors
* Professional fire monitoring infrastructure
* Reliable communication systems
* Proper emergency-response procedures

AI should support human decision-making rather than become the sole source of emergency decisions.

---

# 🌱 Environmental Impact

Forest fires can damage:

🌳 Forest ecosystems
🐘 Wildlife habitats
🌬️ Air quality
💧 Water systems
🌾 Agricultural areas
🏘️ Nearby communities

ForestGuard explores how AI can contribute to earlier awareness and potentially reduce response time.

The broader vision is:

```text
          🤖 Artificial Intelligence
                    +
             👁️ Computer Vision
                    +
              📡 IoT Sensors
                    +
              📍 Location Data
                    +
             🚨 Real-Time Alerts
                    │
                    ▼
          🌲 Smarter Forest Monitoring
                    │
                    ▼
             🔥 Earlier Response
                    │
                    ▼
            🌍 Environmental Protection
```

---

# 🧩 Future System Architecture

A larger version of ForestGuard could look like this:

```text
                  🌲 FOREST AREA
                        │
        ┌───────────────┼────────────────┐
        │               │                │
      📷 Camera       🌡️ Sensor        💨 Smoke Sensor
        │               │                │
        └───────────────┼────────────────┘
                        │
                        ▼
                 📡 IoT / Network
                        │
                        ▼
                 🤖 AI Detection
                        │
              ┌─────────┴─────────┐
              │                   │
              ▼                   ▼
        🔥 Fire Detected      🟢 Normal
              │
              ▼
          📍 Location
              │
              ▼
        🚨 Alert System
              │
       ┌──────┼───────┐
       ▼      ▼       ▼
      📱     📧      🗺️
    Mobile  Email    Map
```

---

# 💡 Why This Project Matters

ForestGuard is more than just a machine-learning model.

It demonstrates how multiple areas of computer science can come together:

```text
Python
  +
Machine Learning
  +
Computer Vision
  +
Web Development
  +
Data Processing
  +
Environmental Technology
       ↓
   FORESTGUARD
```

The project provides a foundation for exploring **AI-driven environmental monitoring** and demonstrates how a trained model can be converted into a usable application.

---

# 🧑‍💻 Author

## Adi

**Information Science Engineering**

GitHub:

https://github.com/Adi20-06

Project Repository:

https://github.com/Adi20-06/forestguard

---

# 🤝 Contributing

Contributions and ideas are welcome.

You can contribute by:

1. ⭐ Starring the repository
2. 🍴 Forking the project
3. 🐛 Reporting bugs
4. 💡 Suggesting features
5. 🔧 Improving the application
6. 🧠 Experimenting with better models
7. 📊 Improving model evaluation
8. 📡 Exploring IoT integration

Example workflow:

```bash
git clone https://github.com/Adi20-06/forestguard.git

cd forestguard

git checkout -b feature/my-feature

# Make your changes

git add .

git commit -m "Add my feature"

git push origin feature/my-feature
```

Then open a Pull Request on GitHub.

---

# 📜 License

This repository currently does not specify an open-source license.

If you plan to distribute ForestGuard publicly or accept external contributions, consider adding a suitable license such as MIT, Apache-2.0, or another license appropriate for the project.

---

# ⭐ Support ForestGuard

If you find this project interesting:

⭐ **Star the repository**

🍴 **Fork the project**

🐛 **Report an issue**

💡 **Share an idea**

🤝 **Contribute**

Every contribution can help improve the project.

---

<div align="center">

# 🌲 ForestGuard

### **Smarter Detection. Safer Forests.**

**AI • Computer Vision • TensorFlow • Keras • Streamlit**

🌍 *Technology for a safer and greener future.*

</div>
