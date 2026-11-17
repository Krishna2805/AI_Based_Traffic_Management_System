# Intelligent AI-Based Traffic Management System

A comprehensive hybrid computer vision and reinforcement learning system designed to dynamically optimize traffic signal timings at 4-way urban intersections based on real-time vehicle counts, vehicle priority weightings, and Q-Learning reinforcement learning policies.

---

## 🌟 Key Features

- **YOLOv8 Object Detection**: Detects and counts vehicle types (`car`, `motorcycle`, `bus`, `truck`) from intersection lane images.
- **Priority-Based Dynamic Signal Allocation**: Calculates lane priority scores based on vehicle weights (`truck: 4`, `bus: 3`, `car: 2`, `motorcycle: 1`) and dynamically assigns green signal durations (10s – 30s).
- **Reinforcement Learning (Q-Learning)**: Tabular Q-Learning agent operating over a 1,024-state space to minimize total intersection wait times compared to a static round-robin signal baseline.
- **Pygame 2D Intersection Simulation**: Real-time multi-lane 2D physics visualization of vehicles navigating an intersection governed by dynamic traffic signal state transitions.

---

## 📂 Project Structure

```
AI-Based-Traffic-Management-System/
├── README.md                           # Comprehensive project quickstart & setup guide
├── requirements.txt                    # Project dependencies (ultralytics, opencv, pygame, pandas, etc.)
├── .gitignore                          # Git exclusions
├── Object Detection/
│   ├── set1/                           # Sample test images for intersection lanes
│   └── src/
│       ├── classes.txt                 # COCO vehicle class labels
│       ├── main.py                     # Object detection, vehicle counting & priority scoring
│       └── trafficsignal.py            # Tkinter GUI for displaying calculated green signal times
├── Q-Learning model/
│   └── src/
│       └── QLearning.py                # RL environment, tabular Q-table agent & training pipeline
└── Simulation/
    ├── final.py                        # Pygame 2D traffic intersection simulation
    └── images/                         # Vehicle sprites & signal graphic assets
```

---

## ⚙️ Prerequisites & Setup

### 1. Prerequisites
- **Python 3.8+** installed on your system.
- An active internet connection on first run (for Ultralytics YOLOv8 weights auto-download) or pre-downloaded `yolov8m.pt`.

### 2. Installation

Clone the repository and install required Python packages:

```bash
git clone https://github.com/Krishna2805/AI_Based_Traffic_Management_System.git
cd AI_Based_Traffic_Management_System
pip install -r requirements.txt
```

### 3. Model Weights & Pre-Trained RL Q-Table
- **YOLOv8 Medium (`yolov8m.pt`)**: Ultralytics automatically downloads official weights on first execution of `main.py` if not present in `Object Detection/src/`.
- **Pre-Trained Q-Table (`q_table.npy`)**: A pre-trained 1,024-state Q-table is bundled directly in `Q-Learning model/src/q_table.npy` (32 KB), allowing instant deterministic evaluation without requiring a 2,000-episode training run.

**1-Line Verification Command:**
```bash
cd "Q-Learning model/src"; python QLearning.py --demo-only
```

---

## 🚀 How to Run the Modules

### Module 1: Object Detection & Priority Calculation
Processes 4 intersection lane images, detects vehicle counts, and computes total priority scores per lane.

```bash
cd "Object Detection/src"
python main.py
```
*Note: A file dialog will prompt you to select 4 lane images (you can select sample images from `Object Detection/set1/`).*

---

### Module 2: Traffic Signal Timing GUI
Renders a Tkinter dashboard displaying the dynamically allocated green signal duration for each lane based on vehicle priorities.

```bash
cd "Object Detection/src"
python trafficsignal.py
```

---

### Module 3: Q-Learning Signal Optimization (Reinforcement Learning)
Trains a tabular Q-learning model to learn optimal signal switching policies over 2,000 episodes and compares performance against a round-robin baseline.

```bash
cd "Q-Learning model/src"
python QLearning.py
```

**CLI Options:**
```bash
python QLearning.py --episodes 5000     # Train for 5,000 episodes
python QLearning.py --demo-only          # Evaluate existing trained model without re-training
python QLearning.py --verbose            # Display step-by-step training logs
```

---

### Module 4: 2D Traffic Intersection Simulation
Launches an interactive Pygame simulation visualizing vehicles arriving, stopping at red signals, and passing through green signal lanes.

```bash
cd Simulation
python final.py
```

---

## 📊 System Architecture & Priorities

- **Vehicle Priority Weightings**:
  - `Truck`: Weight 4
  - `Bus`: Weight 3
  - `Car`: Weight 2
  - `Motorcycle / Bike`: Weight 1

- **Green Time Mapping Rule**:
  - Priority = `0` $\rightarrow$ Green Time = `10s` (Minimum duration)
  - Priority `< 10` $\rightarrow$ Green Time = `15s`
  - Priority `< 20` $\rightarrow$ Green Time = `20s`
  - Priority `< 30` $\rightarrow$ Green Time = `25s`
  - Priority $\ge 30$ $\rightarrow$ Green Time = `30s` (Maximum allocated duration)
