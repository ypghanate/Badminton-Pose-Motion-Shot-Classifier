* 🦴 **Hip-Centric Normalization:** Feature vectors calculate joint distances and displacement relative to the hip center root-joint, achieving full **translation invariance** regardless of court position.
* 🎯 **Aesthetic Visualization:** Generates activation heatmaps to verify prediction confidence across subtle shot transitions.
* 🎥 **Annotated Video Export:** Processes video frame sequences and renders real-time stroke labels overlaid onto output MP4 clips.
* 📈 **Fine-Grained Classification:** Achieved **76% Accuracy** and an **80% F1-Score** across close shot classes (smash, drop, clear, drive, lift).

---

## 📊 Performance Metrics

| Metric | Score |
| :--- | :--- |
| **Accuracy** | `76.0%` |
| **F1-Score** | `80.0%` |
| **Backbone** | `YOLO-Pose 26` |
| **Input Format** | `Broadcast Video (MP4/AVI)` |

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

# Install dependencies
pip install -r requirements.txt
