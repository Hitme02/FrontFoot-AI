# AthleteRise – Real-Time Cover Drive Analysis (MediaPipe)

## What this does
- Downloads and processes the **full video** frame-by-frame (no keyframes).
- Runs **MediaPipe Pose** to get 33 landmarks each frame.
- Computes per-frame metrics:
  - Front elbow angle (shoulder–elbow–wrist)
  - Spine lean vs vertical
  - Head-over-knee alignment (projected horizontal gap)
  - Front foot direction (toe→heel vs x-axis)
- Renders live overlays + feedback cues on the video.
- Writes **/output/annotated_video.mp4** and **/output/evaluation.json**.

## Setup
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
