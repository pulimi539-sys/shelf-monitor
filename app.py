import gradio as gr
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import datetime
import sqlite3
import os

# Load your trained model
model = YOLO("best.pt")

# Class names and colors
CLASS_NAMES = {0: "Apple", 1: "Choco Fills", 2: "Choco Pie"}
COLORS = {
    0: (0, 200, 0),    # green for Apple
    1: (255, 165, 0),  # orange for Choco Fills
    2: (0, 0, 220)     # red for Choco Pie
}

# Setup database
def init_db():
    conn = sqlite3.connect("scans.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            status TEXT,
            fill_pct REAL,
            product_count INTEGER
        )
    """)
    conn.commit()
    conn.close()

def log_scan(status, fill_pct, count):
    conn = sqlite3.connect("scans.db")
    conn.execute("""
        INSERT INTO scans (timestamp, status, fill_pct, product_count)
        VALUES (?, ?, ?, ?)
    """, (datetime.datetime.now().isoformat(), status, fill_pct, count))
    conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect("scans.db")
    try:
        df = pd.read_sql(
            "SELECT * FROM scans ORDER BY id DESC LIMIT 20", conn)
    except:
        df = pd.DataFrame()
    conn.close()
    return df

# Main analysis function
def analyze_shelf(pil_image):
    if pil_image is None:
        return None, pd.DataFrame(), "Please upload an image"

    # Convert PIL to OpenCV
    img_array = np.array(pil_image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Run detection
    results = model(img_bgr, conf=0.25)[0]
    boxes = results.boxes
    h, w = img_bgr.shape[:2]

    # Draw boxes and count detections
    detection_list = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = CLASS_NAMES.get(cls, "Unknown")
        color = COLORS.get(cls, (255, 255, 255))

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf:.0%}"
        cv2.putText(img_bgr, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        detection_list.append({
            "Product": name,
            "Confidence": f"{conf:.0%}",
            "Location": f"({x1},{y1})"
        })

    # Calculate stock level
    total_area = w * h
    detected_area = sum(
        (int(b.xyxy[0][2]) - int(b.xyxy[0][0])) *
        (int(b.xyxy[0][3]) - int(b.xyxy[0][1]))
        for b in boxes
    )
    fill_ratio = min(detected_area / total_area, 1.0)
    count = len(detection_list)

    if fill_ratio >= 0.3:
        status = "FULLY STOCKED"
        status_color = (0, 200, 0)
    elif fill_ratio >= 0.1:
        status = "LOW STOCK"
        status_color = (0, 165, 255)
    else:
        status = "OUT OF STOCK"
        status_color = (0, 0, 220)

    # Add status banner
    cv2.rectangle(img_bgr, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(img_bgr, f"Status: {status} | Products: {count}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, status_color, 2)

    # Log to database
    log_scan(status, round(fill_ratio * 100, 1), count)

    # Convert back to PIL
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_out = Image.fromarray(img_rgb)

    # Build report
    report_df = pd.DataFrame(detection_list) if detection_list else \
        pd.DataFrame(columns=["Product", "Confidence", "Location"])

    summary = f"Status: {status} | Products detected: {count} | Fill level: {fill_ratio:.0%}"

    return pil_out, report_df, summary

def show_history():
    df = get_history()
    if df.empty:
        return pd.DataFrame({"Message": ["No scans yet. Upload an image first."]})
    return df

# Initialize DB
init_db()

# Build Gradio app
with gr.Blocks(title="Retail Shelf Monitor") as demo:
    gr.Markdown("# Retail Shelf Monitor")
    gr.Markdown("Upload a shelf image to detect products and check stock levels.")

    with gr.Tab("Analyze Shelf"):
        with gr.Row():
            img_input = gr.Image(type="pil", label="Upload shelf image")
            img_output = gr.Image(label="Detection result")
        analyze_btn = gr.Button("Analyze", variant="primary")
        summary_out = gr.Textbox(label="Summary")
        report_out = gr.DataFrame(label="Detected products")
        analyze_btn.click(
            fn=analyze_shelf,
            inputs=img_input,
            outputs=[img_output, report_out, summary_out]
        )

    with gr.Tab("Scan History"):
        history_btn = gr.Button("Load history")
        history_out = gr.DataFrame(label="Recent scans")
        history_btn.click(fn=show_history, outputs=history_out)

demo.launch(server_name="0.0.0.0", server_port=7860)