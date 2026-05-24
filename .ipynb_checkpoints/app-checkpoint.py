import os
import io
import base64
import tempfile
import cv2
import numpy as np
from PIL import Image
from collections import Counter
import pandas as pd
import streamlit as st
from ultralytics import YOLO
import google.genai as genai
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.units import inch

# ── Page Configuration ────────────────────────────────────────
st.set_page_config(
    page_title="Intelligent Image Analysis System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    h1 { color: #1a1a2e; font-family: 'Segoe UI', sans-serif; font-weight: 700; }
    h2, h3 { color: #16213e; font-family: 'Segoe UI', sans-serif; }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-value { font-size: 28px; font-weight: 700; color: #1a1a2e; }
    .metric-label { font-size: 13px; color: #666666; margin-top: 4px; }
    .section-divider { border-top: 2px solid #e0e0e0; margin: 24px 0; }
    .stButton > button {
        background-color: #1a1a2e;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 24px;
        font-weight: 500;
    }
    .stButton > button:hover { background-color: #16213e; }
</style>
""", unsafe_allow_html=True)

# ── Initialize Models ─────────────────────────────────────────
@st.cache_resource
def load_models():
    api_key = st.secrets["GOOGLE_API_KEY"]
    client = genai.Client(api_key=api_key)
    yolo = YOLO("yolov8n.pt")
    return client, yolo

client, yolo_model = load_models()

# ── Helper Functions ──────────────────────────────────────────
def image_to_bytes(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

def image_to_base64(img_bytes):
    return base64.b64encode(img_bytes).decode()

def run_detection(image):
    results = yolo_model(image)
    return results[0]

def draw_bounding_boxes(image, result):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    colors_map = {}
    np.random.seed(42)
    for class_id in yolo_model.names:
        colors_map[class_id] = tuple(np.random.randint(60, 200, 3).tolist())

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = yolo_model.names[class_id]
        confidence = float(box.conf[0]) * 100
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = colors_map[class_id]
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {confidence:.1f}%"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image_cv, (x1, y1 - h - 8), (x1 + w + 4, y1), color, -1)
        cv2.putText(image_cv, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

def generate_heatmap(image, result):
    img_array = np.array(image)
    heatmap = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.float32)
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        heatmap[y1:y2, x1:x2] += confidence
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    heatmap_colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_array, 0.5, heatmap_colored, 0.5, 0)

def estimate_distances(result, image_width, image_height):
    objects = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = yolo_model.names[class_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) / 2 / image_width
        cy = (y1 + y2) / 2 / image_height
        objects.append({"name": class_name, "cx": cx, "cy": cy})

    distances = []
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            obj1, obj2 = objects[i], objects[j]
            dist = np.sqrt((obj1["cx"] - obj2["cx"])**2 + (obj1["cy"] - obj2["cy"])**2)
            proximity = "Very Close" if dist < 0.3 else "Close" if dist < 0.5 else "Far"
            distances.append({
                "obj1": obj1["name"], "obj2": obj2["name"],
                "distance": dist, "proximity": proximity
            })

    distances.sort(key=lambda x: x["distance"])
    return distances[:8]

def gemini_request(prompt, img_bytes):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[{
            "parts": [
                {"text": prompt},
                {"inline_data": {
                    "mime_type": "image/jpeg",
                    "data": image_to_base64(img_bytes)
                }}
            ]
        }]
    )
    return response.text

def get_scene_description(img_bytes, objects_text):
    prompt = f"""
You are an intelligent image analysis assistant.
Objects detected by YOLOv8: {objects_text}
Provide a concise, professional description of:
1. The scene and setting
2. What is happening in the image
3. Overall context
Keep it factual and under 150 words.
"""
    return gemini_request(prompt, img_bytes)

def get_risk_assessment(img_bytes, objects_text):
    prompt = f"""
You are a public safety expert.
Objects detected: {objects_text}
Provide a structured safety assessment:
1. Overall Risk Level (Low / Medium / High)
2. Key risks identified (maximum 4 points)
3. Recommendations (maximum 3 points)
Be concise and professional. Under 200 words total.
"""
    return gemini_request(prompt, img_bytes)

def ask_question(img_bytes, question):
    prompt = f"""
Look at this image carefully and answer the following question directly.
Question: {question}
Give a direct, concise answer based only on what you can see.
"""
    return gemini_request(prompt, img_bytes)

def compare_images(result1, result2):
    counts1 = Counter(yolo_model.names[int(box.cls[0])] for box in result1.boxes)
    counts2 = Counter(yolo_model.names[int(box.cls[0])] for box in result2.boxes)
    all_classes = set(counts1.keys()) | set(counts2.keys())
    comparison = []
    for cls in all_classes:
        c1, c2 = counts1.get(cls, 0), counts2.get(cls, 0)
        comparison.append({
            "object": cls, "image1": c1, "image2": c2, "change": c2 - c1
        })
    comparison.sort(key=lambda x: abs(x["change"]), reverse=True)
    return comparison

def generate_pdf_report(image, result, scene_desc, risk_assess, distances):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=40, leftMargin=40,
                            topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('Title', parent=styles['Title'],
        fontSize=20, textColor=colors.HexColor('#1a1a2e'), spaceAfter=6)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'],
        fontSize=13, textColor=colors.HexColor('#16213e'), spaceBefore=12, spaceAfter=6)
    body_style = ParagraphStyle('Body', parent=styles['Normal'],
        fontSize=10, textColor=colors.HexColor('#333333'), spaceAfter=4)

    story.append(Paragraph("Intelligent Image Analysis Report", title_style))
    story.append(Spacer(1, 12))

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        story.append(RLImage(tmp.name, width=5*inch, height=3*inch))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Detection Summary", heading_style))
    class_counts = Counter(yolo_model.names[int(box.cls[0])] for box in result.boxes)
    data = [["Object", "Count"]] + [[cls.capitalize(), str(cnt)] for cls, cnt in class_counts.most_common()]
    table = Table(data, colWidths=[3*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e0e0e0')),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Scene Description", heading_style))
    story.append(Paragraph(scene_desc.replace('\n', '<br/>'), body_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Risk Assessment", heading_style))
    story.append(Paragraph(risk_assess.replace('\n', '<br/>'), body_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Distance Estimation", heading_style))
    dist_data = [["Object 1", "Object 2", "Proximity"]] + [
        [d["obj1"].capitalize(), d["obj2"].capitalize(), d["proximity"]]
        for d in distances[:5]
    ]
    dist_table = Table(dist_data, colWidths=[2*inch, 2*inch, 1.5*inch])
    dist_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e0e0e0')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(dist_table)
    doc.build(story)
    buffer.seek(0)
    return buffer

# ── Main App ──────────────────────────────────────────────────
st.title("Intelligent Image Analysis System")
st.markdown("Upload an image to detect objects, analyze the scene, assess risks, and generate a detailed report.")
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Analysis Mode")
    mode = st.radio("Select mode", ["Single Image Analysis", "Image Comparison"],
                    label_visibility="collapsed")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Confidence Threshold")
    confidence_threshold = st.slider("Minimum confidence score",
        min_value=0.1, max_value=0.9, value=0.25, step=0.05,
        label_visibility="collapsed")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### About")
    st.markdown("""
This system combines **YOLOv8** for object detection with **Google Gemini**
for intelligent scene understanding.

**Capabilities:**
- Object detection (80+ classes)
- Scene description
- Risk assessment
- Distance estimation
- Heatmap visualization
- PDF report export
""")

# ── Single Image Analysis ─────────────────────────────────────
if mode == "Single Image Analysis":
    uploaded_file = st.file_uploader("Upload an image",
        type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_bytes = image_to_bytes(image)

        with st.spinner("Running object detection..."):
            result = run_detection(image)
            result.boxes = [box for box in result.boxes
                           if float(box.conf[0]) >= confidence_threshold]

        # Compute shared variables
        class_counts = Counter(yolo_model.names[int(box.cls[0])] for box in result.boxes)
        objects_text = ", ".join([
            f"{yolo_model.names[int(box.cls[0])]} ({float(box.conf[0])*100:.1f}%)"
            for box in result.boxes
        ])

        # ── Metrics Row ───────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        avg_conf = np.mean([float(box.conf[0]) for box in result.boxes]) * 100 if result.boxes else 0

        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(result.boxes)}</div><div class="metric-label">Objects Detected</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(class_counts)}</div><div class="metric-label">Unique Classes</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_conf:.1f}%</div><div class="metric-label">Avg Confidence</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{image.size[0]}x{image.size[1]}</div><div class="metric-label">Image Resolution</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # ── Tabs ──────────────────────────────────────────────
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Detection", "Heatmap", "Scene Analysis",
            "Risk Assessment", "Distance Estimation", "Ask a Question"
        ])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Image**")
                st.image(image, use_container_width=True)
            with col2:
                st.markdown("**Detected Objects**")
                st.image(draw_bounding_boxes(image, result), use_container_width=True)

            st.markdown("**Detection Results**")
            det_data = [{
                "Object": yolo_model.names[int(box.cls[0])].capitalize(),
                "Confidence": f"{float(box.conf[0])*100:.1f}%",
                "Location": f"({int(box.xyxy[0][0])}, {int(box.xyxy[0][1])}) to ({int(box.xyxy[0][2])}, {int(box.xyxy[0][3])})"
            } for box in result.boxes]
            st.dataframe(det_data, use_container_width=True)

        with tab2:
            st.markdown("**Object Density Heatmap**")
            st.markdown("Warmer colors indicate areas with higher object concentration.")
            st.image(generate_heatmap(image, result), use_container_width=True)

        with tab3:
            st.markdown("**Scene Description**")
            st.markdown("Click the button to generate an AI-powered description of the scene.")
            if st.button("Generate Scene Description"):
                with st.spinner("Analyzing scene..."):
                    st.session_state.scene_desc = get_scene_description(img_bytes, objects_text)
            if "scene_desc" in st.session_state:
                st.write(st.session_state.scene_desc)

        with tab4:
            st.markdown("**Safety Risk Assessment**")
            st.markdown("Click the button to run an AI-powered safety analysis.")
            if st.button("Run Risk Assessment"):
                with st.spinner("Assessing risks..."):
                    st.session_state.risk_assess = get_risk_assessment(img_bytes, objects_text)
            if "risk_assess" in st.session_state:
                st.write(st.session_state.risk_assess)

        with tab5:
            st.markdown("**Distance Estimation**")
            st.markdown("Relative distances between detected objects based on their positions in the image.")
            distances = estimate_distances(result, image.size[0], image.size[1])
            if distances:
                dist_df = pd.DataFrame([{
                    "Object 1": d["obj1"].capitalize(),
                    "Object 2": d["obj2"].capitalize(),
                    "Proximity": d["proximity"],
                    "Distance Score": f"{d['distance']:.3f}"
                } for d in distances])
                st.dataframe(dist_df, use_container_width=True)
            else:
                st.info("Not enough objects detected for distance estimation.")

        with tab6:
            st.markdown("**Ask a Question About the Image**")
            question = st.text_input("Enter your question",
                placeholder="e.g. How many people are in the image?")
            if st.button("Get Answer"):
                if question:
                    with st.spinner("Analyzing..."):
                        answer = ask_question(img_bytes, question)
                    st.session_state.last_answer = answer
                else:
                    st.warning("Please enter a question.")
            if "last_answer" in st.session_state:
                st.markdown("**Answer:**")
                st.write(st.session_state.last_answer)

        # ── PDF Export ────────────────────────────────────────
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### Export Report")
        st.markdown("Generate a PDF report containing the full analysis.")
        if st.button("Generate PDF Report"):
            with st.spinner("Generating report..."):
                scene_desc = st.session_state.get("scene_desc") or get_scene_description(img_bytes, objects_text)
                risk_assess = st.session_state.get("risk_assess") or get_risk_assessment(img_bytes, objects_text)
                distances = estimate_distances(result, image.size[0], image.size[1])
                pdf_buffer = generate_pdf_report(image, result, scene_desc, risk_assess, distances)
            st.download_button(
                label="Download PDF Report",
                data=pdf_buffer,
                file_name="image_analysis_report.pdf",
                mime="application/pdf"
            )

    else:
        st.info("Upload an image to begin analysis.")

# ── Image Comparison Mode ─────────────────────────────────────
elif mode == "Image Comparison":
    st.markdown("### Image Comparison")
    st.markdown("Upload two images to compare object detections between them.")

    col1, col2 = st.columns(2)
    with col1:
        file1 = st.file_uploader("Upload first image", type=["jpg", "jpeg", "png"], key="img1")
    with col2:
        file2 = st.file_uploader("Upload second image", type=["jpg", "jpeg", "png"], key="img2")

    if file1 and file2:
        image1 = Image.open(file1).convert("RGB")
        image2 = Image.open(file2).convert("RGB")

        with st.spinner("Analyzing both images..."):
            result1 = run_detection(image1)
            result2 = run_detection(image2)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Image 1**")
            st.image(draw_bounding_boxes(image1, result1), use_container_width=True)
            st.markdown(f"Objects detected: **{len(result1.boxes)}**")
        with col2:
            st.markdown("**Image 2**")
            st.image(draw_bounding_boxes(image2, result2), use_container_width=True)
            st.markdown(f"Objects detected: **{len(result2.boxes)}**")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("**Comparison Results**")

        comparison = compare_images(result1, result2)
        comp_df = pd.DataFrame([{
            "Object": c["object"].capitalize(),
            "Image 1 Count": c["image1"],
            "Image 2 Count": c["image2"],
            "Change": f"+{c['change']}" if c["change"] > 0 else str(c["change"])
        } for c in comparison])
        st.dataframe(comp_df, use_container_width=True)
    else:
        st.info("Upload two images to begin comparison.")