import os
import io
import json
import requests
import pandas as pd
from PIL import Image
import streamlit as st

# ── Configuration & Backend URL ─────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="Intelligent Image Analysis System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    h1 { color: #1a1a2e; font-family: 'Segoe UI', sans-serif; font-weight: 700; }
    h2, h3 { color: #16213e; font-family: 'Segoe UI', sans-serif; }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .metric-value { font-size: 24px; font-weight: 700; color: #1a1a2e; }
    .metric-label { font-size: 12px; color: #666666; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
    .badge-high { background-color: #fee2e2; color: #991b1b; padding: 4px 8px; border-radius: 4px; font-weight: 600; }
    .badge-med { background-color: #fef3c7; color: #92400e; padding: 4px 8px; border-radius: 4px; font-weight: 600; }
    .badge-low { background-color: #d1fae5; color: #065f46; padding: 4px 8px; border-radius: 4px; font-weight: 600; }
    .section-divider { border-top: 1.5px solid #e2e8f0; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)


# ── Backend Helper Functions ────────────────────────────────────
def check_backend_health(url: str):
    try:
        resp = requests.get(f"{url}/health", timeout=3)
        return resp.status_code == 200, resp.json() if resp.status_code == 200 else {}
    except Exception:
        return False, {}


@st.cache_data(ttl=600)
def get_supported_classes(url: str):
    try:
        resp = requests.get(f"{url}/api/v1/classes", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


# ── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 System Control")
    backend_input = st.text_input("Backend REST API URL", value=BACKEND_URL)
    is_healthy, health_info = check_backend_health(backend_input)

    if is_healthy:
        st.success(f"🟢 Connected to API v{health_info.get('version', '2.0.0')}")
    else:
        st.error("🔴 Backend Offline. Ensure FastAPI is running.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    mode = st.radio(
        "Analysis Mode",
        ["Single Image Analysis", "Batch Image Processing", "Image Comparison"],
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
    )

    # Class filtering
    available_classes = get_supported_classes(backend_input)
    selected_filters = st.multiselect(
        "Filter Specific Classes (Optional)",
        options=available_classes,
        default=[],
        help="Leave empty to detect all 80 object classes.",
    )
    filter_param = ",".join(selected_filters) if selected_filters else None

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
**Architecture:**
- **Frontend**: Streamlit HTTP Client
- **Backend**: FastAPI REST Microservice
- **Vision Models**: YOLOv8n + Gemini 2.5 Flash
""")


# ── Main Application ────────────────────────────────────────────
st.title("Intelligent Image Analysis System")
st.caption("Decoupled Microservice Architecture · Real-time YOLOv8 Inference · Gemini Multimodal Intelligence · Enterprise PDF Reporting")

if not is_healthy:
    st.warning(f"Cannot reach the backend service at `{backend_input}`. Please ensure the backend is started.")
    st.stop()


# ── MODE 1: Single Image Analysis ──────────────────────────────
if mode == "Single Image Analysis":
    uploaded_file = st.file_uploader("Upload an image for analysis", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

        # Run primary detection call
        with st.spinner("Invoking FastAPI detection pipeline..."):
            data = {"confidence_threshold": confidence_threshold}
            if filter_param:
                data["filter_classes"] = filter_param
            files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}

            resp = requests.post(f"{backend_input}/api/v1/detect", data=data, files=files)

        if resp.status_code != 200:
            st.error(f"Error from backend: {resp.text}")
            st.stop()

        detection_data = resp.json()
        metrics = detection_data["metrics"]

        # ── KPI Metrics Cards ───────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{detection_data["total_objects"]}</div><div class="metric-label">Objects Found</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{detection_data["unique_classes"]}</div><div class="metric-label">Unique Classes</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{detection_data["average_confidence"]*100:.1f}%</div><div class="metric-label">Avg Confidence</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{metrics["inference_ms"]} ms</div><div class="metric-label">Inference Latency</div></div>', unsafe_allow_html=True)
        with c5:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{metrics["total_ms"]} ms</div><div class="metric-label">Total Latency</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # ── Analysis Tabs ───────────────────────────────────────
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "🎯 Detection",
            "🌡️ Heatmap",
            "📏 Spatial Distances",
            "🧠 Scene Understanding",
            "⚠️ Safety Risk Assessment",
            "💬 Visual Q&A",
            "📥 Dual Export (PDF & JSON)"
        ])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Image**")
                st.image(image, use_container_width=True)
            with col2:
                st.markdown("**Annotated Bounding Boxes**")
                annotated_resp = requests.post(
                    f"{backend_input}/api/v1/annotated-image",
                    data=data,
                    files={"file": (uploaded_file.name, file_bytes, uploaded_file.type)},
                )
                if annotated_resp.status_code == 200:
                    st.image(annotated_resp.content, use_container_width=True)

            st.markdown("#### Detailed Detection List")
            if detection_data["detections"]:
                det_df = pd.DataFrame([
                    {
                        "Object Class": d["class_name"].capitalize(),
                        "Confidence": f"{d['confidence']*100:.1f}%",
                        "Box Coordinates [x1, y1, x2, y2]": str(d["box"]),
                        "Normalized Centroid": f"({d['normalized_centroid'][0]}, {d['normalized_centroid'][1]})"
                    }
                    for d in detection_data["detections"]
                ])
                st.dataframe(det_df, use_container_width=True)
            else:
                st.info("No objects detected with the selected threshold and filter.")

        with tab2:
            st.markdown("#### Object Density Heatmap")
            st.caption("Generated via 2D confidence matrix accumulation, Gaussian blur, and OpenCV JET colormapping.")
            heatmap_resp = requests.post(
                f"{backend_input}/api/v1/heatmap",
                data={"confidence_threshold": confidence_threshold},
                files={"file": (uploaded_file.name, file_bytes, uploaded_file.type)},
            )
            if heatmap_resp.status_code == 200:
                st.image(heatmap_resp.content, use_container_width=True)

        with tab3:
            st.markdown("#### Spatial Distance Estimation")
            dist_resp = requests.post(
                f"{backend_input}/api/v1/distances",
                data={"confidence_threshold": confidence_threshold},
                files={"file": (uploaded_file.name, file_bytes, uploaded_file.type)},
            )
            if dist_resp.status_code == 200:
                dist_data = dist_resp.json()
                if dist_data["distances"]:
                    dist_table = pd.DataFrame(dist_data["distances"])
                    st.dataframe(dist_table, use_container_width=True)
                else:
                    st.info("Need at least 2 detected objects to calculate pairwise distance.")

        with tab4:
            st.markdown("#### AI Scene Understanding (Google Gemini 2.5 Flash)")
            if st.button("Generate Scene Description"):
                with st.spinner("Analyzing scene context with Gemini..."):
                    desc_resp = requests.post(
                        f"{backend_input}/api/v1/scene-description",
                        data={"confidence_threshold": confidence_threshold},
                        files={"file": (uploaded_file.name, file_bytes, uploaded_file.type)},
                    )
                    if desc_resp.status_code == 200:
                        st.session_state.scene_desc = desc_resp.json()["description"]
                    else:
                        st.error(f"Error from Gemini service: {desc_resp.text}")

            if "scene_desc" in st.session_state:
                st.info(st.session_state.scene_desc)

        with tab5:
            st.markdown("#### Public Safety Risk Assessment")
            if st.button("Run Safety Analysis"):
                with st.spinner("Assessing hazards & compliance risks..."):
                    risk_resp = requests.post(
                        f"{backend_input}/api/v1/risk-assessment",
                        data={"confidence_threshold": confidence_threshold},
                        files={"file": (uploaded_file.name, file_bytes, uploaded_file.type)},
                    )
                    if risk_resp.status_code == 200:
                        st.session_state.risk_data = risk_resp.json()
                    else:
                        st.error(f"Error from Risk service: {risk_resp.text}")

            if "risk_data" in st.session_state:
                rd = st.session_state.risk_data
                badge_class = "badge-high" if rd["overall_risk"] == "High" else "badge-med" if rd["overall_risk"] == "Medium" else "badge-low"
                st.markdown(f"**Overall Risk Level:** <span class='{badge_class}'>{rd['overall_risk']}</span>", unsafe_allow_html=True)
                st.markdown("##### Key Hazards Identified")
                for r in rd.get("key_risks", []):
                    st.markdown(f"- ⚠️ {r}")
                st.markdown("##### Actionable Recommendations")
                for rec in rd.get("recommendations", []):
                    st.markdown(f"- ✅ {rec}")

        with tab6:
            st.markdown("#### Grounded Visual Q&A")
            user_question = st.text_input("Ask a question about the image", placeholder="e.g. Is the pedestrian on a crosswalk?")
            if st.button("Submit Question"):
                if user_question.strip():
                    with st.spinner("Consulting vision model..."):
                        qa_resp = requests.post(
                            f"{backend_input}/api/v1/qa",
                            data={"question": user_question},
                            files={"file": (uploaded_file.name, file_bytes, uploaded_file.type)},
                        )
                        if qa_resp.status_code == 200:
                            st.session_state.qa_answer = qa_resp.json()["answer"]
                        else:
                            st.error(qa_resp.text)

            if "qa_answer" in st.session_state:
                st.markdown(f"**Answer:** {st.session_state.qa_answer}")

        with tab7:
            st.markdown("#### Enterprise Data & Report Export")
            st.write("Export analysis results as an executive PDF document or as standardized JSON for downstream systems.")

            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                st.markdown("##### 📄 Executive PDF Report")
                if st.button("Generate & Download PDF"):
                    with st.spinner("Compiling PDF with ReportLab..."):
                        pdf_resp = requests.post(
                            f"{backend_input}/api/v1/export/pdf",
                            data={"confidence_threshold": confidence_threshold, "include_scene_desc": True, "include_risk_assess": True},
                            files={"file": (uploaded_file.name, file_bytes, uploaded_file.type)},
                        )
                        if pdf_resp.status_code == 200:
                            st.download_button(
                                label="💾 Click to Save PDF Report",
                                data=pdf_resp.content,
                                file_name="image_analysis_report.pdf",
                                mime="application/pdf",
                            )

            with exp_col2:
                st.markdown("##### 🗄️ Standardized JSON Schema")
                if st.button("Export JSON Audit Data"):
                    with st.spinner("Generating JSON payload..."):
                        json_resp = requests.post(
                            f"{backend_input}/api/v1/export/json",
                            data={"confidence_threshold": confidence_threshold, "include_scene_desc": True, "include_risk_assess": True},
                            files={"file": (uploaded_file.name, file_bytes, uploaded_file.type)},
                        )
                        if json_resp.status_code == 200:
                            json_str = json.dumps(json_resp.json(), indent=2)
                            st.download_button(
                                label="💾 Click to Save JSON Audit",
                                data=json_str,
                                file_name="image_analysis_audit.json",
                                mime="application/json",
                            )


# ── MODE 2: Batch Image Processing ─────────────────────────────
elif mode == "Batch Image Processing":
    st.markdown("### 📦 Batch Object Detection")
    st.write("Upload multiple images to run sequential batch detection and aggregate performance metrics.")

    uploaded_files = st.file_uploader(
        "Upload images for batch processing",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button(f"Process {len(uploaded_files)} Images"):
            with st.spinner("Processing batch on backend..."):
                multi_files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
                batch_resp = requests.post(
                    f"{backend_input}/api/v1/detect/batch",
                    data={"confidence_threshold": confidence_threshold},
                    files=multi_files,
                )

            if batch_resp.status_code == 200:
                b_data = batch_resp.json()
                st.success(f"Processed {b_data['total_images_processed']} images in {b_data['total_processing_time_ms']} ms!")

                summary_table = pd.DataFrame([
                    {
                        "File Name": item["filename"],
                        "Objects Found": item["total_objects"],
                        "Unique Classes": item["unique_classes"],
                        "Avg Confidence": f"{item['average_confidence']*100:.1f}%",
                        "Inference Time": f"{item['metrics']['inference_ms']} ms",
                    }
                    for item in b_data["results"]
                ])
                st.dataframe(summary_table, use_container_width=True)
            else:
                st.error(batch_resp.text)


# ── MODE 3: Image Comparison ───────────────────────────────────
elif mode == "Image Comparison":
    st.markdown("### 🔄 Image Comparison & Delta Tracking")
    st.write("Upload two images to compute class count differentials between scenes.")

    col1, col2 = st.columns(2)
    with col1:
        img1_file = st.file_uploader("Baseline Image 1", type=["jpg", "jpeg", "png"], key="b1")
    with col2:
        img2_file = st.file_uploader("Comparison Image 2", type=["jpg", "jpeg", "png"], key="b2")

    if img1_file and img2_file:
        if st.button("Compare Scenes"):
            with st.spinner("Running differential analysis..."):
                files_payload = {
                    "file1": (img1_file.name, img1_file.getvalue(), img1_file.type),
                    "file2": (img2_file.name, img2_file.getvalue(), img2_file.type),
                }
                comp_resp = requests.post(
                    f"{backend_input}/api/v1/compare",
                    data={"confidence_threshold": confidence_threshold},
                    files=files_payload,
                )

            if comp_resp.status_code == 200:
                comp_data = comp_resp.json()
                st.markdown(f"**Image 1 Objects:** {comp_data['image1_total']} | **Image 2 Objects:** {comp_data['image2_total']}")

                diff_df = pd.DataFrame([
                    {
                        "Object": c["object"].capitalize(),
                        "Image 1 Count": c["image1_count"],
                        "Image 2 Count": c["image2_count"],
                        "Net Change": f"+{c['change']}" if c["change"] > 0 else str(c["change"]),
                    }
                    for c in comp_data["changes"]
                ])
                st.dataframe(diff_df, use_container_width=True)
            else:
                st.error(comp_resp.text)
