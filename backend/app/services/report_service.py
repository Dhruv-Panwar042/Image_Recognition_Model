import io
import tempfile
from datetime import datetime
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.units import inch

from app.schemas.analysis import DetectionResponse, DistanceItem, RiskAssessmentResponse


class ReportService:
    @staticmethod
    def generate_pdf(
        image: Image.Image,
        detection: DetectionResponse,
        distances: list[DistanceItem],
        scene_desc: str = "N/A",
        risk_assess: RiskAssessmentResponse = None,
    ) -> bytes:
        """
        Generates an executive PDF report summarizing the multimodal image analysis.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        styles = getSampleStyleSheet()
        story = []

        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Title"],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#1a1a2e"),
            alignment=0,
            spaceAfter=4,
        )
        subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=styles["Normal"],
            fontSize=10,
            textColor=colors.HexColor("#666666"),
            spaceAfter=14,
        )
        heading_style = ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#16213e"),
            spaceBefore=12,
            spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "BodyDark",
            parent=styles["Normal"],
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#333333"),
            spaceAfter=4,
        )

        # Header Title
        story.append(Paragraph("Intelligent Image Analysis Report", title_style))
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        meta_line = (
            f"Generated: <b>{now_str}</b> | Objects Detected: <b>{detection.total_objects}</b> "
            f"| Avg Confidence: <b>{detection.average_confidence * 100:.1f}%</b> "
            f"| Inference Latency: <b>{detection.metrics.inference_ms}ms</b>"
        )
        story.append(Paragraph(meta_line, subtitle_style))

        # Render Annotated Image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name, format="JPEG")
            story.append(RLImage(tmp.name, width=6.5 * inch, height=3.2 * inch))
        story.append(Spacer(1, 10))

        # Detection Summary Table
        story.append(Paragraph("Object Detection Breakdown", heading_style))
        table_data = [["Object Class", "Count", "Class Frequency"]]
        total = detection.total_objects or 1
        for cls, count in sorted(detection.class_counts.items(), key=lambda x: x[1], reverse=True)[:8]:
            freq = f"{(count / total) * 100:.1f}%"
            table_data.append([cls.capitalize(), str(count), freq])

        if len(table_data) == 1:
            table_data.append(["None Detected", "0", "0%"])

        t = Table(table_data, colWidths=[3.0 * inch, 1.7 * inch, 1.8 * inch])
        t.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e0e0e0")),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("PADDING", (0, 0), (-1, -1), 5),
            ])
        )
        story.append(t)
        story.append(Spacer(1, 10))

        # Scene Understanding Section
        if scene_desc and scene_desc != "N/A":
            story.append(Paragraph("AI Scene Understanding", heading_style))
            story.append(Paragraph(scene_desc.replace("\n", "<br/>"), body_style))
            story.append(Spacer(1, 8))

        # Risk Assessment Section
        if risk_assess:
            story.append(Paragraph(f"Public Safety Risk Assessment ({risk_assess.overall_risk} Risk)", heading_style))
            risk_text = f"<b>Overall Risk:</b> {risk_assess.overall_risk}<br/>"
            if risk_assess.key_risks:
                risk_text += "<b>Key Risks:</b><br/>" + "<br/>".join([f"• {r}" for r in risk_assess.key_risks]) + "<br/>"
            if risk_assess.recommendations:
                risk_text += "<b>Recommendations:</b><br/>" + "<br/>".join([f"• {rec}" for rec in risk_assess.recommendations])
            story.append(Paragraph(risk_text, body_style))
            story.append(Spacer(1, 8))

        # Spatial Distance Estimation Table
        if distances:
            story.append(Paragraph("Spatial Proximity Estimation", heading_style))
            dist_data = [["Entity A", "Entity B", "Proximity Rating", "Normalized Distance"]]
            for d in distances[:6]:
                dist_data.append([d.obj1, d.obj2, d.proximity, f"{d.distance:.3f}"])

            dt = Table(dist_data, colWidths=[2.0 * inch, 2.0 * inch, 1.3 * inch, 1.2 * inch])
            dt.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e0e0e0")),
                    ("ALIGN", (2, 0), (-1, -1), "CENTER"),
                    ("PADDING", (0, 0), (-1, -1), 4),
                ])
            )
            story.append(dt)

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()


report_service = ReportService()
