import base64
import re
from typing import Optional, List
import google.genai as genai

from app.core.config import settings
from app.schemas.analysis import (
    SceneDescriptionResponse,
    RiskAssessmentResponse,
    QAResponse,
)


class GeminiService:
    def __init__(self):
        self._client: Optional[genai.Client] = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            if not settings.GOOGLE_API_KEY:
                raise ValueError(
                    "GOOGLE_API_KEY environment variable is not configured. "
                    "Please provide a valid Google Gemini API Key in your .env or cloud environment."
                )
            self._client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        return self._client

    def _generate(self, prompt: str, image_bytes: bytes) -> str:
        """Helper to invoke Gemini multimodal vision model."""
        base64_img = base64.b64encode(image_bytes).decode("utf-8")
        response = self.client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=[{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64_img,
                        }
                    },
                ]
            }],
        )
        return response.text or ""

    def get_scene_description(self, image_bytes: bytes, objects_summary: str) -> SceneDescriptionResponse:
        """Generates contextual natural language description of the scene."""
        prompt = f"""
You are an intelligent computer vision analysis system.
Objects detected by YOLOv8: {objects_summary}

Provide a concise, professional description covering:
1. The scene and physical setting
2. Primary actions, state, or interactions observed
3. Overall environmental context

Keep the answer factual, objective, and under 150 words.
"""
        desc = self._generate(prompt, image_bytes)
        return SceneDescriptionResponse(description=desc.strip(), detected_context=objects_summary)

    def get_risk_assessment(self, image_bytes: bytes, objects_summary: str) -> RiskAssessmentResponse:
        """Generates structured public safety risk evaluation."""
        prompt = f"""
You are an expert safety inspector and risk analyst.
Objects detected by YOLOv8: {objects_summary}

Provide a structured safety assessment with the following exact format:
Overall Risk: [Low / Medium / High]
Key Risks:
- [Risk 1]
- [Risk 2]
Recommendations:
- [Recommendation 1]
- [Recommendation 2]

Keep points concise and actionable. Under 200 words total.
"""
        raw_text = self._generate(prompt, image_bytes)

        # Parse risk level
        overall_risk = "Medium"
        if re.search(r"Overall Risk:?\s*High", raw_text, re.IGNORECASE):
            overall_risk = "High"
        elif re.search(r"Overall Risk:?\s*Low", raw_text, re.IGNORECASE):
            overall_risk = "Low"

        # Extract risks and recommendations
        key_risks: List[str] = []
        recommendations: List[str] = []

        risk_section = False
        rec_section = False

        for line in raw_text.splitlines():
            line_clean = line.strip()
            if "Key Risks" in line_clean:
                risk_section = True
                rec_section = False
                continue
            elif "Recommendations" in line_clean:
                risk_section = False
                rec_section = True
                continue

            if line_clean.startswith(("-", "*", "•")) and len(line_clean) > 3:
                point = line_clean.lstrip("-*• ").strip()
                if risk_section:
                    key_risks.append(point)
                elif rec_section:
                    recommendations.append(point)

        return RiskAssessmentResponse(
            overall_risk=overall_risk,
            key_risks=key_risks[:4],
            recommendations=recommendations[:3],
            raw_text=raw_text.strip(),
        )

    def ask_question(self, image_bytes: bytes, question: str) -> QAResponse:
        """Answers arbitrary questions grounded in visual observation."""
        prompt = f"""
Look at this image carefully and answer the following question directly based on visual evidence:
Question: {question}

Provide a direct, concise, and truthful answer.
"""
        ans = self._generate(prompt, image_bytes)
        return QAResponse(question=question, answer=ans.strip())


gemini_service = GeminiService()
