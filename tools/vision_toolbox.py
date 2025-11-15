"""Light-weight vision tool executor for crop/zoom/rotate operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PIL import Image


@dataclass
class ToolView:
    """Represents an image produced by a tool operation."""

    op_name: str
    description: str
    image: Image.Image
    metadata: Dict[str, Any]


class VisionToolbox:
    """Applies heuristic cropping/zooming/rotation operations based on the question text."""

    REGION_KEYWORDS = {
        "top left": (0.0, 0.0, 0.5, 0.5),
        "top right": (0.5, 0.0, 1.0, 0.5),
        "bottom left": (0.0, 0.5, 0.5, 1.0),
        "bottom right": (0.5, 0.5, 1.0, 1.0),
        "center": (0.25, 0.25, 0.75, 0.75),
        "middle": (0.25, 0.25, 0.75, 0.75),
    }

    ROTATION_KEYWORDS = {
        "rotate 90": 90,
        "rotate 180": 180,
        "rotate 270": 270,
        "clockwise": -90,
        "counterclockwise": 90,
        "upside down": 180,
    }

    def __init__(
        self,
        enable_crop: bool = True,
        enable_zoom: bool = True,
        enable_rotate: bool = True,
        max_extra_views: int = 2,
    ) -> None:
        self.enable_crop = enable_crop
        self.enable_zoom = enable_zoom
        self.enable_rotate = enable_rotate
        self.max_extra_views = max_extra_views

    def describe(self) -> Dict[str, Any]:
        return {
            "enable_crop": self.enable_crop,
            "enable_zoom": self.enable_zoom,
            "enable_rotate": self.enable_rotate,
            "max_extra_views": self.max_extra_views,
        }

    def apply(self, image: Image.Image, question_text: str) -> List[ToolView]:
        """Returns a list of tool views for the provided image."""

        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")

        text = question_text.lower()
        views: List[ToolView] = [ToolView("original", "Original resolution image.", image, metadata={})]
        planned_ops: List[ToolView] = []

        if self.enable_crop:
            region_ops = self._plan_crop_operations(image, text)
            planned_ops.extend(region_ops)

        if self.enable_zoom and ("zoom" in text or "small" in text or "tiny" in text or "detail" in text):
            planned_ops.append(self._zoom_center(image))

        if self.enable_rotate:
            rotation = self._plan_rotation_angle(text)
            if rotation is not None:
                planned_ops.append(self._rotate_view(image, rotation))

        for view in planned_ops[: self.max_extra_views]:
            views.append(view)

        return views

    def _plan_crop_operations(self, image: Image.Image, text: str) -> List[ToolView]:
        width, height = image.size
        planned: List[ToolView] = []
        for keyword, box in self.REGION_KEYWORDS.items():
            if keyword in text:
                left = int(box[0] * width)
                top = int(box[1] * height)
                right = int(box[2] * width)
                bottom = int(box[3] * height)
                crop = image.crop((left, top, right, bottom))
                planned.append(
                    ToolView(
                        op_name="crop",
                        description=f"Cropped {keyword} region.",
                        image=crop,
                        metadata={
                            "keyword": keyword,
                            "box": [left, top, right, bottom],
                        },
                    )
                )
        return planned

    def _zoom_center(self, image: Image.Image) -> ToolView:
        width, height = image.size
        crop_box = (
            int(width * 0.2),
            int(height * 0.2),
            int(width * 0.8),
            int(height * 0.8),
        )
        cropped = image.crop(crop_box)
        zoomed = cropped.resize(image.size)
        return ToolView(
            op_name="zoom",
            description="Zoomed into the central region for small details.",
            image=zoomed,
            metadata={"box": list(crop_box)},
        )

    def _plan_rotation_angle(self, text: str) -> Optional[int]:
        for keyword, angle in self.ROTATION_KEYWORDS.items():
            if keyword in text:
                return angle
        if "rotate" in text:
            return 90
        return None

    def _rotate_view(self, image: Image.Image, angle: int) -> ToolView:
        rotated = image.rotate(angle, expand=True)
        return ToolView(
            op_name="rotate",
            description=f"Rotated image by {angle} degrees.",
            image=rotated,
            metadata={"angle": angle},
        )
