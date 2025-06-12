"""
Image Analyzer for Agentical

This module provides comprehensive image analysis capabilities including
computer vision, content detection, text extraction (OCR), object recognition,
and AI-powered image understanding using multiple vision models.

Features:
- Multi-provider vision AI integration (OpenAI Vision, Google Vision, etc.)
- Object detection and recognition
- Text extraction from images (OCR)
- Image classification and tagging
- Face detection and analysis
- Scene understanding and description
- Image quality assessment
- Batch processing for multiple images
- Metadata extraction (EXIF data)
- Image preprocessing and enhancement
- Enterprise features (audit logging, monitoring, caching)
"""

import asyncio
import base64
import io
import json
import os
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple, BinaryIO
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib

# Optional dependencies
try:
    from PIL import Image, ImageEnhance, ImageFilter
    from PIL.ExifTags import TAGS
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from google.cloud import vision
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_VISION_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class AnalysisType(Enum):
    """Types of image analysis."""
    OBJECT_DETECTION = "object_detection"
    TEXT_EXTRACTION = "text_extraction"
    FACE_DETECTION = "face_detection"
    SCENE_ANALYSIS = "scene_analysis"
    IMAGE_CLASSIFICATION = "image_classification"
    QUALITY_ASSESSMENT = "quality_assessment"
    CONTENT_MODERATION = "content_moderation"
    SIMILARITY_SEARCH = "similarity_search"


class VisionProvider(Enum):
    """Supported vision AI providers."""
    OPENAI = "openai"
    GOOGLE_VISION = "google_vision"
    AZURE_VISION = "azure_vision"
    AWS_REKOGNITION = "aws_rekognition"
    OPENCV = "opencv"
    TESSERACT = "tesseract"
    CUSTOM = "custom"


class ImageFormat(Enum):
    """Supported image formats."""
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"
    BMP = "bmp"
    TIFF = "tiff"
    GIF = "gif"


class ContentType(Enum):
    """Types of content detected in images."""
    TEXT = "text"
    FACE = "face"
    OBJECT = "object"
    SCENE = "scene"
    LANDMARK = "landmark"
    LOGO = "logo"
    BARCODE = "barcode"
    DOCUMENT = "document"


@dataclass
class BoundingBox:
    """Bounding box coordinates."""
    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)

    @property
    def area(self) -> float:
        """Calculate area of bounding box."""
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        """Get center coordinates."""
        return (self.x + self.width / 2, self.y + self.height / 2)


@dataclass
class DetectedObject:
    """Object detected in image."""
    label: str
    confidence: float
    bounding_box: BoundingBox
    attributes: Optional[Dict[str, Any]] = None
    content_type: ContentType = ContentType.OBJECT

    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data['content_type'] = self.content_type.value
        data['bounding_box'] = self.bounding_box.to_dict()
        return data


@dataclass
class ExtractedText:
    """Text extracted from image."""
    text: str
    confidence: float
    bounding_box: Optional[BoundingBox] = None
    language: Optional[str] = None
    font_size: Optional[float] = None
    orientation: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        if self.bounding_box:
            data['bounding_box'] = self.bounding_box.to_dict()
        return data


@dataclass
class ImageMetadata:
    """Image metadata and EXIF data."""
    filename: str
    format: str
    size: Tuple[int, int]
    mode: str
    file_size: int
    creation_date: Optional[datetime] = None
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    iso: Optional[int] = None
    focal_length: Optional[float] = None
    aperture: Optional[float] = None
    shutter_speed: Optional[str] = None
    gps_coordinates: Optional[Tuple[float, float]] = None
    orientation: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        if self.creation_date:
            data['creation_date'] = self.creation_date.isoformat()
        return data


@dataclass
class QualityMetrics:
    """Image quality assessment metrics."""
    sharpness: float
    brightness: float
    contrast: float
    noise_level: float
    blur_detection: float
    overall_score: float
    recommendations: List[str]

    def __post_init__(self):
        if not hasattr(self, 'recommendations'):
            self.recommendations = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class AnalysisResult:
    """Result from image analysis operation."""
    image_path: str
    success: bool
    analysis_types: List[AnalysisType]
    provider_used: VisionProvider
    detected_objects: List[DetectedObject]
    extracted_text: List[ExtractedText]
    scene_description: Optional[str] = None
    image_metadata: Optional[ImageMetadata] = None
    quality_metrics: Optional[QualityMetrics] = None
    tags: List[str] = None
    confidence_score: float = 0.0
    processing_time: float = 0.0
    errors: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.errors is None:
            self.errors = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data['analysis_types'] = [t.value for t in self.analysis_types]
        data['provider_used'] = self.provider_used.value
        data['detected_objects'] = [obj.to_dict() for obj in self.detected_objects]
        data['extracted_text'] = [text.to_dict() for text in self.extracted_text]
        if self.image_metadata:
            data['image_metadata'] = self.image_metadata.to_dict()
        if self.quality_metrics:
            data['quality_metrics'] = self.quality_metrics.to_dict()
        return data


class VisionAnalyzer(ABC):
    """Abstract base class for vision analyzers."""

    @abstractmethod
    async def analyze_image(self, image_data: bytes, analysis_types: List[AnalysisType]) -> Dict[str, Any]:
        """Analyze image using provider's API."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if analyzer dependencies are available."""
        pass

    @abstractmethod
    def get_supported_types(self) -> List[AnalysisType]:
        """Get list of supported analysis types."""
        pass


class OpenAIVisionAnalyzer(VisionAnalyzer):
    """OpenAI Vision API analyzer."""

    def __init__(self, api_key: Optional[str] = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai library required for OpenAI Vision")

        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = "gpt-4-vision-preview"

    async def analyze_image(self, image_data: bytes, analysis_types: List[AnalysisType]) -> Dict[str, Any]:
        """Analyze image using OpenAI Vision."""
        try:
            # Convert image to base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')

            # Create prompt based on analysis types
            prompt = self._create_analysis_prompt(analysis_types)

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )

            content = response.choices[0].message.content
            return self._parse_openai_response(content, analysis_types)

        except Exception as e:
            raise RuntimeError(f"OpenAI Vision analysis failed: {e}")

    def _create_analysis_prompt(self, analysis_types: List[AnalysisType]) -> str:
        """Create analysis prompt based on requested types."""
        prompts = []

        if AnalysisType.OBJECT_DETECTION in analysis_types:
            prompts.append("Identify and list all objects visible in this image with their approximate locations.")

        if AnalysisType.SCENE_ANALYSIS in analysis_types:
            prompts.append("Describe the overall scene, setting, and context of this image.")

        if AnalysisType.TEXT_EXTRACTION in analysis_types:
            prompts.append("Extract any text visible in the image.")

        if AnalysisType.IMAGE_CLASSIFICATION in analysis_types:
            prompts.append("Classify this image and provide relevant tags.")

        if AnalysisType.CONTENT_MODERATION in analysis_types:
            prompts.append("Assess if this image contains any inappropriate or sensitive content.")

        return " ".join(prompts) + " Please provide your response in a structured format."

    def _parse_openai_response(self, content: str, analysis_types: List[AnalysisType]) -> Dict[str, Any]:
        """Parse OpenAI response into structured data."""
        return {
            'scene_description': content,
            'detected_objects': [],  # Would need more sophisticated parsing
            'extracted_text': [],
            'tags': [],
            'confidence': 0.8  # OpenAI doesn't provide confidence scores
        }

    def is_available(self) -> bool:
        """Check if OpenAI Vision is available."""
        return OPENAI_AVAILABLE

    def get_supported_types(self) -> List[AnalysisType]:
        """Get supported analysis types."""
        return [
            AnalysisType.OBJECT_DETECTION,
            AnalysisType.SCENE_ANALYSIS,
            AnalysisType.TEXT_EXTRACTION,
            AnalysisType.IMAGE_CLASSIFICATION,
            AnalysisType.CONTENT_MODERATION
        ]


class TesseractOCRAnalyzer(VisionAnalyzer):
    """Tesseract OCR analyzer for text extraction."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if not OCR_AVAILABLE or not PIL_AVAILABLE:
            raise ImportError("pytesseract and PIL required for OCR")

        self.config = config or {}
        self.language = self.config.get('language', 'eng')
        self.tesseract_config = self.config.get('tesseract_config', '--oem 3 --psm 6')

    async def analyze_image(self, image_data: bytes, analysis_types: List[AnalysisType]) -> Dict[str, Any]:
        """Extract text using Tesseract OCR."""
        if AnalysisType.TEXT_EXTRACTION not in analysis_types:
            return {'extracted_text': []}

        try:
            # Load image
            image = Image.open(io.BytesIO(image_data))

            # Extract text
            text = pytesseract.image_to_string(
                image,
                lang=self.language,
                config=self.tesseract_config
            )

            # Get detailed data with coordinates
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

            extracted_text = []
            for i, word in enumerate(data['text']):
                if word.strip() and int(data['conf'][i]) > 30:  # Confidence threshold
                    bbox = BoundingBox(
                        x=data['left'][i],
                        y=data['top'][i],
                        width=data['width'][i],
                        height=data['height'][i],
                        confidence=int(data['conf'][i]) / 100.0
                    )

                    extracted_text.append(ExtractedText(
                        text=word,
                        confidence=int(data['conf'][i]) / 100.0,
                        bounding_box=bbox
                    ))

            return {
                'extracted_text': extracted_text,
                'full_text': text.strip(),
                'confidence': sum(int(c) for c in data['conf'] if int(c) > 0) / len([c for c in data['conf'] if int(c) > 0]) / 100.0 if any(int(c) > 0 for c in data['conf']) else 0
            }

        except Exception as e:
            raise RuntimeError(f"Tesseract OCR failed: {e}")

    def is_available(self) -> bool:
        """Check if Tesseract is available."""
        return OCR_AVAILABLE and PIL_AVAILABLE

    def get_supported_types(self) -> List[AnalysisType]:
        """Get supported analysis types."""
        return [AnalysisType.TEXT_EXTRACTION]


class OpenCVAnalyzer(VisionAnalyzer):
    """OpenCV-based image analyzer."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if not OPENCV_AVAILABLE:
            raise ImportError("opencv-python required for OpenCV analysis")

        self.config = config or {}

    async def analyze_image(self, image_data: bytes, analysis_types: List[AnalysisType]) -> Dict[str, Any]:
        """Analyze image using OpenCV."""
        try:
            # Convert bytes to cv2 image
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            results = {}

            if AnalysisType.FACE_DETECTION in analysis_types:
                results['faces'] = await self._detect_faces(image)

            if AnalysisType.QUALITY_ASSESSMENT in analysis_types:
                results['quality'] = await self._assess_quality(image)

            if AnalysisType.OBJECT_DETECTION in analysis_types:
                results['objects'] = await self._detect_objects_opencv(image)

            return results

        except Exception as e:
            raise RuntimeError(f"OpenCV analysis failed: {e}")

    async def _detect_faces(self, image) -> List[DetectedObject]:
        """Detect faces using OpenCV."""
        # Load face cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        detected_faces = []
        for (x, y, w, h) in faces:
            bbox = BoundingBox(x=float(x), y=float(y), width=float(w), height=float(h))
            detected_faces.append(DetectedObject(
                label="face",
                confidence=0.8,  # OpenCV doesn't provide confidence
                bounding_box=bbox,
                content_type=ContentType.FACE
            ))

        return detected_faces

    async def _assess_quality(self, image) -> QualityMetrics:
        """Assess image quality using OpenCV."""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate various quality metrics
        # Sharpness (Laplacian variance)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Brightness (mean pixel value)
        brightness = np.mean(gray)

        # Contrast (standard deviation)
        contrast = np.std(gray)

        # Noise level (using high-pass filter)
        kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
        noise = cv2.filter2D(gray, -1, kernel)
        noise_level = np.mean(np.abs(noise))

        # Blur detection (variance of Laplacian)
        blur_detection = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Normalize scores (0-100)
        sharpness_score = min(sharpness / 100, 100)
        brightness_score = 100 - abs(brightness - 127.5) / 127.5 * 100
        contrast_score = min(contrast / 50, 100)
        noise_score = max(0, 100 - noise_level / 10)
        blur_score = min(blur_detection / 100, 100)

        # Calculate overall score
        overall_score = (sharpness_score + brightness_score + contrast_score + noise_score + blur_score) / 5

        # Generate recommendations
        recommendations = []
        if sharpness_score < 50:
            recommendations.append("Image appears to be blurry or out of focus")
        if brightness_score < 50:
            recommendations.append("Image brightness could be improved")
        if contrast_score < 30:
            recommendations.append("Image has low contrast")
        if noise_score < 70:
            recommendations.append("Image has significant noise")

        return QualityMetrics(
            sharpness=sharpness_score,
            brightness=brightness_score,
            contrast=contrast_score,
            noise_level=100 - noise_score,
            blur_detection=blur_score,
            overall_score=overall_score,
            recommendations=recommendations
        )

    async def _detect_objects_opencv(self, image) -> List[DetectedObject]:
        """Basic object detection using OpenCV (simplified)."""
        # This is a simplified implementation
        # In practice, you'd use pre-trained models like YOLO, SSD, etc.
        objects = []

        # Example: detect edges as "objects"
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                bbox = BoundingBox(x=float(x), y=float(y), width=float(w), height=float(h))
                objects.append(DetectedObject(
                    label="object",
                    confidence=0.6,
                    bounding_box=bbox,
                    content_type=ContentType.OBJECT
                ))

        return objects[:10]  # Limit to 10 objects

    def is_available(self) -> bool:
        """Check if OpenCV is available."""
        return OPENCV_AVAILABLE

    def get_supported_types(self) -> List[AnalysisType]:
        """Get supported analysis types."""
        return [
            AnalysisType.FACE_DETECTION,
            AnalysisType.QUALITY_ASSESSMENT,
            AnalysisType.OBJECT_DETECTION
        ]


class ImageAnalyzer:
    """
    Comprehensive image analysis system.

    Provides advanced image processing and analysis using multiple vision providers
    and computer vision techniques.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize image analyzer.

        Args:
            config: Configuration dictionary with analysis settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Core configuration
        self.default_providers = self.config.get('providers', ['opencv', 'tesseract'])
        self.max_image_size_mb = self.config.get('max_image_size_mb', 10)
        self.supported_formats = self.config.get('supported_formats', ['jpeg', 'png', 'webp', 'bmp'])

        # Performance settings
        self.enable_caching = self.config.get('enable_caching', True)
        self.batch_size = self.config.get('batch_size', 5)
        self.enable_preprocessing = self.config.get('enable_preprocessing', True)

        # Enterprise features
        self.audit_logging = self.config.get('audit_logging', False)
        self.monitoring_enabled = self.config.get('monitoring_enabled', False)

        # Initialize analyzers
        self.analyzers = self._initialize_analyzers()
        self.cache: Dict[str, AnalysisResult] = {}
        self.metrics: Dict[str, Any] = defaultdict(int)

    def _initialize_analyzers(self) -> Dict[VisionProvider, VisionAnalyzer]:
        """Initialize available vision analyzers."""
        analyzers = {}

        # Initialize OpenCV analyzer
        try:
            analyzers[VisionProvider.OPENCV] = OpenCVAnalyzer(self.config)
        except ImportError:
            pass

        # Initialize Tesseract OCR
        try:
            analyzers[VisionProvider.TESSERACT] = TesseractOCRAnalyzer(self.config)
        except ImportError:
            pass

        # Initialize OpenAI Vision
        try:
            openai_key = self.config.get('openai_api_key', os.getenv('OPENAI_API_KEY'))
            if openai_key:
                analyzers[VisionProvider.OPENAI] = OpenAIVisionAnalyzer(openai_key)
        except ImportError:
            pass

        return analyzers

    async def analyze_image(self, image_path: str, analysis_types: Optional[List[AnalysisType]] = None) -> AnalysisResult:
        """
        Analyze a single image file.

        Args:
            image_path: Path to the image file
            analysis_types: List of analysis types to perform

        Returns:
            Analysis result with detected content and metadata
        """
        start_time = time.time()

        try:
            self.logger.info(f"Analyzing image: {image_path}")

            # Validate image file
            await self._validate_image(image_path)

            # Check cache
            cache_key = self._get_cache_key(image_path, analysis_types)
            if self.enable_caching and cache_key in self.cache:
                self.metrics['cache_hits'] += 1
                return self.cache[cache_key]

            # Load image data
            with open(image_path, 'rb') as f:
                image_data = f.read()

            # Preprocess image if enabled
            if self.enable_preprocessing:
                image_data = await self._preprocess_image(image_data)

            # Extract metadata
            metadata = await self._extract_metadata(image_path)

            # Set default analysis types
            if analysis_types is None:
                analysis_types = [
                    AnalysisType.OBJECT_DETECTION,
                    AnalysisType.TEXT_EXTRACTION,
                    AnalysisType.SCENE_ANALYSIS,
                    AnalysisType.QUALITY_ASSESSMENT
                ]

            # Perform analysis using available providers
            all_results = {}
            detected_objects = []
            extracted_text = []
            scene_description = None
            quality_metrics = None
            tags = []
            confidence_scores = []

            for provider, analyzer in self.analyzers.items():
                supported_types = [t for t in analysis_types if t in analyzer.get_supported_types()]
                if not supported_types:
                    continue

                try:
                    result = await analyzer.analyze_image(image_data, supported_types)
                    all_results[provider.value] = result

                    # Merge results
                    if 'detected_objects' in result:
                        detected_objects.extend(result['detected_objects'])
                    if 'extracted_text' in result:
                        extracted_text.extend(result['extracted_text'])
                    if 'scene_description' in result:
                        scene_description = result['scene_description']
                    if 'quality' in result:
                        quality_metrics = result['quality']
                    if 'tags' in result:
                        tags.extend(result['tags'])
                    if 'confidence' in result:
                        confidence_scores.append(result['confidence'])

                except Exception as e:
                    self.logger.warning(f"Analysis failed with {provider.value}: {e}")

            # Calculate overall confidence
            overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

            # Create result
            processing_time = time.time() - start_time
            primary_provider = list(self.analyzers.keys())[0] if self.analyzers else VisionProvider.OPENCV

            result = AnalysisResult(
                image_path=image_path,
                success=True,
                analysis_types=analysis_types,
                provider_used=primary_provider,
                detected_objects=detected_objects,
                extracted_text=extracted_text,
                scene_description=scene_description,
                image_metadata=metadata,
                quality_metrics=quality_metrics,
                tags=list(set(tags)),  # Remove duplicates
                confidence_score=overall_confidence,
                processing_time=processing_time,
                errors=[]
            )

            # Cache result
            if self.enable_caching:
                self.cache[cache_key] = result

            # Log audit
            if self.audit_logging:
                self._log_operation('analyze_image', {
                    'image_path': image_path,
                    'analysis_types': [t.value for t in analysis_types],
                    'confidence': overall_confidence
                })

            self.metrics['images_analyzed'] += 1
            return result

        except Exception as e:
            self.logger.error(f"Image analysis failed: {e}")

            # Create failure result
            processing_time = time.time() - start_time
            return AnalysisResult(
                image_path=image_path,
                success=False,
                analysis_types=analysis_types or [],
                provider_used=VisionProvider.OPENCV,
                detected_objects=[],
                extracted_text=[],
                processing_time=processing_time,
                errors=[str(e)]
            )

    async def _validate_image(self, image_path: str):
        """Validate image file."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Check file size
        file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
        if file_size_mb > self.max_image_size_mb:
            raise ValueError(f"Image too large: {file_size_mb:.1f}MB > {self.max_image_size_mb}MB")

        # Check format
        try:
            with Image.open(image_path) as img:
                if img.format.lower() not in [f.upper() for f in self.supported_formats]:
                    raise ValueError(f"Unsupported image format: {img.format}")
        except Exception as e:
            raise ValueError(f"Invalid image file: {e}")

    async def _preprocess_image(self, image_data: bytes) -> bytes:
        """Preprocess image for better analysis results."""
        if not PIL_AVAILABLE:
            return image_data

        try:
            # Load image
            image = Image.open(io.BytesIO(image_data))

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Auto-enhance
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)

            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)

            # Save back to bytes
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=95)
            return output.getvalue()

        except Exception as e:
            self.logger.warning(f"Image preprocessing failed: {e}")
            return image_data

    async def _extract_metadata(self, image_path: str) -> Optional[ImageMetadata]:
        """Extract image metadata and EXIF data."""
        if not PIL_AVAILABLE:
            return None

        try:
            with Image.open(image_path) as image:
                # Basic metadata
                metadata = ImageMetadata(
                    filename=os.path.basename(image_path),
                    format=image.format,
                    size=image.size,
                    mode=image.mode,
                    file_size=os.path.getsize(image_path)
                )

                # EXIF data
                exifdata = image.getexif()
                if exifdata:
                    for tag_id, value in exifdata.items():
                        tag = TAGS.get(tag_id, tag_id)

                        if tag == 'Make':
                            metadata.camera_make = str(value)
                        elif tag == 'Model':
                            metadata.camera_model = str(value)
                        elif tag == 'ISOSpeedRatings':
                            metadata.iso = int(value) if isinstance(value, (int, float)) else None
                        elif tag == 'FocalLength':
                            metadata.focal_length = float(value) if isinstance(value, (int, float)) else None
                        elif tag == 'FNumber':
                            metadata.aperture = float(value) if isinstance(value, (int, float)) else None
                        elif tag == 'ExposureTime':
                            metadata.shutter_speed = str(value)
                        elif tag == 'DateTime':
                            try:
                                metadata.creation_date = datetime.strptime(str(value), '%Y:%m:%d %H:%M:%S')
                            except:
                                pass
                        elif tag == 'Orientation':
                            metadata.orientation = int(value) if isinstance(value, (int, float)) else None

                return metadata

        except Exception as e:
            self.logger.error(f"Metadata extraction failed: {e}")
            return None

    async def batch_analyze(self, image_paths: List[str], analysis_types: Optional[List[AnalysisType]] = None) -> List[AnalysisResult]:
        """
        Analyze multiple images in batch.

        Args:
            image_paths: List of image file paths
            analysis_types: List of analysis types to perform

        Returns:
            List of analysis results
        """
        results = []

        # Process in batches to manage memory
        for i in range(0, len(image_paths), self.batch_size):
            batch = image_paths[i:i + self.batch_size]

            # Process batch concurrently
            batch_tasks = [self.analyze_image(path, analysis_types) for path in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    self.logger.error(f"Batch analysis error: {result}")
                else:
                    results.append(result)

        return results

    def search_images_by_content(self, results: List[AnalysisResult], query: str) -> List[Dict[str, Any]]:
        """
        Search images by detected content or text.

        Args:
            results: List of analysis results to search
            query: Search query string

        Returns:
            List of matching images with relevance scores
        """
        matches = []
        query_lower = query.lower()

        for result in results:
            relevance_score = 0.0
            match_details = []

            # Search in scene description
            if result.scene_description and query_lower in result.scene_description.lower():
                relevance_score += 0.3
                match_details.append(f"Scene: {result.scene_description}")

            # Search in extracted text
            for text_obj in result.extracted_text:
                if query_lower in text_obj.text.lower():
                    relevance_score += 0.4 * text_obj.confidence
                    match_details.append(f"Text: {text_obj.text}")

            # Search in object labels
            for obj in result.detected_objects:
                if query_lower in obj.label.lower():
                    relevance_score += 0.2 * obj.confidence
                    match_details.append(f"Object: {obj.label}")

            # Search in tags
            for tag in result.tags:
                if query_lower in tag.lower():
                    relevance_score += 0.1
                    match_details.append(f"Tag: {tag}")

            if relevance_score > 0:
                matches.append({
                    'image_path': result.image_path,
                    'relevance_score': relevance_score,
                    'match_details': match_details,
                    'confidence': result.confidence_score
                })

        # Sort by relevance score
        matches.sort(key=lambda x: x['relevance_score'], reverse=True)
        return matches

    def filter_by_content_type(self, results: List[AnalysisResult], content_type: ContentType) -> List[AnalysisResult]:
        """
        Filter analysis results by content type.

        Args:
            results: List of analysis results
            content_type: Content type to filter by

        Returns:
            Filtered list of results
        """
        filtered_results = []

        for result in results:
            # Check if any detected objects match the content type
            has_content_type = any(
                obj.content_type == content_type for obj in result.detected_objects
            )

            if has_content_type:
                filtered_results.append(result)

        return filtered_results

    def get_quality_summary(self, results: List[AnalysisResult]) -> Dict[str, Any]:
        """
        Get quality assessment summary across multiple images.

        Args:
            results: List of analysis results

        Returns:
            Quality summary statistics
        """
        quality_scores = []
        recommendations = defaultdict(int)

        for result in results:
            if result.quality_metrics:
                quality_scores.append(result.quality_metrics.overall_score)
                for rec in result.quality_metrics.recommendations:
                    recommendations[rec] += 1

        if not quality_scores:
            return {'message': 'No quality metrics available'}

        return {
            'total_images': len(results),
            'average_quality': sum(quality_scores) / len(quality_scores),
            'min_quality': min(quality_scores),
            'max_quality': max(quality_scores),
            'images_analyzed': len(quality_scores),
            'common_issues': dict(recommendations),
            'quality_distribution': {
                'excellent': len([s for s in quality_scores if s >= 80]),
                'good': len([s for s in quality_scores if 60 <= s < 80]),
                'fair': len([s for s in quality_scores if 40 <= s < 60]),
                'poor': len([s for s in quality_scores if s < 40])
            }
        }

    def _get_cache_key(self, image_path: str, analysis_types: Optional[List[AnalysisType]]) -> str:
        """Generate cache key for analysis results."""
        file_stats = os.stat(image_path)
        key_data = {
            'image_path': image_path,
            'file_size': file_stats.st_size,
            'file_mtime': file_stats.st_mtime,
            'analysis_types': [t.value for t in analysis_types] if analysis_types else [],
            'config': self.config
        }
        key_json = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_json.encode()).hexdigest()

    def _log_operation(self, operation: str, details: Dict[str, Any]):
        """Log operations for audit purposes."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'operation': operation,
            'details': details
        }
        self.logger.info(f"AUDIT: {json.dumps(log_entry)}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics."""
        return dict(self.metrics)

    def clear_cache(self):
        """Clear analysis cache."""
        self.cache.clear()
        self.logger.info("Image analyzer cache cleared")

    async def cleanup(self):
        """Cleanup image analyzer resources."""
        try:
            self.clear_cache()
            self.metrics.clear()
            self.logger.info("Image analyzer cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            if hasattr(self, 'cache') and self.cache:
                self.logger.info("ImageAnalyzer being destroyed - cleanup recommended")
        except:
            pass
