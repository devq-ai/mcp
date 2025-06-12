"""
PDF Processor for Agentical

This module provides comprehensive PDF processing capabilities including
text extraction, OCR, table detection, form processing, and metadata analysis.

Features:
- Text extraction from native PDF files
- OCR for scanned documents and images
- Table detection and extraction
- Form field recognition and data extraction
- Metadata and structure analysis
- Multi-page processing with progress tracking
- Batch processing for multiple PDFs
- Image extraction and analysis
- Document classification and categorization
- Integration with AI/ML models for content analysis
"""

import asyncio
import io
import json
import os
import tempfile
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
import base64

# Optional dependencies
try:
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    from pdfminer.high_level import extract_text
    from pdfminer.layout import LAParams
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False


class ProcessingMethod(Enum):
    """PDF processing methods."""
    NATIVE_TEXT = "native_text"
    OCR = "ocr"
    HYBRID = "hybrid"
    AUTO_DETECT = "auto_detect"


class ContentType(Enum):
    """Types of content found in PDFs."""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    FORM = "form"
    HEADER = "header"
    FOOTER = "footer"
    WATERMARK = "watermark"
    SIGNATURE = "signature"


class DocumentType(Enum):
    """Document classification types."""
    INVOICE = "invoice"
    CONTRACT = "contract"
    REPORT = "report"
    FORM = "form"
    RECEIPT = "receipt"
    LETTER = "letter"
    MANUAL = "manual"
    PRESENTATION = "presentation"
    UNKNOWN = "unknown"


@dataclass
class PDFMetadata:
    """PDF document metadata."""
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    keywords: Optional[str] = None
    page_count: int = 0
    file_size: int = 0
    encrypted: bool = False
    pdf_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        if self.creation_date:
            data['creation_date'] = self.creation_date.isoformat()
        if self.modification_date:
            data['modification_date'] = self.modification_date.isoformat()
        return data


@dataclass
class ExtractedContent:
    """Content extracted from PDF."""
    content_type: ContentType
    text: str
    page_number: int
    coordinates: Optional[Tuple[float, float, float, float]] = None  # x1, y1, x2, y2
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data['content_type'] = self.content_type.value
        return data


@dataclass
class TableData:
    """Extracted table data."""
    page_number: int
    table_index: int
    headers: List[str]
    rows: List[List[str]]
    coordinates: Optional[Tuple[float, float, float, float]] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)

    def to_dataframe(self):
        """Convert to pandas DataFrame if available."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required for DataFrame conversion")

        return pd.DataFrame(self.rows, columns=self.headers)


@dataclass
class ProcessingResult:
    """Result from PDF processing operation."""
    file_path: str
    success: bool
    processing_method: ProcessingMethod
    document_type: DocumentType
    metadata: PDFMetadata
    extracted_content: List[ExtractedContent]
    tables: List[TableData]
    images: List[Dict[str, Any]]
    processing_time: float
    page_count: int
    error_count: int
    errors: List[str]
    confidence_score: float
    language_detected: Optional[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data['processing_method'] = self.processing_method.value
        data['document_type'] = self.document_type.value
        data['metadata'] = self.metadata.to_dict()
        data['extracted_content'] = [content.to_dict() for content in self.extracted_content]
        data['tables'] = [table.to_dict() for table in self.tables]
        return data


class PDFExtractor(ABC):
    """Abstract base class for PDF extraction methods."""

    @abstractmethod
    async def extract_text(self, file_path: str) -> List[ExtractedContent]:
        """Extract text from PDF."""
        pass

    @abstractmethod
    async def extract_metadata(self, file_path: str) -> PDFMetadata:
        """Extract PDF metadata."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if extractor dependencies are available."""
        pass


class NativeTextExtractor(PDFExtractor):
    """Extract native text from PDF files."""

    def __init__(self):
        if not PDF_AVAILABLE:
            raise ImportError("PyPDF2 and pdfplumber required for native text extraction")

    async def extract_text(self, file_path: str) -> List[ExtractedContent]:
        """Extract native text using pdfplumber."""
        content = []

        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        content.append(ExtractedContent(
                            content_type=ContentType.TEXT,
                            text=text.strip(),
                            page_number=page_num + 1,
                            confidence=1.0
                        ))
        except Exception as e:
            logging.error(f"Native text extraction failed: {e}")

        return content

    async def extract_metadata(self, file_path: str) -> PDFMetadata:
        """Extract PDF metadata."""
        metadata = PDFMetadata()

        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                if pdf_reader.metadata:
                    meta = pdf_reader.metadata
                    metadata.title = meta.get('/Title')
                    metadata.author = meta.get('/Author')
                    metadata.subject = meta.get('/Subject')
                    metadata.creator = meta.get('/Creator')
                    metadata.producer = meta.get('/Producer')
                    metadata.keywords = meta.get('/Keywords')

                    # Handle dates
                    if '/CreationDate' in meta:
                        try:
                            metadata.creation_date = meta['/CreationDate']
                        except:
                            pass

                    if '/ModDate' in meta:
                        try:
                            metadata.modification_date = meta['/ModDate']
                        except:
                            pass

                metadata.page_count = len(pdf_reader.pages)
                metadata.encrypted = pdf_reader.is_encrypted

            metadata.file_size = os.path.getsize(file_path)

        except Exception as e:
            logging.error(f"Metadata extraction failed: {e}")

        return metadata

    def is_available(self) -> bool:
        """Check if dependencies are available."""
        return PDF_AVAILABLE


class OCRExtractor(PDFExtractor):
    """Extract text using OCR for scanned documents."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if not OCR_AVAILABLE or not PYMUPDF_AVAILABLE:
            raise ImportError("PIL, pytesseract, and PyMuPDF required for OCR extraction")

        self.config = config or {}
        self.tesseract_config = self.config.get('tesseract_config', '--oem 3 --psm 6')
        self.language = self.config.get('language', 'eng')
        self.dpi = self.config.get('dpi', 300)

    async def extract_text(self, file_path: str) -> List[ExtractedContent]:
        """Extract text using OCR."""
        content = []

        try:
            pdf_document = fitz.open(file_path)

            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)

                # Convert page to image
                mat = fitz.Matrix(self.dpi/72, self.dpi/72)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("ppm")

                # Convert to PIL Image
                image = Image.open(io.BytesIO(img_data))

                # Perform OCR
                ocr_text = pytesseract.image_to_string(
                    image,
                    lang=self.language,
                    config=self.tesseract_config
                )

                # Get confidence score
                try:
                    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    confidence = avg_confidence / 100.0
                except:
                    confidence = 0.5

                if ocr_text and ocr_text.strip():
                    content.append(ExtractedContent(
                        content_type=ContentType.TEXT,
                        text=ocr_text.strip(),
                        page_number=page_num + 1,
                        confidence=confidence
                    ))

            pdf_document.close()

        except Exception as e:
            logging.error(f"OCR extraction failed: {e}")

        return content

    async def extract_metadata(self, file_path: str) -> PDFMetadata:
        """Extract PDF metadata using PyMuPDF."""
        metadata = PDFMetadata()

        try:
            pdf_document = fitz.open(file_path)
            meta = pdf_document.metadata

            metadata.title = meta.get('title')
            metadata.author = meta.get('author')
            metadata.subject = meta.get('subject')
            metadata.creator = meta.get('creator')
            metadata.producer = meta.get('producer')
            metadata.keywords = meta.get('keywords')
            metadata.page_count = len(pdf_document)
            metadata.encrypted = pdf_document.needs_pass

            pdf_document.close()
            metadata.file_size = os.path.getsize(file_path)

        except Exception as e:
            logging.error(f"Metadata extraction failed: {e}")

        return metadata

    def is_available(self) -> bool:
        """Check if dependencies are available."""
        return OCR_AVAILABLE and PYMUPDF_AVAILABLE


class HybridExtractor(PDFExtractor):
    """Combine native text extraction with OCR fallback."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.native_extractor = NativeTextExtractor() if PDF_AVAILABLE else None
        self.ocr_extractor = OCRExtractor(config) if OCR_AVAILABLE and PYMUPDF_AVAILABLE else None
        self.text_confidence_threshold = self.config.get('text_confidence_threshold', 0.1)

    async def extract_text(self, file_path: str) -> List[ExtractedContent]:
        """Extract text using hybrid approach."""
        content = []

        # Try native text extraction first
        if self.native_extractor:
            try:
                native_content = await self.native_extractor.extract_text(file_path)

                # Check if we got meaningful text
                total_text_length = sum(len(c.text) for c in native_content)

                if total_text_length > 100:  # Threshold for meaningful content
                    return native_content

            except Exception as e:
                logging.warning(f"Native extraction failed, falling back to OCR: {e}")

        # Fall back to OCR if native extraction failed or produced little text
        if self.ocr_extractor:
            try:
                content = await self.ocr_extractor.extract_text(file_path)
            except Exception as e:
                logging.error(f"OCR extraction also failed: {e}")

        return content

    async def extract_metadata(self, file_path: str) -> PDFMetadata:
        """Extract metadata using available method."""
        if self.native_extractor:
            return await self.native_extractor.extract_metadata(file_path)
        elif self.ocr_extractor:
            return await self.ocr_extractor.extract_metadata(file_path)
        else:
            return PDFMetadata()

    def is_available(self) -> bool:
        """Check if any extractor is available."""
        return (self.native_extractor and self.native_extractor.is_available()) or \
               (self.ocr_extractor and self.ocr_extractor.is_available())


class TableExtractor:
    """Extract tables from PDF documents."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.table_detection_method = self.config.get('table_detection_method', 'pdfplumber')

    async def extract_tables(self, file_path: str) -> List[TableData]:
        """Extract tables from PDF."""
        tables = []

        if self.table_detection_method == 'pdfplumber' and PDF_AVAILABLE:
            tables = await self._extract_with_pdfplumber(file_path)
        elif self.table_detection_method == 'camelot':
            # Camelot integration would go here
            pass

        return tables

    async def _extract_with_pdfplumber(self, file_path: str) -> List[TableData]:
        """Extract tables using pdfplumber."""
        tables = []

        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()

                    for table_index, table in enumerate(page_tables):
                        if table and len(table) > 1:  # Must have header + at least one row
                            headers = table[0] if table[0] else []
                            rows = table[1:] if len(table) > 1 else []

                            table_data = TableData(
                                page_number=page_num + 1,
                                table_index=table_index,
                                headers=headers,
                                rows=rows,
                                confidence=0.8  # pdfplumber generally reliable
                            )
                            tables.append(table_data)

        except Exception as e:
            logging.error(f"Table extraction failed: {e}")

        return tables


class DocumentClassifier:
    """Classify PDF documents by type."""

    # Keywords for document type classification
    TYPE_KEYWORDS = {
        DocumentType.INVOICE: ['invoice', 'bill', 'payment', 'amount due', 'total'],
        DocumentType.CONTRACT: ['contract', 'agreement', 'terms', 'conditions', 'party'],
        DocumentType.REPORT: ['report', 'analysis', 'summary', 'findings', 'conclusion'],
        DocumentType.FORM: ['form', 'application', 'submit', 'field', 'checkbox'],
        DocumentType.RECEIPT: ['receipt', 'purchase', 'transaction', 'paid', 'thank you'],
        DocumentType.LETTER: ['dear', 'sincerely', 'regards', 'letter', 'correspondence'],
        DocumentType.MANUAL: ['manual', 'instructions', 'guide', 'how to', 'procedure'],
        DocumentType.PRESENTATION: ['slide', 'presentation', 'agenda', 'overview']
    }

    @classmethod
    def classify_document(cls, content: List[ExtractedContent], metadata: PDFMetadata) -> DocumentType:
        """Classify document based on content and metadata."""

        # Combine all text content
        full_text = ' '.join([c.text.lower() for c in content]).lower()

        # Check title and metadata
        title = (metadata.title or '').lower()
        subject = (metadata.subject or '').lower()

        combined_text = f"{full_text} {title} {subject}"

        # Score each document type
        type_scores = {}

        for doc_type, keywords in cls.TYPE_KEYWORDS.items():
            score = sum(combined_text.count(keyword) for keyword in keywords)
            type_scores[doc_type] = score

        # Return type with highest score, or UNKNOWN if no clear match
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            if type_scores[best_type] > 0:
                return best_type

        return DocumentType.UNKNOWN


class PDFProcessor:
    """
    Comprehensive PDF processing system.

    Provides advanced PDF analysis with text extraction, OCR, table detection,
    and document classification capabilities.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize PDF processor.

        Args:
            config: Configuration dictionary with processing settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Core configuration
        self.processing_method = ProcessingMethod(
            self.config.get('processing_method', 'auto_detect')
        )
        self.extract_tables = self.config.get('extract_tables', True)
        self.extract_images = self.config.get('extract_images', True)
        self.classify_documents = self.config.get('classify_documents', True)

        # Performance settings
        self.enable_caching = self.config.get('enable_caching', True)
        self.batch_size = self.config.get('batch_size', 10)
        self.max_file_size_mb = self.config.get('max_file_size_mb', 100)

        # Enterprise features
        self.audit_logging = self.config.get('audit_logging', False)
        self.monitoring_enabled = self.config.get('monitoring_enabled', False)

        # Initialize components
        self.extractors = self._initialize_extractors()
        self.table_extractor = TableExtractor(config) if self.extract_tables else None
        self.cache: Dict[str, ProcessingResult] = {}
        self.metrics: Dict[str, Any] = defaultdict(int)

    def _initialize_extractors(self) -> Dict[ProcessingMethod, PDFExtractor]:
        """Initialize available PDF extractors."""
        extractors = {}

        try:
            extractors[ProcessingMethod.NATIVE_TEXT] = NativeTextExtractor()
        except ImportError:
            pass

        try:
            extractors[ProcessingMethod.OCR] = OCRExtractor(self.config)
        except ImportError:
            pass

        try:
            extractors[ProcessingMethod.HYBRID] = HybridExtractor(self.config)
        except ImportError:
            pass

        return extractors

    async def process_file(self, file_path: str) -> ProcessingResult:
        """
        Process a single PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Processing result with extracted content and analysis
        """
        start_time = time.time()

        try:
            self.logger.info(f"Processing PDF: {file_path}")

            # Validate file
            await self._validate_file(file_path)

            # Check cache
            cache_key = self._get_cache_key(file_path)
            if self.enable_caching and cache_key in self.cache:
                self.metrics['cache_hits'] += 1
                return self.cache[cache_key]

            # Determine processing method
            method = await self._determine_processing_method(file_path)

            # Extract content
            extractor = self.extractors.get(method)
            if not extractor:
                raise ValueError(f"No extractor available for method: {method}")

            # Extract text content
            content = await extractor.extract_text(file_path)

            # Extract metadata
            metadata = await extractor.extract_metadata(file_path)

            # Extract tables if enabled
            tables = []
            if self.table_extractor:
                tables = await self.table_extractor.extract_tables(file_path)

            # Extract images if enabled
            images = []
            if self.extract_images:
                images = await self._extract_images(file_path)

            # Classify document if enabled
            doc_type = DocumentType.UNKNOWN
            if self.classify_documents:
                doc_type = DocumentClassifier.classify_document(content, metadata)

            # Calculate confidence score
            confidence = self._calculate_confidence_score(content, method)

            # Create result
            processing_time = time.time() - start_time
            result = ProcessingResult(
                file_path=file_path,
                success=True,
                processing_method=method,
                document_type=doc_type,
                metadata=metadata,
                extracted_content=content,
                tables=tables,
                images=images,
                processing_time=processing_time,
                page_count=metadata.page_count,
                error_count=0,
                errors=[],
                confidence_score=confidence
            )

            # Cache result
            if self.enable_caching:
                self.cache[cache_key] = result

            # Log audit
            if self.audit_logging:
                self._log_operation('process_file', {
                    'file_path': file_path,
                    'method': method.value,
                    'pages': metadata.page_count,
                    'confidence': confidence
                })

            self.metrics['files_processed'] += 1
            return result

        except Exception as e:
            self.logger.error(f"PDF processing failed: {e}")

            # Create failure result
            processing_time = time.time() - start_time
            return ProcessingResult(
                file_path=file_path,
                success=False,
                processing_method=self.processing_method,
                document_type=DocumentType.UNKNOWN,
                metadata=PDFMetadata(),
                extracted_content=[],
                tables=[],
                images=[],
                processing_time=processing_time,
                page_count=0,
                error_count=1,
                errors=[str(e)],
                confidence_score=0.0
            )

    async def _validate_file(self, file_path: str):
        """Validate PDF file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.lower().endswith('.pdf'):
            raise ValueError(f"File is not a PDF: {file_path}")

        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise ValueError(f"File too large: {file_size_mb:.1f}MB > {self.max_file_size_mb}MB")

    async def _determine_processing_method(self, file_path: str) -> ProcessingMethod:
        """Determine the best processing method for a PDF."""
        if self.processing_method != ProcessingMethod.AUTO_DETECT:
            return self.processing_method

        # Try to determine if PDF has native text or is scanned
        if ProcessingMethod.HYBRID in self.extractors:
            return ProcessingMethod.HYBRID
        elif ProcessingMethod.NATIVE_TEXT in self.extractors:
            # Quick check for native text
            try:
                extractor = self.extractors[ProcessingMethod.NATIVE_TEXT]
                content = await extractor.extract_text(file_path)
                total_text = sum(len(c.text) for c in content)

                if total_text > 100:  # Has meaningful native text
                    return ProcessingMethod.NATIVE_TEXT
                elif ProcessingMethod.OCR in self.extractors:
                    return ProcessingMethod.OCR
                else:
                    return ProcessingMethod.NATIVE_TEXT
            except:
                if ProcessingMethod.OCR in self.extractors:
                    return ProcessingMethod.OCR
                else:
                    return ProcessingMethod.NATIVE_TEXT
        elif ProcessingMethod.OCR in self.extractors:
            return ProcessingMethod.OCR
        else:
            raise ValueError("No PDF extractors available")

    async def _extract_images(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract images from PDF."""
        images = []

        if not PYMUPDF_AVAILABLE:
            return images

        try:
            pdf_document = fitz.open(file_path)

            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                image_list = page.get_images()

                for img_index, img in enumerate(image_list):
                    try:
                        # Get image data
                        xref = img[0]
                        pix = fitz.Pixmap(pdf_document, xref)

                        if pix.n - pix.alpha < 4:  # Skip if not RGB or grayscale
                            img_data = pix.tobytes("png")
                            img_b64 = base64.b64encode(img_data).decode()

                            images.append({
                                'page_number': page_num + 1,
                                'image_index': img_index,
                                'width': pix.width,
                                'height': pix.height,
                                'data': img_b64,
                                'format': 'png'
                            })

                        pix = None

                    except Exception as e:
                        self.logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")

            pdf_document.close()

        except Exception as e:
            self.logger.error(f"Image extraction failed: {e}")

        return images

    def _calculate_confidence_score(self, content: List[ExtractedContent], method: ProcessingMethod) -> float:
        """Calculate overall confidence score for extraction."""
        if not content:
            return 0.0

        # Base confidence by method
        method_confidence = {
            ProcessingMethod.NATIVE_TEXT: 0.95,
            ProcessingMethod.OCR: 0.7,
            ProcessingMethod.HYBRID: 0.85,
            ProcessingMethod.AUTO_DETECT: 0.8
        }

        base_confidence = method_confidence.get(method, 0.5)

        # Average individual content confidence
        content_confidences = [c.confidence for c in content if c.confidence > 0]
        avg_content_confidence = sum(content_confidences) / len(content_confidences) if content_confidences else 0.5

        # Combine scores
        overall_confidence = (base_confidence + avg_content_confidence) / 2

        return min(overall_confidence, 1.0)

    async def process_batch(self, file_paths: List[str]) -> List[ProcessingResult]:
        """
        Process multiple PDF files in batch.

        Args:
            file_paths: List of PDF file paths

        Returns:
            List of processing results
        """
        results = []

        # Process in batches to manage memory
        for i in range(0, len(file_paths), self.batch_size):
            batch = file_paths[i:i + self.batch_size]

            # Process batch concurrently
            batch_tasks = [self.process_file(file_path) for file_path in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    self.logger.error(f"Batch processing error: {result}")
                else:
                    results.append(result)

        return results

    def search_content(self, results: List[ProcessingResult], query: str) -> List[Dict[str, Any]]:
        """
        Search extracted content across processed documents.

        Args:
            results: List of processing results to search
            query: Search query string

        Returns:
            List of search matches with context
        """
        matches = []
        query_lower = query.lower()

        for result in results:
            for content in result.extracted_content:
                if query_lower in content.text.lower():
                    # Find context around match
                    text_lower = content.text.lower()
                    match_pos = text_lower.find(query_lower)

                    start = max(0, match_pos - 100)
                    end = min(len(content.text), match_pos + len(query) + 100)
                    context = content.text[start:end]

                    matches.append({
                        'file_path': result.file_path,
                        'page_number': content.page_number,
                        'content_type': content.content_type.value,
                        'match_text': query,
                        'context': context,
                        'confidence': content.confidence
                    })

        return matches

    def _get_cache_key(self, file_path: str) -> str:
        """Generate cache key for processing results."""
        file_stats = os.stat(file_path)
        key_data = {
            'file_path': file_path,
            'file_size': file_stats.st_size,
            'file_mtime': file_stats.st_mtime,
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
        """Clear processing cache."""
        self.cache.clear()
        self.logger.info("PDF processor cache cleared")

    async def cleanup(self):
        """Cleanup PDF processor resources."""
        try:
            self.clear_cache()
            self.metrics.clear()
            self.logger.info("PDF processor cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            if hasattr(self, 'cache') and self.cache:
                self.logger.info("PDFProcessor being destroyed - cleanup recommended")
        except:
            pass
