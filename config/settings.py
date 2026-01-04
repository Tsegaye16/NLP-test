import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class ProcessingMode(Enum):
    FULL = "full"
    FAST = "fast"
    ACCURATE = "accurate"

@dataclass
class PipelineConfig:
    """Configuration for NLP pipeline"""
    mode: ProcessingMode = ProcessingMode.ACCURATE
    batch_size: int = 10
    max_document_length: int = 1000000
    min_sentence_length: int = 10
    max_sentence_length: int = 500
    language: str = "en"
    
    # Cleaning options
    remove_footnotes: bool = True
    preserve_dates: bool = True
    normalize_whitespace: bool = True
    remove_special_chars: bool = True
    
    # Tokenization
    spacy_model: str = "en_core_web_lg"
    custom_abbreviations: List[str] = None
    
    # NER
    use_custom_ner: bool = True
    historical_gazetteers: List[str] = None
    
    def __post_init__(self):
        if self.custom_abbreviations is None:
            self.custom_abbreviations = [
                "e.g.", "i.e.", "U.S.", "U.S.S.R.", "B.C.", "A.D.",
                "Mr.", "Mrs.", "Dr.", "Prof.", "St.", "No.", "pp.",
                "vol.", "chap.", "fig.", "etc.", "cf.", "viz."
            ]
        
        if self.historical_gazetteers is None:
            self.historical_gazetteers = [
                "historical_figures.txt",
                "historical_events.txt",
                "historical_organizations.txt"
            ]

@dataclass
class PathConfig:
    """Path configuration"""
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = project_root / "data"
    raw_documents: Path = data_dir / "raw_documents"
    processed_dir: Path = data_dir / "processed"
    outputs_dir: Path = data_dir / "outputs"
    gazetteers_dir: Path = data_dir / "gazetteers"
    
    src_dir: Path = project_root / "src"
    logs_dir: Path = project_root / "logs"
    tests_dir: Path = project_root / "tests"
    
    def create_directories(self):
        """Create all required directories"""
        directories = [
            self.raw_documents,
            self.processed_dir,
            self.outputs_dir,
            self.gazetteers_dir,
            self.logs_dir,
            self.processed_dir / "cleaned",
            self.processed_dir / "tokenized",
            self.processed_dir / "vectors",
            self.outputs_dir / "summaries",
            self.outputs_dir / "entities",
            self.outputs_dir / "timelines",
            self.outputs_dir / "reports"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} - {message}"
    rotation: str = "10 MB"
    retention: str = "30 days"
    compression: str = "zip"
    
    # File paths
    main_log: Path = PathConfig().logs_dir / "pipeline.log"
    error_log: Path = PathConfig().logs_dir / "errors.log"
    performance_log: Path = PathConfig().logs_dir / "performance.log"

# Global configuration instances
PATHS = PathConfig()
LOGGING = LoggingConfig()
PIPELINE = PipelineConfig()

# Create directories
PATHS.create_directories()