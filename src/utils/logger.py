import logging
import logging.config
import sys
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime
from contextlib import contextmanager
import time

class PipelineLogger:
    """Comprehensive logging system for the NLP pipeline"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PipelineLogger, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize logging configuration"""
        self.loggers = {}
        self.setup_logging()
        
        # Performance tracking
        self.performance_data = []
        self.start_times = {}
    
    def setup_logging(self):
        """Setup logging configuration from YAML file"""
        config_path = Path(__file__).parent.parent.parent / "config" / "logging_config.yaml"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Update log file paths
            logs_dir = Path(__file__).parent.parent.parent / "logs"
            logs_dir.mkdir(exist_ok=True)
            
            config['handlers']['file_handler']['filename'] = str(logs_dir / "pipeline.log")
            config['handlers']['error_handler']['filename'] = str(logs_dir / "errors.log")
            
            logging.config.dictConfig(config)
        else:
            # Default configuration
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(logs_dir / "pipeline.log"),
                    logging.StreamHandler(sys.stdout)
                ]
            )
        
        # Create main logger
        self.main_logger = logging.getLogger("historical_analysis")
        self.main_logger.info("Logging system initialized")
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger with given name"""
        if name not in self.loggers:
            logger = logging.getLogger(f"historical_analysis.{name}")
            self.loggers[name] = logger
        return self.loggers[name]
    
    def log_step(self, step_name: str, document_id: Optional[str] = None, 
                 details: Optional[Dict] = None, level: str = "INFO"):
        """Log a pipeline step with details"""
        logger = self.get_logger("pipeline")
        message = f"Step: {step_name}"
        
        if document_id:
            message += f" | Document: {document_id}"
        
        if details:
            details_str = json.dumps(details, default=str)
            message += f" | Details: {details_str}"
        
        if level == "INFO":
            logger.info(message)
        elif level == "DEBUG":
            logger.debug(message)
        elif level == "WARNING":
            logger.warning(message)
        elif level == "ERROR":
            logger.error(message)
        elif level == "CRITICAL":
            logger.critical(message)
    
    @contextmanager
    def time_step(self, step_name: str, document_id: Optional[str] = None):
        """Context manager to time a pipeline step"""
        start_time = time.time()
        self.start_times[step_name] = start_time
        self.log_step(f"START_{step_name}", document_id, level="DEBUG")
        
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            performance_record = {
                "step": step_name,
                "document": document_id,
                "start_time": datetime.fromtimestamp(start_time).isoformat(),
                "end_time": datetime.fromtimestamp(end_time).isoformat(),
                "duration_seconds": duration,
                "duration_human": f"{duration:.2f}s"
            }
            
            self.performance_data.append(performance_record)
            self.log_step(
                f"END_{step_name}", 
                document_id, 
                {"duration": f"{duration:.2f}s"},
                level="DEBUG"
            )
    
    def log_error(self, error_type: str, message: str, 
                  document_id: Optional[str] = None, exception: Optional[Exception] = None,
                  details: Optional[Dict[str, Any]] = None):
        """Log an error with context"""
        error_logger = self.get_logger("errors")
        
        error_details = {
            "type": error_type,
            "message": message,
            "document": document_id,
            "exception": str(exception) if exception else None,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add details if provided
        if details:
            error_details.update(details)
        
        error_logger.error(json.dumps(error_details))
    
    def log_performance_summary(self):
        """Log performance summary of all steps"""
        if not self.performance_data:
            return
        
        summary_logger = self.get_logger("performance")
        
        # Calculate statistics
        steps = {}
        for record in self.performance_data:
            step = record["step"]
            if step not in steps:
                steps[step] = {
                    "count": 0,
                    "total_duration": 0,
                    "durations": []
                }
            
            steps[step]["count"] += 1
            steps[step]["total_duration"] += record["duration_seconds"]
            steps[step]["durations"].append(record["duration_seconds"])
        
        # Log summary
        summary = {
            "total_steps": len(self.performance_data),
            "unique_steps": len(steps),
            "step_breakdown": {}
        }
        
        for step, data in steps.items():
            avg_duration = data["total_duration"] / data["count"]
            max_duration = max(data["durations"])
            min_duration = min(data["durations"])
            
            summary["step_breakdown"][step] = {
                "count": data["count"],
                "avg_duration": f"{avg_duration:.2f}s",
                "min_duration": f"{min_duration:.2f}s",
                "max_duration": f"{max_duration:.2f}s",
                "total_duration": f"{data['total_duration']:.2f}s"
            }
        
        summary_logger.info(f"Performance Summary: {json.dumps(summary, indent=2)}")
        
        # Save to file
        logs_dir = Path(__file__).parent.parent.parent / "logs"
        performance_file = logs_dir / "performance_summary.json"
        
        with open(performance_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def save_logs_to_file(self, filename: Optional[str] = None):
        """Save all logs to a JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pipeline_logs_{timestamp}.json"
        
        logs_dir = Path(__file__).parent.parent.parent / "logs"
        log_file = logs_dir / filename
        
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "performance_data": self.performance_data,
            "step_count": len(self.performance_data)
        }
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        self.main_logger.info(f"Logs saved to {log_file}")

# Global logger instance
logger = PipelineLogger()