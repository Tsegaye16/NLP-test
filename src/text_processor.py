import re
import unicodedata
from typing import List, Dict, Tuple, Optional, Set, Any
from pathlib import Path
import json
from dataclasses import dataclass
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
import stanza

from src.utils.logger import logger
from config.settings import PIPELINE, PATHS

@dataclass
class ProcessedDocument:
    """Container for processed document data"""
    document_id: str
    original_text: str
    cleaned_text: str
    tokens: List[str]
    sentences: List[str]
    lemmas: List[str]
    pos_tags: List[Tuple[str, str]]
    normalized_tokens: List[str]
    stats: Dict[str, Any]
    metadata: Dict[str, Any]

class TextCleaner:
    """Comprehensive text cleaning with historical text considerations"""
    
    def __init__(self, preserve_dates: bool = True):
        self.preserve_dates = preserve_dates
        self.date_patterns = self._compile_date_patterns()
        self.entity_placeholders = {}
        
        # Download NLTK resources if needed
        self._download_nltk_resources()
    
    def _download_nltk_resources(self):
        """Download required NLTK resources"""
        resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
            except LookupError:
                logger.log_step("Downloading NLTK resource", details={"resource": resource})
                nltk.download(resource, quiet=True)
    
    def _compile_date_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for historical dates"""
        patterns = [
            # Full dates: 1945-08-15, 15/08/1945
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
            
            # Month names: August 15, 1945
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b',
            
            # Years: 1945, 1960s
            r'\b(1[0-9]{3}|20[0-9]{2})s?\b',
            
            # Centuries: 19th century
            r'\b\d{1,2}(?:st|nd|rd|th)\s+century\b',
            
            # BCE/CE dates: 500 BCE, 200 CE
            r'\b\d+\s+(?:BCE?|CE|AD)\b',
            
            # Decade ranges: 1960-1970
            r'\b(1[0-9]{3}|20[0-9]{2})[-–—](1[0-9]{3}|20[0-9]{2})\b',
            
            # Approximate dates: circa 1500, c. 1500
            r'\b(?:circa|c\.?|approx\.?|approximately)\s+\d+\b'
        ]
        
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def clean(self, text: str, document_id: Optional[str] = None) -> Tuple[str, Dict]:
        """Clean text while preserving important historical information"""
        with logger.time_step("text_cleaning", document_id):
            
            # Step 1: Normalize unicode and encoding
            cleaned = self._normalize_unicode(text)
            
            # Step 2: Extract and protect temporal and numerical entities
            protected_entities = self._protect_entities(cleaned)
            cleaned = protected_entities['text']
            
            # Step 3: Remove unwanted characters and patterns
            cleaned = self._remove_unwanted_patterns(cleaned)
            
            # Step 4: Fix common OCR errors
            cleaned = self._fix_ocr_errors(cleaned)
            
            # Step 5: Normalize whitespace
            cleaned = self._normalize_whitespace(cleaned)
            
            # Step 6: Restore protected entities
            cleaned = self._restore_entities(cleaned, protected_entities)
            
            # Step 7: Final cleanup
            cleaned = cleaned.strip()
            
            # Collect statistics
            stats = self._collect_cleaning_stats(text, cleaned, protected_entities)
            
            logger.log_step("Text cleaning completed", document_id, {
                "original_length": len(text),
                "cleaned_length": len(cleaned),
                "entities_protected": len(protected_entities['entities'])
            })
            
            return cleaned, stats
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters"""
        # Normalize to NFKC (compatibility decomposition followed by composition)
        text = unicodedata.normalize('NFKC', text)
        
        # Replace curly quotes and apostrophes with straight ones
        replacements = {
            '“': '"', '”': '"', '‘': "'", '’': "'",
            '…': '...', '–': '-', '—': '-', '―': '-',
            '«': '"', '»': '"'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _protect_entities(self, text: str) -> Dict:
        """Protect dates, numbers, and special entities from cleaning"""
        entities = []
        protected_text = text
        
        # Protect dates
        for pattern in self.date_patterns:
            for match in pattern.finditer(text):
                entity = {
                    'type': 'DATE',
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end()
                }
                entities.append(entity)
        
        # Protect Roman numerals (common in historical texts)
        roman_pattern = r'\b[IVXLCDM]+\b'
        for match in re.finditer(roman_pattern, text, re.IGNORECASE):
            entity = {
                'type': 'ROMAN_NUMERAL',
                'text': match.group(),
                'start': match.start(),
                'end': match.end()
            }
            entities.append(entity)
        
        # Protect common historical abbreviations
        abbreviations = [
            'U.S.', 'U.S.S.R.', 'U.K.', 'B.C.', 'A.D.', 'C.E.', 'B.C.E.',
            'e.g.', 'i.e.', 'etc.', 'vs.', 'viz.', 'cf.', 'et al.'
        ]
        
        for abbr in abbreviations:
            pattern = re.compile(r'\b' + re.escape(abbr) + r'\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                entity = {
                    'type': 'ABBREVIATION',
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end()
                }
                entities.append(entity)
        
        # Sort entities by start position (descending) to avoid offset issues
        entities.sort(key=lambda x: x['start'], reverse=True)
        
        # Replace entities with placeholders
        for i, entity in enumerate(entities):
            placeholder = f"__ENTITY_{i:04d}_{entity['type']}__"
            protected_text = (
                protected_text[:entity['start']] + 
                placeholder + 
                protected_text[entity['end']:]
            )
            entity['placeholder'] = placeholder
        
        return {
            'text': protected_text,
            'entities': entities,
            'original_text': text
        }
    
    def _remove_unwanted_patterns(self, text: str) -> str:
        """Remove unwanted patterns while preserving structure"""
        # Remove page numbers and headers/footers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # Standalone page numbers
        text = re.sub(r'-\s*\d+\s*-', '', text)  # Page numbers in footer
        
        # Remove URL-like patterns
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        
        # Remove control characters but keep tabs and newlines
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        # Remove excessive punctuation (keep single instances)
        text = re.sub(r'([!?.]){3,}', r'\1', text)  # Multiple !!! to single !
        text = re.sub(r'[-]{3,}', '--', text)  # Multiple --- to --
        
        return text
    
    def _fix_ocr_errors(self, text: str) -> str:
        """Fix common OCR errors in historical texts"""
        # Common OCR error mappings
        ocr_corrections = {
            r'\b([A-Z])l\b': r'\1I',  # AI vs Al
            r'\b([A-Z])1\b': r'\1I',  # AI vs A1
            r'\b([A-Z])0\b': r'\1O',  # AO vs A0
            r'\b([A-Z])5\b': r'\1S',  # AS vs A5
            r'\bthe\s*([A-Z])': r'the \1',  # Fix spacing
            r'\band\s*([A-Z])': r'and \1',
            r'\bof\s*([A-Z])': r'of \1',
        }
        
        for pattern, replacement in ocr_corrections.items():
            text = re.sub(pattern, replacement, text)
        
        # Fix common word errors
        word_corrections = {
            'rn': 'm',  # rn often misinterpreted as m in OCR
            'cl': 'd',  # cl as d
            'vv': 'w',  # vv as w
        }
        
        for wrong, correct in word_corrections.items():
            text = re.sub(rf'\b{wrong}\b', correct, text, flags=re.IGNORECASE)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving paragraph structure"""
        # Replace multiple newlines with paragraph break (double newline)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Normalize spaces (but keep single newlines for line breaks)
        lines = text.split('\n')
        normalized_lines = []
        
        for line in lines:
            # Remove leading/trailing whitespace
            line = line.strip()
            
            # Normalize internal spaces
            line = re.sub(r'\s+', ' ', line)
            
            if line:  # Keep non-empty lines
                normalized_lines.append(line)
        
        # Join back with appropriate spacing
        return '\n'.join(normalized_lines)
    
    def _restore_entities(self, text: str, protected_entities: Dict) -> str:
        """Restore protected entities to cleaned text"""
        restored_text = text
        
        for entity in protected_entities['entities']:
            placeholder = entity['placeholder']
            original_text = entity['text']
            
            if placeholder in restored_text:
                restored_text = restored_text.replace(placeholder, original_text)
            else:
                logger.log_step("Entity placeholder not found", details={
                    "placeholder": placeholder,
                    "entity": entity['text']
                })
        
        return restored_text
    
    def _collect_cleaning_stats(self, original: str, cleaned: str, 
                               protected_entities: Dict) -> Dict:
        """Collect statistics about the cleaning process"""
        return {
            'original_length': len(original),
            'cleaned_length': len(cleaned),
            'reduction_percentage': (len(original) - len(cleaned)) / len(original) * 100,
            'entities_protected': len(protected_entities['entities']),
            'date_count': sum(1 for e in protected_entities['entities'] if e['type'] == 'DATE'),
            'abbreviation_count': sum(1 for e in protected_entities['entities'] if e['type'] == 'ABBREVIATION'),
            'roman_numeral_count': sum(1 for e in protected_entities['entities'] if e['type'] == 'ROMAN_NUMERAL')
        }

class TextTokenizer:
    """Advanced tokenization with historical text support"""
    
    def __init__(self, spacy_model: str = "en_core_web_lg"):
        self.spacy_model = spacy_model
        self.nlp = self._load_spacy_model()
        self.custom_abbreviations = PIPELINE.custom_abbreviations
        
        # Configure tokenizer
        self._configure_tokenizer()
    
    def _load_spacy_model(self):
        """Load spaCy model with error handling"""
        try:
            nlp = spacy.load(self.spacy_model)
            logger.log_step("spaCy model loaded", details={"model": self.spacy_model})
            return nlp
        except OSError:
            logger.log_step("Downloading spaCy model", details={"model": self.spacy_model})
            import subprocess
            import sys
            
            subprocess.run([sys.executable, "-m", "spacy", "download", self.spacy_model])
            nlp = spacy.load(self.spacy_model)
            return nlp
    
    def _configure_tokenizer(self):
        """Configure spaCy tokenizer for historical text"""
        # Add custom abbreviations using proper spaCy API
        for abbr in self.custom_abbreviations:
            try:
                # Use add_special_case with proper format
                # Format: list of dicts with token attributes
                self.nlp.tokenizer.add_special_case(abbr, [{"ORTH": abbr}])
            except Exception as e:
                # If special case addition fails, log and continue
                logger.log_step("Failed to add abbreviation", details={
                    "abbreviation": abbr,
                    "error": str(e)
                })
        
        # Note: Modifying prefix/suffix search can cause issues with special cases
        # We'll skip this modification to avoid conflicts
        # The default tokenizer should handle most cases correctly
        
        logger.log_step("Tokenizer configured", details={
            "custom_abbreviations": len(self.custom_abbreviations)
        })
    
    def tokenize(self, text: str, document_id: Optional[str] = None) -> Dict:
        """Tokenize text with comprehensive processing"""
        with logger.time_step("tokenization", document_id):
            
            doc = self.nlp(text)
            
            # Extract tokens and linguistic features
            tokens = []
            lemmas = []
            pos_tags = []
            dependencies = []
            named_entities = []
            
            for token in doc:
                tokens.append(token.text)
                lemmas.append(token.lemma_)
                pos_tags.append((token.text, token.pos_, token.tag_))
                dependencies.append((token.text, token.dep_, token.head.text))
            
            # Extract sentences
            sentences = [sent.text for sent in doc.sents]
            
            # Extract named entities
            for ent in doc.ents:
                named_entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
            
            # Collect statistics
            stats = {
                'token_count': len(tokens),
                'sentence_count': len(sentences),
                'unique_tokens': len(set(tokens)),
                'avg_sentence_length': len(tokens) / len(sentences) if sentences else 0,
                'entity_count': len(named_entities)
            }
            
            logger.log_step("Tokenization completed", document_id, {
                "tokens": stats['token_count'],
                "sentences": stats['sentence_count'],
                "entities": stats['entity_count']
            })
            
            return {
                'tokens': tokens,
                'lemmas': lemmas,
                'pos_tags': pos_tags,
                'dependencies': dependencies,
                'sentences': sentences,
                'named_entities': named_entities,
                'stats': stats,
                'spacy_doc': doc  # Keep for further processing if needed
            }

class TextNormalizer:
    """Normalize text for historical analysis"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.stemmer = nltk.PorterStemmer()
        
        # Historical spelling variations
        self.historical_variations = self._load_historical_variations()
        
        # Contractions expansion
        self.contractions = {
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "can't": "cannot",
            "won't": "will not",
            "shouldn't": "should not",
            "couldn't": "could not",
            "wouldn't": "would not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "it's": "it is",
            "he's": "he is",
            "she's": "she is",
            "that's": "that is",
            "what's": "what is",
            "where's": "where is",
            "who's": "who is",
            "i'm": "i am",
            "you're": "you are",
            "we're": "we are",
            "they're": "they are",
            "i've": "i have",
            "you've": "you have",
            "we've": "we have",
            "they've": "they have",
            "i'll": "i will",
            "you'll": "you will",
            "he'll": "he will",
            "she'll": "she will",
            "we'll": "we will",
            "they'll": "they will",
            "i'd": "i would",
            "you'd": "you would",
            "he'd": "he would",
            "she'd": "she would",
            "we'd": "we would",
            "they'd": "they would"
        }
    
    def _load_historical_variations(self) -> Dict[str, str]:
        """Load historical spelling variations"""
        return {
            'colour': 'color',
            'centre': 'center',
            'theatre': 'theater',
            'labour': 'labor',
            'favour': 'favor',
            'honour': 'honor',
            'programme': 'program',
            'travelled': 'traveled',
            'travelling': 'traveling',
            'cancelled': 'canceled',
            'cancelling': 'canceling',
            'defence': 'defense',
            'offence': 'offense',
            'licence': 'license',
            'practise': 'practice',
            'organise': 'organize',
            'realise': 'realize',
            'recognise': 'recognize',
            'analyse': 'analyze',
            'catalogue': 'catalog',
            'dialogue': 'dialog',
            'monologue': 'monolog',
            'analogue': 'analog'
        }
    
    def normalize(self, tokens: List[str], 
                  document_id: Optional[str] = None,
                  options: Optional[Dict] = None) -> Dict:
        """Normalize tokens with various options"""
        with logger.time_step("normalization", document_id):
            
            if options is None:
                options = {
                    'lowercase': True,
                    'remove_stopwords': False,
                    'lemmatize': True,
                    'expand_contractions': True,
                    'fix_historical_spelling': True,
                    'remove_punctuation': False,
                    'min_length': 2
                }
            
            normalized_tokens = []
            original_to_normalized = {}
            normalization_log = []
            
            for token in tokens:
                original_token = token
                
                # Expand contractions
                if options['expand_contractions']:
                    token = self.contractions.get(token.lower(), token)
                
                # Convert to lowercase
                if options['lowercase']:
                    token = token.lower()
                
                # Fix historical spelling
                if options['fix_historical_spelling']:
                    token = self.historical_variations.get(token, token)
                
                # Remove punctuation (but keep some for entity recognition)
                if options['remove_punctuation']:
                    token = re.sub(r'[^\w\s-]', '', token)
                
                # Apply lemmatization
                if options['lemmatize'] and token.isalpha():
                    token = self.lemmatizer.lemmatize(token)
                
                # Filter by length
                if len(token) < options['min_length']:
                    normalization_log.append({
                        'original': original_token,
                        'normalized': token,
                        'action': 'removed (too short)'
                    })
                    continue
                
                # Remove stopwords
                if options['remove_stopwords'] and token in self.stop_words:
                    normalization_log.append({
                        'original': original_token,
                        'normalized': token,
                        'action': 'removed (stopword)'
                    })
                    continue
                
                # Keep the token
                if token and token.strip():
                    normalized_tokens.append(token)
                    original_to_normalized[original_token] = token
                    normalization_log.append({
                        'original': original_token,
                        'normalized': token,
                        'action': 'normalized'
                    })
            
            # Statistics
            stats = {
                'original_token_count': len(tokens),
                'normalized_token_count': len(normalized_tokens),
                'reduction_percentage': (len(tokens) - len(normalized_tokens)) / len(tokens) * 100,
                'unique_normalized': len(set(normalized_tokens)),
                'normalization_operations': len(normalization_log)
            }
            
            logger.log_step("Normalization completed", document_id, {
                "original_tokens": stats['original_token_count'],
                "normalized_tokens": stats['normalized_token_count'],
                "reduction": f"{stats['reduction_percentage']:.1f}%"
            })
            
            return {
                'normalized_tokens': normalized_tokens,
                'original_to_normalized': original_to_normalized,
                'normalization_log': normalization_log[:100],  # Limit log size
                'stats': stats,
                'options_used': options
            }

class TextProcessor:
    """Main text processor coordinating all cleaning, tokenization, and normalization"""
    
    def __init__(self):
        self.cleaner = TextCleaner(preserve_dates=PIPELINE.preserve_dates)
        self.tokenizer = TextTokenizer(spacy_model=PIPELINE.spacy_model)
        self.normalizer = TextNormalizer()
        
        logger.log_step("TextProcessor initialized")
    
    def process_document(self, document_id: str, text: str, 
                        metadata: Optional[Any] = None) -> ProcessedDocument:
        """Process a single document through the complete pipeline"""
        with logger.time_step("full_text_processing", document_id):
            
            # Step 1: Clean text
            cleaned_text, cleaning_stats = self.cleaner.clean(text, document_id)
            
            # Step 2: Tokenize
            tokenization_result = self.tokenizer.tokenize(cleaned_text, document_id)
            
            # Step 3: Normalize
            normalization_result = self.normalizer.normalize(
                tokenization_result['tokens'], 
                document_id
            )
            
            # Combine all statistics
            combined_stats = {
                'cleaning': cleaning_stats,
                'tokenization': tokenization_result['stats'],
                'normalization': normalization_result['stats'],
                'overall': {
                    'original_length': len(text),
                    'cleaned_length': len(cleaned_text),
                    'token_count': len(tokenization_result['tokens']),
                    'normalized_token_count': len(normalization_result['normalized_tokens']),
                    'sentence_count': len(tokenization_result['sentences'])
                }
            }
            
            # Convert metadata to dict if it's a DocumentMetadata object
            if metadata is None:
                metadata_dict = {}
            elif hasattr(metadata, 'to_dict'):
                # It's a DocumentMetadata object
                metadata_dict = metadata.to_dict()
            elif isinstance(metadata, dict):
                metadata_dict = metadata
            else:
                # Try to convert using dataclasses.asdict
                try:
                    from dataclasses import asdict
                    metadata_dict = asdict(metadata)
                except:
                    metadata_dict = {}
            
            # Create processed document
            processed_doc = ProcessedDocument(
                document_id=document_id,
                original_text=text,
                cleaned_text=cleaned_text,
                tokens=tokenization_result['tokens'],
                sentences=tokenization_result['sentences'],
                lemmas=tokenization_result['lemmas'],
                pos_tags=tokenization_result['pos_tags'],
                normalized_tokens=normalization_result['normalized_tokens'],
                stats=combined_stats,
                metadata=metadata_dict
            )
            
            logger.log_step("Document processing complete", document_id, {
                "sentences": combined_stats['overall']['sentence_count'],
                "tokens": combined_stats['overall']['token_count'],
                "normalized_tokens": combined_stats['overall']['normalized_token_count']
            })
            
            return processed_doc
    
    def process_batch(self, documents: Dict[str, Tuple[str, Dict]]) -> Dict[str, ProcessedDocument]:
        """Process multiple documents"""
        processed_docs = {}
        
        logger.log_step("Starting batch processing", details={
            "document_count": len(documents)
        })
        
        for i, (doc_id, (text, metadata)) in enumerate(documents.items(), 1):
            logger.log_step(f"Processing document {i}/{len(documents)}", doc_id)
            
            try:
                processed_doc = self.process_document(doc_id, text, metadata)
                processed_docs[doc_id] = processed_doc
                
                # Save intermediate results periodically
                if i % 5 == 0:
                    self._save_intermediate_results(processed_docs, i)
                    
            except Exception as e:
                logger.log_error("ProcessingError", 
                               f"Error processing document: {str(e)}", 
                               doc_id, e)
        
        logger.log_step("Batch processing complete", details={
            "successful": len(processed_docs),
            "failed": len(documents) - len(processed_docs)
        })
        
        return processed_docs
    
    def _save_intermediate_results(self, processed_docs: Dict[str, ProcessedDocument], 
                                  batch_num: int):
        """Save intermediate processing results"""
        import pickle
        
        output_dir = PATHS.processed_dir / "intermediate"
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"processed_batch_{batch_num}.pkl"
        
        # Convert to serializable format
        serializable_docs = {}
        for doc_id, doc in processed_docs.items():
            serializable_docs[doc_id] = {
                'document_id': doc.document_id,
                'cleaned_text': doc.cleaned_text,
                'tokens': doc.tokens,
                'sentences': doc.sentences,
                'normalized_tokens': doc.normalized_tokens,
                'stats': doc.stats,
                'metadata': doc.metadata
            }
        
        with open(output_file, 'wb') as f:
            pickle.dump(serializable_docs, f)
        
        logger.log_step("Intermediate results saved", details={
            "file": str(output_file),
            "documents_saved": len(serializable_docs)
        })