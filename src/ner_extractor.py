import spacy
from spacy.tokens import Span
from spacy.language import Language
from spacy.pipeline import EntityRuler
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
import json
import re
from datetime import datetime
import dateparser
from collections import defaultdict
import itertools

from src.utils.logger import logger
from config.settings import PATHS

@dataclass
class Entity:
    """Named entity container"""
    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float = 1.0
    source: str = "spacy"  # spacy, gazetteer, rule
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DocumentEntities:
    """All entities for a document"""
    document_id: str
    entities: List[Entity]
    entity_counts: Dict[str, int]
    entity_cooccurrence: Dict[Tuple[str, str], int]
    metadata: Dict[str, Any]

class HistoricalGazetteer:
    """Gazetteer for historical entities"""
    
    def __init__(self, gazetteer_dir: Optional[Path] = None):
        self.gazetteer_dir = gazetteer_dir or PATHS.gazetteers_dir
        self.gazetteers = self._load_gazetteers()
        
        logger.log_step("HistoricalGazetteer initialized", details={
            "gazetteer_dir": str(self.gazetteer_dir),
            "loaded_gazetteers": list(self.gazetteers.keys())
        })
    
    def _load_gazetteers(self) -> Dict[str, Set[str]]:
        """Load all gazetteer files"""
        gazetteers = {}
        
        # Default gazetteer files
        default_files = {
            'historical_figures': [
                'presidents.txt', 'monarchs.txt', 'revolutionaries.txt',
                'philosophers.txt', 'scientists.txt', 'military_leaders.txt'
            ],
            'historical_events': [
                'wars.txt', 'revolutions.txt', 'treaties.txt',
                'discoveries.txt', 'movements.txt'
            ],
            'historical_organizations': [
                'political_parties.txt', 'governments.txt', 'military_alliances.txt',
                'international_organizations.txt', 'movements.txt'
            ],
            'historical_locations': [
                'battles.txt', 'cities.txt', 'countries.txt', 'regions.txt',
                'empires.txt', 'colonies.txt'
            ]
        }
        
        for category, files in default_files.items():
            entities = set()
            for file in files:
                file_path = self.gazetteer_dir / file
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                entities.add(line)
            
            if entities:
                gazetteers[category] = entities
        
        # Load additional custom gazetteers
        for file_path in self.gazetteer_dir.glob("*.txt"):
            if file_path.name not in itertools.chain(*default_files.values()):
                category = file_path.stem
                entities = set()
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            entities.add(line)
                
                if entities:
                    gazetteers[category] = entities
        
        return gazetteers
    
    def search_entities(self, text: str) -> List[Tuple[str, str]]:
        """Search for gazetteer entities in text"""
        found_entities = []
        text_lower = text.lower()
        
        for category, entities in self.gazetteers.items():
            for entity in entities:
                entity_lower = entity.lower()
                
                # Simple substring matching (can be enhanced)
                if entity_lower in text_lower:
                    # Find all occurrences
                    start = 0
                    while True:
                        pos = text_lower.find(entity_lower, start)
                        if pos == -1:
                            break
                        
                        # Get the actual text (preserving case)
                        actual_text = text[pos:pos + len(entity)]
                        
                        found_entities.append((
                            actual_text,
                            self._map_category_to_label(category),
                            pos,
                            pos + len(entity)
                        ))
                        
                        start = pos + 1
        
        # Remove overlapping entities (keep longer ones)
        found_entities.sort(key=lambda x: (x[2], -len(x[0])))
        filtered_entities = []
        
        for i, (text, label, start, end) in enumerate(found_entities):
            overlap = False
            for _, _, other_start, other_end in filtered_entities:
                if not (end <= other_start or start >= other_end):
                    overlap = True
                    break
            
            if not overlap:
                filtered_entities.append((text, label, start, end))
        
        return filtered_entities
    
    def _map_category_to_label(self, category: str) -> str:
        """Map gazetteer category to NER label"""
        mapping = {
            'historical_figures': 'PERSON',
            'historical_events': 'EVENT',
            'historical_organizations': 'ORG',
            'historical_locations': 'LOC'
        }
        return mapping.get(category, category.upper())

class HistoricalNER:
    """Named Entity Recognition for historical documents"""
    
    def __init__(self, spacy_model: str = "en_core_web_lg"):
        self.nlp = self._load_spacy_model(spacy_model)
        self.gazetteer = HistoricalGazetteer()
        
        # Enhance spaCy pipeline
        self._enhance_pipeline()
        
        # Historical date patterns
        self.date_patterns = self._compile_date_patterns()
        
        logger.log_step("HistoricalNER initialized", details={
            "spacy_model": spacy_model,
            "gazetteer_entities": sum(len(v) for v in self.gazetteer.gazetteers.values())
        })
    
    def _load_spacy_model(self, model_name: str):
        """Load spaCy model with customizations"""
        try:
            nlp = spacy.load(model_name)
        except OSError:
            import subprocess
            import sys
            subprocess.run([sys.executable, "-m", "spacy", "download", model_name])
            nlp = spacy.load(model_name)
        
        return nlp
    
    def _enhance_pipeline(self):
        """Enhance spaCy pipeline for historical text"""
        # Add custom entity ruler
        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        else:
            ruler = self.nlp.get_pipe("entity_ruler")
        
        # Add patterns for historical entities
        patterns = self._create_historical_patterns()
        ruler.add_patterns(patterns)
        
        # Add custom NER component
        if "historical_ner" not in self.nlp.pipe_names:
            self.nlp.add_pipe("historical_ner", after="ner")
    
    def _create_historical_patterns(self) -> List[Dict]:
        """Create patterns for historical entity recognition"""
        patterns = []
        
        # Historical titles and honorifics
        titles = [
            "King", "Queen", "Prince", "Princess", "Emperor", "Empress",
            "President", "Prime Minister", "Chancellor", "Pope", "Cardinal",
            "General", "Admiral", "Field Marshal", "Commander", "Captain",
            "Sir", "Lord", "Lady", "Duke", "Duchess", "Baron", "Count", "Earl"
        ]
        
        for title in titles:
            patterns.append({
                "label": "PERSON",
                "pattern": [
                    {"LOWER": title.lower()},
                    {"POS": "PROPN", "OP": "+"}
                ]
            })
        
        # Historical events
        event_indicators = ["War", "Revolution", "Treaty", "Conference", 
                          "Congress", "Council", "Alliance", "League"]
        
        for indicator in event_indicators:
            patterns.append({
                "label": "EVENT",
                "pattern": [
                    {"POS": "PROPN", "OP": "+"},
                    {"LOWER": indicator.lower()}
                ]
            })
        
        # Historical periods
        period_patterns = [
            {"label": "DATE", "pattern": [{"LOWER": {"IN": ["century", "era", "age", "period"]}}]},
            {"label": "DATE", "pattern": [{"SHAPE": "dddd"}, {"LOWER": {"IN": ["bc", "ad", "bce", "ce"]}}]},
        ]
        patterns.extend(period_patterns)
        
        return patterns
    
    def _compile_date_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for historical dates"""
        patterns = [
            # Centuries: 19th century, 20th century
            r'\b(\d{1,2})(?:st|nd|rd|th)\s+century\b',
            
            # Decade ranges: 1960-1970, 1914-1918
            r'\b(1[0-9]{3}|20[0-9]{2})[-–—](1[0-9]{3}|20[0-9]{2})\b',
            
            # Approximate dates: circa 1500, c. 1500, approx. 1500
            r'\b(?:circa|c\.?|approx\.?|approximately|around|about)\s+(\d{3,4})\b',
            
            # Season-year: Spring 1945, Summer of 1945
            r'\b(?:Spring|Summer|Autumn|Fall|Winter)(?:\s+of)?\s+(\d{4})\b',
            
            # Early/mid/late periods: early 20th century, mid-19th century
            r'\b(?:early|mid|middle|late)\s+(?:\d{1,2}(?:st|nd|rd|th)\s+)?century\b',
            
            # Reign periods: reign of Louis XIV, during the reign of
            r'\b(?:reign|rule|era)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
        ]
        
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    @Language.component("historical_ner")
    def historical_ner_component(doc):
        """Custom NER component for historical entities"""
        # This will be added to the spaCy pipeline
        return doc
    
    def extract_entities(self, text: str, document_id: Optional[str] = None) -> DocumentEntities:
        """Extract entities from text using multiple methods"""
        with logger.time_step("entity_extraction", document_id):
            
            # Method 1: spaCy NER
            spacy_entities = self._extract_spacy_entities(text)
            
            # Method 2: Gazetteer lookup
            gazetteer_entities = self._extract_gazetteer_entities(text)
            
            # Method 3: Rule-based extraction for dates and events
            rule_entities = self._extract_rule_based_entities(text)
            
            # Combine and deduplicate entities
            all_entities = self._combine_entities(
                spacy_entities, gazetteer_entities, rule_entities, text=text
            )
            
            # Analyze entity relationships
            entity_counts = self._count_entities(all_entities)
            cooccurrence = self._calculate_cooccurrence(all_entities, text)
            
            # Create document entities container
            doc_entities = DocumentEntities(
                document_id=document_id or "unknown",
                entities=all_entities,
                entity_counts=entity_counts,
                entity_cooccurrence=cooccurrence,
                metadata={
                    'extraction_methods': ['spacy', 'gazetteer', 'rule_based'],
                    'text_length': len(text),
                    'extraction_timestamp': datetime.now().isoformat()
                }
            )
            
            logger.log_step("Entity extraction complete", document_id, {
                "total_entities": len(all_entities),
                "unique_labels": len(entity_counts),
                "entity_distribution": {k: v for k, v in entity_counts.items() if v > 0}
            })
            
            return doc_entities
    
    def _extract_spacy_entities(self, text: str) -> List[Entity]:
        """Extract entities using spaCy"""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entity = Entity(
                text=ent.text,
                label=ent.label_,
                start_char=ent.start_char,
                end_char=ent.end_char,
                confidence=0.9,  # Default confidence for spaCy
                source="spacy",
                metadata={
                    'spacy_label': ent.label_,
                    'lemma': ent.lemma_ if hasattr(ent, 'lemma_') else None
                }
            )
            entities.append(entity)
        
        return entities
    
    def _extract_gazetteer_entities(self, text: str) -> List[Entity]:
        """Extract entities using gazetteer"""
        gazetteer_matches = self.gazetteer.search_entities(text)
        entities = []
        
        for entity_text, label, start, end in gazetteer_matches:
            entity = Entity(
                text=entity_text,
                label=label,
                start_char=start,
                end_char=end,
                confidence=0.95,  # High confidence for gazetteer matches
                source="gazetteer",
                metadata={
                    'gazetteer_source': 'historical',
                    'matched_text': entity_text
                }
            )
            entities.append(entity)
        
        return entities
    
    def _extract_rule_based_entities(self, text: str) -> List[Entity]:
        """Extract entities using rule-based patterns"""
        entities = []
        
        # Extract dates using patterns
        for pattern in self.date_patterns:
            for match in pattern.finditer(text):
                entity_text = match.group()
                
                # Try to parse the date
                try:
                    parsed_date = dateparser.parse(entity_text)
                    date_info = {
                        'parsed_date': parsed_date.isoformat() if parsed_date else None,
                        'date_type': 'historical_period'
                    }
                except:
                    date_info = {'date_type': 'historical_period'}
                
                entity = Entity(
                    text=entity_text,
                    label='DATE',
                    start_char=match.start(),
                    end_char=match.end(),
                    confidence=0.8,
                    source="rule",
                    metadata=date_info
                )
                entities.append(entity)
        
        # Extract historical events using contextual patterns
        event_patterns = [
            (r'\b(?:Battle|Siege|War|Revolution|Treaty|Conference)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', 'EVENT'),
            (r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:War|Revolution|Treaty|Alliance)\b', 'EVENT'),
            (r'\b(?:Proclamation|Declaration|Manifesto|Charter|Constitution)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', 'DOCUMENT'),
        ]
        
        for pattern, label in event_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity = Entity(
                    text=match.group(),
                    label=label,
                    start_char=match.start(),
                    end_char=match.end(),
                    confidence=0.7,
                    source="rule",
                    metadata={'pattern': pattern}
                )
                entities.append(entity)
        
        return entities
    
    def _combine_entities(self, *entity_lists: List[Entity], text: str) -> List[Entity]:
        """Combine entities from different sources, removing duplicates"""
        all_entities = []
        seen_positions = set()
        
        # Sort entities by start position
        flat_entities = [ent for sublist in entity_lists for ent in sublist]
        flat_entities.sort(key=lambda x: (x.start_char, -x.end_char))
        
        for entity in flat_entities:
            # Check for overlap with already selected entities
            overlap = False
            positions_to_remove = []
            
            # First pass: identify overlapping positions (don't modify set during iteration)
            for (start, end) in seen_positions:
                if not (entity.end_char <= start or entity.start_char >= end):
                    overlap = True
                    
                    # If overlapping, keep the entity with higher confidence or longer text
                    if entity.confidence > 0.8 or len(entity.text) > (end - start):
                        # Mark for removal
                        positions_to_remove.append((start, end))
                    else:
                        # Keep existing entity, skip this one
                        overlap = True
                        break
            
            # Remove marked positions (outside the iteration)
            for pos in positions_to_remove:
                seen_positions.remove(pos)
                all_entities = [e for e in all_entities 
                              if not (e.start_char == pos[0] and e.end_char == pos[1])]
            
            if not overlap:
                all_entities.append(entity)
                seen_positions.add((entity.start_char, entity.end_char))
        
        # Validate entities
        validated_entities = []
        for entity in all_entities:
            if self._validate_entity(entity, text):
                validated_entities.append(entity)
        
        return validated_entities
    
    def _validate_entity(self, entity: Entity, text: str) -> bool:
        """Validate extracted entity"""
        # Check if entity text matches actual text
        actual_text = text[entity.start_char:entity.end_char]
        if actual_text != entity.text:
            logger.log_step("Entity text mismatch", details={
                "expected": entity.text,
                "actual": actual_text
            })
            return False
        
        # Check entity length
        if len(entity.text) < 2:
            return False
        
        # Check if entity is mostly punctuation or numbers
        if sum(1 for c in entity.text if c.isalnum()) < len(entity.text) * 0.5:
            return False
        
        return True
    
    def _count_entities(self, entities: List[Entity]) -> Dict[str, int]:
        """Count entities by label"""
        counts = defaultdict(int)
        for entity in entities:
            counts[entity.label] += 1
        return dict(counts)
    
    def _calculate_cooccurrence(self, entities: List[Entity], 
                               text: str, window_size: int = 100) -> Dict[Tuple[str, str], int]:
        """Calculate entity co-occurrence within a window"""
        cooccurrence = defaultdict(int)
        
        # Sort entities by position
        sorted_entities = sorted(entities, key=lambda x: x.start_char)
        
        for i, entity1 in enumerate(sorted_entities):
            for entity2 in sorted_entities[i+1:]:
                # Check if entities are within window
                if entity2.start_char - entity1.end_char <= window_size:
                    pair = tuple(sorted([entity1.label, entity2.label]))
                    cooccurrence[pair] += 1
        
        return dict(cooccurrence)
    
    def extract_batch(self, documents: Dict[str, str]) -> Dict[str, DocumentEntities]:
        """Extract entities from multiple documents"""
        all_entities = {}
        
        logger.log_step("Starting batch entity extraction", details={
            "document_count": len(documents)
        })
        
        for i, (doc_id, text) in enumerate(documents.items(), 1):
            logger.log_step(f"Extracting entities from document {i}/{len(documents)}", doc_id)
            
            try:
                doc_entities = self.extract_entities(text, doc_id)
                all_entities[doc_id] = doc_entities
                
                # Save intermediate results
                if i % 5 == 0:
                    self._save_intermediate_entities(all_entities, i)
                    
            except Exception as e:
                logger.log_error("EntityExtractionError", 
                               f"Error extracting entities: {str(e)}", 
                               doc_id, e)
        
        logger.log_step("Batch entity extraction complete", details={
            "successful": len(all_entities),
            "failed": len(documents) - len(all_entities)
        })
        
        return all_entities
    
    def _save_intermediate_entities(self, entities: Dict[str, DocumentEntities], 
                                   batch_num: int):
        """Save intermediate entity extraction results"""
        output_dir = PATHS.outputs_dir / "entities" / "intermediate"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"entities_batch_{batch_num}.json"
        
        serializable_entities = {}
        for doc_id, doc_entities in entities.items():
            serializable_entities[doc_id] = {
                'document_id': doc_entities.document_id,
                'entities': [
                    {
                        'text': ent.text,
                        'label': ent.label,
                        'start_char': ent.start_char,
                        'end_char': ent.end_char,
                        'confidence': ent.confidence,
                        'source': ent.source
                    }
                    for ent in doc_entities.entities[:500]  # Limit for JSON
                ],
                'entity_counts': doc_entities.entity_counts,
                'metadata': doc_entities.metadata
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_entities, f, indent=2, ensure_ascii=False)
        
        logger.log_step("Intermediate entities saved", details={
            "file": str(output_file),
            "documents_saved": len(serializable_entities)
        })
    
    def analyze_entity_network(self, entities: Dict[str, DocumentEntities]) -> Dict[str, Any]:
        """Analyze entity relationships across documents"""
        # Aggregate entities across all documents
        all_entities = []
        entity_doc_map = defaultdict(set)
        
        for doc_id, doc_entities in entities.items():
            for entity in doc_entities.entities:
                all_entities.append(entity)
                entity_doc_map[entity.text].add(doc_id)
        
        # Calculate global statistics
        label_counts = defaultdict(int)
        for entity in all_entities:
            label_counts[entity.label] += 1
        
        # Find most frequent entities
        entity_freq = defaultdict(int)
        for entity in all_entities:
            key = f"{entity.text} ({entity.label})"
            entity_freq[key] += 1
        
        top_entities = dict(sorted(
            entity_freq.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:50])
        
        # Find entity relationships (co-occurrence across documents)
        relationship_graph = defaultdict(lambda: defaultdict(int))
        
        for doc_id, doc_entities in entities.items():
            doc_entity_list = doc_entities.entities
            for i, entity1 in enumerate(doc_entity_list):
                for entity2 in doc_entity_list[i+1:]:
                    key1 = f"{entity1.text} ({entity1.label})"
                    key2 = f"{entity2.text} ({entity2.label})"
                    pair = tuple(sorted([key1, key2]))
                    relationship_graph[pair[0]][pair[1]] += 1
        
        analysis = {
            'total_entities': len(all_entities),
            'unique_entities': len(set(e.text for e in all_entities)),
            'label_distribution': dict(label_counts),
            'top_entities': top_entities,
            'documents_with_entities': len(entities),
            'average_entities_per_doc': len(all_entities) / len(entities) if entities else 0,
            'entity_network': {
                'nodes': list(set(
                    f"{e.text} ({e.label})" for e in all_entities
                )),
                'edges': relationship_graph
            }
        }
        
        # Save analysis
        analysis_file = PATHS.outputs_dir / "entities" / "entity_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.log_step("Entity network analysis complete", details={
            "analysis_file": str(analysis_file),
            "total_entities": len(all_entities),
            "unique_entities": len(set(e.text for e in all_entities))
        })
        
        return analysis
    
    def save_entities(self, entities: Dict[str, DocumentEntities], 
                     output_dir: Optional[Path] = None):
        """Save extracted entities"""
        if output_dir is None:
            output_dir = PATHS.outputs_dir / "entities"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save in multiple formats
        formats = ['json', 'csv', 'txt']
        
        for fmt in formats:
            if fmt == 'json':
                self._save_entities_json(entities, output_dir)
            elif fmt == 'csv':
                self._save_entities_csv(entities, output_dir)
            elif fmt == 'txt':
                self._save_entities_txt(entities, output_dir)
        
        # Save consolidated summary
        self._save_entity_summary(entities, output_dir)
        
        logger.log_step("Entities saved in multiple formats", details={
            "output_dir": str(output_dir),
            "formats": formats,
            "document_count": len(entities)
        })
    
    def _save_entities_json(self, entities: Dict[str, DocumentEntities], output_dir: Path):
        """Save entities as JSON"""
        json_data = {}
        
        for doc_id, doc_entities in entities.items():
            json_data[doc_id] = {
                'document_id': doc_entities.document_id,
                'entities': [
                    {
                        'text': ent.text,
                        'label': ent.label,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': ent.confidence,
                        'source': ent.source
                    }
                    for ent in doc_entities.entities
                ],
                'entity_counts': doc_entities.entity_counts,
                'metadata': doc_entities.metadata
            }
        
        json_file = output_dir / "entities.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    def _save_entities_csv(self, entities: Dict[str, DocumentEntities], output_dir: Path):
        """Save entities as CSV"""
        import pandas as pd
        
        rows = []
        for doc_id, doc_entities in entities.items():
            for entity in doc_entities.entities:
                rows.append({
                    'document_id': doc_id,
                    'entity_text': entity.text,
                    'entity_label': entity.label,
                    'start_char': entity.start_char,
                    'end_char': entity.end_char,
                    'confidence': entity.confidence,
                    'source': entity.source,
                    'context': '...'  # Could add context window
                })
        
        df = pd.DataFrame(rows)
        csv_file = output_dir / "entities.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
    
    def _save_entities_txt(self, entities: Dict[str, DocumentEntities], output_dir: Path):
        """Save entities as readable text"""
        txt_file = output_dir / "entities.txt"
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            for doc_id, doc_entities in entities.items():
                f.write(f"=== Document: {doc_id} ===\n")
                f.write(f"Total entities: {len(doc_entities.entities)}\n")
                f.write("=" * 50 + "\n")
                
                # Group by label
                by_label = defaultdict(list)
                for entity in doc_entities.entities:
                    by_label[entity.label].append(entity)
                
                for label, label_entities in by_label.items():
                    f.write(f"\n{label.upper()} ({len(label_entities)}):\n")
                    for entity in label_entities[:20]:  # Limit per label
                        f.write(f"  - {entity.text}\n")
                
                f.write("\n" + "=" * 50 + "\n\n")
    
    def _save_entity_summary(self, entities: Dict[str, DocumentEntities], output_dir: Path):
        """Save entity summary statistics"""
        summary = {
            'total_documents': len(entities),
            'total_entities': sum(len(doc.entities) for doc in entities.values()),
            'entity_distribution': defaultdict(int),
            'documents_by_period': defaultdict(int),
            'top_entities_by_label': defaultdict(list)
        }
        
        # Aggregate statistics
        all_entities = []
        for doc_id, doc_entities in entities.items():
            # Entity distribution
            for label, count in doc_entities.entity_counts.items():
                summary['entity_distribution'][label] += count
            
            # Period distribution
            period = doc_entities.metadata.get('period', 'Unknown')
            summary['documents_by_period'][period] += 1
            
            # Collect all entities
            all_entities.extend(doc_entities.entities)
        
        # Find top entities by label
        entities_by_label = defaultdict(list)
        for entity in all_entities:
            entities_by_label[entity.label].append(entity)
        
        for label, label_entities in entities_by_label.items():
            # Count frequency
            freq = defaultdict(int)
            for entity in label_entities:
                freq[entity.text] += 1
            
            # Get top 10
            top_entities = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
            summary['top_entities_by_label'][label] = top_entities
        
        # Save summary
        summary_file = output_dir / "entity_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)