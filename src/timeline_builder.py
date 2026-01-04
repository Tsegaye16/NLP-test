import re
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
import json
import dateparser
from collections import defaultdict
import numpy as np
from enum import Enum

from src.utils.logger import logger
from config.settings import PATHS

class DatePrecision(Enum):
    """Precision level for dates"""
    YEAR = "year"
    MONTH = "month"
    DAY = "day"
    DECADE = "decade"
    CENTURY = "century"
    APPROXIMATE = "approximate"
    UNKNOWN = "unknown"

@dataclass
class TimelineEvent:
    """Timeline event container"""
    event_id: str
    description: str
    date_str: str
    parsed_date: Optional[date]
    date_precision: DatePrecision
    year: Optional[int]
    decade: Optional[int]
    century: Optional[int]
    sources: List[str]  # Document IDs where event was found
    entities: List[Dict[str, Any]]
    context: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Timeline:
    """Complete timeline container"""
    timeline_id: str
    events: List[TimelineEvent]
    period_range: Tuple[int, int]  # (start_year, end_year)
    event_density: Dict[int, int]  # Events per year
    metadata: Dict[str, Any]

class HistoricalDateParser:
    """Parser for historical dates with various formats"""
    
    def __init__(self):
        self.date_patterns = self._compile_date_patterns()
        self.month_abbreviations = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        
        logger.log_step("HistoricalDateParser initialized")
    
    def _compile_date_patterns(self) -> List[Tuple[str, DatePrecision]]:
        """Compile regex patterns for historical dates"""
        patterns = [
            # Full dates: 1945-08-15, 15/08/1945
            (r'\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b', DatePrecision.DAY),
            (r'\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b', DatePrecision.DAY),
            
            # Month-year: August 1945, Aug 1945
            (r'\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})\b', DatePrecision.MONTH),
            
            # Year only: 1945
            (r'\b(1[0-9]{3}|20[0-9]{2})\b', DatePrecision.YEAR),
            
            # Decades: 1960s, 1960's
            (r'\b(1[0-9]{3}|20[0-9]{2})s\'?', DatePrecision.DECADE),
            
            # Centuries: 19th century, 20th century
            (r'\b(\d{1,2})(?:st|nd|rd|th)\s+century\b', DatePrecision.CENTURY),
            
            # Year ranges: 1914-1918, 1939-1945
            (r'\b(1[0-9]{3}|20[0-9]{2})[-–—](1[0-9]{3}|20[0-9]{2})\b', DatePrecision.YEAR),
            
            # BCE/CE dates: 500 BCE, 200 CE, 500 BC, 200 AD
            (r'\b(\d+)\s+(BCE?|CE|AD)\b', DatePrecision.YEAR),
            
            # Approximate dates: circa 1500, c. 1500
            (r'\b(?:circa|c\.?|approx\.?|approximately|around|about)\s+(\d{3,4})\b', DatePrecision.APPROXIMATE),
            
            # Season-year: Spring 1945, Summer of 1945
            (r'\b(?:Spring|Summer|Autumn|Fall|Winter)(?:\s+of)?\s+(\d{4})\b', DatePrecision.MONTH),
            
            # Early/mid/late: early 20th century, mid-19th century
            (r'\b(early|mid|middle|late)\s+(\d{1,2})(?:st|nd|rd|th)\s+century\b', DatePrecision.CENTURY),
        ]
        
        return [(re.compile(pattern, re.IGNORECASE), precision) 
                for pattern, precision in patterns]
    
    def parse_date(self, date_str: str) -> Tuple[Optional[date], DatePrecision, Optional[int]]:
        """Parse historical date string"""
        date_str = date_str.strip()
        
        # Try each pattern
        for pattern, precision in self.date_patterns:
            match = pattern.match(date_str)
            if match:
                try:
                    if precision == DatePrecision.DAY:
                        # Full date
                        groups = match.groups()
                        if len(groups) == 3:
                            if len(groups[0]) == 4:  # YYYY-MM-DD
                                year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                            else:  # DD-MM-YYYY
                                day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
                            
                            parsed_date = date(year, month, day)
                            return parsed_date, precision, year
                    
                    elif precision == DatePrecision.MONTH:
                        # Month year
                        month_str, year_str = match.groups()
                        month = self._parse_month(month_str)
                        year = int(year_str)
                        
                        # Use first day of month
                        parsed_date = date(year, month, 1)
                        return parsed_date, precision, year
                    
                    elif precision == DatePrecision.YEAR:
                        # Year only
                        year_str = match.groups()[0]
                        year = int(year_str)
                        
                        # Use middle of year
                        parsed_date = date(year, 6, 15)
                        return parsed_date, precision, year
                    
                    elif precision == DatePrecision.DECADE:
                        # Decade
                        decade_str = match.groups()[0]
                        decade = int(decade_str)
                        
                        # Use middle of decade
                        year = decade + 5
                        parsed_date = date(year, 6, 15)
                        return parsed_date, precision, year
                    
                    elif precision == DatePrecision.CENTURY:
                        # Century
                        century_str = match.groups()[-1]  # Last group is century number
                        century = int(century_str)
                        
                        # Use middle of century
                        year = (century - 1) * 100 + 50
                        parsed_date = date(year, 6, 15)
                        return parsed_date, precision, year
                    
                    elif precision == DatePrecision.APPROXIMATE:
                        # Approximate date
                        year_str = match.groups()[-1]
                        year = int(year_str)
                        
                        parsed_date = date(year, 6, 15)
                        return parsed_date, precision, year
                
                except (ValueError, TypeError) as e:
                    logger.log_step("Date parsing error", details={
                        "date_str": date_str,
                        "error": str(e),
                        "pattern": pattern.pattern
                    })
                    continue
        
        # Try dateparser as fallback
        try:
            parsed = dateparser.parse(date_str)
            if parsed:
                year = parsed.year
                precision = self._infer_precision_from_date(parsed)
                return parsed.date(), precision, year
        except:
            pass
        
        return None, DatePrecision.UNKNOWN, None
    
    def _parse_month(self, month_str: str) -> int:
        """Parse month string to integer"""
        month_str = month_str.lower()[:3]  # First three letters
        return self.month_abbreviations.get(month_str, 1)
    
    def _infer_precision_from_date(self, dt: datetime) -> DatePrecision:
        """Infer precision from parsed datetime"""
        if dt.day != 1:
            return DatePrecision.DAY
        elif dt.month != 1:
            return DatePrecision.MONTH
        else:
            return DatePrecision.YEAR
    
    def extract_dates_from_text(self, text: str) -> List[Tuple[str, DatePrecision, Optional[int]]]:
        """Extract all dates from text"""
        dates = []
        
        for pattern, precision in self.date_patterns:
            for match in pattern.finditer(text):
                date_str = match.group()
                parsed_date, actual_precision, year = self.parse_date(date_str)
                
                if parsed_date or year:
                    dates.append((date_str, actual_precision, year))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_dates = []
        for date_tuple in dates:
            if date_tuple not in seen:
                seen.add(date_tuple)
                unique_dates.append(date_tuple)
        
        return unique_dates

class TimelineBuilder:
    """Build chronological timeline from extracted entities and dates"""
    
    def __init__(self):
        self.date_parser = HistoricalDateParser()
        self.event_extractor = EventExtractor()
        
        logger.log_step("TimelineBuilder initialized")
    
    def build_timeline(self, documents: Dict[str, Dict[str, Any]], 
                      entities: Optional[Dict[str, Any]] = None) -> Timeline:
        """Build timeline from documents and entities"""
        with logger.time_step("timeline_building"):
            
            # Extract events from documents
            all_events = self._extract_events_from_documents(documents, entities)
            
            # Sort events chronologically
            sorted_events = self._sort_events_chronologically(all_events)
            
            # Calculate timeline statistics
            period_range = self._calculate_period_range(sorted_events)
            event_density = self._calculate_event_density(sorted_events)
            
            # Create timeline
            timeline = Timeline(
                timeline_id=f"timeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                events=sorted_events,
                period_range=period_range,
                event_density=event_density,
                metadata={
                    'total_events': len(sorted_events),
                    'date_range': f"{period_range[0]}-{period_range[1]}",
                    'build_timestamp': datetime.now().isoformat(),
                    'source_documents': len(documents)
                }
            )
            
            logger.log_step("Timeline built", details={
                "total_events": len(sorted_events),
                "period_range": period_range,
                "source_documents": len(documents)
            })
            
            return timeline
    
    def _extract_events_from_documents(self, documents: Dict[str, Dict[str, Any]], 
                                      entities: Optional[Dict[str, Any]]) -> List[TimelineEvent]:
        """Extract events from documents"""
        all_events = []
        
        for doc_id, doc_data in documents.items():
            text = doc_data.get('text', '')
            metadata = doc_data.get('metadata', {})
            
            # Extract dates from text
            dates = self.date_parser.extract_dates_from_text(text)
            
            # Extract events near dates
            for date_str, precision, year in dates:
                # Find context around date
                context_start = max(0, text.find(date_str) - 200)
                context_end = min(len(text), text.find(date_str) + len(date_str) + 200)
                context = text[context_start:context_end]
                
                # Extract event description from context
                event_description = self.event_extractor.extract_event_description(context, date_str)
                
                # Parse date
                parsed_date, _, _ = self.date_parser.parse_date(date_str)
                
                # Extract entities in context
                context_entities = self._extract_entities_in_context(
                    context, entities.get(doc_id, []) if entities else []
                )
                
                # Create event
                event = TimelineEvent(
                    event_id=f"{doc_id}_{len(all_events)}",
                    description=event_description,
                    date_str=date_str,
                    parsed_date=parsed_date,
                    date_precision=precision,
                    year=year,
                    decade=(year // 10) * 10 if year else None,
                    century=((year - 1) // 100) + 1 if year else None,
                    sources=[doc_id],
                    entities=context_entities,
                    context=context,
                    confidence=self._calculate_event_confidence(
                        event_description, date_str, context_entities
                    ),
                    metadata={
                        'source_document': doc_id,
                        'extraction_method': 'contextual',
                        'document_period': metadata.get('period', 'Unknown')
                    }
                )
                
                all_events.append(event)
        
        return all_events
    
    def _extract_entities_in_context(self, context: str, 
                                    entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract entities that appear in the context"""
        context_entities = []
        context_lower = context.lower()
        
        for entity in entities:
            entity_text = entity.get('text', '').lower()
            if entity_text and entity_text in context_lower:
                context_entities.append(entity)
        
        return context_entities[:10]  # Limit to 10 entities
    
    def _calculate_event_confidence(self, description: str, date_str: str, 
                                   entities: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for event"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on description quality
        if len(description.split()) >= 5:
            confidence += 0.2
        
        # Boost confidence based on entities found
        if entities:
            confidence += min(0.3, len(entities) * 0.05)
        
        # Boost confidence based on date specificity
        if any(term in date_str.lower() for term in ['century', 'decade', 'circa']):
            confidence -= 0.1  # Less specific dates
        elif re.search(r'\d{4}', date_str):
            confidence += 0.1  # Specific year
        
        return min(1.0, max(0.1, confidence))
    
    def _sort_events_chronologically(self, events: List[TimelineEvent]) -> List[TimelineEvent]:
        """Sort events by date"""
        # Separate events with and without dates
        dated_events = []
        undated_events = []
        
        for event in events:
            if event.parsed_date:
                dated_events.append(event)
            elif event.year:
                # Create pseudo-date for sorting
                dated_events.append(event)
            else:
                undated_events.append(event)
        
        # Sort dated events
        dated_events.sort(key=lambda x: (
            x.parsed_date if x.parsed_date else date(x.year, 6, 15) if x.year else date(1, 1, 1)
        ))
        
        # Append undated events at the end
        return dated_events + undated_events
    
    def _calculate_period_range(self, events: List[TimelineEvent]) -> Tuple[int, int]:
        """Calculate the time period covered by events"""
        years = []
        
        for event in events:
            if event.year:
                years.append(event.year)
            elif event.decade:
                years.append(event.decade)
            elif event.century:
                years.append((event.century - 1) * 100 + 50)
        
        if years:
            return (min(years), max(years))
        else:
            return (1900, 2000)  # Default range
    
    def _calculate_event_density(self, events: List[TimelineEvent]) -> Dict[int, int]:
        """Calculate number of events per year"""
        density = defaultdict(int)
        
        for event in events:
            if event.year:
                density[event.year] += 1
            elif event.decade:
                for year in range(event.decade, event.decade + 10):
                    density[year] += 0.1  # Distribute across decade
        
        return dict(density)
    
    def merge_similar_events(self, timeline: Timeline, 
                            similarity_threshold: float = 0.7) -> Timeline:
        """Merge similar events in the timeline"""
        merged_events = []
        used_indices = set()
        
        for i, event1 in enumerate(timeline.events):
            if i in used_indices:
                continue
            
            similar_events = [event1]
            
            # Find similar events
            for j, event2 in enumerate(timeline.events[i+1:], start=i+1):
                if j in used_indices:
                    continue
                
                similarity = self._calculate_event_similarity(event1, event2)
                if similarity >= similarity_threshold:
                    similar_events.append(event2)
                    used_indices.add(j)
            
            # Merge similar events
            if len(similar_events) > 1:
                merged_event = self._merge_events(similar_events)
                merged_events.append(merged_event)
            else:
                merged_events.append(event1)
            
            used_indices.add(i)
        
        # Re-sort merged events
        merged_events_sorted = self._sort_events_chronologically(merged_events)
        
        # Update timeline
        timeline.events = merged_events_sorted
        timeline.metadata['merged_events'] = len(merged_events)
        timeline.metadata['original_events'] = len(timeline.events)
        
        logger.log_step("Events merged", details={
            "original": timeline.metadata['original_events'],
            "merged": timeline.metadata['merged_events'],
            "reduction": f"{100 * (1 - len(merged_events) / timeline.metadata['original_events']):.1f}%"
        })
        
        return timeline
    
    def _calculate_event_similarity(self, event1: TimelineEvent, 
                                   event2: TimelineEvent) -> float:
        """Calculate similarity between two events"""
        similarity = 0.0
        
        # Date similarity
        if event1.year and event2.year:
            year_diff = abs(event1.year - event2.year)
            if year_diff <= 1:
                similarity += 0.4
            elif year_diff <= 5:
                similarity += 0.2
            elif year_diff <= 10:
                similarity += 0.1
        
        # Description similarity (simple word overlap)
        words1 = set(event1.description.lower().split())
        words2 = set(event2.description.lower().split())
        
        if words1 and words2:
            overlap = len(words1.intersection(words2))
            total = len(words1.union(words2))
            similarity += (overlap / total) * 0.4
        
        # Entity overlap
        entities1 = set((e.get('text', '') if isinstance(e, dict) else getattr(e, 'text', '')).lower() for e in event1.entities)
        entities2 = set((e.get('text', '') if isinstance(e, dict) else getattr(e, 'text', '')).lower() for e in event2.entities)
        
        if entities1 and entities2:
            overlap = len(entities1.intersection(entities2))
            total = len(entities1.union(entities2))
            similarity += (overlap / total) * 0.2
        
        return similarity
    
    def _merge_events(self, events: List[TimelineEvent]) -> TimelineEvent:
        """Merge multiple events into one"""
        # Use the most specific date
        primary_event = max(events, key=lambda x: (
            3 if x.date_precision == DatePrecision.DAY else
            2 if x.date_precision == DatePrecision.MONTH else
            1 if x.date_precision == DatePrecision.YEAR else 0
        ))
        
        # Combine descriptions
        descriptions = [e.description for e in events]
        combined_description = self._combine_descriptions(descriptions)
        
        # Combine sources
        all_sources = []
        for event in events:
            all_sources.extend(event.sources)
        
        # Combine entities
        all_entities = []
        for event in events:
            all_entities.extend(event.entities)
        
        # Remove duplicate entities
        unique_entities = []
        seen_entity_texts = set()
        for entity in all_entities:
            entity_text = entity.get('text', '').lower()
            if entity_text and entity_text not in seen_entity_texts:
                seen_entity_texts.add(entity_text)
                unique_entities.append(entity)
        
        # Update primary event
        primary_event.description = combined_description
        primary_event.sources = list(set(all_sources))
        primary_event.entities = unique_entities
        primary_event.confidence = np.mean([e.confidence for e in events])
        primary_event.metadata['merged_from'] = len(events)
        
        return primary_event
    
    def _combine_descriptions(self, descriptions: List[str]) -> str:
        """Combine multiple event descriptions"""
        if not descriptions:
            return "Historical event"
        
        # Remove duplicates and similar descriptions
        unique_descriptions = []
        seen_words = set()
        
        for desc in descriptions:
            words = set(desc.lower().split())
            if not words.intersection(seen_words):
                unique_descriptions.append(desc)
                seen_words.update(words)
        
        if len(unique_descriptions) == 1:
            return unique_descriptions[0]
        else:
            # Combine with semicolons
            return "; ".join(unique_descriptions[:3])  # Limit to 3
    
    def save_timeline(self, timeline: Timeline, 
                     output_dir: Optional[Path] = None):
        """Save timeline to files"""
        if output_dir is None:
            output_dir = PATHS.outputs_dir / "timelines"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save in multiple formats
        self._save_timeline_json(timeline, output_dir)
        self._save_timeline_csv(timeline, output_dir)
        self._save_timeline_html(timeline, output_dir)
        self._save_timeline_visualization(timeline, output_dir)
        
        logger.log_step("Timeline saved", details={
            "output_dir": str(output_dir),
            "events": len(timeline.events),
            "formats": ["JSON", "CSV", "HTML", "Visualization"]
        })
    
    def _save_timeline_json(self, timeline: Timeline, output_dir: Path):
        """Save timeline as JSON"""
        json_file = output_dir / "timeline.json"
        
        timeline_dict = {
            'timeline_id': timeline.timeline_id,
            'events': [
                {
                    'event_id': event.event_id,
                    'description': event.description,
                    'date': event.date_str,
                    'parsed_date': event.parsed_date.isoformat() if event.parsed_date else None,
                    'year': event.year,
                    'decade': event.decade,
                    'century': event.century,
                    'precision': event.date_precision.value,
                    'sources': event.sources,
                    'entities': [
                        {
                            'text': ent.get('text'),
                            'label': ent.get('label'),
                            'type': ent.get('type')
                        }
                        for ent in event.entities[:10]  # Limit
                    ],
                    'context': event.context[:500],  # Limit
                    'confidence': event.confidence,
                    'metadata': event.metadata
                }
                for event in timeline.events
            ],
            'period_range': timeline.period_range,
            'event_density': timeline.event_density,
            'metadata': timeline.metadata
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(timeline_dict, f, indent=2, default=str)
    
    def _save_timeline_csv(self, timeline: Timeline, output_dir: Path):
        """Save timeline as CSV"""
        import pandas as pd
        
        rows = []
        for event in timeline.events:
            rows.append({
                'event_id': event.event_id,
                'date': event.date_str,
                'year': event.year,
                'decade': event.decade,
                'century': event.century,
                'description': event.description,
                'sources': '; '.join(event.sources),
                'entity_count': len(event.entities),
                'entities': '; '.join([e.get('text', '') for e in event.entities[:5]]),
                'confidence': event.confidence,
                'precision': event.date_precision.value
            })
        
        df = pd.DataFrame(rows)
        csv_file = output_dir / "timeline.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
    
    def _save_timeline_html(self, timeline: Timeline, output_dir: Path):
        """Save timeline as interactive HTML"""
        # Escape curly braces in CSS to avoid format() issues
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Historical Timeline</title>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .timeline {{ border-left: 3px solid #333; padding-left: 20px; }}
                .event {{ margin: 20px 0; padding: 15px; border-radius: 5px; background: #f5f5f5; }}
                .event-date {{ font-weight: bold; color: #2c3e50; }}
                .event-desc {{ margin: 5px 0; }}
                .event-meta {{ font-size: 0.9em; color: #666; }}
                .event-entities {{ margin-top: 10px; font-size: 0.85em; }}
                .entity-tag {{ display: inline-block; background: #e0e0e0; padding: 2px 8px; margin: 2px; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <h1>Historical Timeline</h1>
            <p>Generated on {generation_date}</p>
            <p>Total events: {event_count} | Period: {period_start}-{period_end}</p>
            
            <div class="timeline">
                {events_html}
            </div>
        </body>
        </html>
        """
        
        events_html = ""
        for event in timeline.events:
            # Escape HTML special characters in event data
            import html
            date_str = html.escape(str(event.date_str))
            description = html.escape(str(event.description))
            
            entities_html = "".join([
                f'<span class="entity-tag">{html.escape(str(ent.get("text", "") if isinstance(ent, dict) else getattr(ent, "text", "")))} ({html.escape(str(ent.get("label", "") if isinstance(ent, dict) else getattr(ent, "label", "")))})</span>'
                for ent in event.entities[:5]
            ])
            
            sources_str = ", ".join([html.escape(str(s)) for s in event.sources[:3]])
            
            events_html += f"""
            <div class="event">
                <div class="event-date">{date_str}</div>
                <div class="event-desc">{description}</div>
                <div class="event-meta">
                    Sources: {sources_str} | 
                    Confidence: {event.confidence:.2f} | 
                    Precision: {event.date_precision.value}
                </div>
                <div class="event-entities">
                    Entities: {entities_html if entities_html else "None"}
                </div>
            </div>
            """
        
        html_content = html_template.format(
            generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            event_count=len(timeline.events),
            period_start=timeline.period_range[0],
            period_end=timeline.period_range[1],
            events_html=events_html
        )
        
        html_file = output_dir / "timeline.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _save_timeline_visualization(self, timeline: Timeline, output_dir: Path):
        """Save timeline visualization as image"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            # Prepare data
            dates = []
            descriptions = []
            
            for event in timeline.events:
                if event.parsed_date:
                    dates.append(event.parsed_date)
                    descriptions.append(event.description[:50] + "..." if len(event.description) > 50 else event.description)
            
            if dates:
                # Create figure
                fig, ax = plt.subplots(figsize=(15, 10))
                
                # Plot events
                y_positions = list(range(len(dates)))
                ax.scatter(dates, y_positions, alpha=0.6, s=100)
                
                # Add labels
                for i, (date, desc, y) in enumerate(zip(dates, descriptions, y_positions)):
                    ax.text(date, y + 0.1, desc, fontsize=8, ha='left', va='bottom')
                
                # Format x-axis
                ax.xaxis.set_major_locator(mdates.YearLocator(base=10))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                plt.xticks(rotation=45)
                
                # Labels and title
                ax.set_ylabel('Event Index')
                ax.set_title(f'Historical Timeline ({timeline.period_range[0]}-{timeline.period_range[1]})')
                ax.grid(True, alpha=0.3)
                
                # Save figure
                plt.tight_layout()
                viz_file = output_dir / "timeline_visualization.png"
                plt.savefig(viz_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.log_step("Timeline visualization saved", details={
                    "file": str(viz_file)
                })
        
        except ImportError:
            logger.log_step("Matplotlib not available, skipping visualization")

class EventExtractor:
    """Extract event descriptions from text"""
    
    def __init__(self):
        self.event_keywords = [
            'battle', 'war', 'treaty', 'revolution', 'rebellion', 'uprising',
            'conference', 'congress', 'meeting', 'summit', 'alliance',
            'declaration', 'proclamation', 'manifesto', 'charter', 'constitution',
            'discovery', 'invention', 'innovation', 'breakthrough',
            'election', 'inauguration', 'coronation', 'accession',
            'earthquake', 'flood', 'famine', 'epidemic', 'pandemic',
            'assassination', 'murder', 'death', 'birth', 'marriage'
        ]
    
    def extract_event_description(self, context: str, date_str: str) -> str:
        """Extract event description from context"""
        # Find sentences containing the date
        sentences = re.split(r'[.!?]+', context)
        
        for sentence in sentences:
            if date_str in sentence:
                # Clean up the sentence
                sentence = sentence.strip()
                
                # Remove excessive whitespace
                sentence = re.sub(r'\s+', ' ', sentence)
                
                # Check if it contains event keywords
                if any(keyword in sentence.lower() for keyword in self.event_keywords):
                    return sentence
        
        # If no sentence with date and keywords found, return the context
        return context[:200].strip() + "..."

# Continue with Comparative Analyzer and Main Script...