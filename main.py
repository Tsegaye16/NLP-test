#!/usr/bin/env python3
"""
Main execution script for Historical Text Analysis System
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.logger import logger
from config.settings import PATHS
from src.nlp_pipeline import NLPPipeline
from src.summarizer import HistoricalSummarizer
from src.ner_extractor import HistoricalNER
from src.timeline_builder import TimelineBuilder
from src.analyzer import ComparativeAnalyzer
from src.visualization.timeline_plot import TimelineVisualizer

class HistoricalAnalysisSystem:
    """Main system orchestrator"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the historical analysis system"""
        self.config = self._load_config(config_path)
        self.pipeline = NLPPipeline()
        self.summarizer = HistoricalSummarizer()
        self.ner = HistoricalNER()
        self.timeline_builder = TimelineBuilder()
        self.analyzer = ComparativeAnalyzer()
        self.visualizer = TimelineVisualizer()
        
        logger.log_step("Historical Analysis System initialized")
    
    def _load_config(self, config_path: Optional[Path]) -> dict:
        """Load configuration file"""
        default_config = {
            'pipeline': {
                'mode': 'accurate',
                'max_documents': None,
                'skip_existing': True
            },
            'summarization': {
                'method': 'textrank',
                'compression_ratio': 0.3
            },
            'ner': {
                'use_gazetteers': True,
                'extract_dates': True
            },
            'timeline': {
                'merge_similar_events': True,
                'similarity_threshold': 0.7
            },
            'analysis': {
                'perform_clustering': True,
                'create_visualizations': True
            }
        }
        
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def run_full_analysis(self, document_paths: Optional[List[Path]] = None,
                         output_dir: Optional[Path] = None):
        """Run complete historical text analysis"""
        
        logger.log_step("Starting full historical analysis")
        
        # Step 1: Run NLP Pipeline
        logger.log_step("Step 1: Running NLP Pipeline")
        pipeline_results = self.pipeline.run(document_paths)
        
        # Prepare documents for further processing
        documents_for_summary = {}
        documents_for_ner = {}
        
        for doc_id, doc in pipeline_results.processed_documents.items():
            documents_for_summary[doc_id] = (
                doc.cleaned_text,
                doc.sentences
            )
            documents_for_ner[doc_id] = doc.cleaned_text
        
        # Step 2: Document Summarization
        logger.log_step("Step 2: Document Summarization")
        summaries = self.summarizer.summarize_batch(
            documents_for_summary,
            method=self.config['summarization']['method'],
            compression_ratio=self.config['summarization']['compression_ratio']
        )
        self.summarizer.save_summaries(summaries)
        
        # Step 3: Named Entity Recognition
        logger.log_step("Step 3: Named Entity Recognition")
        entities = self.ner.extract_batch(documents_for_ner)
        self.ner.save_entities(entities)
        
        # Step 4: Timeline Construction
        logger.log_step("Step 4: Timeline Construction")
        
        # Prepare documents for timeline
        timeline_documents = {}
        for doc_id, doc in pipeline_results.processed_documents.items():
            timeline_documents[doc_id] = {
                'text': doc.cleaned_text,
                'metadata': doc.metadata
            }
        
        # Prepare entities for timeline
        timeline_entities = {}
        for doc_id, doc_entities in entities.items():
            timeline_entities[doc_id] = [
                {'text': ent.text, 'label': ent.label}
                for ent in doc_entities.entities
            ]
        
        # Build timeline
        timeline = self.timeline_builder.build_timeline(
            timeline_documents,
            timeline_entities
        )
        
        # Merge similar events if configured
        if self.config['timeline']['merge_similar_events']:
            timeline = self.timeline_builder.merge_similar_events(
                timeline,
                similarity_threshold=self.config['timeline']['similarity_threshold']
            )
        
        self.timeline_builder.save_timeline(timeline)
        
        # Step 5: Comparative Analysis
        logger.log_step("Step 5: Comparative Analysis")
        
        # Prepare documents for analysis
        analysis_documents = {}
        for doc_id, doc in pipeline_results.processed_documents.items():
            analysis_documents[doc_id] = {
                'cleaned_text': doc.cleaned_text,
                'normalized_tokens': doc.normalized_tokens,
                'metadata': doc.metadata
            }
        
        comparative_analysis = self.analyzer.analyze_documents(analysis_documents)
        self.analyzer.save_analysis(comparative_analysis)
        
        # Step 6: Create Visualizations
        if self.config['analysis']['create_visualizations']:
            logger.log_step("Step 6: Creating Visualizations")
            self._create_visualizations(
                pipeline_results,
                summaries,
                entities,
                timeline,
                comparative_analysis
            )
        
        # Step 7: Generate Comparative Analytical Report
        logger.log_step("Step 7: Generating Comparative Analytical Report")
        from src.comparative_report import ComparativeReportGenerator
        report_generator = ComparativeReportGenerator()
        
        comparative_report = report_generator.generate_report(
            comparative_analysis,
            summaries,
            entities,
            timeline
        )
        report_generator.save_report(comparative_report)
        
        # Step 8: Generate Final Report
        logger.log_step("Step 8: Generating Final Report")
        self._generate_final_report(
            pipeline_results,
            summaries,
            entities,
            timeline,
            comparative_analysis
        )
        
        logger.log_step("Historical analysis completed successfully")
        
        return {
            'pipeline_results': pipeline_results,
            'summaries': summaries,
            'entities': entities,
            'timeline': timeline,
            'comparative_analysis': comparative_analysis
        }
    
    def _create_visualizations(self, pipeline_results, summaries, 
                             entities, timeline, comparative_analysis):
        """Create all visualizations"""
        viz_dir = PATHS.outputs_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Create timeline visualization
        # Convert Timeline object to dictionary format expected by visualizer
        timeline_dict = {
            'events': [
                {
                    'date': event.date_str,
                    'year': event.year,
                    'description': event.description,
                    'sources': event.sources,
                    'entities': event.entities,  # Already List[Dict[str, Any]]
                    'confidence': event.confidence,
                    'precision': event.date_precision.value if hasattr(event.date_precision, 'value') else str(event.date_precision)
                }
                for event in timeline.events
            ],
            'period_range': timeline.period_range,
            'metadata': timeline.metadata
        }
        
        self.visualizer.create_interactive_timeline(timeline_dict, viz_dir)
        self.visualizer.create_density_timeline(timeline_dict, viz_dir)
        
        # Create entity network visualization
        from src.visualization.timeline_plot import EntityNetworkVisualizer
        entity_viz = EntityNetworkVisualizer()
        
        # Prepare entity data
        entity_data = {'entities_by_document': {}}
        for doc_id, doc_entities in entities.items():
            entity_data['entities_by_document'][doc_id] = [
                {'text': ent.text, 'label': ent.label}
                for ent in doc_entities.entities[:50]  # Limit for visualization
            ]
        
        entity_viz.create_entity_network(entity_data, viz_dir)
        
        # Create analysis dashboard
        from src.visualization.timeline_plot import AnalysisDashboard
        dashboard = AnalysisDashboard()
        
        # Prepare analysis data
        analysis_data = {
            'dimension_scores': {},
            'cluster_distribution': {},
            'temporal_evolution': comparative_analysis.temporal_evolution,
            'correlation_matrix': []
        }
        
        # Extract dimension scores
        if comparative_analysis.analyses:
            sample_doc = next(iter(comparative_analysis.analyses.values()))
            for dim_name, dimension in sample_doc.dimensions.items():
                analysis_data['dimension_scores'][dim_name] = dimension.score
        
        # Extract cluster distribution
        for cluster_id, doc_ids in comparative_analysis.clusters.items():
            analysis_data['cluster_distribution'][f'Cluster {cluster_id}'] = len(doc_ids)
        
        dashboard.create_dashboard(analysis_data, viz_dir)
    
    def _generate_final_report(self, pipeline_results, summaries, 
                             entities, timeline, comparative_analysis):
        """Generate comprehensive final report"""
        report_dir = PATHS.outputs_dir / "reports"
        report_dir.mkdir(exist_ok=True)
        
        report = []
        report.append("=" * 100)
        report.append("HISTORICAL TEXT ANALYSIS - FINAL REPORT")
        report.append("=" * 100)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Pipeline Statistics
        report.append("1. PIPELINE STATISTICS")
        report.append("-" * 50)
        report.append(f"Documents Processed: {len(pipeline_results.processed_documents)}")
        
        if pipeline_results.performance_metrics:
            metrics = pipeline_results.performance_metrics
            report.append(f"Total Words: {metrics.get('total_words', 0):,}")
            report.append(f"Average Document Length: {metrics.get('average_document_length', 0):.0f} words")
            report.append(f"Vocabulary Size: {metrics.get('vocabulary_size', 0):,}")
        
        # Summarization Statistics
        report.append("\n2. SUMMARIZATION")
        report.append("-" * 50)
        report.append(f"Documents Summarized: {len(summaries)}")
        
        if summaries:
            avg_compression = sum(s.compression_ratio for s in summaries.values()) / len(summaries)
            report.append(f"Average Compression Ratio: {avg_compression:.1%}")
            
            methods_used = set(s.method for s in summaries.values())
            report.append(f"Methods Used: {', '.join(methods_used)}")
        
        # Entity Recognition Statistics
        report.append("\n3. NAMED ENTITY RECOGNITION")
        report.append("-" * 50)
        
        total_entities = 0
        entity_counts = {}
        
        for doc_entities in entities.values():
            total_entities += len(doc_entities.entities)
            for label, count in doc_entities.entity_counts.items():
                entity_counts[label] = entity_counts.get(label, 0) + count
        
        report.append(f"Total Entities Extracted: {total_entities:,}")
        report.append("Entity Distribution:")
        for label, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
            report.append(f"  - {label}: {count:,}")
        
        # Timeline Statistics
        report.append("\n4. TIMELINE")
        report.append("-" * 50)
        
        if hasattr(timeline, 'events'):
            report.append(f"Events in Timeline: {len(timeline.events)}")
            report.append(f"Time Period: {timeline.period_range[0]} - {timeline.period_range[1]}")
            
            # Count events by century
            century_counts = {}
            for event in timeline.events:
                if event.century:
                    century_counts[event.century] = century_counts.get(event.century, 0) + 1
            
            if century_counts:
                report.append("Events by Century:")
                for century, count in sorted(century_counts.items()):
                    report.append(f"  - {century}th century: {count}")
        
        # Comparative Analysis Statistics
        report.append("\n5. COMPARATIVE ANALYSIS")
        report.append("-" * 50)
        
        if hasattr(comparative_analysis, 'clusters'):
            report.append(f"Clusters Identified: {len(comparative_analysis.clusters)}")
            
            for cluster_id, doc_ids in comparative_analysis.clusters.items():
                report.append(f"\nCluster {cluster_id} ({len(doc_ids)} documents):")
                
                # Get cluster characteristics
                cluster_docs = [comparative_analysis.analyses[doc_id] for doc_id in doc_ids[:3]]
                for doc_analysis in cluster_docs:
                    report.append(f"  - {doc_analysis.document_id[:20]}... "
                                 f"(Score: {doc_analysis.overall_score:.2f})")
        
        # Key Insights
        report.append("\n6. KEY INSIGHTS")
        report.append("-" * 50)
        
        # Add insights based on analysis
        if comparative_analysis.analyses:
            # Find most frequent political ideology
            ideologies = {}
            for analysis in comparative_analysis.analyses.values():
                ideology = analysis.dimensions['political_ideology'].metadata.get('dominant_ideology')
                if ideology:
                    ideologies[ideology] = ideologies.get(ideology, 0) + 1
            
            if ideologies:
                dominant_ideology = max(ideologies.items(), key=lambda x: x[1])
                report.append(f"Dominant Political Ideology: {dominant_ideology[0]} "
                             f"({dominant_ideology[1]} documents)")
            
            # Find temporal trends
            if comparative_analysis.temporal_evolution:
                years = sorted(comparative_analysis.temporal_evolution.keys())
                if len(years) >= 2:
                    first = years[0]
                    last = years[-1]
                    report.append(f"\nTemporal Analysis ({first} - {last}):")
                    
                    for dim in ['political_ideology', 'socioeconomic_structure']:
                        first_score = comparative_analysis.temporal_evolution[first].get(dim, 0)
                        last_score = comparative_analysis.temporal_evolution[last].get(dim, 0)
                        
                        if first_score != 0:
                            change = ((last_score - first_score) / first_score) * 100
                            direction = "increased" if change > 0 else "decreased"
                            report.append(f"  - {dim.replace('_', ' ').title()} {direction} by "
                                         f"{abs(change):.1f}%")
        
        # Recommendations
        report.append("\n7. RECOMMENDATIONS FOR FURTHER ANALYSIS")
        report.append("-" * 50)
        report.append("1. Investigate clusters with similar political ideologies")
        report.append("2. Analyze entity networks for historical relationships")
        report.append("3. Compare socioeconomic structures across different periods")
        report.append("4. Examine governance changes over time")
        report.append("5. Study impact of major historical events on social structures")
        
        # Save report
        report_file = report_dir / "final_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        logger.log_step("Final report generated", details={
            "report_file": str(report_file)
        })
        
        # Also save as JSON for programmatic access
        json_report = {
            'metadata': {
                'generated': datetime.now().isoformat(),
                'documents_processed': len(pipeline_results.processed_documents),
                'entities_extracted': total_entities,
                'events_in_timeline': len(timeline.events) if hasattr(timeline, 'events') else 0,
                'clusters_identified': len(comparative_analysis.clusters) if hasattr(comparative_analysis, 'clusters') else 0
            },
            'key_findings': {
                'dominant_ideology': dominant_ideology[0] if 'dominant_ideology' in locals() else None,
                'time_period': timeline.period_range if hasattr(timeline, 'period_range') else None,
                'entity_distribution': entity_counts
            }
        }
        
        json_file = report_dir / "final_report.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2, default=str)

def main():
    """Main entry point with command-line interface"""
    parser = argparse.ArgumentParser(
        description="Historical Text Analysis System"
    )
    
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=PATHS.raw_documents,
        help='Directory containing historical documents'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=PATHS.outputs_dir,
        help='Directory for output files'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--documents',
        type=Path,
        nargs='+',
        help='Specific document files to process'
    )
    
    parser.add_argument(
        '--skip-pipeline',
        action='store_true',
        help='Skip NLP pipeline if results exist'
    )
    
    parser.add_argument(
        '--only-summarize',
        action='store_true',
        help='Only run summarization'
    )
    
    parser.add_argument(
        '--only-ner',
        action='store_true',
        help='Only run named entity recognition'
    )
    
    parser.add_argument(
        '--only-timeline',
        action='store_true',
        help='Only build timeline'
    )
    
    parser.add_argument(
        '--only-analysis',
        action='store_true',
        help='Only run comparative analysis'
    )
    
    parser.add_argument(
        '--max-documents',
        type=int,
        help='Maximum number of documents to process'
    )
    
    args = parser.parse_args()
    
    # Update paths if specified
    if args.input_dir:
        PATHS.raw_documents = args.input_dir
    
    if args.output_dir:
        PATHS.outputs_dir = args.output_dir
        PATHS.create_directories()
    
    # Initialize system
    system = HistoricalAnalysisSystem(args.config)
    
    # Get document paths
    if args.documents:
        document_paths = args.documents
    else:
        # Discover documents in input directory
        from src.document_loader import DocumentLoader
        loader = DocumentLoader()
        discovered = loader.discover_documents()
        document_paths = discovered[:args.max_documents] if args.max_documents else discovered
    
    if not document_paths:
        logger.log_error("NoDocuments", "No documents found to process")
        sys.exit(1)
    
    logger.log_step("Starting analysis", details={
        "documents": len(document_paths),
        "input_dir": str(PATHS.raw_documents),
        "output_dir": str(PATHS.outputs_dir)
    })
    
    try:
        # Run analysis based on arguments
        if args.only_summarize:
            # Load existing pipeline results
            from src.nlp_pipeline import PipelineResults
            pipeline_results = PipelineResults.load(PATHS.processed_dir)
            
            # Prepare for summarization
            documents_for_summary = {}
            for doc_id, doc in pipeline_results.processed_documents.items():
                documents_for_summary[doc_id] = (doc.cleaned_text, doc.sentences)
            
            # Run summarization
            summaries = system.summarizer.summarize_batch(documents_for_summary)
            system.summarizer.save_summaries(summaries)
            
        elif args.only_ner:
            # Load existing pipeline results
            from src.nlp_pipeline import PipelineResults
            pipeline_results = PipelineResults.load(PATHS.processed_dir)
            
            # Prepare for NER
            documents_for_ner = {
                doc_id: doc.cleaned_text
                for doc_id, doc in pipeline_results.processed_documents.items()
            }
            
            # Run NER
            entities = system.ner.extract_batch(documents_for_ner)
            system.ner.save_entities(entities)
            
        elif args.only_timeline:
            # Load existing pipeline results and entities
            from src.nlp_pipeline import PipelineResults
            pipeline_results = PipelineResults.load(PATHS.processed_dir)
            
            # Load entities
            import pickle
            entities_file = PATHS.outputs_dir / "entities" / "entities.pkl"
            if entities_file.exists():
                with open(entities_file, 'rb') as f:
                    entities = pickle.load(f)
            else:
                logger.log_error("MissingData", "Entity file not found")
                sys.exit(1)
            
            # Build timeline
            timeline_documents = {}
            for doc_id, doc in pipeline_results.processed_documents.items():
                timeline_documents[doc_id] = {
                    'text': doc.cleaned_text,
                    'metadata': doc.metadata
                }
            
            timeline_entities = {}
            for doc_id, doc_entities in entities.items():
                timeline_entities[doc_id] = [
                    {'text': ent.text, 'label': ent.label}
                    for ent in doc_entities.entities[:100]
                ]
            
            timeline = system.timeline_builder.build_timeline(
                timeline_documents,
                timeline_entities
            )
            system.timeline_builder.save_timeline(timeline)
            
        elif args.only_analysis:
            # Load existing pipeline results
            from src.nlp_pipeline import PipelineResults
            pipeline_results = PipelineResults.load(PATHS.processed_dir)
            
            # Run comparative analysis
            analysis_documents = {}
            for doc_id, doc in pipeline_results.processed_documents.items():
                analysis_documents[doc_id] = {
                    'cleaned_text': doc.cleaned_text,
                    'normalized_tokens': doc.normalized_tokens,
                    'metadata': doc.metadata
                }
            
            comparative_analysis = system.analyzer.analyze_documents(analysis_documents)
            system.analyzer.save_analysis(comparative_analysis)
            
        else:
            # Run full analysis
            results = system.run_full_analysis(document_paths)
            logger.log_step("Full analysis completed successfully")
        
        # Log performance summary
        logger.log_performance_summary()
        
        # Save logs
        logger.save_logs_to_file()
        
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Output files saved to: {PATHS.outputs_dir}")
        print(f"Logs saved to: {PATHS.logs_dir}")
        print(f"Reports saved to: {PATHS.outputs_dir / 'reports'}")
        print("="*60)
        
    except Exception as e:
        logger.log_error("MainExecution", f"Error during execution: {str(e)}", exception=e)
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()