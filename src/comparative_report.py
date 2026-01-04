"""
Comparative Analytical Report Generator

Generates comprehensive comparative reports addressing:
- Political orientation or ideology
- Socioeconomic structure and policies
- Power distribution and governance style
- Social impact and economic advantage or disadvantage

All comparisons are grounded in evidence extracted via NLP.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json
import numpy as np

from src.utils.logger import logger
from src.analyzer import ComparativeAnalysis, DocumentAnalysis
from config.settings import PATHS


class ComparativeReportGenerator:
    """Generate comprehensive comparative analytical reports"""
    
    def __init__(self):
        logger.log_step("ComparativeReportGenerator initialized")
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {self._convert_numpy_types(k): self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def generate_report(self, comparative_analysis: ComparativeAnalysis,
                       summaries: Dict[str, Any],
                       entities: Dict[str, Any],
                       timeline: Any) -> Dict[str, Any]:
        """Generate comprehensive comparative report"""
        
        report = {
            'metadata': {
                'generated': datetime.now().isoformat(),
                'documents_analyzed': len(comparative_analysis.analyses),
                'report_type': 'comparative_historical_analysis'
            },
            'executive_summary': self._generate_executive_summary(comparative_analysis),
            'political_ideology_analysis': self._analyze_political_ideology(comparative_analysis),
            'socioeconomic_analysis': self._analyze_socioeconomic_structure(comparative_analysis),
            'governance_analysis': self._analyze_governance_power(comparative_analysis),
            'social_impact_analysis': self._analyze_social_impact(comparative_analysis),
            'cross_document_comparison': self._cross_document_comparison(comparative_analysis),
            'temporal_evolution': self._analyze_temporal_evolution(comparative_analysis),
            'evidence_citations': self._extract_evidence_citations(comparative_analysis, summaries, entities),
            'methodology': self._document_methodology(),
            'limitations': self._document_limitations()
        }
        
        return report
    
    def _generate_executive_summary(self, analysis: ComparativeAnalysis) -> Dict[str, Any]:
        """Generate executive summary of findings"""
        summary = {
            'total_documents': len(analysis.analyses),
            'clusters_identified': len(analysis.clusters),
            'time_period_covered': self._get_time_period(analysis),
            'key_findings': []
        }
        
        # Find dominant political ideology
        ideologies = defaultdict(int)
        for doc_analysis in analysis.analyses.values():
            ideology = doc_analysis.dimensions['political_ideology'].metadata.get('dominant_ideology')
            if ideology:
                ideologies[ideology] += 1
        
        if ideologies:
            dominant = max(ideologies.items(), key=lambda x: x[1])
            summary['dominant_political_ideology'] = {
                'ideology': dominant[0],
                'document_count': dominant[1],
                'percentage': (dominant[1] / len(analysis.analyses)) * 100
            }
        
        # Find dominant socioeconomic system
        systems = defaultdict(int)
        for doc_analysis in analysis.analyses.values():
            system = doc_analysis.dimensions['socioeconomic_structure'].metadata.get('dominant_system')
            if system:
                systems[system] += 1
        
        if systems:
            dominant = max(systems.items(), key=lambda x: x[1])
            summary['dominant_socioeconomic_system'] = {
                'system': dominant[0],
                'document_count': dominant[1],
                'percentage': (dominant[1] / len(analysis.analyses)) * 100
            }
        
        # Find dominant governance system
        gov_systems = defaultdict(int)
        for doc_analysis in analysis.analyses.values():
            system = doc_analysis.dimensions['governance_power'].metadata.get('dominant_system')
            if system:
                gov_systems[system] += 1
        
        if gov_systems:
            dominant = max(gov_systems.items(), key=lambda x: x[1])
            summary['dominant_governance_system'] = {
                'system': dominant[0],
                'document_count': dominant[1],
                'percentage': (dominant[1] / len(analysis.analyses)) * 100
            }
        
        return summary
    
    def _analyze_political_ideology(self, analysis: ComparativeAnalysis) -> Dict[str, Any]:
        """Analyze political orientation and ideology across documents"""
        ideology_analysis = {
            'ideology_distribution': defaultdict(int),
            'ideology_scores': defaultdict(list),
            'ideological_themes': defaultdict(list),
            'cluster_ideology_patterns': {},
            'comparative_insights': []
        }
        
        # Collect ideology data
        for doc_id, doc_analysis in analysis.analyses.items():
            ideology_dim = doc_analysis.dimensions['political_ideology']
            dominant = ideology_dim.metadata.get('dominant_ideology')
            
            if dominant:
                ideology_analysis['ideology_distribution'][dominant] += 1
                ideology_analysis['ideology_scores'][dominant].append(ideology_dim.score)
                ideology_analysis['ideological_themes'][dominant].extend(
                    ideology_dim.metadata.get('themes', [])
                )
        
        # Analyze cluster patterns
        for cluster_id, doc_ids in analysis.clusters.items():
            cluster_ideologies = []
            for doc_id in doc_ids:
                if doc_id in analysis.analyses:
                    ideology = analysis.analyses[doc_id].dimensions['political_ideology']
                    dominant = ideology.metadata.get('dominant_ideology')
                    if dominant:
                        cluster_ideologies.append(dominant)
            
            if cluster_ideologies:
                most_common = max(set(cluster_ideologies), key=cluster_ideologies.count)
                ideology_analysis['cluster_ideology_patterns'][cluster_id] = {
                    'dominant_ideology': most_common,
                    'ideology_count': len(set(cluster_ideologies)),
                    'documents': len(doc_ids)
                }
        
        # Generate comparative insights
        if len(ideology_analysis['ideology_distribution']) > 1:
            ideology_analysis['comparative_insights'].append(
                f"Documents exhibit {len(ideology_analysis['ideology_distribution'])} distinct "
                f"political ideologies, with {max(ideology_analysis['ideology_distribution'].items(), key=lambda x: x[1])[0]} "
                f"being the most prevalent."
            )
        
        # Calculate average scores
        ideology_analysis['average_scores'] = {
            ideology: sum(scores) / len(scores) if scores else 0
            for ideology, scores in ideology_analysis['ideology_scores'].items()
        }
        
        return ideology_analysis
    
    def _analyze_socioeconomic_structure(self, analysis: ComparativeAnalysis) -> Dict[str, Any]:
        """Analyze socioeconomic structure and policies"""
        socioeconomic_analysis = {
            'economic_systems': defaultdict(int),
            'development_indices': [],
            'inequality_indices': [],
            'indicator_scores': defaultdict(list),
            'comparative_insights': []
        }
        
        # Collect socioeconomic data
        for doc_id, doc_analysis in analysis.analyses.items():
            se_dim = doc_analysis.dimensions['socioeconomic_structure']
            system = se_dim.metadata.get('dominant_system')
            
            if system:
                socioeconomic_analysis['economic_systems'][system] += 1
            
            # Collect indicator scores
            for indicator, score in se_dim.indicators.items():
                socioeconomic_analysis['indicator_scores'][indicator].append(score)
            
            # Collect indices
            dev_index = se_dim.metadata.get('development_index', 0)
            ineq_index = se_dim.metadata.get('inequality_index', 0)
            
            if dev_index > 0:
                socioeconomic_analysis['development_indices'].append(dev_index)
            if ineq_index > 0:
                socioeconomic_analysis['inequality_indices'].append(ineq_index)
        
        # Calculate averages
        socioeconomic_analysis['average_development_index'] = (
            sum(socioeconomic_analysis['development_indices']) / 
            len(socioeconomic_analysis['development_indices'])
            if socioeconomic_analysis['development_indices'] else 0
        )
        
        socioeconomic_analysis['average_inequality_index'] = (
            sum(socioeconomic_analysis['inequality_indices']) / 
            len(socioeconomic_analysis['inequality_indices'])
            if socioeconomic_analysis['inequality_indices'] else 0
        )
        
        # Average indicator scores
        socioeconomic_analysis['average_indicator_scores'] = {
            indicator: sum(scores) / len(scores) if scores else 0
            for indicator, scores in socioeconomic_analysis['indicator_scores'].items()
        }
        
        # Generate insights
        if socioeconomic_analysis['economic_systems']:
            dominant_system = max(socioeconomic_analysis['economic_systems'].items(), key=lambda x: x[1])
            socioeconomic_analysis['comparative_insights'].append(
                f"The dominant economic system across documents is {dominant_system[0]}, "
                f"appearing in {dominant_system[1]} documents."
            )
        
        return socioeconomic_analysis
    
    def _analyze_governance_power(self, analysis: ComparativeAnalysis) -> Dict[str, Any]:
        """Analyze power distribution and governance style"""
        governance_analysis = {
            'governance_systems': defaultdict(int),
            'power_distribution': defaultdict(list),
            'governance_complexity': [],
            'comparative_insights': []
        }
        
        # Collect governance data
        for doc_id, doc_analysis in analysis.analyses.items():
            gov_dim = doc_analysis.dimensions['governance_power']
            system = gov_dim.metadata.get('dominant_system')
            
            if system:
                governance_analysis['governance_systems'][system] += 1
            
            # Collect power distribution
            power_dist = gov_dim.metadata.get('power_distribution', {})
            for power_type, value in power_dist.items():
                governance_analysis['power_distribution'][power_type].append(value)
            
            # Collect complexity
            complexity = gov_dim.metadata.get('complexity', 0)
            if complexity > 0:
                governance_analysis['governance_complexity'].append(complexity)
        
        # Calculate averages
        governance_analysis['average_power_distribution'] = {
            power_type: sum(values) / len(values) if values else 0
            for power_type, values in governance_analysis['power_distribution'].items()
        }
        
        governance_analysis['average_complexity'] = (
            sum(governance_analysis['governance_complexity']) / 
            len(governance_analysis['governance_complexity'])
            if governance_analysis['governance_complexity'] else 0
        )
        
        # Generate insights
        if governance_analysis['governance_systems']:
            dominant = max(governance_analysis['governance_systems'].items(), key=lambda x: x[1])
            governance_analysis['comparative_insights'].append(
                f"The most common governance system is {dominant[0]}, "
                f"found in {dominant[1]} documents."
            )
        
        # Analyze power distribution patterns
        if governance_analysis['average_power_distribution']:
            max_power = max(governance_analysis['average_power_distribution'].items(), key=lambda x: x[1])
            governance_analysis['comparative_insights'].append(
                f"On average, {max_power[0]} power has the highest distribution "
                f"({max_power[1]:.2%}) across documents."
            )
        
        return governance_analysis
    
    def _analyze_social_impact(self, analysis: ComparativeAnalysis) -> Dict[str, Any]:
        """Analyze social impact and economic advantage/disadvantage"""
        impact_analysis = {
            'impact_categories': defaultdict(int),
            'net_impact_scores': [],
            'positive_impacts': [],
            'negative_impacts': [],
            'impact_magnitude': [],
            'comparative_insights': []
        }
        
        # Collect impact data
        for doc_id, doc_analysis in analysis.analyses.items():
            impact_dim = doc_analysis.dimensions['social_impact']
            category = impact_dim.metadata.get('dominant_category')
            
            if category:
                impact_analysis['impact_categories'][category] += 1
            
            # Collect scores
            net_impact = impact_dim.score
            if net_impact != 0:
                impact_analysis['net_impact_scores'].append(net_impact)
            
            # Extract from indicators
            for indicator, score in impact_dim.indicators.items():
                if indicator in ['social_mobility', 'social_cohesion', 'cultural_development', 
                               'technological_impact', 'quality_of_life']:
                    if score > 0:
                        impact_analysis['positive_impacts'].append(score)
                elif indicator in ['social_conflict', 'environmental_impact']:
                    if score > 0:
                        impact_analysis['negative_impacts'].append(score)
            
            magnitude = impact_dim.metadata.get('impact_magnitude', 0)
            if magnitude > 0:
                impact_analysis['impact_magnitude'].append(magnitude)
        
        # Calculate averages
        impact_analysis['average_net_impact'] = (
            sum(impact_analysis['net_impact_scores']) / 
            len(impact_analysis['net_impact_scores'])
            if impact_analysis['net_impact_scores'] else 0
        )
        
        impact_analysis['average_positive_impact'] = (
            sum(impact_analysis['positive_impacts']) / 
            len(impact_analysis['positive_impacts'])
            if impact_analysis['positive_impacts'] else 0
        )
        
        impact_analysis['average_negative_impact'] = (
            sum(impact_analysis['negative_impacts']) / 
            len(impact_analysis['negative_impacts'])
            if impact_analysis['negative_impacts'] else 0
        )
        
        impact_analysis['average_magnitude'] = (
            sum(impact_analysis['impact_magnitude']) / 
            len(impact_analysis['impact_magnitude'])
            if impact_analysis['impact_magnitude'] else 0
        )
        
        # Generate insights
        if impact_analysis['average_net_impact'] > 0:
            impact_analysis['comparative_insights'].append(
                "On average, documents describe positive social impacts."
            )
        elif impact_analysis['average_net_impact'] < 0:
            impact_analysis['comparative_insights'].append(
                "On average, documents describe negative social impacts."
            )
        else:
            impact_analysis['comparative_insights'].append(
                "Social impacts are balanced across documents."
            )
        
        return impact_analysis
    
    def _cross_document_comparison(self, analysis: ComparativeAnalysis) -> Dict[str, Any]:
        """Compare documents across all dimensions"""
        comparison = {
            'similarity_clusters': {},
            'dimension_correlations': {},
            'document_pairs': []
        }
        
        # Analyze clusters
        for cluster_id, doc_ids in analysis.clusters.items():
            if len(doc_ids) > 1:
                cluster_docs = [analysis.analyses[doc_id] for doc_id in doc_ids if doc_id in analysis.analyses]
                
                # Calculate cluster characteristics
                cluster_characteristics = {
                    'size': len(cluster_docs),
                    'average_overall_score': sum(d.overall_score for d in cluster_docs) / len(cluster_docs),
                    'common_ideology': self._get_most_common(
                        [d.dimensions['political_ideology'].metadata.get('dominant_ideology') 
                         for d in cluster_docs]
                    ),
                    'common_system': self._get_most_common(
                        [d.dimensions['socioeconomic_structure'].metadata.get('dominant_system')
                         for d in cluster_docs]
                    )
                }
                
                comparison['similarity_clusters'][cluster_id] = cluster_characteristics
        
        # Document dimension correlations
        comparison['dimension_correlations'] = {
            f"{dim1}_{dim2}": corr
            for (dim1, dim2), corr in analysis.dimension_correlation.items()
        }
        
        return comparison
    
    def _analyze_temporal_evolution(self, analysis: ComparativeAnalysis) -> Dict[str, Any]:
        """Analyze evolution of dimensions over time"""
        # Convert numpy types to native Python types for JSON serialization
        temporal_data = {}
        for year, scores in analysis.temporal_evolution.items():
            # Convert year key to native Python int
            year_key = int(year) if isinstance(year, (np.integer, np.int32, np.int64)) else year
            # Convert scores dict values to native Python types
            scores_dict = {}
            for dim_name, score in scores.items():
                if isinstance(score, (np.floating, np.float32, np.float64)):
                    scores_dict[dim_name] = float(score)
                else:
                    scores_dict[dim_name] = score
            temporal_data[year_key] = scores_dict
        
        evolution = {
            'temporal_data': temporal_data,
            'trends': {},
            'insights': []
        }
        
        if not temporal_data:
            return evolution
        
        years = sorted(temporal_data.keys())
        
        if len(years) >= 2:
            # Calculate trends for each dimension
            for dim_name in ['political_ideology', 'socioeconomic_structure', 
                           'governance_power', 'social_impact']:
                scores = [temporal_data[year].get(dim_name, 0) for year in years]
                
                if scores and any(s > 0 for s in scores):
                    first_score = scores[0]
                    last_score = scores[-1]
                    
                    if first_score != 0:
                        change_pct = ((last_score - first_score) / first_score) * 100
                        evolution['trends'][dim_name] = {
                            'first_year': int(years[0]) if isinstance(years[0], (np.integer, np.int32, np.int64)) else years[0],
                            'last_year': int(years[-1]) if isinstance(years[-1], (np.integer, np.int32, np.int64)) else years[-1],
                            'first_score': float(first_score) if isinstance(first_score, (np.floating, np.float32, np.float64)) else first_score,
                            'last_score': float(last_score) if isinstance(last_score, (np.floating, np.float32, np.float64)) else last_score,
                            'change_percentage': float(change_pct) if isinstance(change_pct, (np.floating, np.float32, np.float64)) else change_pct,
                            'direction': 'increasing' if change_pct > 0 else 'decreasing'
                        }
                        
                        evolution['insights'].append(
                            f"{dim_name.replace('_', ' ').title()} {evolution['trends'][dim_name]['direction']} "
                            f"by {abs(change_pct):.1f}% from {years[0]} to {years[-1]}."
                        )
        
        return evolution
    
    def _extract_evidence_citations(self, analysis: ComparativeAnalysis,
                                   summaries: Dict[str, Any],
                                   entities: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """Extract textual evidence for analytical claims"""
        citations = {
            'political_ideology': [],
            'socioeconomic_structure': [],
            'governance_power': [],
            'social_impact': []
        }
        
        # Extract evidence from summaries and entities
        for doc_id, doc_analysis in analysis.analyses.items():
            # Get summary if available
            summary_text = ""
            if doc_id in summaries:
                if hasattr(summaries[doc_id], 'summary_text'):
                    summary_text = summaries[doc_id].summary_text
                elif isinstance(summaries[doc_id], dict):
                    summary_text = summaries[doc_id].get('summary_text', '')
            
            # Extract key entities
            key_entities = []
            if doc_id in entities:
                if hasattr(entities[doc_id], 'entities'):
                    key_entities = [e.text for e in entities[doc_id].entities[:10]]
                elif isinstance(entities[doc_id], dict):
                    ents = entities[doc_id].get('entities', [])
                    key_entities = [e.get('text', '') for e in ents[:10] if isinstance(e, dict)]
            
            # Create citations for each dimension
            for dim_name in citations.keys():
                dim = doc_analysis.dimensions.get(dim_name)
                if dim:
                    citation = {
                        'document_id': doc_id,
                        'dimension': dim_name,
                        'score': dim.score,
                        'key_indicators': list(dim.indicators.keys())[:5],
                        'summary_excerpt': summary_text[:200] if summary_text else '',
                        'key_entities': key_entities[:5],
                        'metadata': doc_analysis.metadata
                    }
                    citations[dim_name].append(citation)
        
        return citations
    
    def _document_methodology(self) -> Dict[str, Any]:
        """Document the methodology used"""
        return {
            'nlp_pipeline': {
                'text_cleaning': 'Unicode normalization, entity protection, OCR error correction',
                'tokenization': 'spaCy-based with historical text support',
                'normalization': 'Lowercasing, lemmatization, historical spelling correction',
                'feature_extraction': 'TF-IDF, BOW, semantic embeddings, topic modeling'
            },
            'analysis_methods': {
                'political_ideology': 'Lexicon-based scoring with dimension analysis',
                'socioeconomic_structure': 'Indicator-based scoring with system classification',
                'governance_power': 'Power distribution analysis with system classification',
                'social_impact': 'Impact indicator scoring with net impact calculation'
            },
            'comparative_methods': {
                'clustering': 'K-means on feature vectors',
                'similarity': 'Cosine similarity on embeddings',
                'correlation': 'Pearson correlation on dimension scores'
            },
            'evidence_extraction': {
                'summaries': 'Extractive summarization (TextRank, TF-IDF, LexRank)',
                'entities': 'Multi-method NER (spaCy, gazetteers, rule-based)',
                'citations': 'Text excerpts from summaries and key entities'
            }
        }
    
    def _document_limitations(self) -> Dict[str, Any]:
        """Document limitations and trade-offs"""
        return {
            'methodological_limitations': [
                'Lexicon-based analysis may miss nuanced ideological expressions',
                'Historical spelling variations may not all be captured',
                'OCR quality affects text extraction accuracy',
                'Entity recognition depends on gazetteer completeness'
            ],
            'data_limitations': [
                'Analysis limited to provided documents',
                'Temporal coverage depends on document selection',
                'Language primarily English',
                'Document quality varies'
            ],
            'analytical_limitations': [
                'Scores are relative, not absolute',
                'Comparisons assume comparable document types',
                'Temporal evolution limited by document distribution',
                'Clustering results depend on feature selection'
            ],
            'trade_offs': [
                'Speed vs accuracy in feature extraction',
                'Preservation vs normalization in text processing',
                'Feature richness vs computational efficiency',
                'Generalization vs specificity in analysis'
            ]
        }
    
    def _get_time_period(self, analysis: ComparativeAnalysis) -> Dict[str, Any]:
        """Extract time period covered"""
        years = []
        for doc_analysis in analysis.analyses.values():
            year = doc_analysis.metadata.get('year')
            if year:
                years.append(year)
        
        if years:
            return {
                'start_year': min(years),
                'end_year': max(years),
                'span_years': max(years) - min(years),
                'document_count': len(years)
            }
        else:
            return {
                'start_year': None,
                'end_year': None,
                'span_years': 0,
                'document_count': 0
            }
    
    def _get_most_common(self, items: List[Any]) -> Any:
        """Get most common item from list"""
        if not items:
            return None
        items = [item for item in items if item is not None]
        if not items:
            return None
        return max(set(items), key=items.count)
    
    def save_report(self, report: Dict[str, Any], output_dir: Optional[Path] = None):
        """Save comparative report in multiple formats"""
        if output_dir is None:
            output_dir = PATHS.outputs_dir / "reports" / "comparative_analysis"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        json_file = output_dir / "comparative_report.json"
        # Convert numpy types to native Python types before JSON serialization
        report_serializable = self._convert_numpy_types(report)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report_serializable, f, indent=2, default=str, ensure_ascii=False)
        
        # Save as formatted text
        text_file = output_dir / "comparative_report.txt"
        self._save_text_report(report, text_file)
        
        logger.log_step("Comparative report saved", details={
            "output_dir": str(output_dir),
            "formats": ["JSON", "TXT"]
        })
    
    def _save_text_report(self, report: Dict[str, Any], output_file: Path):
        """Save report as formatted text"""
        lines = []
        
        lines.append("=" * 100)
        lines.append("COMPARATIVE HISTORICAL ANALYSIS REPORT")
        lines.append("=" * 100)
        lines.append(f"Generated: {report['metadata']['generated']}")
        lines.append(f"Documents Analyzed: {report['metadata']['documents_analyzed']}")
        lines.append("")
        
        # Executive Summary
        lines.append("EXECUTIVE SUMMARY")
        lines.append("-" * 100)
        exec_sum = report['executive_summary']
        lines.append(f"Total Documents: {exec_sum['total_documents']}")
        lines.append(f"Clusters Identified: {exec_sum['clusters_identified']}")
        
        if 'dominant_political_ideology' in exec_sum:
            lines.append(f"\nDominant Political Ideology: {exec_sum['dominant_political_ideology']['ideology']} "
                        f"({exec_sum['dominant_political_ideology']['percentage']:.1f}% of documents)")
        
        if 'dominant_socioeconomic_system' in exec_sum:
            lines.append(f"Dominant Socioeconomic System: {exec_sum['dominant_socioeconomic_system']['system']} "
                        f"({exec_sum['dominant_socioeconomic_system']['percentage']:.1f}% of documents)")
        
        lines.append("")
        
        # Political Ideology Analysis
        lines.append("POLITICAL IDEOLOGY ANALYSIS")
        lines.append("-" * 100)
        pol_analysis = report['political_ideology_analysis']
        lines.append("Ideology Distribution:")
        for ideology, count in pol_analysis['ideology_distribution'].items():
            lines.append(f"  - {ideology}: {count} documents")
        
        if pol_analysis['comparative_insights']:
            lines.append("\nKey Insights:")
            for insight in pol_analysis['comparative_insights']:
                lines.append(f"  • {insight}")
        lines.append("")
        
        # Socioeconomic Analysis
        lines.append("SOCIOECONOMIC STRUCTURE ANALYSIS")
        lines.append("-" * 100)
        se_analysis = report['socioeconomic_analysis']
        lines.append(f"Average Development Index: {se_analysis['average_development_index']:.2f}")
        lines.append(f"Average Inequality Index: {se_analysis['average_inequality_index']:.2f}")
        
        if se_analysis['comparative_insights']:
            lines.append("\nKey Insights:")
            for insight in se_analysis['comparative_insights']:
                lines.append(f"  • {insight}")
        lines.append("")
        
        # Governance Analysis
        lines.append("GOVERNANCE AND POWER DISTRIBUTION ANALYSIS")
        lines.append("-" * 100)
        gov_analysis = report['governance_analysis']
        lines.append("Average Power Distribution:")
        for power_type, value in gov_analysis['average_power_distribution'].items():
            lines.append(f"  - {power_type}: {value:.2%}")
        
        if gov_analysis['comparative_insights']:
            lines.append("\nKey Insights:")
            for insight in gov_analysis['comparative_insights']:
                lines.append(f"  • {insight}")
        lines.append("")
        
        # Social Impact Analysis
        lines.append("SOCIAL IMPACT ANALYSIS")
        lines.append("-" * 100)
        impact_analysis = report['social_impact_analysis']
        lines.append(f"Average Net Impact: {impact_analysis['average_net_impact']:.2f}")
        lines.append(f"Average Positive Impact: {impact_analysis['average_positive_impact']:.2f}")
        lines.append(f"Average Negative Impact: {impact_analysis['average_negative_impact']:.2f}")
        
        if impact_analysis['comparative_insights']:
            lines.append("\nKey Insights:")
            for insight in impact_analysis['comparative_insights']:
                lines.append(f"  • {insight}")
        lines.append("")
        
        # Temporal Evolution
        if report['temporal_evolution']['trends']:
            lines.append("TEMPORAL EVOLUTION")
            lines.append("-" * 100)
            for dim_name, trend in report['temporal_evolution']['trends'].items():
                lines.append(f"{dim_name.replace('_', ' ').title()}: "
                           f"{trend['direction']} by {abs(trend['change_percentage']):.1f}% "
                           f"({trend['first_year']} to {trend['last_year']})")
            lines.append("")
        
        # Methodology
        lines.append("METHODOLOGY")
        lines.append("-" * 100)
        lines.append("This analysis uses an NLP pipeline with:")
        lines.append("  - Text cleaning and normalization")
        lines.append("  - Tokenization and sentence segmentation")
        lines.append("  - Feature extraction (TF-IDF, embeddings, topic modeling)")
        lines.append("  - Lexicon-based dimension analysis")
        lines.append("  - Statistical comparison and clustering")
        lines.append("")
        
        # Limitations
        lines.append("LIMITATIONS")
        lines.append("-" * 100)
        limitations = report['limitations']
        lines.append("Methodological Limitations:")
        for lim in limitations['methodological_limitations']:
            lines.append(f"  • {lim}")
        lines.append("\nAnalytical Limitations:")
        for lim in limitations['analytical_limitations']:
            lines.append(f"  • {lim}")
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

