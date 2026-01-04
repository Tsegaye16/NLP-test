import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None

from src.utils.logger import logger
from config.settings import PATHS

@dataclass
class AnalysisDimension:
    """Analysis dimension container"""
    name: str
    indicators: Dict[str, float]  # indicator_name -> value
    score: float
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DocumentAnalysis:
    """Complete analysis for a document"""
    document_id: str
    dimensions: Dict[str, AnalysisDimension]  # dimension_name -> AnalysisDimension
    overall_score: float
    cluster_label: Optional[int] = None
    similar_documents: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ComparativeAnalysis:
    """Comparative analysis across all documents"""
    analyses: Dict[str, DocumentAnalysis]  # document_id -> DocumentAnalysis
    clusters: Dict[int, List[str]]  # cluster_id -> document_ids
    dimension_correlation: Dict[Tuple[str, str], float]
    temporal_evolution: Dict[int, Dict[str, float]]  # year -> dimension_scores
    metadata: Dict[str, Any] = field(default_factory=dict)

class PoliticalIdeologyAnalyzer:
    """Analyze political ideology in historical documents"""
    
    def __init__(self):
        # Political lexicons (expanded for historical analysis)
        self.lexicons = self._load_political_lexicons()
        
        # Ideology dimensions
        self.dimensions = {
            'liberalism': ['freedom', 'rights', 'democracy', 'equality', 'reform', 'progress'],
            'conservatism': ['tradition', 'order', 'authority', 'stability', 'heritage', 'custom'],
            'socialism': ['collective', 'workers', 'public', 'cooperative', 'redistribution', 'class'],
            'nationalism': ['nation', 'sovereignty', 'identity', 'patriot', 'independence', 'unity'],
            'imperialism': ['empire', 'colony', 'dominion', 'expansion', 'superiority', 'civilize'],
            'republicanism': ['republic', 'virtue', 'civic', 'commonwealth', 'citizen', 'public_good'],
            'monarchism': ['monarchy', 'crown', 'royal', 'loyalty', 'dynasty', 'throne'],
            'federalism': ['federal', 'state_rights', 'decentralization', 'union', 'confederation'],
            'anarchism': ['anarchy', 'stateless', 'voluntary', 'cooperation', 'mutual_aid', 'hierarchy'],
            'feminism': ['women', 'gender', 'suffrage', 'equality', 'patriarchy', 'rights']
        }
        
        # Historical political movements
        self.movements = {
            'enlightenment': ['reason', 'science', 'liberty', 'progress', 'secular', 'toleration'],
            'romanticism': ['emotion', 'nature', 'individual', 'tradition', 'folk', 'national'],
            'industrialization': ['industry', 'machine', 'progress', 'technology', 'urban', 'factory'],
            'colonialism': ['colony', 'empire', 'civilize', 'mission', 'exploration', 'trade'],
            'anti_colonialism': ['independence', 'sovereignty', 'liberation', 'resistance', 'self_rule'],
            'cold_war': ['communism', 'capitalism', 'democracy', 'totalitarian', 'bloc', 'deterrence']
        }
        
        logger.log_step("PoliticalIdeologyAnalyzer initialized", details={
            "dimensions": len(self.dimensions),
            "movements": len(self.movements)
        })
    
    def _load_political_lexicons(self) -> Dict[str, List[str]]:
        """Load political lexicons from files"""
        lexicons = {}
        lexicon_dir = PATHS.gazetteers_dir / "lexicons"
        
        if lexicon_dir.exists():
            for lexicon_file in lexicon_dir.glob("*.txt"):
                category = lexicon_file.stem
                with open(lexicon_file, 'r', encoding='utf-8') as f:
                    terms = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                    lexicons[category] = terms
        
        return lexicons
    
    def analyze_document(self, text: str, normalized_tokens: List[str]) -> Dict[str, Any]:
        """Analyze political ideology in a document"""
        text_lower = text.lower()
        tokens_lower = [token.lower() for token in normalized_tokens]
        
        # Calculate dimension scores
        dimension_scores = {}
        for dimension, terms in self.dimensions.items():
            # Count occurrences
            count = sum(1 for term in terms if term in text_lower)
            # Normalize by document length
            score = count / len(tokens_lower) * 1000 if tokens_lower else 0
            dimension_scores[dimension] = score
        
        # Calculate movement scores
        movement_scores = {}
        for movement, terms in self.movements.items():
            count = sum(1 for term in terms if term in text_lower)
            score = count / len(tokens_lower) * 1000 if tokens_lower else 0
            movement_scores[movement] = score
        
        # Identify dominant ideology
        dominant_ideology = max(dimension_scores.items(), key=lambda x: x[1])[0] if dimension_scores else None
        
        # Calculate ideological complexity (number of ideologies mentioned)
        ideology_count = sum(1 for score in dimension_scores.values() if score > 0)
        
        # Identify political themes
        themes = self._identify_themes(text_lower)
        
        return {
            'dimension_scores': dimension_scores,
            'movement_scores': movement_scores,
            'dominant_ideology': dominant_ideology,
            'ideology_count': ideology_count,
            'themes': themes,
            'metadata': {
                'analyzed_tokens': len(tokens_lower),
                'analysis_timestamp': datetime.now().isoformat()
            }
        }
    
    def _identify_themes(self, text: str) -> List[str]:
        """Identify political themes in text"""
        themes = []
        
        # Economic themes
        economic_terms = {
            'capitalism': ['market', 'capital', 'profit', 'private', 'enterprise', 'competition'],
            'socialism': ['public', 'collective', 'state', 'plan', 'redistribution', 'welfare'],
            'mercantilism': ['trade', 'export', 'import', 'balance', 'protection', 'tariff'],
            'keynesianism': ['demand', 'spending', 'employment', 'intervention', 'stimulus'],
            'monetarism': ['money', 'inflation', 'supply', 'interest', 'bank', 'currency']
        }
        
        for theme, terms in economic_terms.items():
            if any(term in text for term in terms):
                themes.append(theme)
        
        # Social themes
        social_terms = {
            'civil_rights': ['rights', 'equality', 'discrimination', 'segregation', 'freedom'],
            'labor': ['worker', 'union', 'strike', 'wage', 'labor', 'employment'],
            'education': ['school', 'education', 'university', 'literacy', 'knowledge'],
            'healthcare': ['health', 'medicine', 'hospital', 'disease', 'public_health'],
            'immigration': ['immigrant', 'migration', 'border', 'citizen', 'naturalization']
        }
        
        for theme, terms in social_terms.items():
            if any(term in text for term in terms):
                themes.append(theme)
        
        # Foreign policy themes
        foreign_terms = {
            'isolationism': ['isolation', 'neutral', 'non_intervention', 'sovereignty'],
            'interventionism': ['intervene', 'aid', 'assistance', 'alliance', 'treaty'],
            'imperialism': ['empire', 'colony', 'dominion', 'expansion', 'territory'],
            'diplomacy': ['diplomat', 'negotiation', 'treaty', 'accord', 'agreement']
        }
        
        for theme, terms in foreign_terms.items():
            if any(term in text for term in terms):
                themes.append(theme)
        
        return list(set(themes))[:10]  # Limit to 10 themes

class SocioeconomicAnalyzer:
    """Analyze socioeconomic structure in historical documents"""
    
    def __init__(self):
        # Socioeconomic indicators
        self.indicators = {
            'economic_development': [
                'industry', 'manufacturing', 'technology', 'innovation', 'growth',
                'development', 'progress', 'modernization', 'infrastructure'
            ],
            'wealth_distribution': [
                'wealth', 'poverty', 'inequality', 'rich', 'poor', 'class',
                'distribution', 'gap', 'disparity', 'privilege'
            ],
            'labor_conditions': [
                'worker', 'labor', 'wage', 'union', 'strike', 'employment',
                'unemployment', 'condition', 'exploitation', 'rights'
            ],
            'social_welfare': [
                'welfare', 'benefit', 'pension', 'healthcare', 'education',
                'social_security', 'assistance', 'support', 'aid'
            ],
            'trade_commerce': [
                'trade', 'commerce', 'market', 'export', 'import', 'tariff',
                'merchant', 'business', 'transaction', 'exchange'
            ],
            'agriculture': [
                'farm', 'agriculture', 'crop', 'harvest', 'land', 'peasant',
                'rural', 'agrarian', 'plantation', 'cultivation'
            ],
            'urbanization': [
                'city', 'urban', 'town', 'population', 'migration', 'slum',
                'metropolis', 'suburb', 'housing', 'development'
            ],
            'education_literacy': [
                'education', 'school', 'literacy', 'university', 'knowledge',
                'learning', 'instruction', 'academy', 'enlightenment'
            ],
            'health_public_health': [
                'health', 'disease', 'medicine', 'hospital', 'sanitation',
                'epidemic', 'hygiene', 'public_health', 'mortality'
            ],
            'social_mobility': [
                'mobility', 'opportunity', 'advancement', 'merit', 'talent',
                'achievement', 'success', 'progress', 'upward'
            ]
        }
        
        # Historical economic systems
        self.economic_systems = {
            'feudalism': ['feudal', 'lord', 'serf', 'manor', 'vassal', 'fief'],
            'mercantilism': ['mercantile', 'bullion', 'protection', 'balance', 'monopoly'],
            'capitalism': ['capital', 'market', 'profit', 'enterprise', 'competition'],
            'socialism': ['socialist', 'collective', 'public', 'state', 'plan'],
            'communism': ['communist', 'classless', 'revolution', 'proletariat', 'collective']
        }
        
        logger.log_step("SocioeconomicAnalyzer initialized", details={
            "indicators": len(self.indicators),
            "economic_systems": len(self.economic_systems)
        })
    
    def analyze_document(self, text: str, normalized_tokens: List[str]) -> Dict[str, Any]:
        """Analyze socioeconomic structure in a document"""
        text_lower = text.lower()
        tokens_lower = [token.lower() for token in normalized_tokens]
        
        # Calculate indicator scores
        indicator_scores = {}
        for indicator, terms in self.indicators.items():
            # Count occurrences (including partial matches)
            count = 0
            for term in terms:
                if '_' in term:
                    # Handle multi-word terms
                    if term.replace('_', ' ') in text_lower:
                        count += 1
                else:
                    if term in text_lower:
                        count += 1
            
            # Normalize score
            score = count / len(tokens_lower) * 1000 if tokens_lower else 0
            indicator_scores[indicator] = score
        
        # Identify economic system
        system_scores = {}
        for system, terms in self.economic_systems.items():
            count = sum(1 for term in terms if term in text_lower)
            score = count / len(tokens_lower) * 1000 if tokens_lower else 0
            system_scores[system] = score
        
        dominant_system = max(system_scores.items(), key=lambda x: x[1])[0] if system_scores else None
        
        # Calculate development index
        development_indicators = ['economic_development', 'education_literacy', 'health_public_health']
        development_score = np.mean([indicator_scores.get(ind, 0) for ind in development_indicators])
        
        # Calculate inequality index
        inequality_indicators = ['wealth_distribution', 'social_mobility']
        inequality_score = np.mean([indicator_scores.get(ind, 0) for ind in inequality_indicators])
        
        # Identify socioeconomic themes
        themes = self._identify_themes(text_lower)
        
        return {
            'indicator_scores': indicator_scores,
            'system_scores': system_scores,
            'dominant_system': dominant_system,
            'development_index': development_score,
            'inequality_index': inequality_score,
            'themes': themes,
            'metadata': {
                'analyzed_tokens': len(tokens_lower),
                'analysis_timestamp': datetime.now().isoformat()
            }
        }
    
    def _identify_themes(self, text: str) -> List[str]:
        """Identify socioeconomic themes"""
        themes = []
        
        # Production themes
        production_themes = {
            'industrialization': ['factory', 'machine', 'industry', 'manufacturing', 'technology'],
            'agricultural_revolution': ['enclosure', 'crop_rotation', 'mechanization', 'yield'],
            'service_economy': ['service', 'bank', 'insurance', 'trade', 'commerce']
        }
        
        for theme, terms in production_themes.items():
            if any(term in text for term in terms):
                themes.append(theme)
        
        # Social structure themes
        social_themes = {
            'class_struggle': ['class', 'struggle', 'conflict', 'revolution', 'protest'],
            'social_reform': ['reform', 'improvement', 'progress', 'change', 'movement'],
            'urban_crisis': ['slum', 'poverty', 'overcrowding', 'sanitation', 'disease']
        }
        
        for theme, terms in social_themes.items():
            if any(term in text for term in terms):
                themes.append(theme)
        
        return list(set(themes))[:10]

class GovernanceAnalyzer:
    """Analyze governance and power distribution"""
    
    def __init__(self):
        # Governance indicators
        self.indicators = {
            'democratic_participation': [
                'vote', 'election', 'representation', 'parliament', 'congress',
                'assembly', 'democracy', 'participation', 'franchise', 'suffrage'
            ],
            'authoritarian_control': [
                'authority', 'control', 'power', 'dictator', 'autocrat',
                'tyranny', 'repression', 'censorship', 'surveillance', 'force'
            ],
            'legal_framework': [
                'law', 'legal', 'court', 'justice', 'constitution',
                'right', 'legislation', 'statute', 'judiciary', 'trial'
            ],
            'bureaucratic_administration': [
                'bureaucracy', 'administration', 'official', 'department',
                'agency', 'civil_service', 'government', 'ministry', 'office'
            ],
            'military_power': [
                'military', 'army', 'navy', 'general', 'soldier',
                'war', 'defense', 'security', 'force', 'weapon'
            ],
            'religious_influence': [
                'church', 'religion', 'clergy', 'priest', 'bishop',
                'faith', 'doctrine', 'theology', 'spiritual', 'divine'
            ],
            'economic_power': [
                'wealth', 'economic', 'financial', 'bank', 'commerce',
                'industry', 'capital', 'trade', 'market', 'business'
            ],
            'social_institutions': [
                'family', 'community', 'society', 'organization', 'association',
                'institution', 'custom', 'tradition', 'culture', 'norm'
            ]
        }
        
        # Governance systems
        self.governance_systems = {
            'monarchy': ['king', 'queen', 'monarch', 'royal', 'crown', 'throne', 'dynasty'],
            'republic': ['republic', 'president', 'democracy', 'citizen', 'election', 'vote'],
            'aristocracy': ['aristocrat', 'noble', 'lord', 'elite', 'privilege', 'hereditary'],
            'theocracy': ['theocracy', 'religious', 'clerical', 'priestly', 'divine', 'ecclesiastical'],
            'oligarchy': ['oligarchy', 'few', 'elite', 'wealthy', 'powerful', 'privileged'],
            'democracy': ['democracy', 'people', 'popular', 'majority', 'representation', 'vote'],
            'dictatorship': ['dictator', 'autocrat', 'tyrant', 'authoritarian', 'totalitarian', 'despot']
        }
        
        logger.log_step("GovernanceAnalyzer initialized", details={
            "indicators": len(self.indicators),
            "governance_systems": len(self.governance_systems)
        })
    
    def analyze_document(self, text: str, normalized_tokens: List[str]) -> Dict[str, Any]:
        """Analyze governance structure in a document"""
        text_lower = text.lower()
        tokens_lower = [token.lower() for token in normalized_tokens]
        
        # Calculate indicator scores
        indicator_scores = {}
        for indicator, terms in self.indicators.items():
            count = sum(1 for term in terms if term in text_lower)
            score = count / len(tokens_lower) * 1000 if tokens_lower else 0
            indicator_scores[indicator] = score
        
        # Identify governance system
        system_scores = {}
        for system, terms in self.governance_systems.items():
            count = sum(1 for term in terms if term in text_lower)
            score = count / len(tokens_lower) * 1000 if tokens_lower else 0
            system_scores[system] = score
        
        dominant_system = max(system_scores.items(), key=lambda x: x[1])[0] if system_scores else None
        
        # Calculate power distribution index
        power_scores = [
            indicator_scores.get('democratic_participation', 0),
            indicator_scores.get('authoritarian_control', 0),
            indicator_scores.get('military_power', 0),
            indicator_scores.get('economic_power', 0)
        ]
        
        if sum(power_scores) > 0:
            power_distribution = {
                'democratic': power_scores[0] / sum(power_scores),
                'authoritarian': power_scores[1] / sum(power_scores),
                'military': power_scores[2] / sum(power_scores),
                'economic': power_scores[3] / sum(power_scores)
            }
        else:
            power_distribution = {}
        
        # Calculate governance complexity
        governance_complexity = sum(1 for score in indicator_scores.values() if score > 0)
        
        return {
            'indicator_scores': indicator_scores,
            'system_scores': system_scores,
            'dominant_system': dominant_system,
            'power_distribution': power_distribution,
            'governance_complexity': governance_complexity,
            'metadata': {
                'analyzed_tokens': len(tokens_lower),
                'analysis_timestamp': datetime.now().isoformat()
            }
        }

class SocialImpactAnalyzer:
    """Analyze social impact and consequences"""
    
    def __init__(self):
        # Impact indicators
        self.indicators = {
            'social_mobility': [
                'mobility', 'opportunity', 'advancement', 'progress', 'upward',
                'success', 'achievement', 'merit', 'talent', 'ability'
            ],
            'social_cohesion': [
                'unity', 'harmony', 'cooperation', 'community', 'solidarity',
                'togetherness', 'collaboration', 'mutual', 'shared', 'common'
            ],
            'social_conflict': [
                'conflict', 'tension', 'struggle', 'violence', 'protest',
                'revolt', 'revolution', 'strike', 'riot', 'disorder'
            ],
            'cultural_development': [
                'culture', 'art', 'literature', 'music', 'education',
                'knowledge', 'learning', 'enlightenment', 'civilization', 'heritage'
            ],
            'technological_impact': [
                'technology', 'innovation', 'invention', 'machine', 'industrial',
                'progress', 'advancement', 'modernization', 'development', 'change'
            ],
            'environmental_impact': [
                'environment', 'nature', 'resource', 'pollution', 'conservation',
                'sustainability', 'ecological', 'climate', 'land', 'water'
            ],
            'demographic_change': [
                'population', 'migration', 'birth', 'death', 'growth',
                'urbanization', 'movement', 'settlement', 'immigration', 'emigration'
            ],
            'quality_of_life': [
                'living', 'standard', 'quality', 'life', 'wellbeing',
                'happiness', 'health', 'comfort', 'security', 'safety'
            ]
        }
        
        # Historical impact categories
        self.impact_categories = {
            'revolutionary': ['revolution', 'radical', 'transform', 'overthrow', 'change'],
            'reformist': ['reform', 'improve', 'modify', 'amend', 'change'],
            'conservative': ['preserve', 'maintain', 'tradition', 'stability', 'continuity'],
            'progressive': ['progress', 'advance', 'develop', 'modernize', 'improve'],
            'regressive': ['regress', 'decline', 'deteriorate', 'worsen', 'degenerate']
        }
        
        logger.log_step("SocialImpactAnalyzer initialized", details={
            "indicators": len(self.indicators),
            "impact_categories": len(self.impact_categories)
        })
    
    def analyze_document(self, text: str, normalized_tokens: List[str]) -> Dict[str, Any]:
        """Analyze social impact in a document"""
        text_lower = text.lower()
        tokens_lower = [token.lower() for token in normalized_tokens]
        
        # Calculate impact scores
        impact_scores = {}
        for impact, terms in self.indicators.items():
            count = sum(1 for term in terms if term in text_lower)
            score = count / len(tokens_lower) * 1000 if tokens_lower else 0
            impact_scores[impact] = score
        
        # Calculate net impact (positive vs negative)
        positive_impacts = ['social_mobility', 'social_cohesion', 'cultural_development', 
                          'technological_impact', 'quality_of_life']
        negative_impacts = ['social_conflict', 'environmental_impact']
        
        positive_score = sum(impact_scores.get(imp, 0) for imp in positive_impacts)
        negative_score = sum(impact_scores.get(imp, 0) for imp in negative_impacts)
        
        net_impact = positive_score - negative_score
        
        # Identify impact category
        category_scores = {}
        for category, terms in self.impact_categories.items():
            count = sum(1 for term in terms if term in text_lower)
            score = count / len(tokens_lower) * 1000 if tokens_lower else 0
            category_scores[category] = score
        
        dominant_category = max(category_scores.items(), key=lambda x: x[1])[0] if category_scores else None
        
        # Calculate impact magnitude
        impact_magnitude = np.sqrt(sum(score**2 for score in impact_scores.values()))
        
        return {
            'impact_scores': impact_scores,
            'positive_score': positive_score,
            'negative_score': negative_score,
            'net_impact': net_impact,
            'category_scores': category_scores,
            'dominant_category': dominant_category,
            'impact_magnitude': impact_magnitude,
            'metadata': {
                'analyzed_tokens': len(tokens_lower),
                'analysis_timestamp': datetime.now().isoformat()
            }
        }

class ComparativeAnalyzer:
    """Main comparative analyzer coordinating all analyses"""
    
    def __init__(self):
        self.political_analyzer = PoliticalIdeologyAnalyzer()
        self.socioeconomic_analyzer = SocioeconomicAnalyzer()
        self.governance_analyzer = GovernanceAnalyzer()
        self.social_impact_analyzer = SocialImpactAnalyzer()
        
        logger.log_step("ComparativeAnalyzer initialized")
    
    def analyze_documents(self, processed_docs: Dict[str, Any]) -> ComparativeAnalysis:
        """Analyze all documents comparatively"""
        with logger.time_step("comparative_analysis"):
            
            document_analyses = {}
            
            # Analyze each document
            for doc_id, doc_data in processed_docs.items():
                logger.log_step(f"Analyzing document {doc_id}")
                
                text = doc_data.get('cleaned_text', '')
                tokens = doc_data.get('normalized_tokens', [])
                metadata = doc_data.get('metadata', {})
                
                # Run all analyses
                political_analysis = self.political_analyzer.analyze_document(text, tokens)
                socioeconomic_analysis = self.socioeconomic_analyzer.analyze_document(text, tokens)
                governance_analysis = self.governance_analyzer.analyze_document(text, tokens)
                social_impact_analysis = self.social_impact_analyzer.analyze_document(text, tokens)
                
                # Create dimension objects
                dimensions = {
                    'political_ideology': AnalysisDimension(
                        name='political_ideology',
                        indicators=political_analysis['dimension_scores'],
                        score=np.mean(list(political_analysis['dimension_scores'].values())),
                        weight=1.0,
                        metadata={
                            'dominant_ideology': political_analysis['dominant_ideology'],
                            'themes': political_analysis['themes']
                        }
                    ),
                    'socioeconomic_structure': AnalysisDimension(
                        name='socioeconomic_structure',
                        indicators=socioeconomic_analysis['indicator_scores'],
                        score=socioeconomic_analysis['development_index'],
                        weight=1.0,
                        metadata={
                            'dominant_system': socioeconomic_analysis['dominant_system'],
                            'inequality_index': socioeconomic_analysis['inequality_index'],
                            'themes': socioeconomic_analysis['themes']
                        }
                    ),
                    'governance_power': AnalysisDimension(
                        name='governance_power',
                        indicators=governance_analysis['indicator_scores'],
                        score=np.mean(list(governance_analysis['indicator_scores'].values())),
                        weight=1.0,
                        metadata={
                            'dominant_system': governance_analysis['dominant_system'],
                            'power_distribution': governance_analysis['power_distribution'],
                            'complexity': governance_analysis['governance_complexity']
                        }
                    ),
                    'social_impact': AnalysisDimension(
                        name='social_impact',
                        indicators=social_impact_analysis['impact_scores'],
                        score=social_impact_analysis['net_impact'],
                        weight=1.0,
                        metadata={
                            'dominant_category': social_impact_analysis['dominant_category'],
                            'impact_magnitude': social_impact_analysis['impact_magnitude']
                        }
                    )
                }
                
                # Calculate overall score (weighted average)
                overall_score = np.mean([dim.score * dim.weight for dim in dimensions.values()])
                
                # Create document analysis
                doc_analysis = DocumentAnalysis(
                    document_id=doc_id,
                    dimensions=dimensions,
                    overall_score=overall_score,
                    metadata={
                        **metadata,
                        'analysis_timestamp': datetime.now().isoformat()
                    }
                )
                
                document_analyses[doc_id] = doc_analysis
            
            # Perform comparative analysis across documents
            comparative_analysis = self._perform_comparative_analysis(document_analyses)
            
            logger.log_step("Comparative analysis complete", details={
                "documents_analyzed": len(document_analyses),
                "clusters_found": len(comparative_analysis.clusters)
            })
            
            return comparative_analysis
    
    def _perform_comparative_analysis(self, document_analyses: Dict[str, DocumentAnalysis]) -> ComparativeAnalysis:
        """Perform comparative analysis across all documents"""
        
        # Extract feature vectors for clustering
        feature_vectors = []
        doc_ids = []
        
        for doc_id, analysis in document_analyses.items():
            # Create feature vector from all dimension scores
            features = []
            for dim_name, dimension in analysis.dimensions.items():
                features.extend(list(dimension.indicators.values()))
            
            feature_vectors.append(features)
            doc_ids.append(doc_id)
        
        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(feature_vectors)
        
        # Perform clustering
        n_clusters = min(5, len(doc_ids))
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(features_normalized)
            
            # Calculate silhouette score
            silhouette = silhouette_score(features_normalized, cluster_labels)
        else:
            cluster_labels = [0] * len(doc_ids)
            silhouette = 0
        
        # Assign clusters to documents
        clusters = defaultdict(list)
        for doc_id, label in zip(doc_ids, cluster_labels):
            clusters[label].append(doc_id)
            document_analyses[doc_id].cluster_label = label
        
        # Calculate dimension correlations
        dimension_correlation = self._calculate_dimension_correlations(document_analyses)
        
        # Analyze temporal evolution
        temporal_evolution = self._analyze_temporal_evolution(document_analyses)
        
        # Find similar documents
        self._find_similar_documents(document_analyses, features_normalized)
        
        return ComparativeAnalysis(
            analyses=document_analyses,
            clusters=dict(clusters),
            dimension_correlation=dimension_correlation,
            temporal_evolution=temporal_evolution,
            metadata={
                'silhouette_score': silhouette,
                'total_clusters': len(clusters),
                'analysis_timestamp': datetime.now().isoformat()
            }
        )
    
    def _calculate_dimension_correlations(self, document_analyses: Dict[str, DocumentAnalysis]) -> Dict[Tuple[str, str], float]:
        """Calculate correlations between dimensions"""
        # Extract dimension scores
        dimension_scores = defaultdict(list)
        
        for analysis in document_analyses.values():
            for dim_name, dimension in analysis.dimensions.items():
                dimension_scores[dim_name].append(dimension.score)
        
        # Calculate correlations
        correlations = {}
        dimensions = list(dimension_scores.keys())
        
        for i, dim1 in enumerate(dimensions):
            for dim2 in dimensions[i+1:]:
                scores1 = dimension_scores[dim1]
                scores2 = dimension_scores[dim2]
                
                if len(scores1) > 1 and len(scores2) > 1:
                    correlation = np.corrcoef(scores1, scores2)[0, 1]
                    correlations[(dim1, dim2)] = correlation
        
        return correlations
    
    def _analyze_temporal_evolution(self, document_analyses: Dict[str, DocumentAnalysis]) -> Dict[int, Dict[str, float]]:
        """Analyze evolution of dimensions over time"""
        # Group by year
        documents_by_year = defaultdict(list)
        
        for doc_id, analysis in document_analyses.items():
            year = analysis.metadata.get('year')
            if year:
                documents_by_year[year].append(analysis)
        
        # Calculate average scores per year
        temporal_evolution = {}
        
        for year, analyses in documents_by_year.items():
            if len(analyses) > 0:
                year_scores = defaultdict(list)
                
                for analysis in analyses:
                    for dim_name, dimension in analysis.dimensions.items():
                        year_scores[dim_name].append(dimension.score)
                
                # Calculate averages
                avg_scores = {}
                for dim_name, scores in year_scores.items():
                    avg_scores[dim_name] = np.mean(scores)
                
                temporal_evolution[year] = avg_scores
        
        return temporal_evolution
    
    def _find_similar_documents(self, document_analyses: Dict[str, DocumentAnalysis], 
                               feature_vectors: np.ndarray, top_n: int = 3):
        """Find similar documents for each document"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarity_matrix = cosine_similarity(feature_vectors)
        doc_ids = list(document_analyses.keys())
        
        for i, doc_id in enumerate(doc_ids):
            # Get similarity scores for this document
            similarities = similarity_matrix[i]
            
            # Find top N similar documents (excluding self)
            similar_indices = np.argsort(similarities)[::-1][1:top_n+1]
            similar_docs = [doc_ids[idx] for idx in similar_indices]
            
            document_analyses[doc_id].similar_documents = similar_docs
    
    def save_analysis(self, analysis: ComparativeAnalysis, 
                     output_dir: Optional[Path] = None):
        """Save comparative analysis results"""
        if output_dir is None:
            output_dir = PATHS.outputs_dir / "analysis"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save in multiple formats
        self._save_analysis_json(analysis, output_dir)
        self._save_analysis_csv(analysis, output_dir)
        self._save_analysis_visualizations(analysis, output_dir)
        self._save_analysis_report(analysis, output_dir)
        
        logger.log_step("Comparative analysis saved", details={
            "output_dir": str(output_dir),
            "documents_analyzed": len(analysis.analyses),
            "clusters": len(analysis.clusters)
        })
    
    def _save_analysis_json(self, analysis: ComparativeAnalysis, output_dir: Path):
        """Save analysis as JSON"""
        json_file = output_dir / "comparative_analysis.json"
        
        # Convert to serializable format
        analysis_dict = {
            'metadata': analysis.metadata,
            'clusters': {
                str(cluster_id): docs 
                for cluster_id, docs in analysis.clusters.items()
            },
            'dimension_correlation': {
                f"{dim1}_{dim2}": corr 
                for (dim1, dim2), corr in analysis.dimension_correlation.items()
            },
            'temporal_evolution': analysis.temporal_evolution,
            'document_analyses': {}
        }
        
        for doc_id, doc_analysis in analysis.analyses.items():
            analysis_dict['document_analyses'][doc_id] = {
                'document_id': doc_analysis.document_id,
                'overall_score': doc_analysis.overall_score,
                'cluster_label': doc_analysis.cluster_label,
                'similar_documents': doc_analysis.similar_documents,
                'dimensions': {
                    dim_name: {
                        'name': dim.name,
                        'score': dim.score,
                        'indicators': dim.indicators,
                        'metadata': dim.metadata
                    }
                    for dim_name, dim in doc_analysis.dimensions.items()
                },
                'metadata': doc_analysis.metadata
            }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_dict, f, indent=2, default=str)
    
    def _save_analysis_csv(self, analysis: ComparativeAnalysis, output_dir: Path):
        """Save analysis as CSV"""
        import pandas as pd
        
        rows = []
        for doc_id, doc_analysis in analysis.analyses.items():
            row = {
                'document_id': doc_id,
                'overall_score': doc_analysis.overall_score,
                'cluster_label': doc_analysis.cluster_label,
                'similar_documents': '; '.join(doc_analysis.similar_documents),
                'year': doc_analysis.metadata.get('year', 'Unknown'),
                'period': doc_analysis.metadata.get('period', 'Unknown'),
                'source_type': doc_analysis.metadata.get('source_type', 'Unknown')
            }
            
            # Add dimension scores
            for dim_name, dimension in doc_analysis.dimensions.items():
                row[f'{dim_name}_score'] = dimension.score
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_file = output_dir / "analysis_summary.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
    
    def _save_analysis_visualizations(self, analysis: ComparativeAnalysis, output_dir: Path):
        """Save analysis visualizations"""
        if plt is None or sns is None:
            logger.log_step("Visualization libraries not available, skipping visualizations")
            return
        
        try:
            # Set style
            sns.set_style("whitegrid")
            
            # 1. Cluster visualization
            self._visualize_clusters(analysis, output_dir)
            
            # 2. Dimension correlation heatmap
            self._visualize_correlations(analysis, output_dir)
            
            # 3. Temporal evolution chart
            self._visualize_temporal_evolution(analysis, output_dir)
            
            # 4. Dimension radar charts
            self._visualize_dimension_radar(analysis, output_dir)
            
        except Exception as e:
            logger.log_error("VisualizationError", 
                           f"Error creating visualizations: {str(e)}", 
                           details={"exception": str(e)})
    
    def _visualize_clusters(self, analysis: ComparativeAnalysis, output_dir: Path):
        """Visualize document clusters"""
        if plt is None:
            return
        
        # Extract features for PCA
        features = []
        cluster_labels = []
        
        for doc_id, doc_analysis in analysis.analyses.items():
            # Create feature vector
            doc_features = []
            for dim_name, dimension in doc_analysis.dimensions.items():
                doc_features.append(dimension.score)
            
            features.append(doc_features)
            cluster_labels.append(doc_analysis.cluster_label)
        
        # Apply PCA
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        
        unique_clusters = set(cluster_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
        
        for i, cluster_id in enumerate(unique_clusters):
            cluster_points = [features_2d[j] for j, label in enumerate(cluster_labels) if label == cluster_id]
            if cluster_points:
                x_vals = [p[0] for p in cluster_points]
                y_vals = [p[1] for p in cluster_points]
                plt.scatter(x_vals, y_vals, c=[colors[i]], label=f'Cluster {cluster_id}', alpha=0.7, s=100)
        
        plt.title('Document Clusters (PCA Visualization)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.tight_layout()
        
        cluster_file = output_dir / "clusters_visualization.png"
        plt.savefig(cluster_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_correlations(self, analysis: ComparativeAnalysis, output_dir: Path):
        """Visualize dimension correlations"""
        if plt is None or sns is None:
            return
        
        # Create correlation matrix
        dimensions = ['political_ideology', 'socioeconomic_structure', 
                     'governance_power', 'social_impact']
        
        corr_matrix = np.ones((len(dimensions), len(dimensions)))
        
        for (dim1, dim2), corr in analysis.dimension_correlation.items():
            i = dimensions.index(dim1)
            j = dimensions.index(dim2)
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   xticklabels=dimensions, yticklabels=dimensions,
                   center=0, vmin=-1, vmax=1)
        
        plt.title('Dimension Correlations')
        plt.tight_layout()
        
        corr_file = output_dir / "correlation_heatmap.png"
        plt.savefig(corr_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_temporal_evolution(self, analysis: ComparativeAnalysis, output_dir: Path):
        """Visualize temporal evolution of dimensions"""
        if plt is None:
            return
        
        if not analysis.temporal_evolution:
            return
        
        # Prepare data
        years = sorted(analysis.temporal_evolution.keys())
        dimensions = ['political_ideology', 'socioeconomic_structure', 
                     'governance_power', 'social_impact']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, dim in enumerate(dimensions):
            scores = [analysis.temporal_evolution.get(year, {}).get(dim, 0) for year in years]
            
            axes[i].plot(years, scores, marker='o', linewidth=2)
            axes[i].set_title(f'{dim.replace("_", " ").title()} Evolution')
            axes[i].set_xlabel('Year')
            axes[i].set_ylabel('Score')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        evolution_file = output_dir / "temporal_evolution.png"
        plt.savefig(evolution_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_dimension_radar(self, analysis: ComparativeAnalysis, output_dir: Path):
        """Create radar charts for document dimensions"""
        if plt is None:
            return
        
        # Select representative documents from each cluster
        representative_docs = []
        
        for cluster_id, doc_ids in analysis.clusters.items():
            if doc_ids:
                # Select the document with median overall score
                scores = [analysis.analyses[doc_id].overall_score for doc_id in doc_ids]
                median_idx = np.argsort(scores)[len(scores) // 2]
                representative_docs.append(doc_ids[median_idx])
        
        # Create radar chart for each representative document
        for doc_id in representative_docs[:5]:  # Limit to 5 documents
            doc_analysis = analysis.analyses[doc_id]
            
            # Prepare data for radar chart
            dimensions = list(doc_analysis.dimensions.keys())
            scores = [dim.score for dim in doc_analysis.dimensions.values()]
            
            # Close the radar chart
            angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
            scores += scores[:1]
            angles += angles[:1]
            
            # Create radar chart
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            ax.plot(angles, scores, 'o-', linewidth=2)
            ax.fill(angles, scores, alpha=0.25)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([dim.replace('_', '\n').title() for dim in dimensions])
            ax.set_ylim(0, max(scores) * 1.2)
            
            title = f"Document: {doc_id[:20]}...\nCluster: {doc_analysis.cluster_label}"
            plt.title(title, size=14, y=1.1)
            plt.tight_layout()
            
            radar_file = output_dir / f"radar_{doc_id[:10]}.png"
            plt.savefig(radar_file, dpi=300, bbox_inches='tight')
            plt.close()
    
    def _save_analysis_report(self, analysis: ComparativeAnalysis, output_dir: Path):
        """Generate comprehensive analysis report"""
        report = []
        
        report.append("=" * 80)
        report.append("COMPARATIVE HISTORICAL ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Documents Analyzed: {len(analysis.analyses)}")
        report.append(f"Clusters Identified: {len(analysis.clusters)}")
        report.append(f"Silhouette Score: {analysis.metadata.get('silhouette_score', 0):.3f}")
        report.append("")
        
        # Cluster analysis
        report.append("CLUSTER ANALYSIS")
        report.append("-" * 40)
        for cluster_id, doc_ids in analysis.clusters.items():
            report.append(f"\nCluster {cluster_id} ({len(doc_ids)} documents):")
            
            # Get average scores for this cluster
            cluster_scores = defaultdict(list)
            for doc_id in doc_ids:
                doc_analysis = analysis.analyses[doc_id]
                for dim_name, dimension in doc_analysis.dimensions.items():
                    cluster_scores[dim_name].append(dimension.score)
            
            for dim_name, scores in cluster_scores.items():
                avg_score = np.mean(scores)
                report.append(f"  {dim_name.replace('_', ' ').title()}: {avg_score:.2f}")
            
            # Example documents
            report.append(f"  Example documents: {', '.join(doc_ids[:3])}")
        
        # Dimension correlations
        report.append("\nDIMENSION CORRELATIONS")
        report.append("-" * 40)
        for (dim1, dim2), corr in analysis.dimension_correlation.items():
            strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.3 else "Weak"
            direction = "positive" if corr > 0 else "negative"
            report.append(f"{dim1.replace('_', ' ').title()} vs {dim2.replace('_', ' ').title()}: "
                         f"{corr:.3f} ({strength} {direction} correlation)")
        
        # Key findings
        report.append("\nKEY FINDINGS")
        report.append("-" * 40)
        
        # Find documents with highest scores in each dimension
        for dim_name in ['political_ideology', 'socioeconomic_structure', 
                        'governance_power', 'social_impact']:
            top_docs = sorted(
                analysis.analyses.items(),
                key=lambda x: x[1].dimensions[dim_name].score,
                reverse=True
            )[:3]
            
            report.append(f"\nHighest {dim_name.replace('_', ' ').title()}:")
            for doc_id, doc_analysis in top_docs:
                score = doc_analysis.dimensions[dim_name].score
                report.append(f"  - {doc_id[:20]}...: {score:.2f} "
                             f"(Year: {doc_analysis.metadata.get('year', 'Unknown')})")
        
        # Temporal trends
        if analysis.temporal_evolution:
            report.append("\nTEMPORAL TRENDS")
            report.append("-" * 40)
            
            # Calculate change over time for each dimension
            years = sorted(analysis.temporal_evolution.keys())
            if len(years) >= 2:
                first_year = years[0]
                last_year = years[-1]
                
                for dim_name in ['political_ideology', 'socioeconomic_structure', 
                               'governance_power', 'social_impact']:
                    first_score = analysis.temporal_evolution[first_year].get(dim_name, 0)
                    last_score = analysis.temporal_evolution[last_year].get(dim_name, 0)
                    
                    if first_score != 0:
                        change_pct = ((last_score - first_score) / first_score) * 100
                        direction = "increased" if change_pct > 0 else "decreased"
                        report.append(f"{dim_name.replace('_', ' ').title()} {direction} by "
                                     f"{abs(change_pct):.1f}% from {first_year} to {last_year}")
        
        # Save report
        report_file = output_dir / "analysis_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))