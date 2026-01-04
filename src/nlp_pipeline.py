import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from sentence_transformers import SentenceTransformer
import umap

from src.utils.logger import logger
from config.settings import PATHS
from src.document_loader import DocumentLoader
from src.text_processor import TextProcessor, ProcessedDocument

@dataclass
class PipelineResults:
    """Container for pipeline results"""
    processed_documents: Dict[str, ProcessedDocument]
    features: Dict[str, Any]
    models: Dict[str, Any]
    metadata: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    
    def save(self, output_dir: Path):
        """Save all pipeline results"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save processed documents
        docs_file = output_dir / "processed_documents.json"
        docs_data = {}
        for doc_id, doc in self.processed_documents.items():
            docs_data[doc_id] = {
                'document_id': doc.document_id,
                'cleaned_text': doc.cleaned_text[:5000] + "..." if len(doc.cleaned_text) > 5000 else doc.cleaned_text,
                'sentences': doc.sentences,
                'tokens': doc.tokens[:1000],  # Limit for JSON
                'normalized_tokens': doc.normalized_tokens[:1000],
                'stats': doc.stats,
                'metadata': doc.metadata
            }
        
        with open(docs_file, 'w', encoding='utf-8') as f:
            json.dump(docs_data, f, indent=2, ensure_ascii=False)
        
        # Save features
        features_file = output_dir / "features.pkl"
        with open(features_file, 'wb') as f:
            pickle.dump(self.features, f)
        
        # Save metadata
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        # Save performance metrics
        metrics_file = output_dir / "performance_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.performance_metrics, f, indent=2, default=str)
        
        logger.log_step("Pipeline results saved", details={
            "output_dir": str(output_dir),
            "documents": len(self.processed_documents)
        })
    
    @classmethod
    def load(cls, output_dir: Path):
        """Load pipeline results"""
        docs_file = output_dir / "processed_documents.json"
        features_file = output_dir / "features.pkl"
        metadata_file = output_dir / "metadata.json"
        metrics_file = output_dir / "performance_metrics.json"
        
        with open(docs_file, 'r', encoding='utf-8') as f:
            docs_data = json.load(f)
        
        with open(features_file, 'rb') as f:
            features = pickle.load(f)
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        with open(metrics_file, 'r', encoding='utf-8') as f:
            performance_metrics = json.load(f)
        
        # Recreate processed documents
        processed_documents = {}
        for doc_id, doc_data in docs_data.items():
            doc = ProcessedDocument(
                document_id=doc_data['document_id'],
                original_text="",  # Not saved in JSON
                cleaned_text=doc_data['cleaned_text'],
                tokens=doc_data['tokens'],
                sentences=doc_data['sentences'],
                lemmas=[],  # Not saved
                pos_tags=[],  # Not saved
                normalized_tokens=doc_data['normalized_tokens'],
                stats=doc_data['stats'],
                metadata=doc_data['metadata']
            )
            processed_documents[doc_id] = doc
        
        return cls(
            processed_documents=processed_documents,
            features=features,
            models={},  # Models not saved
            metadata=metadata,
            performance_metrics=performance_metrics
        )

class FeatureExtractor:
    """Extract various features from processed documents"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.sentence_transformer = None
        self.features_cache = {}
        
        logger.log_step("FeatureExtractor initialized")
    
    def extract_all_features(self, processed_docs: Dict[str, ProcessedDocument]) -> Dict[str, Any]:
        """Extract all feature types"""
        with logger.time_step("feature_extraction"):
            
            features = {}
            
            # 1. Basic Text Statistics
            features['basic_stats'] = self._extract_basic_stats(processed_docs)
            
            # 2. TF-IDF Features
            features['tfidf'] = self._extract_tfidf_features(processed_docs)
            
            # 3. Bag-of-Words Features
            features['bow'] = self._extract_bow_features(processed_docs)
            
            # 4. Semantic Embeddings
            features['embeddings'] = self._extract_semantic_embeddings(processed_docs)
            
            # 5. Topic Modeling
            features['topics'] = self._extract_topics(processed_docs)
            
            # 6. Temporal Features
            features['temporal'] = self._extract_temporal_features(processed_docs)
            
            # 7. Readability Scores
            features['readability'] = self._extract_readability_scores(processed_docs)
            
            # 8. Sentiment Analysis
            features['sentiment'] = self._extract_sentiment_features(processed_docs)
            
            # Cache features
            self.features_cache = features
            
            logger.log_step("All features extracted", details={
                "feature_types": list(features.keys()),
                "document_count": len(processed_docs)
            })
            
            return features
    
    def _extract_basic_stats(self, processed_docs: Dict[str, ProcessedDocument]) -> Dict[str, Any]:
        """Extract basic text statistics"""
        stats = {}
        
        for doc_id, doc in processed_docs.items():
            doc_stats = {
                'word_count': len(doc.tokens),
                'sentence_count': len(doc.sentences),
                'avg_sentence_length': len(doc.tokens) / len(doc.sentences) if doc.sentences else 0,
                'avg_word_length': sum(len(word) for word in doc.tokens) / len(doc.tokens) if doc.tokens else 0,
                'vocabulary_size': len(set(doc.tokens)),
                'type_token_ratio': len(set(doc.tokens)) / len(doc.tokens) if doc.tokens else 0,
                'unique_normalized_tokens': len(set(doc.normalized_tokens)),
                'normalized_ttr': len(set(doc.normalized_tokens)) / len(doc.normalized_tokens) if doc.normalized_tokens else 0
            }
            stats[doc_id] = doc_stats
        
        return stats
    
    def _extract_tfidf_features(self, processed_docs: Dict[str, ProcessedDocument]) -> Dict[str, Any]:
        """Extract TF-IDF features"""
        texts = [doc.cleaned_text for doc in processed_docs.values()]
        doc_ids = list(processed_docs.keys())
        
        # Initialize vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            use_idf=True
        )
        
        # Fit and transform
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # Get top features per document
        top_features = {}
        for i, doc_id in enumerate(doc_ids):
            feature_array = tfidf_matrix[i].toarray().flatten()
            top_indices = feature_array.argsort()[-20:][::-1]
            top_features[doc_id] = [
                (feature_names[idx], feature_array[idx])
                for idx in top_indices if feature_array[idx] > 0
            ]
        
        # Document similarity matrix
        similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
        
        return {
            'matrix': tfidf_matrix,
            'feature_names': feature_names,
            'top_features': top_features,
            'similarity_matrix': similarity_matrix,
            'document_ids': doc_ids
        }
    
    def _extract_bow_features(self, processed_docs: Dict[str, ProcessedDocument]) -> Dict[str, Any]:
        """Extract Bag-of-Words features"""
        texts = [doc.cleaned_text for doc in processed_docs.values()]
        doc_ids = list(processed_docs.keys())
        
        # Initialize vectorizer
        self.count_vectorizer = CountVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.98
        )
        
        # Fit and transform
        bow_matrix = self.count_vectorizer.fit_transform(texts)
        feature_names = self.count_vectorizer.get_feature_names_out()
        
        # Get document-term frequency
        doc_term_freq = {}
        for i, doc_id in enumerate(doc_ids):
            feature_array = bow_matrix[i].toarray().flatten()
            nonzero_indices = feature_array.nonzero()[0]
            doc_term_freq[doc_id] = {
                feature_names[idx]: int(feature_array[idx])
                for idx in nonzero_indices
            }
        
        return {
            'matrix': bow_matrix,
            'feature_names': feature_names,
            'doc_term_freq': doc_term_freq,
            'document_ids': doc_ids
        }
    
    def _extract_semantic_embeddings(self, processed_docs: Dict[str, ProcessedDocument]) -> Dict[str, Any]:
        """Extract semantic embeddings using Sentence Transformers"""
        texts = [doc.cleaned_text for doc in processed_docs.values()]
        doc_ids = list(processed_docs.keys())
        
        # Load model (cached)
        if self.sentence_transformer is None:
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate embeddings
        embeddings = self.sentence_transformer.encode(
            texts, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Dimensionality reduction for visualization
        reducer = umap.UMAP(n_components=2, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)
        
        # Clustering
        n_clusters = min(5, len(doc_ids))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # Calculate silhouette score
        if len(set(clusters)) > 1:
            silhouette = silhouette_score(embeddings, clusters)
        else:
            silhouette = 0
        
        # Document similarity based on embeddings
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(embeddings)
        
        return {
            'embeddings': embeddings,
            'reduced_embeddings': reduced_embeddings,
            'clusters': clusters,
            'cluster_centers': kmeans.cluster_centers_,
            'silhouette_score': silhouette,
            'similarity_matrix': similarity_matrix,
            'document_ids': doc_ids
        }
    
    def _extract_topics(self, processed_docs: Dict[str, ProcessedDocument]) -> Dict[str, Any]:
        """Extract topics using multiple methods"""
        texts = [doc.cleaned_text for doc in processed_docs.values()]
        doc_ids = list(processed_docs.keys())
        
        # Prepare corpus for LDA
        tokenized_texts = [doc.normalized_tokens for doc in processed_docs.values()]
        dictionary = corpora.Dictionary(tokenized_texts)
        dictionary.filter_extremes(no_below=2, no_above=0.95)
        corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
        
        # Method 1: LDA (Gensim)
        num_topics = min(10, len(doc_ids))
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        
        # Calculate coherence score
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=tokenized_texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        
        # Get topics
        lda_topics = []
        for topic_id in range(num_topics):
            topic_words = lda_model.show_topic(topic_id, topn=10)
            lda_topics.append({
                'topic_id': topic_id,
                'words': [word for word, _ in topic_words],
                'weights': [weight for _, weight in topic_words]
            })
        
        # Get document-topic distribution
        doc_topics = {}
        for i, doc_id in enumerate(doc_ids):
            doc_bow = corpus[i]
            topic_dist = lda_model.get_document_topics(doc_bow)
            doc_topics[doc_id] = {
                topic_id: prob for topic_id, prob in topic_dist
            }
        
        # Method 2: NMF (sklearn)
        if hasattr(self, 'count_vectorizer') and self.count_vectorizer is not None:
            bow_matrix = self.count_vectorizer.transform(texts)
            # Use alpha_W and alpha_H instead of alpha in newer scikit-learn versions
            try:
                # Try newer API first (scikit-learn >= 1.0)
                nmf = NMF(n_components=num_topics, random_state=42, alpha_W=0.1, alpha_H=0.1, l1_ratio=0.5)
            except TypeError:
                # Fallback for older versions (scikit-learn < 1.0)
                try:
                    nmf = NMF(n_components=num_topics, random_state=42, alpha=0.1, l1_ratio=0.5)
                except TypeError:
                    # Minimal parameters for maximum compatibility
                    nmf = NMF(n_components=num_topics, random_state=42)
            nmf_features = nmf.fit_transform(bow_matrix)
            
            nmf_topics = []
            feature_names = self.count_vectorizer.get_feature_names_out()
            for topic_idx, topic in enumerate(nmf.components_):
                top_indices = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_indices]
                top_weights = [topic[i] for i in top_indices]
                nmf_topics.append({
                    'topic_id': topic_idx,
                    'words': top_words,
                    'weights': top_weights
                })
        else:
            nmf_topics = []
            nmf_features = None
        
        return {
            'lda_model': lda_model,
            'lda_topics': lda_topics,
            'doc_topics': doc_topics,
            'coherence_score': coherence_score,
            'nmf_topics': nmf_topics,
            'nmf_features': nmf_features,
            'num_topics': num_topics,
            'dictionary': dictionary,
            'corpus': corpus
        }
    
    def _extract_temporal_features(self, processed_docs: Dict[str, ProcessedDocument]) -> Dict[str, Any]:
        """Extract temporal features from documents"""
        temporal_features = {}
        
        for doc_id, doc in processed_docs.items():
            # Extract dates from metadata
            year = doc.metadata.get('year')
            period = doc.metadata.get('period')
            
            # Count temporal references in text
            import re
            date_patterns = [
                r'\b(1[0-9]{3}|20[0-9]{2})\b',  # Years
                r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b',
                r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b'  # Full dates
            ]
            
            temporal_refs = []
            for pattern in date_patterns:
                matches = re.findall(pattern, doc.cleaned_text, re.IGNORECASE)
                temporal_refs.extend(matches)
            
            temporal_features[doc_id] = {
                'year': year,
                'period': period,
                'temporal_references': len(temporal_refs),
                'temporal_terms': list(set(temporal_refs))[:20],  # Limit
                'temporal_density': len(temporal_refs) / len(doc.tokens) if doc.tokens else 0
            }
        
        return temporal_features
    
    def _extract_readability_scores(self, processed_docs: Dict[str, ProcessedDocument]) -> Dict[str, Any]:
        """Calculate readability scores"""
        import textstat
        
        readability_scores = {}
        
        for doc_id, doc in processed_docs.items():
            text = doc.cleaned_text[:10000]  # Limit for performance
            
            scores = {
                'flesch_reading_ease': textstat.flesch_reading_ease(text),
                'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
                'gunning_fog': textstat.gunning_fog(text),
                'smog_index': textstat.smog_index(text),
                'automated_readability_index': textstat.automated_readability_index(text),
                'coleman_liau_index': textstat.coleman_liau_index(text),
                'linsear_write_formula': textstat.linsear_write_formula(text),
                'dale_chall_readability_score': textstat.dale_chall_readability_score(text),
                'text_standard': textstat.text_standard(text, float_output=True)
            }
            
            readability_scores[doc_id] = scores
        
        return readability_scores
    
    def _extract_sentiment_features(self, processed_docs: Dict[str, ProcessedDocument]) -> Dict[str, Any]:
        """Extract sentiment features"""
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer
        
        # Download VADER if needed
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
        
        sia = SentimentIntensityAnalyzer()
        sentiment_features = {}
        
        for doc_id, doc in processed_docs.items():
            # Analyze full text
            full_sentiment = sia.polarity_scores(doc.cleaned_text)
            
            # Analyze by sentence
            sentence_sentiments = []
            for sentence in doc.sentences[:100]:  # Limit for performance
                sent_scores = sia.polarity_scores(sentence)
                sentence_sentiments.append(sent_scores)
            
            # Calculate statistics
            if sentence_sentiments:
                compound_scores = [s['compound'] for s in sentence_sentiments]
                pos_scores = [s['pos'] for s in sentence_sentiments]
                neg_scores = [s['neg'] for s in sentence_sentiments]
                
                stats = {
                    'mean_compound': np.mean(compound_scores),
                    'std_compound': np.std(compound_scores),
                    'mean_pos': np.mean(pos_scores),
                    'mean_neg': np.mean(neg_scores),
                    'pos_neg_ratio': np.mean(pos_scores) / (np.mean(neg_scores) + 1e-10),
                    'sentiment_variance': np.var(compound_scores)
                }
            else:
                stats = {}
            
            sentiment_features[doc_id] = {
                'full_text_sentiment': full_sentiment,
                'sentence_sentiments': sentence_sentiments[:10],  # Limit
                'statistics': stats,
                'overall_sentiment': 'positive' if full_sentiment['compound'] > 0.05 
                                   else 'negative' if full_sentiment['compound'] < -0.05 
                                   else 'neutral'
            }
        
        return sentiment_features

class NLPPipeline:
    """Main NLP Pipeline orchestrator"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.document_loader = DocumentLoader()
        self.text_processor = TextProcessor()
        self.feature_extractor = FeatureExtractor()
        self.results = None
        
        logger.log_step("NLPPipeline initialized")
    
    def run(self, document_paths: Optional[List[Path]] = None) -> PipelineResults:
        """Run complete NLP pipeline"""
        with logger.time_step("full_pipeline_execution"):
            
            # Step 1: Load documents
            logger.log_step("Starting pipeline: Document Loading")
            if document_paths:
                loaded_docs = {}
                for path in document_paths:
                    text, metadata = self.document_loader.load_document(path)
                    if text and metadata:
                        loaded_docs[metadata.document_id] = (text, metadata)
            else:
                loaded_docs = self.document_loader.load_all_documents()
            
            if not loaded_docs:
                raise ValueError("No documents loaded successfully")
            
            # Save metadata
            self.document_loader.save_metadata()
            
            # Step 2: Process documents
            logger.log_step("Processing documents")
            processed_docs = self.text_processor.process_batch(loaded_docs)
            
            # Step 3: Extract features
            logger.log_step("Extracting features")
            features = self.feature_extractor.extract_all_features(processed_docs)
            
            # Step 4: Collect performance metrics
            performance_metrics = self._collect_performance_metrics(processed_docs, features)
            
            # Step 5: Create results
            metadata = {
                'pipeline_version': '1.0',
                'execution_timestamp': datetime.now().isoformat(),
                'document_count': len(processed_docs),
                'config': self.config
            }
            
            self.results = PipelineResults(
                processed_documents=processed_docs,
                features=features,
                models={},  # Will be populated by specific tasks
                metadata=metadata,
                performance_metrics=performance_metrics
            )
            
            # Step 6: Save results
            logger.log_step("Saving pipeline results")
            self.results.save(PATHS.processed_dir)
            
            # Step 7: Generate summary report
            self._generate_summary_report()
            
            logger.log_step("Pipeline execution complete", details={
                "documents_processed": len(processed_docs),
                "feature_types": len(features),
                "output_dir": str(PATHS.processed_dir)
            })
            
            return self.results
    
    def _collect_performance_metrics(self, processed_docs: Dict[str, ProcessedDocument], 
                                    features: Dict[str, Any]) -> Dict[str, Any]:
        """Collect performance metrics"""
        metrics = {
            'document_count': len(processed_docs),
            'total_words': sum(len(doc.tokens) for doc in processed_docs.values()),
            'total_sentences': sum(len(doc.sentences) for doc in processed_docs.values()),
            'average_document_length': np.mean([len(doc.tokens) for doc in processed_docs.values()]),
            'vocabulary_size': len(set(
                token for doc in processed_docs.values() for token in doc.tokens
            )),
            'feature_dimensions': {
                'tfidf': features.get('tfidf', {}).get('matrix', None).shape[1] if features.get('tfidf') else 0,
                'bow': features.get('bow', {}).get('matrix', None).shape[1] if features.get('bow') else 0,
                'embeddings': features.get('embeddings', {}).get('embeddings', None).shape[1] if features.get('embeddings') else 0
            },
            'topic_quality': {
                'coherence_score': features.get('topics', {}).get('coherence_score', 0),
                'num_topics': features.get('topics', {}).get('num_topics', 0)
            },
            'clustering_quality': {
                'silhouette_score': features.get('embeddings', {}).get('silhouette_score', 0),
                'num_clusters': len(set(features.get('embeddings', {}).get('clusters', [])))
            }
        }
        
        return metrics
    
    def _generate_summary_report(self):
        """Generate summary report of pipeline execution"""
        report = {
            'pipeline_summary': {
                'timestamp': datetime.now().isoformat(),
                'status': 'completed',
                'documents_processed': len(self.results.processed_documents) if self.results else 0
            },
            'document_statistics': {},
            'feature_statistics': {}
        }
        
        if self.results:
            # Document statistics
            docs = self.results.processed_documents
            report['document_statistics'] = {
                'total_documents': len(docs),
                'total_words': sum(len(doc.tokens) for doc in docs.values()),
                'average_words_per_document': np.mean([len(doc.tokens) for doc in docs.values()]),
                'average_sentences_per_document': np.mean([len(doc.sentences) for doc in docs.values()]),
                'period_distribution': {},
                'source_type_distribution': {}
            }
            
            # Period distribution
            periods = [doc.metadata.get('period', 'Unknown') for doc in docs.values()]
            period_counts = {}
            for period in periods:
                period_counts[period] = period_counts.get(period, 0) + 1
            report['document_statistics']['period_distribution'] = period_counts
            
            # Source type distribution
            source_types = [doc.metadata.get('source_type', 'Unknown') for doc in docs.values()]
            source_counts = {}
            for stype in source_types:
                source_counts[stype] = source_counts.get(stype, 0) + 1
            report['document_statistics']['source_type_distribution'] = source_counts
            
            # Feature statistics
            features = self.results.features
            report['feature_statistics'] = {
                'feature_types': list(features.keys()),
                'tfidf_features': features.get('tfidf', {}).get('matrix', None).shape[1] if features.get('tfidf') else 0,
                'embedding_dimensions': features.get('embeddings', {}).get('embeddings', None).shape[1] if features.get('embeddings') else 0,
                'num_topics': features.get('topics', {}).get('num_topics', 0)
            }
        
        # Save report
        report_file = PATHS.outputs_dir / "pipeline_summary_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.log_step("Summary report generated", details={
            "report_file": str(report_file)
        })
    
    def load_previous_results(self, results_dir: Optional[Path] = None) -> PipelineResults:
        """Load previously saved pipeline results"""
        if results_dir is None:
            results_dir = PATHS.processed_dir
        
        self.results = PipelineResults.load(results_dir)
        logger.log_step("Previous results loaded", details={
            "results_dir": str(results_dir),
            "documents": len(self.results.processed_documents)
        })
        
        return self.results