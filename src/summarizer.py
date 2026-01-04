import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
import heapq

from src.utils.logger import logger
from config.settings import PATHS

@dataclass
class Summary:
    """Document summary container"""
    document_id: str
    original_sentences: List[str]
    summary_sentences: List[str]
    summary_text: str
    method: str
    compression_ratio: float
    sentence_scores: Dict[str, float]
    metadata: Dict[str, Any]

class HistoricalSummarizer:
    """Advanced summarizer for historical documents without LLMs"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.methods = ['textrank', 'tfidf', 'lexrank', 'lsa', 'kl']
        
        logger.log_step("HistoricalSummarizer initialized", details={
            "model": model_name,
            "methods": self.methods
        })
    
    def summarize_document(self, document_id: str, text: str, 
                          sentences: Optional[List[str]] = None,
                          method: str = 'textrank',
                          compression_ratio: float = 0.3,
                          min_sentences: int = 3,
                          max_sentences: int = 10) -> Summary:
        """Summarize a single document"""
        with logger.time_step("document_summarization", document_id):
            
            if sentences is None:
                sentences = sent_tokenize(text)
            
            if len(sentences) <= min_sentences:
                # Return all sentences if document is short
                summary_sentences = sentences
                method = 'full'
            else:
                # Select method
                if method == 'textrank':
                    summary_sentences = self._textrank_summarize(
                        sentences, compression_ratio, min_sentences, max_sentences
                    )
                elif method == 'tfidf':
                    summary_sentences = self._tfidf_summarize(
                        sentences, compression_ratio, min_sentences, max_sentences
                    )
                elif method == 'lexrank':
                    summary_sentences = self._lexrank_summarize(
                        sentences, compression_ratio, min_sentences, max_sentences
                    )
                elif method == 'lsa':
                    summary_sentences = self._lsa_summarize(
                        sentences, compression_ratio, min_sentences, max_sentences
                    )
                elif method == 'kl':
                    summary_sentences = self._kl_summarize(
                        sentences, compression_ratio, min_sentences, max_sentences
                    )
                else:
                    raise ValueError(f"Unknown method: {method}")
            
            # Calculate sentence scores
            sentence_scores = self._calculate_sentence_scores(sentences, summary_sentences)
            
            # Create summary
            summary = Summary(
                document_id=document_id,
                original_sentences=sentences,
                summary_sentences=summary_sentences,
                summary_text=' '.join(summary_sentences),
                method=method,
                compression_ratio=len(summary_sentences) / len(sentences),
                sentence_scores=sentence_scores,
                metadata={
                    'original_sentence_count': len(sentences),
                    'summary_sentence_count': len(summary_sentences),
                    'compression_ratio_actual': len(summary_sentences) / len(sentences)
                }
            )
            
            logger.log_step("Document summarized", document_id, {
                "method": method,
                "original_sentences": len(sentences),
                "summary_sentences": len(summary_sentences),
                "compression_ratio": f"{summary.compression_ratio:.2%}"
            })
            
            return summary
    
    def _textrank_summarize(self, sentences: List[str], 
                           compression_ratio: float,
                           min_sentences: int,
                           max_sentences: int) -> List[str]:
        """TextRank algorithm for extractive summarization"""
        # Generate sentence embeddings
        embeddings = self.model.encode(sentences, show_progress_bar=False)
        
        # Build similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(similarity_matrix, 0)
        
        # Create graph and apply PageRank
        graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(graph)
        
        # Select top sentences
        n_sentences = max(
            min_sentences,
            min(max_sentences, int(len(sentences) * compression_ratio))
        )
        
        top_indices = heapq.nlargest(
            n_sentences, 
            range(len(scores)), 
            key=scores.get
        )
        
        # Sort by original order
        top_indices.sort()
        
        return [sentences[i] for i in top_indices]
    
    def _tfidf_summarize(self, sentences: List[str],
                        compression_ratio: float,
                        min_sentences: int,
                        max_sentences: int) -> List[str]:
        """TF-IDF based summarization"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Score sentences by sum of TF-IDF scores
        sentence_scores = tfidf_matrix.sum(axis=1).A1
        
        # Select top sentences
        n_sentences = max(
            min_sentences,
            min(max_sentences, int(len(sentences) * compression_ratio))
        )
        
        top_indices = heapq.nlargest(
            n_sentences, 
            range(len(sentence_scores)), 
            key=lambda i: sentence_scores[i]
        )
        
        # Sort by original order
        top_indices.sort()
        
        return [sentences[i] for i in top_indices]
    
    def _lexrank_summarize(self, sentences: List[str],
                          compression_ratio: float,
                          min_sentences: int,
                          max_sentences: int) -> List[str]:
        """LexRank algorithm"""
        # Similar to TextRank but with cosine similarity threshold
        embeddings = self.model.encode(sentences, show_progress_bar=False)
        similarity_matrix = cosine_similarity(embeddings)
        
        # Apply threshold
        similarity_matrix[similarity_matrix < 0.1] = 0
        
        # Normalize
        row_sums = similarity_matrix.sum(axis=1)
        similarity_matrix = similarity_matrix / row_sums[:, np.newaxis]
        
        # Power method for stationary distribution
        scores = np.ones(len(sentences)) / len(sentences)
        for _ in range(50):
            scores = similarity_matrix.T.dot(scores)
        
        # Select top sentences
        n_sentences = max(
            min_sentences,
            min(max_sentences, int(len(sentences) * compression_ratio))
        )
        
        top_indices = heapq.nlargest(
            n_sentences, 
            range(len(scores)), 
            key=lambda i: scores[i]
        )
        
        top_indices.sort()
        
        return [sentences[i] for i in top_indices]
    
    def _lsa_summarize(self, sentences: List[str],
                      compression_ratio: float,
                      min_sentences: int,
                      max_sentences: int) -> List[str]:
        """Latent Semantic Analysis summarization"""
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english'
        )
        
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Apply SVD
        n_components = min(10, len(sentences) - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        lsa_matrix = svd.fit_transform(tfidf_matrix)
        
        # Score sentences by their importance in the concept space
        sentence_scores = np.linalg.norm(lsa_matrix, axis=1)
        
        # Select top sentences
        n_sentences = max(
            min_sentences,
            min(max_sentences, int(len(sentences) * compression_ratio))
        )
        
        top_indices = heapq.nlargest(
            n_sentences, 
            range(len(sentence_scores)), 
            key=lambda i: sentence_scores[i]
        )
        
        top_indices.sort()
        
        return [sentences[i] for i in top_indices]
    
    def _kl_summarize(self, sentences: List[str],
                     compression_ratio: float,
                     min_sentences: int,
                     max_sentences: int) -> List[str]:
        """KL-divergence based summarization"""
        from sklearn.feature_extraction.text import CountVectorizer
        
        # Create document-term matrix
        vectorizer = CountVectorizer(
            max_features=1000,
            stop_words='english'
        )
        
        count_matrix = vectorizer.fit_transform(sentences)
        vocab = vectorizer.get_feature_names_out()
        
        # Calculate document probabilities
        doc_probs = count_matrix.toarray() / count_matrix.sum(axis=1)
        
        # Calculate overall document probabilities
        overall_probs = count_matrix.sum(axis=0).A1 / count_matrix.sum()
        
        # Calculate KL divergence for each sentence
        kl_divergences = []
        for i in range(len(sentences)):
            sentence_probs = doc_probs[i]
            # Avoid division by zero
            mask = sentence_probs > 0
            if mask.any():
                kl = np.sum(sentence_probs[mask] * np.log(
                    sentence_probs[mask] / overall_probs[mask]
                ))
            else:
                kl = 0
            kl_divergences.append(kl)
        
        # Select sentences that minimize KL divergence with overall distribution
        n_sentences = max(
            min_sentences,
            min(max_sentences, int(len(sentences) * compression_ratio))
        )
        
        # Greedy selection
        selected = []
        remaining = set(range(len(sentences)))
        
        while len(selected) < n_sentences and remaining:
            best_idx = -1
            best_score = float('inf')
            
            for idx in remaining:
                if not selected:
                    score = kl_divergences[idx]
                else:
                    # Calculate combined distribution
                    selected_indices = selected + [idx]
                    combined_counts = count_matrix[selected_indices].sum(axis=0).A1
                    combined_probs = combined_counts / combined_counts.sum()
                    
                    # Calculate KL divergence
                    mask = combined_probs > 0
                    if mask.any():
                        score = np.sum(combined_probs[mask] * np.log(
                            combined_probs[mask] / overall_probs[mask]
                        ))
                    else:
                        score = float('inf')
                
                if score < best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx != -1:
                selected.append(best_idx)
                remaining.remove(best_idx)
            else:
                break
        
        selected.sort()
        
        return [sentences[i] for i in selected]
    
    def _calculate_sentence_scores(self, all_sentences: List[str], 
                                  summary_sentences: List[str]) -> Dict[str, float]:
        """Calculate importance scores for sentences"""
        # Simple scoring: 1.0 for summary sentences, 0.0 for others
        scores = {}
        summary_set = set(summary_sentences)
        
        for sentence in all_sentences:
            scores[sentence] = 1.0 if sentence in summary_set else 0.0
        
        return scores
    
    def summarize_batch(self, documents: Dict[str, Tuple[str, List[str]]],
                       method: str = 'textrank',
                       compression_ratio: float = 0.3) -> Dict[str, Summary]:
        """Summarize multiple documents"""
        summaries = {}
        
        logger.log_step("Starting batch summarization", details={
            "document_count": len(documents),
            "method": method,
            "compression_ratio": compression_ratio
        })
        
        for i, (doc_id, (text, sentences)) in enumerate(documents.items(), 1):
            logger.log_step(f"Summarizing document {i}/{len(documents)}", doc_id)
            
            try:
                summary = self.summarize_document(
                    doc_id, text, sentences, method, compression_ratio
                )
                summaries[doc_id] = summary
                
                # Save intermediate results
                if i % 5 == 0:
                    self._save_intermediate_summaries(summaries, i)
                    
            except Exception as e:
                logger.log_error("SummarizationError", 
                               f"Error summarizing document: {str(e)}", 
                               doc_id, e)
        
        logger.log_step("Batch summarization complete", details={
            "successful": len(summaries),
            "failed": len(documents) - len(summaries)
        })
        
        return summaries
    
    def _save_intermediate_summaries(self, summaries: Dict[str, Summary], batch_num: int):
        """Save intermediate summaries"""
        output_dir = PATHS.outputs_dir / "summaries" / "intermediate"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"summaries_batch_{batch_num}.json"
        
        serializable_summaries = {}
        for doc_id, summary in summaries.items():
            serializable_summaries[doc_id] = {
                'document_id': summary.document_id,
                'summary_text': summary.summary_text,
                'method': summary.method,
                'compression_ratio': summary.compression_ratio,
                'metadata': summary.metadata,
                'sentence_count': len(summary.summary_sentences)
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_summaries, f, indent=2, ensure_ascii=False)
        
        logger.log_step("Intermediate summaries saved", details={
            "file": str(output_file),
            "summaries_saved": len(serializable_summaries)
        })
    
    def evaluate_summaries(self, summaries: Dict[str, Summary], 
                          reference_summaries: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Evaluate summary quality"""
        from rouge_score import rouge_scorer
        
        evaluation_results = {}
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        if reference_summaries:
            # Calculate ROUGE scores against reference summaries
            rouge_scores = []
            for doc_id, summary in summaries.items():
                if doc_id in reference_summaries:
                    ref_summary = reference_summaries[doc_id]
                    scores = scorer.score(ref_summary, summary.summary_text)
                    # Convert to dict format for compatibility
                    scores_dict = {
                        'rouge-1': {
                            'f': scores['rouge1'].fmeasure,
                            'p': scores['rouge1'].precision,
                            'r': scores['rouge1'].recall
                        },
                        'rouge-2': {
                            'f': scores['rouge2'].fmeasure,
                            'p': scores['rouge2'].precision,
                            'r': scores['rouge2'].recall
                        },
                        'rouge-l': {
                            'f': scores['rougeL'].fmeasure,
                            'p': scores['rougeL'].precision,
                            'r': scores['rougeL'].recall
                        }
                    }
                    rouge_scores.append(scores_dict)
                    
                    evaluation_results[doc_id] = {
                        'rouge_scores': scores_dict,
                        'summary_length': len(summary.summary_text.split()),
                        'reference_length': len(ref_summary.split())
                    }
            
            # Aggregate scores
            if rouge_scores:
                avg_scores = {
                    'rouge-1': {
                        'f': np.mean([s['rouge-1']['f'] for s in rouge_scores]),
                        'p': np.mean([s['rouge-1']['p'] for s in rouge_scores]),
                        'r': np.mean([s['rouge-1']['r'] for s in rouge_scores])
                    },
                    'rouge-2': {
                        'f': np.mean([s['rouge-2']['f'] for s in rouge_scores]),
                        'p': np.mean([s['rouge-2']['p'] for s in rouge_scores]),
                        'r': np.mean([s['rouge-2']['r'] for s in rouge_scores])
                    },
                    'rouge-l': {
                        'f': np.mean([s['rouge-l']['f'] for s in rouge_scores]),
                        'p': np.mean([s['rouge-l']['p'] for s in rouge_scores]),
                        'r': np.mean([s['rouge-l']['r'] for s in rouge_scores])
                    }
                }
                
                evaluation_results['aggregate'] = {
                    'average_rouge_scores': avg_scores,
                    'document_count': len(rouge_scores)
                }
        
        # Calculate intrinsic metrics
        intrinsic_metrics = {
            'compression_ratios': [s.compression_ratio for s in summaries.values()],
            'summary_lengths': [len(s.summary_text.split()) for s in summaries.values()],
            'methods_used': list(set(s.method for s in summaries.values()))
        }
        
        evaluation_results['intrinsic'] = intrinsic_metrics
        
        # Save evaluation results
        eval_file = PATHS.outputs_dir / "summaries" / "evaluation_results.json"
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        logger.log_step("Summary evaluation complete", details={
            "evaluation_file": str(eval_file),
            "documents_evaluated": len(summaries)
        })
        
        return evaluation_results
    
    def save_summaries(self, summaries: Dict[str, Summary], 
                      output_dir: Optional[Path] = None):
        """Save all summaries to files"""
        if output_dir is None:
            output_dir = PATHS.outputs_dir / "summaries"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual summaries
        for doc_id, summary in summaries.items():
            summary_file = output_dir / f"{doc_id}_summary.json"
            
            summary_data = {
                'document_id': summary.document_id,
                'summary_text': summary.summary_text,
                'summary_sentences': summary.summary_sentences,
                'method': summary.method,
                'compression_ratio': summary.compression_ratio,
                'metadata': summary.metadata
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        # Save consolidated summaries
        consolidated_file = output_dir / "all_summaries.json"
        consolidated_data = {}
        
        for doc_id, summary in summaries.items():
            consolidated_data[doc_id] = {
                'summary': summary.summary_text,
                'method': summary.method,
                'compression': f"{summary.compression_ratio:.2%}",
                'sentences': len(summary.summary_sentences)
            }
        
        with open(consolidated_file, 'w', encoding='utf-8') as f:
            json.dump(consolidated_data, f, indent=2, ensure_ascii=False)
        
        # Save as CSV for easy viewing
        csv_file = output_dir / "summaries.csv"
        csv_data = []
        
        for doc_id, summary in summaries.items():
            csv_data.append({
                'document_id': doc_id,
                'summary': summary.summary_text[:500] + "..." if len(summary.summary_text) > 500 else summary.summary_text,
                'method': summary.method,
                'compression_ratio': summary.compression_ratio,
                'sentence_count': len(summary.summary_sentences),
                'word_count': len(summary.summary_text.split())
            })
        
        import pandas as pd
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        logger.log_step("Summaries saved", details={
            "output_dir": str(output_dir),
            "summary_count": len(summaries),
            "files_created": ["individual JSON", "consolidated JSON", "CSV"]
        })