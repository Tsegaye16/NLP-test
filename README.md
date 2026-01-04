# Historical Text Analysis Using a Structured NLP Pipeline

**Backend Developer (Python – Search, NLP & Data Processing) - Practical Exam**

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Problem Statement](#problem-statement)
- [Solution Architecture](#solution-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Task 1: NLP Pipeline Construction](#task-1-nlp-pipeline-construction)
- [Task 2: Document Summarization](#task-2-document-summarization)
- [Task 3: Named Entity Recognition](#task-3-named-entity-recognition)
- [Task 4: Timeline Construction](#task-4-timeline-construction)
- [Task 5: Comparative Analysis](#task-5-comparative-analysis)
- [Final Deliverables](#final-deliverables)
- [Project Structure](#project-structure)
- [Methodology Documentation](#methodology-documentation)
- [Limitations and Trade-offs](#limitations-and-trade-offs)
- [Performance Metrics](#performance-metrics)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project implements a comprehensive NLP pipeline for analyzing historical documents to extract meaningful insights about political systems, socioeconomic structures, governance patterns, and historical events. The system processes historical texts through a structured pipeline without relying on end-to-end LLM summarization, ensuring all analytical claims are grounded in extracted textual evidence.

**Key Features:**

- Complete NLP pipeline with explicit, documented processing steps
- Multiple extractive summarization methods (TextRank, TF-IDF, LexRank, LSA, KL-divergence)
- Multi-method Named Entity Recognition (spaCy, gazetteers, rule-based)
- Chronological timeline construction with date parsing and event extraction
- Comparative analysis across four key dimensions
- Comprehensive visualizations and reports

---

## Dataset

The system processes **12 historical documents** covering multiple political periods and socioeconomic contexts in Ethiopian history:

### Primary Data Sources (PDS)

1. **Ancient/Classical Period** (`ancient_classical_period.pdf`)
2. **Medieval Period** (`medieval_period.pdf`)
3. **Early Modern Gondarine Period** (`early_modern_gondarine_period.pdf`)
4. **Haile Selassie Era** (`haile_selassie_summary.pdf`)
5. **Derg Regime** (`derg_regime_summary.pdf`)
6. **EPRDF Era** (`eprdf_era_summary.pdf`)
7. **Prosperity Party Period** (`prosperity_party_summary.pdf`)
8. **Ethiopia History Research - Ancient** (`ethiopia_history_research_ancient.pdf`)
9. **Ethiopia History Research - Medieval** (`ethiopia_history_research_medieval.pdf`)
10. **Ethiopia History Research - Early Modern** (`ethiopia_history_research_early_modern.pdf`)
11. **Ethiopia History Research - Modern** (`ethiopia_history_research_modern.pdf`)
12. **Modern Ethiopia Period** (`modern_ethiopia_period.pdf`)

**Document Types:**

- Historical narratives
- Policy texts
- Political statements
- Socioeconomic descriptions

**Time Coverage:** ~50 CE to 2023 CE

**Location:** All documents are stored in `data/raw_documents/`

---

## Problem Statement

The challenge requires designing and executing an end-to-end NLP pipeline to transform historical documents into structured analytical outputs that enable:

1. **Summarization** - Generate document summaries from pipeline outputs
2. **Entity Extraction** - Identify and classify named entities
3. **Timeline Construction** - Create chronological timelines of historical events
4. **Comparative Analysis** - Compare documents across multiple analytical dimensions

**Constraints:**

- Each NLP step must be explicitly implemented and documented
- End-to-end LLM summarization is strictly prohibited
- All analytical claims must be supported by textual evidence
- Solution must demonstrate methodological correctness, interpretability, and cost awareness

---

## Solution Architecture

The system follows a modular, pipeline-based architecture:

```
Historical Documents
    ↓
[NLP Pipeline]
    ├── Document Loading
    ├── Text Cleaning
    ├── Tokenization
    ├── Normalization
    ├── Sentence Segmentation
    └── Feature Extraction
    ↓
[Processing Modules]
    ├── Summarization (Extractive Methods)
    ├── Named Entity Recognition
    ├── Timeline Construction
    └── Comparative Analysis
    ↓
[Output Generation]
    ├── Summaries
    ├── Entities
    ├── Timelines
    ├── Comparative Reports
    └── Visualizations
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Setup Instructions

1. **Clone the repository:**

```bash
git clone https://github.com/Tsegaye16/NLP-test.git
cd NLP-test
```

2. **Create a virtual environment (recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Download spaCy language model:**

```bash
python -m spacy download en_core_web_lg
```

5. **Download NLTK data (if needed):**

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon')"
```

### Configuration

The system uses configuration files in the `config/` directory:

- `config/settings.py` - Main configuration settings
- `config/logging_config.yaml` - Logging configuration

Default settings can be modified in `config/settings.py` if needed.

---

## Usage

### Basic Usage

Run the complete analysis pipeline:

```bash
python main.py
```

This will:

1. Load all documents from `data/raw_documents/`
2. Run the complete NLP pipeline
3. Generate summaries, extract entities, build timelines
4. Perform comparative analysis
5. Generate all outputs in `data/outputs/`

### Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --config PATH       Path to custom configuration file
  --documents PATH    Path to specific documents directory
  --output PATH       Path to output directory
  --skip-pipeline     Skip NLP pipeline (use cached results)
  --skip-visualizations  Skip visualization generation
```

### Running Individual Components

The system can also be used programmatically:

```python
from src.nlp_pipeline import NLPPipeline
from src.summarizer import HistoricalSummarizer
from src.ner_extractor import HistoricalNER
from src.timeline_builder import TimelineBuilder
from src.analyzer import ComparativeAnalyzer

# Initialize components
pipeline = NLPPipeline()
summarizer = HistoricalSummarizer()
ner = HistoricalNER()
timeline_builder = TimelineBuilder()
analyzer = ComparativeAnalyzer()

# Run pipeline
results = pipeline.run()

# Generate summaries
summaries = summarizer.summarize_batch(documents, method='textrank')

# Extract entities
entities = ner.extract_batch(documents)

# Build timeline
timeline = timeline_builder.build_timeline(documents, entities)

# Perform analysis
analysis = analyzer.analyze_documents(documents)
```

---

## Task 1: NLP Pipeline Construction

The NLP pipeline is the foundation of the entire system. Each step is explicitly implemented, documented, and designed to support downstream tasks.

### Pipeline Steps

#### 1. Text Collection and Organization

**Implementation:** `src/document_loader.py`

**Purpose:** Load and organize documents from various formats (PDF, TXT, DOCX, HTML)

**Design Choices:**

- Multi-format support using PyMuPDF, pdfplumber, python-docx, BeautifulSoup
- Document ID generation using hash-based approach for uniqueness
- Metadata extraction (title, author, creation date, page count)
- Error handling for corrupted or unsupported files

**Why Necessary:** Historical documents come in various formats and must be standardized before processing.

**Downstream Impact:** Provides clean text input and metadata for all subsequent steps.

---

#### 2. Text Cleaning

**Implementation:** `src/text_processor.py` → `TextCleaner` class

**Purpose:** Remove noise, normalize formatting, preserve important information

**Design Choices:**

- Unicode normalization (NFKC form)
- Entity protection (dates, numbers, names preserved)
- OCR error correction patterns
- Removal of excessive whitespace, special characters
- Preservation of sentence boundaries

**Cleaning Steps:**

1. Unicode normalization
2. Line break normalization
3. Whitespace cleanup
4. Special character handling
5. Entity protection (dates, numbers, names)

**Why Necessary:** Raw text contains formatting artifacts, encoding issues, and noise that interfere with NLP tasks.

**Downstream Impact:** Clean text improves tokenization accuracy, reduces vocabulary size, and enhances feature quality.

---

#### 3. Tokenization

**Implementation:** `src/text_processor.py` → `TextTokenizer` class

**Purpose:** Split text into meaningful units (tokens and sentences)

**Design Choices:**

- spaCy-based tokenization with custom rules
- Historical text support (archaic abbreviations, titles)
- Sentence segmentation using spaCy's sentence boundary detection
- Preservation of token position information

**Why Necessary:** Tokenization is fundamental for all NLP tasks - it breaks continuous text into discrete units.

**Downstream Impact:** Enables normalization, feature extraction, and entity recognition at token level.

---

#### 4. Normalization

**Implementation:** `src/text_processor.py` → `TextNormalizer` class

**Purpose:** Standardize tokens to canonical forms

**Design Choices:**

- Lowercasing (case-insensitive analysis)
- Lemmatization using spaCy (reduces inflectional variations)
- Stop word removal (configurable)
- Historical spelling correction (limited patterns)

**Normalization Options:**

- Case normalization
- Lemmatization
- Stop word filtering
- Spelling correction

**Why Necessary:** Reduces vocabulary size, handles morphological variations, improves feature quality.

**Downstream Impact:** Improves feature extraction efficiency, reduces sparsity, enhances semantic similarity calculations.

---

#### 5. Sentence Segmentation

**Implementation:** `src/text_processor.py` → `TextTokenizer` class

**Purpose:** Split text into sentences for summarization and analysis

**Design Choices:**

- spaCy's sentence boundary detection
- Custom rules for historical text (handling abbreviations like "Dr.", "Mr.")
- Preservation of sentence-level metadata

**Why Necessary:** Summarization and many analysis tasks operate at sentence level.

**Downstream Impact:** Enables extractive summarization, sentence-level feature extraction, and timeline event extraction.

---

#### 6. Feature Extraction / Vectorization

**Implementation:** `src/nlp_pipeline.py` → `FeatureExtractor` class

**Purpose:** Convert text into numerical representations for analysis

**Feature Types Extracted:**

1. **Basic Statistics**

   - Token counts, sentence counts, vocabulary size
   - Average word length, sentence length
   - Type-token ratio

2. **TF-IDF Features**

   - Term frequency-inverse document frequency vectors
   - Captures important terms relative to corpus

3. **Bag-of-Words (BOW)**

   - Simple word count vectors
   - Foundation for topic modeling

4. **Semantic Embeddings**

   - Sentence Transformer embeddings (all-MiniLM-L6-v2)
   - Captures semantic meaning

5. **Topic Modeling**

   - Latent Dirichlet Allocation (LDA)
   - Non-negative Matrix Factorization (NMF)
   - Identifies latent topics

6. **Temporal Features**

   - Date extraction and parsing
   - Temporal references identification

7. **Readability Scores**

   - Flesch Reading Ease, Flesch-Kincaid Grade Level
   - Automated Readability Index

8. **Sentiment Features**
   - VADER sentiment analysis
   - Polarity scores

**Design Choices:**

- Multiple feature types for different analysis needs
- Dimensionality reduction for efficiency (UMAP, PCA)
- Caching for performance

**Why Necessary:** Numerical features enable machine learning, clustering, similarity analysis, and comparative analysis.

**Downstream Impact:** Powers comparative analysis, clustering, similarity calculations, and visualization.

---

### Pipeline Execution Flow

```
1. Document Loading → Document objects with text and metadata
2. Text Cleaning → Cleaned text strings
3. Tokenization → Token lists and sentence lists
4. Normalization → Normalized token lists
5. Sentence Segmentation → Sentence-level representations
6. Feature Extraction → Multiple feature matrices/vectors
```

**Output:** `ProcessedDocument` objects containing all intermediate results for each document.

---

## Task 2: Document Summarization

### Implementation

**Module:** `src/summarizer.py`  
**Class:** `HistoricalSummarizer`

### Method: Extractive Summarization

The system uses **extractive summarization** methods that select important sentences from the original text, **strictly avoiding end-to-end LLM summarization**.

### Available Methods

1. **TextRank** (Default)

   - Graph-based algorithm inspired by PageRank
   - Builds sentence similarity graph
   - Ranks sentences by importance

2. **TF-IDF**

   - Selects sentences with highest TF-IDF scores
   - Emphasizes unique, important terms

3. **LexRank**

   - Graph-based centrality algorithm
   - Uses cosine similarity between sentences
   - Good for longer documents

4. **LSA (Latent Semantic Analysis)**

   - Singular Value Decomposition (SVD)
   - Identifies semantically important sentences
   - Reduces dimensionality

5. **KL-Divergence**
   - Minimizes KL-divergence between summary and document
   - Information-theoretic approach
   - Preserves key information

### Justification

**Why Extractive Methods:**

1. **Historical Accuracy:** Preserves original wording, crucial for historical documents
2. **Traceability:** Every summary sentence can be traced to source text
3. **No Hallucination:** Cannot introduce facts not in source
4. **Cost-Effective:** No API calls or LLM inference costs
5. **Interpretability:** Clear selection criteria (sentence scores)

**Why Not LLM Summarization:**

- Requirement explicitly prohibits end-to-end LLM summarization
- Risk of hallucination with historical facts
- Loss of traceability to source text
- Higher computational cost

### Implementation Details

Each method:

1. Uses sentence embeddings (Sentence Transformer)
2. Calculates sentence importance scores
3. Selects top sentences based on compression ratio
4. Preserves original sentence order
5. Returns summary with metadata (method, scores, sentences)

### Configuration

```python
config = {
    'summarization': {
        'method': 'textrank',  # Options: textrank, tfidf, lexrank, lsa, kl
        'compression_ratio': 0.3  # 30% of original sentences
    }
}
```

### Output Format

```json
{
    "document_id": "...",
    "summary_text": "...",
    "summary_sentences": ["...", "..."],
    "method": "textrank",
    "compression_ratio": 0.3,
    "sentence_scores": [...]
}
```

**Location:** `data/outputs/summaries/`

---

## Task 3: Named Entity Recognition

### Implementation

**Module:** `src/ner_extractor.py`  
**Class:** `HistoricalNER`

### Multi-Method Approach

The system uses **three complementary methods** for entity extraction:

#### 1. spaCy NER

- Pre-trained model: `en_core_web_lg`
- Detects standard entity types: PERSON, ORG, GPE, DATE, etc.
- High accuracy for modern text
- **Limitation:** May miss historical entities not in training data

#### 2. Gazetteer Lookup

**Implementation:** `HistoricalGazetteer` class

- Custom gazetteers for historical entities:

  - Historical figures (presidents, monarchs, revolutionaries)
  - Historical events (wars, revolutions, treaties)
  - Historical organizations (political parties, governments)
  - Historical locations (battles, cities, countries, empires)

- **Advantages:**
  - Captures domain-specific entities
  - High precision for known entities
  - Customizable for different historical periods

#### 3. Rule-Based Extraction

- Regex patterns for:

  - Historical dates (centuries, decades, approximate dates)
  - Event patterns ("Battle of X", "Treaty of Y")
  - Title patterns ("King X", "President Y")

- **Advantages:**
  - Handles date variations
  - Captures historical periods
  - Good for structured patterns

### Entity Types Extracted

1. **Persons (PERSON)**

   - Historical figures, leaders, individuals
   - Methods: spaCy, gazetteers, title patterns

2. **Locations (LOC, GPE)**

   - Countries, cities, regions, geographical entities
   - Methods: spaCy, gazetteers

3. **Organizations (ORG)**

   - Political parties, governments, institutions
   - Methods: spaCy, gazetteers

4. **Dates (DATE)**

   - Years, centuries, periods, approximate dates
   - Methods: spaCy, regex patterns, date parser

5. **Events (EVENT)**
   - Wars, revolutions, treaties, major historical events
   - Methods: Rule-based patterns, gazetteers

### Entity Combination and Deduplication

- Multiple methods may extract the same entity
- System combines and deduplicates entities:
  - Overlap detection
  - Confidence-based selection
  - Position-based merging

### Output Format

```json
{
  "document_id": "...",
  "entities": [
    {
      "text": "Ethiopia",
      "label": "GPE",
      "start_char": 123,
      "end_char": 130,
      "confidence": 0.9,
      "source": "spacy"
    }
  ],
  "entity_counts": {
    "PERSON": 25,
    "ORG": 30,
    "DATE": 15
  }
}
```

**Location:** `data/outputs/entities/`

### Limitations

1. **Ambiguous Entities:** Same entity may be classified differently (e.g., "Washington" = person or location)
2. **Historical Spelling:** Archaic spellings may not match modern entities
3. **Context-Dependent:** Some entities require context for disambiguation
4. **Language:** Optimized for English; may miss entities in other languages

---

## Task 4: Timeline Construction

### Implementation

**Module:** `src/timeline_builder.py`  
**Class:** `TimelineBuilder`

### Process

1. **Date Extraction**

   - Extracts dates from text using regex patterns
   - Handles various formats: "1945", "19th century", "circa 1500"
   - Parses dates using `dateparser` library

2. **Event Extraction**

   - Identifies events near extracted dates
   - Uses contextual patterns ("Battle of X", "Treaty of Y")
   - Extracts event descriptions from surrounding text

3. **Date Parsing and Normalization**

   - Converts dates to standardized format
   - Handles incomplete dates (year only, century only)
   - Assigns precision levels (DAY, MONTH, YEAR, DECADE, CENTURY, APPROXIMATE)

4. **Event Merging**

   - Identifies similar events from different documents
   - Merges duplicate events
   - Aggregates sources

5. **Chronological Sorting**
   - Sorts events by date
   - Handles undated events (placed at end or context-based)

### Date Handling

**Supported Formats:**

- Full dates: "1945-08-15", "15/08/1945"
- Month-year: "August 1945", "Aug 1945"
- Year only: "1945"
- Centuries: "19th century", "20th century"
- Decades: "1960s", "1960-1970"
- Approximate: "circa 1500", "c. 1500", "around 1500"
- Historical periods: "during the reign of X"

**Precision Levels:**

- `DAY` - Exact date
- `MONTH` - Month and year
- `YEAR` - Year only
- `DECADE` - Decade
- `CENTURY` - Century
- `APPROXIMATE` - Approximate date
- `UNKNOWN` - Cannot determine

### Traceability

Each timeline event includes:

- Source document IDs
- Character positions in source text
- Context excerpt
- Confidence scores
- Extracted entities

### Output Formats

1. **JSON** (`timeline.json`)

   - Structured data format
   - Includes all event metadata

2. **CSV** (`timeline.csv`)

   - Tabular format for analysis
   - Easy to import into other tools

3. **HTML** (`timeline.html`)

   - Interactive visualization
   - Color-coded by precision
   - Entity tags and source links

4. **PNG** (`timeline_visualization.png`)
   - Static visualization
   - Gantt-style timeline chart

**Location:** `data/outputs/timelines/`

---

## Task 5: Comparative Analysis

### Implementation

**Module:** `src/analyzer.py`  
**Class:** `ComparativeAnalyzer`

### Analytical Dimensions

The system performs comparative analysis across **four key dimensions**, all grounded in NLP-extracted evidence:

#### 1. Political Orientation or Ideology

**Indicators:**

- Political terminology frequency (democracy, monarchy, republic, etc.)
- Ideological keywords (liberal, conservative, socialist, etc.)
- Governance-related terms
- Power distribution language

**Analysis:**

- Scores documents on political ideology spectrum
- Classifies dominant ideology
- Identifies ideological shifts over time

**Evidence Sources:**

- Keyword frequencies from TF-IDF
- Lexicon-based scoring
- Entity co-occurrence (political figures, parties)

#### 2. Socioeconomic Structure and Policies

**Indicators:**

- Economic system terms (capitalist, socialist, mixed)
- Policy keywords (taxation, trade, agriculture, industry)
- Social structure indicators (class, caste, equality)
- Development indicators

**Analysis:**

- Classifies economic systems
- Scores development indices
- Analyzes inequality indicators
- Compares policy approaches

**Evidence Sources:**

- Economic terminology extraction
- Policy statement analysis
- Statistical indicators from text

#### 3. Power Distribution and Governance Style

**Indicators:**

- Centralization vs. decentralization terms
- Authority structures (monarchy, democracy, dictatorship)
- Power concentration keywords
- Governance complexity measures

**Analysis:**

- Maps power distribution
- Classifies governance systems
- Analyzes authority structures
- Measures governance complexity

**Evidence Sources:**

- Governance-related entities (rulers, institutions)
- Power-related terminology
- Organizational structure analysis

#### 4. Social Impact and Economic Advantage/Disadvantage

**Indicators:**

- Social impact keywords (improvement, decline, equality, inequality)
- Economic advantage/disadvantage terms
- Quality of life indicators
- Social mobility references

**Analysis:**

- Calculates net social impact scores
- Identifies advantaged/disadvantaged groups
- Analyzes impact magnitude
- Compares outcomes across documents

**Evidence Sources:**

- Sentiment analysis
- Impact-related terminology
- Comparative statements in text

### Comparative Methods

1. **Clustering**

   - K-means clustering on feature vectors
   - Groups similar documents
   - Identifies document clusters

2. **Similarity Analysis**

   - Cosine similarity on embeddings
   - Identifies most similar documents
   - Measures document relationships

3. **Dimension Correlation**

   - Pearson correlation between dimensions
   - Identifies relationships (e.g., ideology vs. governance)
   - Temporal evolution analysis

4. **Temporal Evolution**
   - Tracks dimension scores over time
   - Identifies trends and shifts
   - Compares periods

### Output

**Location:** `data/outputs/analysis/` and `data/outputs/reports/comparative_analysis/`

**Formats:**

- JSON analysis file
- Text report
- Visualizations (clusters, correlations, temporal evolution, radar charts)

---

## Final Deliverables

All deliverables are generated in the `data/outputs/` directory:

### 1. Cleaned and Processed Corpus

**Location:** `data/processed/`

**Contents:**

- `processed_documents.json` - All processed documents with tokens, sentences, metadata
- `document_metadata.json` - Document metadata
- `features.pkl` - Extracted features
- `performance_metrics.json` - Pipeline performance metrics
- `cleaned/` - Cleaned text files
- `tokenized/` - Tokenized representations

### 2. Summaries for All Documents

**Location:** `data/outputs/summaries/`

**Contents:**

- Individual summary JSON files (one per document)
- `all_summaries.json` - Consolidated summaries
- `summaries.csv` - Tabular format
- `intermediate/` - Batch processing intermediate files

**Format:**

- Summary text
- Selected sentences
- Method used
- Compression ratio
- Sentence scores

### 3. Extracted Named Entities

**Location:** `data/outputs/entities/`

**Contents:**

- `entities.json` - All entities by document
- `entities.csv` - Tabular format
- `entities.txt` - Human-readable format
- `entity_summary.json` - Entity statistics
- `intermediate/` - Batch processing files

**Entity Types:**

- Persons, Locations, Organizations, Dates, Events
- Counts, distributions, co-occurrence patterns

### 4. Chronological Timeline of Events

**Location:** `data/outputs/timelines/`

**Contents:**

- `timeline.json` - Structured timeline data
- `timeline.csv` - Tabular format
- `timeline.html` - Interactive HTML visualization
- `timeline_visualization.png` - Static visualization

**Features:**

- Chronologically sorted events
- Date precision levels
- Source document traceability
- Event descriptions and entities

### 5. Comparative Analytical Report

**Location:** `data/outputs/reports/comparative_analysis/`

**Contents:**

- `comparative_report.json` - Structured analysis data
- `comparative_report.txt` - Human-readable report

**Sections:**

- Executive summary
- Political ideology analysis
- Socioeconomic structure analysis
- Governance power analysis
- Social impact analysis
- Cross-document comparison
- Temporal evolution
- Evidence citations
- Methodology
- Limitations

**Supporting Visualizations:** `data/outputs/analysis/`

- Cluster visualizations
- Dimension correlation heatmaps
- Temporal evolution charts
- Radar charts for dimensions

---

## Project Structure

```
v1/
├── config/                 # Configuration files
│   ├── settings.py        # Main configuration
│   └── logging_config.yaml # Logging configuration
│
├── data/
│   ├── raw_documents/     # Input PDF documents
│   ├── processed/         # Processed corpus and features
│   └── outputs/           # All generated outputs
│       ├── summaries/     # Document summaries
│       ├── entities/      # Extracted entities
│       ├── timelines/     # Timeline data and visualizations
│       ├── analysis/      # Comparative analysis results
│       ├── reports/       # Final reports
│       └── visualizations/ # Interactive visualizations
│
├── src/                   # Source code
│   ├── document_loader.py      # Document loading
│   ├── text_processor.py       # Text cleaning, tokenization, normalization
│   ├── nlp_pipeline.py         # Main NLP pipeline orchestrator
│   ├── summarizer.py           # Summarization methods
│   ├── ner_extractor.py        # Named Entity Recognition
│   ├── timeline_builder.py     # Timeline construction
│   ├── analyzer.py             # Comparative analysis
│   ├── comparative_report.py   # Report generation
│   ├── utils/
│   │   └── logger.py           # Logging utilities
│   └── visualization/
│       └── timeline_plot.py    # Visualization modules
│
├── logs/                  # Log files
│   ├── pipeline.log       # Main pipeline log
│   └── errors.log         # Error log
│
├── main.py                # Main execution script
├── requirements.txt       # Python dependencies
├── setup.py              # Setup script
└── README.md             # This file
```

---

## Methodology Documentation

### Pipeline Description

The NLP pipeline follows a sequential processing approach:

1. **Document Loading** → Raw text extraction with metadata
2. **Text Cleaning** → Noise removal and normalization
3. **Tokenization** → Text segmentation into tokens and sentences
4. **Normalization** → Token standardization
5. **Feature Extraction** → Multiple feature representations
6. **Downstream Tasks** → Summarization, NER, Timeline, Analysis

### Justification for Processing Steps

**Text Cleaning:**

- **Why:** Historical documents contain OCR errors, formatting artifacts, encoding issues
- **Design Choice:** Conservative cleaning that preserves entity information
- **Impact:** Improves downstream accuracy, reduces noise

**Tokenization:**

- **Why:** Required for all NLP tasks
- **Design Choice:** spaCy-based with historical text customizations
- **Impact:** Enables accurate feature extraction and entity recognition

**Normalization:**

- **Why:** Reduces vocabulary size, handles morphological variation
- **Design Choice:** Lemmatization over stemming for better interpretability
- **Impact:** Improves feature quality, reduces sparsity

**Feature Extraction:**

- **Why:** Enables numerical analysis and machine learning
- **Design Choice:** Multiple feature types for different analysis needs
- **Impact:** Powers comparative analysis, clustering, similarity calculations

### Modeling and Feature Choices

**TF-IDF:**

- **Choice:** Captures important terms relative to corpus
- **Rationale:** Standard, interpretable, effective for historical text
- **Trade-off:** Does not capture semantic similarity

**Sentence Embeddings:**

- **Choice:** Sentence Transformer (all-MiniLM-L6-v2)
- **Rationale:** Balances quality and speed, good for semantic similarity
- **Trade-off:** Less powerful than large transformers but faster

**Topic Modeling:**

- **Choice:** LDA and NMF
- **Rationale:** LDA for interpretable topics, NMF for efficiency
- **Trade-off:** LDA is slower but more interpretable

**Clustering:**

- **Choice:** K-means on feature vectors
- **Rationale:** Simple, fast, effective for document grouping
- **Trade-off:** Requires pre-specified number of clusters

---

## Limitations and Trade-offs

### Technical Limitations

1. **Language Support:**

   - Optimized for English
   - Historical spellings and archaic language may reduce accuracy

2. **Entity Recognition:**

   - Ambiguous entities (e.g., "Washington" = person or location)
   - Historical entities not in training data may be missed
   - Context-dependent disambiguation limitations

3. **Date Parsing:**

   - Incomplete or approximate dates handled but with lower precision
   - Historical date formats may not always parse correctly
   - Relative dates ("before X", "after Y") require context

4. **Summarization:**

   - Extractive methods may miss implicit information
   - Cannot generate new phrasing (extractive only)
   - May include less important sentences if they score highly

5. **Comparative Analysis:**
   - Depends on terminology and keyword frequencies
   - May miss subtle ideological differences
   - Requires sufficient text volume for reliable analysis

### Design Trade-offs

1. **Extractive vs. Abstractive Summarization:**

   - **Choice:** Extractive
   - **Trade-off:** Preserves accuracy and traceability but may be less coherent

2. **Multiple NER Methods vs. Single Method:**

   - **Choice:** Multiple methods (spaCy + gazetteers + rules)
   - **Trade-off:** Higher coverage but potential duplication

3. **Feature Diversity:**

   - **Choice:** Multiple feature types
   - **Trade-off:** More comprehensive but higher computational cost

4. **Processing Speed vs. Accuracy:**
   - **Choice:** Balanced approach (efficient models with good quality)
   - **Trade-off:** Faster processing but potentially lower accuracy than state-of-the-art

### Known Issues

1. **Memory Usage:** Large documents may require significant memory
2. **Processing Time:** Full pipeline takes time for large document sets
3. **Date Precision:** Some dates cannot be precisely determined
4. **Entity Ambiguity:** Some entities remain ambiguous without additional context

---

## Performance Metrics

### Pipeline Performance

Performance metrics are logged and saved in `data/processed/performance_metrics.json`:

- Document processing times
- Feature extraction times
- Token counts, vocabulary sizes
- Processing throughput

### Example Metrics

For 12 documents:

- Average processing time: ~2-5 seconds per document
- Entity extraction: ~1-3 seconds per document
- Summarization: ~0.5-2 seconds per document
- Full pipeline: ~30-60 seconds total (depending on document size)

### Quality Metrics

- **Summarization:** ROUGE scores (if reference summaries available)
- **Entity Recognition:** Precision/recall (if gold standard available)
- **Timeline:** Event count, date coverage, precision distribution

---

## Contributing

This is a practical exam submission. However, suggestions and improvements are welcome:

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Submit a pull request

---

## License

This project is submitted as part of a practical exam. All code and documentation are provided as-is for evaluation purposes.

---

## Contact and Submission

**Repository URL:** [GitHub Repository URL]

**Access:** If the repository is private, access has been granted to: https://github.com/addisfortunedev

**Documentation:** This README serves as the primary documentation. Additional documentation can be found in:

- Code comments and docstrings
- Generated reports (methodology sections)
- Configuration files

---

## Acknowledgments

- spaCy for NLP models and tools
- NLTK for text processing utilities
- Sentence Transformers for semantic embeddings
- scikit-learn for machine learning algorithms
- All contributors to the open-source libraries used in this project

---

**Last Updated:** 2026-01-04  
**Version:** 1.0  
**Status:** Complete - Ready for Submission
