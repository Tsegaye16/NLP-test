import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from src.utils.logger import logger
from config.settings import PATHS

class TimelineVisualizer:
    """Create interactive timeline visualizations"""
    
    def __init__(self):
        logger.log_step("TimelineVisualizer initialized")
    
    def create_interactive_timeline(self, timeline_data: Dict[str, Any], 
                                   output_dir: Path) -> Path:
        """Create interactive timeline visualization"""
        # Convert to DataFrame
        events = []
        for event in timeline_data.get('events', []):
            # Handle both dict and TimelineEvent object
            if isinstance(event, dict):
                event_dict = event
            else:
                # Convert TimelineEvent object to dict
                event_dict = {
                    'date': getattr(event, 'date_str', ''),
                    'year': getattr(event, 'year', None),
                    'description': getattr(event, 'description', ''),
                    'sources': getattr(event, 'sources', []),
                    'entities': getattr(event, 'entities', []),
                    'confidence': getattr(event, 'confidence', 0),
                    'precision': getattr(event, 'date_precision', None)
                }
                if hasattr(event_dict['precision'], 'value'):
                    event_dict['precision'] = event_dict['precision'].value
                else:
                    event_dict['precision'] = str(event_dict['precision']) if event_dict['precision'] else 'unknown'
            
            events.append({
                'date': event_dict.get('date', ''),
                'year': event_dict.get('year'),
                'description': event_dict.get('description', ''),
                'sources': '; '.join(event_dict.get('sources', [])),
                'entities': '; '.join([e.get('text', '') if isinstance(e, dict) else getattr(e, 'text', '') for e in event_dict.get('entities', [])][:3]),
                'confidence': event_dict.get('confidence', 0),
                'precision': event_dict.get('precision', 'unknown')
            })
        
        df = pd.DataFrame(events)
        
        if df.empty:
            logger.log_step("No events to visualize")
            return None
        
        # Create timeline figure
        fig = go.Figure()
        
        # Add events as scatter points
        fig.add_trace(go.Scatter(
            x=df['year'],
            y=[1] * len(df),  # Constant y for timeline
            mode='markers+text',
            marker=dict(
                size=10,
                color=df['confidence'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Confidence")
            ),
            text=df['description'].str[:50] + "...",
            hovertext=df.apply(lambda row: self._create_hover_text(row), axis=1),
            hoverinfo='text',
            name='Events'
        ))
        
        # Update layout
        fig.update_layout(
            title='Historical Timeline',
            xaxis_title='Year',
            yaxis=dict(showticklabels=False),
            hovermode='closest',
            height=600,
            showlegend=False
        )
        
        # Save as HTML
        output_file = output_dir / "interactive_timeline.html"
        fig.write_html(str(output_file))
        
        logger.log_step("Interactive timeline created", details={
            "output_file": str(output_file),
            "events": len(df)
        })
        
        return output_file
    
    def _create_hover_text(self, row: pd.Series) -> str:
        """Create hover text for timeline events"""
        return f"""
        <b>Date:</b> {row['date']}<br>
        <b>Description:</b> {row['description']}<br>
        <b>Sources:</b> {row['sources']}<br>
        <b>Entities:</b> {row['entities']}<br>
        <b>Confidence:</b> {row['confidence']:.2f}<br>
        <b>Precision:</b> {row['precision']}
        """
    
    def create_density_timeline(self, timeline_data: Dict[str, Any], 
                               output_dir: Path) -> Path:
        """Create event density timeline"""
        # Calculate events per year
        year_counts = {}
        for event in timeline_data.get('events', []):
            # Handle both dict and TimelineEvent object
            if isinstance(event, dict):
                year = event.get('year')
            else:
                year = getattr(event, 'year', None)
            if year:
                year_counts[year] = year_counts.get(year, 0) + 1
        
        if not year_counts:
            return None
        
        # Create DataFrame
        df = pd.DataFrame({
            'year': list(year_counts.keys()),
            'count': list(year_counts.values())
        }).sort_values('year')
        
        # Create figure
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['year'],
            y=df['count'],
            mode='lines+markers',
            line=dict(color='firebrick', width=2),
            marker=dict(size=8),
            name='Event Count'
        ))
        
        fig.update_layout(
            title='Historical Event Density',
            xaxis_title='Year',
            yaxis_title='Number of Events',
            hovermode='x'
        )
        
        # Save as HTML
        output_file = output_dir / "event_density_timeline.html"
        fig.write_html(str(output_file))
        
        return output_file

class EntityNetworkVisualizer:
    """Create entity network visualizations"""
    
    def __init__(self):
        logger.log_step("EntityNetworkVisualizer initialized")
    
    def create_entity_network(self, entity_data: Dict[str, Any], 
                             output_dir: Path) -> Path:
        """Create entity relationship network"""
        import networkx as nx
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes (entities)
        entities_by_doc = entity_data.get('entities_by_document', {})
        all_entities = set()
        
        for doc_id, entities in entities_by_doc.items():
            for entity in entities:
                entity_text = entity.get('text', '')
                entity_label = entity.get('label', '')
                if entity_text:
                    node_id = f"{entity_text} ({entity_label})"
                    all_entities.add(node_id)
                    
                    # Add node if not exists
                    if not G.has_node(node_id):
                        G.add_node(node_id, 
                                  label=entity_label,
                                  size=1)
                    else:
                        # Increase node size for co-occurrence
                        G.nodes[node_id]['size'] += 1
        
        # Add edges (co-occurrence)
        for doc_id, entities in entities_by_doc.items():
            doc_entities = [e for e in entities if e.get('text')]
            
            for i, entity1 in enumerate(doc_entities):
                for entity2 in doc_entities[i+1:]:
                    node1 = f"{entity1['text']} ({entity1.get('label', '')})"
                    node2 = f"{entity2['text']} ({entity2.get('label', '')})"
                    
                    if G.has_edge(node1, node2):
                        G[node1][node2]['weight'] += 1
                    else:
                        G.add_edge(node1, node2, weight=1)
        
        if len(G.nodes) == 0:
            logger.log_step("No entities to visualize")
            return None
        
        # Create Plotly network
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(edge[2]['weight'])
        
        # Create edge traces
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Node positions and sizes
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_size.append(G.nodes[node]['size'] * 10)
            
            # Color by label
            label = G.nodes[node]['label']
            color_map = {
                'PERSON': 'red',
                'LOC': 'blue',
                'ORG': 'green',
                'DATE': 'orange',
                'EVENT': 'purple'
            }
            node_color.append(color_map.get(label, 'gray'))
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="top center",
            marker=dict(
                size=node_size,
                color=node_color,
                line_width=2
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Entity Network',
                           showlegend=False,
                           hovermode='closest',
                           height=800
                       ))
        
        # Save as HTML
        output_file = output_dir / "entity_network.html"
        fig.write_html(str(output_file))
        
        logger.log_step("Entity network created", details={
            "output_file": str(output_file),
            "nodes": len(G.nodes()),
            "edges": len(G.edges())
        })
        
        return output_file

class AnalysisDashboard:
    """Create comprehensive analysis dashboard"""
    
    def __init__(self):
        logger.log_step("AnalysisDashboard initialized")
    
    def create_dashboard(self, analysis_data: Dict[str, Any], 
                        output_dir: Path) -> Path:
        """Create interactive analysis dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Dimension Scores', 'Cluster Distribution',
                           'Temporal Evolution', 'Correlation Heatmap'),
            specs=[[{'type': 'bar'}, {'type': 'pie'}],
                  [{'type': 'scatter'}, {'type': 'heatmap'}]]
        )
        
        # 1. Dimension Scores (Bar Chart)
        dimension_scores = analysis_data.get('dimension_scores', {})
        if dimension_scores:
            fig.add_trace(
                go.Bar(
                    x=list(dimension_scores.keys()),
                    y=list(dimension_scores.values()),
                    name='Dimension Scores'
                ),
                row=1, col=1
            )
        
        # 2. Cluster Distribution (Pie Chart)
        cluster_dist = analysis_data.get('cluster_distribution', {})
        if cluster_dist:
            fig.add_trace(
                go.Pie(
                    labels=list(cluster_dist.keys()),
                    values=list(cluster_dist.values()),
                    name='Clusters'
                ),
                row=1, col=2
            )
        
        # 3. Temporal Evolution (Scatter)
        temporal_data = analysis_data.get('temporal_evolution', {})
        if temporal_data:
            years = list(temporal_data.keys())
            for dim in ['political_ideology', 'socioeconomic_structure']:
                if dim in temporal_data.get(years[0], {}):
                    scores = [temporal_data[year].get(dim, 0) for year in years]
                    fig.add_trace(
                        go.Scatter(
                            x=years,
                            y=scores,
                            mode='lines+markers',
                            name=dim.replace('_', ' ').title()
                        ),
                        row=2, col=1
                    )
        
        # 4. Correlation Heatmap
        correlation_matrix = analysis_data.get('correlation_matrix', [])
        if correlation_matrix:
            fig.add_trace(
                go.Heatmap(
                    z=correlation_matrix,
                    colorscale='RdBu',
                    zmid=0
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text='Historical Analysis Dashboard',
            height=800,
            showlegend=True
        )
        
        # Save dashboard
        output_file = output_dir / "analysis_dashboard.html"
        fig.write_html(str(output_file))
        
        logger.log_step("Analysis dashboard created", details={
            "output_file": str(output_file)
        })
        
        return output_file