#!/usr/bin/env python3
"""
Multi-Symbol Order Flow Engine Dashboard
Real-time dashboard using Dash and Plotly with dark theme and symbol selection.
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import json
import time
import threading
import queue
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logger = logging.getLogger(__name__)

class MultiSymbolOrderFlowDashboard:
    """Multi-Symbol Order Flow Engine Dashboard with real-time updates."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.app = dash.Dash(__name__)
        
        # Get symbols from config
        self.symbols = self.config.get('symbols', ['btcusdt', 'ethusdt', 'xauusdt', 'xagusdt'])
        self.selected_symbol = self.symbols[0]  # Default to first symbol
        
        # Data storage for each symbol
        self.symbol_data = {}
        for symbol in self.symbols:
            self.symbol_data[symbol] = {
                'price': [],
                'vwap': [],
                'cvd': [],
                'imbalance': [],
                'timestamp': [],
                'latest_signal': 'WAIT',
                'latest_confidence': 0,
                'latest_price': 0.0,
                'latest_time': datetime.now()
            }
        
        # Dashboard configuration
        self.update_interval = 1000  # 1 second
        self.max_data_points = 300  # 5 minutes of 1-second data
        self.dark_theme = True
        
        # Initialize dashboard
        self._setup_layout()
        self._setup_callbacks()
        
        logger.info(f"Multi-Symbol Dashboard initialized for symbols: {self.symbols}")
    
    def _setup_layout(self):
        """Setup the dashboard layout with dark theme."""
        # Dark theme colors
        dark_colors = {
            'background': '#1a1a1a',
            'surface': '#2d2d2d',
            'primary': '#00d4aa',
            'secondary': '#ff6b6b',
            'text': '#ffffff',
            'text_secondary': '#b0b0b0',
            'border': '#404040'
        }
        
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Order Flow Engine Dashboard", 
                       style={'color': dark_colors['text'], 'margin': '0', 'fontSize': '28px'}),
                html.P("Real-time multi-symbol order flow analysis", 
                      style={'color': dark_colors['text_secondary'], 'margin': '5px 0 0 0'})
            ], style={'textAlign': 'center', 'marginBottom': '20px'}),
            
            # Symbol selector and signal card
            html.Div([
                # Symbol dropdown
                html.Div([
                    html.Label("Select Symbol:", 
                             style={'color': dark_colors['text'], 'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Dropdown(
                        id='symbol-dropdown',
                        options=[{'label': symbol.upper(), 'value': symbol} for symbol in self.symbols],
                        value=self.selected_symbol,
                        style={'backgroundColor': dark_colors['surface'], 'color': dark_colors['text']},
                        className='symbol-dropdown'
                    )
                ], style={'width': '200px', 'marginRight': '20px'}),
                
                # Signal card
                html.Div([
                    html.Div(id='signal-card', className='signal-card')
                ], style={'flex': '1'})
                
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px'}),
            
            # Charts container
            html.Div([
                # Price vs VWAP chart
                html.Div([
                    html.H3("Price vs VWAP", 
                           style={'color': dark_colors['text'], 'textAlign': 'center', 'marginBottom': '10px'}),
                    dcc.Graph(
                        id='price-vwap-chart',
                        config={'displayModeBar': False},
                        style={'height': '300px'}
                    )
                ], className='chart-container', style={'width': '100%', 'marginBottom': '20px'}),
                
                # CVD and Imbalance charts
                html.Div([
                    # CVD chart
                    html.Div([
                        html.H3("Cumulative Volume Delta (CVD)", 
                               style={'color': dark_colors['text'], 'textAlign': 'center', 'marginBottom': '10px'}),
                        dcc.Graph(
                            id='cvd-chart',
                            config={'displayModeBar': False},
                            style={'height': '250px'}
                        )
                    ], style={'width': '50%', 'paddingRight': '10px'}),
                    
                    # Imbalance chart
                    html.Div([
                        html.H3("Order Book Imbalance", 
                               style={'color': dark_colors['text'], 'textAlign': 'center', 'marginBottom': '10px'}),
                        dcc.Graph(
                            id='imbalance-chart',
                            config={'displayModeBar': False},
                            style={'height': '250px'}
                        )
                    ], style={'width': '50%', 'paddingLeft': '10px'})
                    
                ], style={'display': 'flex', 'marginBottom': '20px'}),
                
                # Metrics summary
                html.Div([
                    html.H3("Current Metrics", 
                           style={'color': dark_colors['text'], 'textAlign': 'center', 'marginBottom': '10px'}),
                    html.Div(id='metrics-summary', className='metrics-summary')
                ], style={'marginBottom': '20px'})
                
            ], className='charts-container'),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval,
                n_intervals=0
            )
            
        ], style={
            'backgroundColor': dark_colors['background'],
            'color': dark_colors['text'],
            'padding': '20px',
            'fontFamily': 'Arial, sans-serif',
            'minHeight': '100vh'
        })
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            [Output('price-vwap-chart', 'figure'),
             Output('cvd-chart', 'figure'),
             Output('imbalance-chart', 'figure'),
             Output('signal-card', 'children'),
             Output('metrics-summary', 'children')],
            [Input('interval-component', 'n_intervals'),
             Input('symbol-dropdown', 'value')]
        )
        def update_dashboard(n_intervals, selected_symbol):
            """Update all dashboard components."""
            if selected_symbol:
                self.selected_symbol = selected_symbol
            
            symbol_data = self.symbol_data[self.selected_symbol]
            
            # Create charts
            price_vwap_fig = self._create_price_vwap_chart(symbol_data)
            cvd_fig = self._create_cvd_chart(symbol_data)
            imbalance_fig = self._create_imbalance_chart(symbol_data)
            
            # Create signal card
            signal_card = self._create_signal_card(symbol_data)
            
            # Create metrics summary
            metrics_summary = self._create_metrics_summary(symbol_data)
            
            return price_vwap_fig, cvd_fig, imbalance_fig, signal_card, metrics_summary
    
    def _create_price_vwap_chart(self, symbol_data: Dict) -> go.Figure:
        """Create price vs VWAP chart."""
        if not symbol_data['timestamp']:
            return self._create_empty_chart("Price vs VWAP")
        
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=symbol_data['timestamp'],
            y=symbol_data['price'],
            mode='lines',
            name='Price',
            line=dict(color='#00d4aa', width=2),
            hovertemplate='<b>Price</b><br>Time: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
        ))
        
        # VWAP line
        fig.add_trace(go.Scatter(
            x=symbol_data['timestamp'],
            y=symbol_data['vwap'],
            mode='lines',
            name='VWAP',
            line=dict(color='#ff6b6b', width=2, dash='dash'),
            hovertemplate='<b>VWAP</b><br>Time: %{x}<br>VWAP: $%{y:,.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="",
            xaxis_title="Time",
            yaxis_title="Price ($)",
            template='plotly_dark',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=300
        )
        
        return fig
    
    def _create_cvd_chart(self, symbol_data: Dict) -> go.Figure:
        """Create CVD chart."""
        if not symbol_data['timestamp']:
            return self._create_empty_chart("CVD")
        
        fig = go.Figure()
        
        # CVD line with fill
        fig.add_trace(go.Scatter(
            x=symbol_data['timestamp'],
            y=symbol_data['cvd'],
            mode='lines',
            name='CVD',
            line=dict(color='#8b5cf6', width=2),
            fill='tonexty',
            fillcolor='rgba(139, 92, 246, 0.2)',
            hovertemplate='<b>CVD</b><br>Time: %{x}<br>CVD: %{y:.4f}<extra></extra>'
        ))
        
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title="",
            xaxis_title="Time",
            yaxis_title="CVD",
            template='plotly_dark',
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            height=250
        )
        
        return fig
    
    def _create_imbalance_chart(self, symbol_data: Dict) -> go.Figure:
        """Create imbalance bar chart."""
        if not symbol_data['timestamp']:
            return self._create_empty_chart("Imbalance")
        
        # Create colors based on imbalance values
        colors = []
        for imbalance in symbol_data['imbalance']:
            if imbalance > 0.1:
                colors.append('#00d4aa')  # Green for positive
            elif imbalance < -0.1:
                colors.append('#ff6b6b')  # Red for negative
            else:
                colors.append('#6b7280')  # Gray for neutral
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=symbol_data['timestamp'],
            y=symbol_data['imbalance'],
            name='Imbalance',
            marker_color=colors,
            hovertemplate='<b>Imbalance</b><br>Time: %{x}<br>Imbalance: %{y:.4f}<extra></extra>'
        ))
        
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title="",
            xaxis_title="Time",
            yaxis_title="Imbalance",
            template='plotly_dark',
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            height=250
        )
        
        return fig
    
    def _create_signal_card(self, symbol_data: Dict) -> html.Div:
        """Create signal card component."""
        signal = symbol_data['latest_signal']
        confidence = symbol_data['latest_confidence']
        price = symbol_data['latest_price']
        timestamp = symbol_data['latest_time']
        
        # Signal emoji and color
        if signal == 'LONG':
            emoji = '🟩'
            color = '#00d4aa'
            bg_color = 'rgba(0, 212, 170, 0.1)'
        elif signal == 'SHORT':
            emoji = '🟥'
            color = '#ff6b6b'
            bg_color = 'rgba(255, 107, 107, 0.1)'
        else:
            emoji = '⚪'
            color = '#6b7280'
            bg_color = 'rgba(107, 114, 128, 0.1)'
        
        # Confidence color
        if confidence >= 80:
            conf_color = '#00d4aa'
        elif confidence >= 60:
            conf_color = '#fbbf24'
        else:
            conf_color = '#ff6b6b'
        
        return html.Div([
            html.Div([
                html.Div([
                    html.Span(emoji, style={'fontSize': '24px', 'marginRight': '10px'}),
                    html.Span(signal, style={'fontSize': '18px', 'fontWeight': 'bold', 'color': color})
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                
                html.Div([
                    html.Span("Confidence: ", style={'color': '#b0b0b0'}),
                    html.Span(f"{confidence}%", style={'color': conf_color, 'fontWeight': 'bold'})
                ], style={'marginBottom': '5px'}),
                
                html.Div([
                    html.Span("Price: ", style={'color': '#b0b0b0'}),
                    html.Span(f"${price:,.2f}", style={'color': '#ffffff', 'fontWeight': 'bold'})
                ], style={'marginBottom': '5px'}),
                
                html.Div([
                    html.Span("Time: ", style={'color': '#b0b0b0'}),
                    html.Span(timestamp.strftime('%H:%M:%S'), style={'color': '#ffffff'})
                ])
                
            ], style={
                'padding': '15px',
                'backgroundColor': bg_color,
                'border': f'2px solid {color}',
                'borderRadius': '8px',
                'minWidth': '200px'
            })
        ])
    
    def _create_metrics_summary(self, symbol_data: Dict) -> html.Div:
        """Create metrics summary component."""
        if not symbol_data['timestamp']:
            return html.Div("No data available", style={'color': '#b0b0b0', 'textAlign': 'center'})
        
        # Get latest values
        latest_price = symbol_data['price'][-1] if symbol_data['price'] else 0
        latest_vwap = symbol_data['vwap'][-1] if symbol_data['vwap'] else 0
        latest_cvd = symbol_data['cvd'][-1] if symbol_data['cvd'] else 0
        latest_imbalance = symbol_data['imbalance'][-1] if symbol_data['imbalance'] else 0
        
        # Calculate price vs VWAP percentage
        price_vwap_pct = ((latest_price / latest_vwap) - 1) * 100 if latest_vwap > 0 else 0
        
        return html.Div([
            html.Div([
                html.Div([
                    html.Span("Price vs VWAP: ", style={'color': '#b0b0b0'}),
                    html.Span(f"{price_vwap_pct:+.2f}%", 
                             style={'color': '#00d4aa' if price_vwap_pct >= 0 else '#ff6b6b', 'fontWeight': 'bold'})
                ], style={'marginBottom': '8px'}),
                
                html.Div([
                    html.Span("CVD: ", style={'color': '#b0b0b0'}),
                    html.Span(f"{latest_cvd:+.4f}", 
                             style={'color': '#8b5cf6', 'fontWeight': 'bold'})
                ], style={'marginBottom': '8px'}),
                
                html.Div([
                    html.Span("Imbalance: ", style={'color': '#b0b0b0'}),
                    html.Span(f"{latest_imbalance:+.4f}", 
                             style={'color': '#00d4aa' if latest_imbalance > 0 else '#ff6b6b' if latest_imbalance < 0 else '#6b7280', 'fontWeight': 'bold'})
                ], style={'marginBottom': '8px'}),
                
                html.Div([
                    html.Span("Data Points: ", style={'color': '#b0b0b0'}),
                    html.Span(f"{len(symbol_data['timestamp'])}", 
                             style={'color': '#ffffff', 'fontWeight': 'bold'})
                ])
                
            ], style={
                'display': 'grid',
                'gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))',
                'gap': '15px',
                'padding': '15px',
                'backgroundColor': '#2d2d2d',
                'borderRadius': '8px',
                'border': '1px solid #404040'
            })
        ])
    
    def _create_empty_chart(self, title: str) -> go.Figure:
        """Create empty chart placeholder."""
        fig = go.Figure()
        fig.add_annotation(
            text=f"No data available for {title}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#b0b0b0')
        )
        fig.update_layout(
            template='plotly_dark',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=0, r=0, t=0, b=0)
        )
        return fig
    
    def add_data(self, symbol: str, data: Dict[str, Any]):
        """Add new data point for a symbol."""
        if symbol not in self.symbol_data:
            logger.warning(f"Unknown symbol: {symbol}")
            return
        
        symbol_data = self.symbol_data[symbol]
        current_time = datetime.now()
        
        # Add new data points
        symbol_data['timestamp'].append(current_time)
        symbol_data['price'].append(data.get('price', 0.0))
        symbol_data['vwap'].append(data.get('metrics', {}).get('vwap', 0.0))
        symbol_data['cvd'].append(data.get('metrics', {}).get('cvd', 0.0))
        symbol_data['imbalance'].append(data.get('metrics', {}).get('imbalance', 0.0))
        
        # Update latest values
        symbol_data['latest_signal'] = data.get('signal', 'WAIT')
        symbol_data['latest_confidence'] = data.get('confidence', 0)
        symbol_data['latest_price'] = data.get('price', 0.0)
        symbol_data['latest_time'] = current_time
        
        # Limit data points
        if len(symbol_data['timestamp']) > self.max_data_points:
            for key in ['timestamp', 'price', 'vwap', 'cvd', 'imbalance']:
                symbol_data[key] = symbol_data[key][-self.max_data_points:]
        
        logger.debug(f"Added data for {symbol}: {symbol_data['latest_signal']} @ ${symbol_data['latest_price']:.2f}")
    
    def run(self, host='127.0.0.1', port=8050, debug=False):
        """Run the dashboard."""
        logger.info(f"Starting Multi-Symbol Dashboard on http://{host}:{port}")
        logger.info(f"Available symbols: {', '.join(self.symbols)}")
        
        # Add CSS for dark theme
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
                <style>
                    body {
                        background-color: #1a1a1a !important;
                        color: #ffffff !important;
                    }
                    .symbol-dropdown .Select-control {
                        background-color: #2d2d2d !important;
                        border-color: #404040 !important;
                        color: #ffffff !important;
                    }
                    .symbol-dropdown .Select-menu-outer {
                        background-color: #2d2d2d !important;
                        border-color: #404040 !important;
                    }
                    .symbol-dropdown .Select-option {
                        background-color: #2d2d2d !important;
                        color: #ffffff !important;
                    }
                    .symbol-dropdown .Select-option.is-focused {
                        background-color: #404040 !important;
                    }
                    .symbol-dropdown .Select-option.is-selected {
                        background-color: #00d4aa !important;
                    }
                </style>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''
        
        self.app.run_server(debug=debug, host=host, port=port)

class MockDataGenerator:
    """Mock data generator for testing the dashboard."""
    
    def __init__(self, dashboard: MultiSymbolOrderFlowDashboard):
        self.dashboard = dashboard
        self.running = False
        self.thread = None
        
        # Mock data parameters
        self.base_prices = {
            'btcusdt': 42500.0,
            'ethusdt': 3200.0,
            'xauusdt': 2000.0,
            'xagusdt': 25.0
        }
        self.price_trends = {symbol: 0.0 for symbol in self.base_prices}
        
    def start(self):
        """Start mock data generation."""
        self.running = True
        self.thread = threading.Thread(target=self._generate_data, daemon=True)
        self.thread.start()
        logger.info("Mock data generator started")
    
    def stop(self):
        """Stop mock data generation."""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("Mock data generator stopped")
    
    def _generate_data(self):
        """Generate mock data for all symbols."""
        import random
        
        while self.running:
            try:
                for symbol in self.dashboard.symbols:
                    # Generate mock data
                    mock_data = self._create_mock_data(symbol)
                    self.dashboard.add_data(symbol, mock_data)
                
                time.sleep(1.0)  # 1 second interval
                
            except Exception as e:
                logger.error(f"Error generating mock data: {e}")
                time.sleep(1.0)
    
    def _create_mock_data(self, symbol: str) -> Dict[str, Any]:
        """Create mock data for a symbol."""
        import random
        
        # Update price trend
        trend_change = random.uniform(-0.001, 0.001)
        self.price_trends[symbol] += trend_change
        self.price_trends[symbol] = max(-0.01, min(0.01, self.price_trends[symbol]))
        
        # Calculate price
        base_price = self.base_prices[symbol]
        price_change = base_price * self.price_trends[symbol]
        current_price = base_price + price_change + random.uniform(-base_price * 0.001, base_price * 0.001)
        
        # Generate metrics
        cvd = random.uniform(-0.2, 0.2)
        imbalance = random.uniform(-0.3, 0.3)
        vwap = current_price * random.uniform(0.998, 1.002)
        
        # Generate signal based on metrics
        signal = "WAIT"
        confidence = random.randint(20, 50)
        
        if cvd > 0.1 and imbalance > 0.15 and current_price >= vwap:
            signal = "LONG"
            confidence = random.randint(70, 95)
        elif cvd < -0.1 and imbalance < -0.15 and current_price <= vwap:
            signal = "SHORT"
            confidence = random.randint(70, 95)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence,
            'price': current_price,
            'metrics': {
                'cvd': cvd,
                'imbalance': imbalance,
                'vwap': vwap,
                'absorption': random.uniform(0.1, 0.4)
            },
            'session_active': True,
            'reasoning': [f"Mock data for {symbol}"]
        }

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}

if __name__ == '__main__':
    # Load configuration
    config = load_config()
    
    # Create dashboard
    dashboard = MultiSymbolOrderFlowDashboard(config)
    
    # Start mock data generator for testing
    mock_generator = MockDataGenerator(dashboard)
    mock_generator.start()
    
    try:
        # Run dashboard
        dashboard.run(debug=False)
    except KeyboardInterrupt:
        logger.info("Dashboard shutting down...")
    finally:
        mock_generator.stop()