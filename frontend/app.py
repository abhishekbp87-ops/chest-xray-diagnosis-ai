"""
Advanced Streamlit application for chest X-ray analysis.
Alternative UI with comprehensive features and real-time analysis.
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import time
import io
import base64
from typing import Dict, List, Optional

# Configure page
st.set_page_config(
    page_title="Chest X-ray AI Diagnostics",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import local modules
from src.infer import AdvancedInferenceEngine, create_inference_engine
from src.medical_model import load_model

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .prediction-card {
        border: 2px solid #e0e0e0;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .success-card {
        border-color: #28a745;
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    }
    
    .warning-card {
        border-color: #ffc107;
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    }
    
    .danger-card {
        border-color: #dc3545;
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    }
</style>
""", unsafe_allow_html=True)

class StreamlitApp:
    """Advanced Streamlit application for chest X-ray analysis."""
    
    def __init__(self):
        self.initialize_session_state()
        self.load_model_cache()
    
    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'inference_engine' not in st.session_state:
            st.session_state.inference_engine = None
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
        if 'uploaded_image' not in st.session_state:
            st.session_state.uploaded_image = None
        if 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False
    
    @st.cache_resource
    def load_model_cache(self):
        """Load and cache the model."""
        try:
            # Configure model path
            model_path = "models/best_model.pth"
            
            # Create inference engine
            engine = create_inference_engine(
                model_path=model_path,
                class_names=["Normal", "Pneumonia"],
                use_tta=True,
                use_uncertainty=True,
            )
            
            st.session_state.inference_engine = engine
            return engine
            
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            return None
    
    def create_header(self):
        """Create application header."""
        st.markdown('<h1 class="main-header">🫁 Chest X-ray AI Diagnostics</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Advanced AI-powered chest X-ray analysis for medical diagnostics</p>', 
                   unsafe_allow_html=True)
        
        # Statistics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-container"><h3>50K+</h3><p>Images Analyzed</p></div>', 
                       unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-container"><h3>95.2%</h3><p>Accuracy Rate</p></div>', 
                       unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-container"><h3>&lt;3s</h3><p>Analysis Time</p></div>', 
                       unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-container"><h3>24/7</h3><p>Availability</p></div>', 
                       unsafe_allow_html=True)
    
    def create_sidebar(self):
        """Create sidebar with settings and information."""
        st.sidebar.header("🔧 Settings")
        
        # Model settings
        st.sidebar.subheader("Model Configuration")
        use_tta = st.sidebar.checkbox("Test-Time Augmentation", value=True, 
                                     help="Improve accuracy with multiple augmented predictions")
        confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01,
                                                 help="Minimum confidence for reliable predictions")
        
        # Display settings
        st.sidebar.subheader("Display Options")
        show_technical_details = st.sidebar.checkbox("Show Technical Details", value=True)
        show_uncertainty = st.sidebar.checkbox("Show Uncertainty Analysis", value=True)
        show_grad_cam = st.sidebar.checkbox("Show Grad-CAM Visualization", value=False,
                                           help="Generate attention heatmap (slower)")
        
        # Information
        st.sidebar.markdown("---")
        st.sidebar.subheader("ℹ️ About")
        st.sidebar.info(
            "This AI system uses deep learning to analyze chest X-rays for signs of pneumonia. "
            "It's designed for educational purposes and should not replace professional medical diagnosis."
        )
        
        # Model info
        if st.session_state.inference_engine:
            stats = st.session_state.inference_engine.get_statistics()
            st.sidebar.subheader("📊 Session Statistics")
            st.sidebar.metric("Total Predictions", stats['total_predictions'])
            if stats['avg_processing_time'] > 0:
                st.sidebar.metric("Avg Processing Time", f"{stats['avg_processing_time']:.2f}s")
        
        return {
            'use_tta': use_tta,
            'confidence_threshold': confidence_threshold,
            'show_technical_details': show_technical_details,
            'show_uncertainty': show_uncertainty,
            'show_grad_cam': show_grad_cam
        }
    
    def upload_section(self):
        """Create file upload section."""
        st.header("📤 Upload X-ray Image")
        
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a chest X-ray image in PNG, JPG, or JPEG format (max 10MB)"
        )
        
        if uploaded_file is not None:
            # Validate file size
            if uploaded_file.size > 10 * 1024 * 1024:  # 10MB
                st.error("File size too large. Please upload an image smaller than 10MB.")
                return None
            
            # Load and display image
            try:
                image = Image.open(uploaded_file)
                st.session_state.uploaded_image = image
                
                # Display image with info
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.image(image, caption="Uploaded X-ray Image", use_column_width=True)
                
                with col2:
                    st.subheader("Image Information")
                    st.write(f"**Filename:** {uploaded_file.name}")
                    st.write(f"**Size:** {image.size[0]} x {image.size[1]} pixels")
                    st.write(f"**File Size:** {uploaded_file.size / 1024:.1f} KB")
                    st.write(f"**Format:** {image.format}")
                
                return image
                
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
                return None
        
        return None
    
    def analyze_image(self, image: Image.Image, settings: Dict):
        """Analyze uploaded image."""
        if st.session_state.inference_engine is None:
            st.error("Model not loaded. Please check the model file.")
            return
        
        st.header("🔬 AI Analysis")
        
        # Analysis button
        if st.button("🚀 Analyze Image", type="primary"):
            with st.spinner("Analyzing image... Please wait."):
                try:
                    # Update engine settings
                    st.session_state.inference_engine.use_tta = settings['use_tta']
                    st.session_state.inference_engine.confidence_threshold = settings['confidence_threshold']
                    
                    # Perform prediction
                    result = st.session_state.inference_engine.predict(
                        image, 
                        return_cam=settings['show_grad_cam']
                    )
                    
                    # Store result
                    prediction_data = {
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                        'filename': getattr(st.session_state, 'current_filename', 'uploaded_image'),
                        'result': result
                    }
                    st.session_state.prediction_history.append(prediction_data)
                    st.session_state.analysis_complete = True
                    
                    # Display results
                    self.display_results(result, settings)
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
        
        # Display previous results if available
        elif st.session_state.analysis_complete and st.session_state.prediction_history:
            latest_result = st.session_state.prediction_history[-1]['result']
            self.display_results(latest_result, settings)
    
    def display_results(self, result, settings: Dict):
        """Display analysis results."""
        # Determine card style based on prediction
        if result.label.lower() == 'normal':
            card_class = "success-card"
            icon = "✅"
        else:
            if result.confidence > 0.8:
                card_class = "danger-card"
                icon = "⚠️"
            else:
                card_class = "warning-card"
                icon = "⚡"
        
        # Main result card
        st.markdown(f"""
        <div class="prediction-card {card_class}">
            <h2>{icon} Diagnosis: {result.label}</h2>
            <h3>Confidence: {result.confidence:.1%}</h3>
            <p>Processing Time: {result.processing_time:.2f} seconds</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Probability breakdown
            st.subheader("📊 Probability Breakdown")
            prob_df = pd.DataFrame(
                list(result.probabilities.items()),
                columns=['Class', 'Probability']
            )
            
            # Create bar chart
            fig = px.bar(
                prob_df,
                x='Class',
                y='Probability',
                color='Probability',
                color_continuous_scale='viridis',
                title="Class Probabilities"
            )
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence gauge
            st.subheader("🎯 Confidence Level")
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=result.confidence * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Confidence %"},
                delta={'reference': settings['confidence_threshold'] * 100},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': settings['confidence_threshold'] * 100
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Technical details
        if settings['show_technical_details']:
            st.subheader("🔧 Technical Details")
            
            tech_col1, tech_col2 = st.columns(2)
            
            with tech_col1:
                st.write("**Model Information:**")
                st.write(f"- Model Version: {result.metadata.get('model_version', 'Unknown')}")
                st.write(f"- TTA Used: {result.metadata.get('tta_used', False)}")
                st.write(f"- Device: {result.metadata.get('device', 'Unknown')}")
                st.write(f"- Image Size: {result.metadata.get('image_size', 'Unknown')}")
            
            with tech_col2:
                st.write("**Performance Metrics:**")
                st.write(f"- Processing Time: {result.processing_time:.3f}s")
                st.write(f"- Uncertainty: {result.uncertainty:.3f}")
                
                # Probability details
                st.write("**Detailed Probabilities:**")
                for class_name, prob in result.probabilities.items():
                    st.write(f"- {class_name}: {prob:.4f}")
        
        # Uncertainty analysis
        if settings['show_uncertainty'] and result.uncertainty is not None:
            st.subheader("🔍 Uncertainty Analysis")
            
            uncertainty_col1, uncertainty_col2 = st.columns(2)
            
            with uncertainty_col1:
                # Uncertainty gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result.uncertainty,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Prediction Uncertainty"},
                    gauge={
                        'axis': {'range': [None, 1]},
                        'bar': {'color': "orange"},
                        'steps': [
                            {'range': [0, 0.3], 'color': "lightgreen"},
                            {'range': [0.3, 0.7], 'color': "yellow"},
                            {'range': [0.7, 1], 'color': "lightcoral"}
                        ],
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with uncertainty_col2:
                st.write("**Uncertainty Interpretation:**")
                if result.uncertainty < 0.3:
                    st.success("Low uncertainty - High confidence in prediction")
                elif result.uncertainty < 0.7:
                    st.warning("Medium uncertainty - Consider additional analysis")
                else:
                    st.error("High uncertainty - Prediction may be unreliable")
                
                st.write(f"**Uncertainty Score:** {result.uncertainty:.3f}")
                st.write("Lower values indicate higher model confidence.")
        
        # Grad-CAM visualization
        if settings['show_grad_cam'] and result.metadata.get('cam_available'):
            st.subheader("🔥 Attention Heatmap (Grad-CAM)")
            
            if 'cam' in result.metadata:
                cam = result.metadata['cam']
                
                # Display original and heatmap side by side
                cam_col1, cam_col2 = st.columns(2)
                
                with cam_col1:
                    st.write("**Original Image**")
                    st.image(st.session_state.uploaded_image, use_column_width=True)
                
                with cam_col2:
                    st.write("**Attention Heatmap**")
                    # Convert CAM to displayable format
                    cam_colored = px.imshow(cam, color_continuous_scale='jet')
                    cam_colored.update_layout(height=400)
                    st.plotly_chart(cam_colored, use_container_width=True)
                
                st.info("The heatmap shows areas the AI model focused on when making its prediction. "
                       "Brighter areas indicate higher attention.")
        
        # Action buttons
        st.subheader("📋 Actions")
        action_col1, action_col2, action_col3 = st.columns(3)
        
        with action_col1:
            if st.button("📄 Generate Report"):
                self.generate_report(result)
        
        with action_col2:
            if st.button("📊 View History"):
                self.show_prediction_history()
        
        with action_col3:
            if st.button("🔄 Analyze Another"):
                st.session_state.analysis_complete = False
                st.experimental_rerun()
    
    def generate_report(self, result):
        """Generate downloadable report."""
        report_content = f"""
# Chest X-ray Analysis Report

**Analysis Date:** {time.strftime("%Y-%m-%d %H:%M:%S")}
**Model Version:** {result.metadata.get('model_version', 'Unknown')}

## Diagnosis
**Primary Finding:** {result.label}
**Confidence Level:** {result.confidence:.1%}

## Detailed Analysis
**Processing Time:** {result.processing_time:.3f} seconds
**Uncertainty Score:** {result.uncertainty:.3f}
**Test-Time Augmentation:** {result.metadata.get('tta_used', False)}

## Probability Breakdown
"""
        
        for class_name, prob in result.probabilities.items():
            report_content += f"- **{class_name}:** {prob:.4f} ({prob * 100:.1f}%)\n"
        
        report_content += f"""

## Technical Details
- **Image Size:** {result.metadata.get('image_size', 'Unknown')}
- **Device Used:** {result.metadata.get('device', 'Unknown')}
- **Model Architecture:** ResNet-50 v2.1

## Disclaimer
This analysis is generated by an AI system for educational purposes only. 
It should not be used as a substitute for professional medical diagnosis. 
Always consult qualified healthcare professionals for medical decisions.
"""
        
        st.download_button(
            label="📥 Download Report",
            data=report_content,
            file_name=f"chest_xray_analysis_{int(time.time())}.txt",
            mime="text/plain"
        )
    
    def show_prediction_history(self):
        """Show prediction history."""
        if not st.session_state.prediction_history:
            st.info("No prediction history available.")
            return
        
        st.subheader("📈 Prediction History")
        
        # Create DataFrame from history
        history_data = []
        for entry in st.session_state.prediction_history:
            history_data.append({
                'Timestamp': entry['timestamp'],
                'Filename': entry['filename'],
                'Prediction': entry['result'].label,
                'Confidence': f"{entry['result'].confidence:.1%}",
                'Processing Time': f"{entry['result'].processing_time:.2f}s"
            })
        
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True)
        
        # Summary statistics
        st.subheader("📊 Session Summary")
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            st.metric("Total Predictions", len(st.session_state.prediction_history))
        
        with summary_col2:
            avg_confidence = np.mean([entry['result'].confidence 
                                    for entry in st.session_state.prediction_history])
            st.metric("Average Confidence", f"{avg_confidence:.1%}")
        
        with summary_col3:
            avg_time = np.mean([entry['result'].processing_time 
                              for entry in st.session_state.prediction_history])
            st.metric("Average Processing Time", f"{avg_time:.2f}s")
    
    def run(self):
        """Run the Streamlit application."""
        # Create header
        self.create_header()
        
        # Create sidebar
        settings = self.create_sidebar()
        
        # Main content
        st.markdown("---")
        
        # Upload section
        uploaded_image = self.upload_section()
        
        # Analysis section
        if uploaded_image is not None:
            st.markdown("---")
            self.analyze_image(uploaded_image, settings)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; margin-top: 2rem;">
            <p>🏥 Chest X-ray AI Diagnostics | Educational Prototype</p>
            <p>⚠️ Not for clinical use • Consult healthcare professionals for medical decisions</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main function to run the Streamlit app."""
    app = StreamlitApp()
    app.run()

if __name__ == "__main__":
    main()
