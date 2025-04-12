"""
URL routes for the network traffic analyzer web application.
"""

import os
import time
from datetime import datetime
from flask import (
    Blueprint, render_template, request, redirect, url_for, 
    flash, current_app, jsonify, send_from_directory
)
from werkzeug.utils import secure_filename
import uuid

# Local imports
from ..utils.logger import setup_logger
from ..pipeline.inference_pipeline import InferencePipeline
from .app import init_model
from .forms import UploadForm

# Set up logger
logger = setup_logger(__name__)

# Create blueprint
main_bp = Blueprint('main', __name__)

# Initialize model
model_loader = init_model()
if model_loader:
    inference_pipeline = InferencePipeline()
else:
    inference_pipeline = None

@main_bp.route('/', methods=['GET'])
def index():
    """Route for the home page."""
    form = UploadForm()
    return render_template('index.html', form=form, model_status=model_loader is not None)

@main_bp.route('/upload', methods=['POST'])
def upload_file():
    """Route for file upload and processing."""
    form = UploadForm()
    
    if form.validate_on_submit():
        # Check if inference pipeline is available
        if not inference_pipeline:
            flash('Model not loaded. Please check logs for errors.', 'error')
            return redirect(url_for('main.index'))
        
        # Get the uploaded file
        file = form.file.data
        if file and allowed_file(file.filename):
            # Generate a unique filename
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Save the file
            file.save(file_path)
            logger.info(f"File saved: {file_path}")
            
            try:
                # Process the file
                result = inference_pipeline.process_pcap_file(file_path)
                
                # Generate visualizations
                vis_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'visualizations', unique_filename)
                os.makedirs(vis_dir, exist_ok=True)
                
                # Create a simple result with just one file
                simple_results = [result]
                vis_files = inference_pipeline.visualize_results(simple_results, vis_dir)
                
                # Add visualization paths to result
                result['visualizations'] = [os.path.basename(f) for f in vis_files]
                result['vis_dir'] = os.path.join('visualizations', unique_filename)
                
                # Store analysis timestamp
                result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                return render_template('results.html', result=result)
                
            except Exception as e:
                logger.error(f"Error processing file: {str(e)}")
                flash(f'Error processing file: {str(e)}', 'error')
                return redirect(url_for('main.index'))
        else:
            flash('Invalid file type. Please upload a PCAP file.', 'error')
            return redirect(url_for('main.index'))
    
    flash('Form validation failed.', 'error')
    return redirect(url_for('main.index'))

@main_bp.route('/dashboard', methods=['GET'])
def dashboard():
    """Route for the dashboard page."""
    # This is a placeholder for a more comprehensive dashboard
    # that could display historical analysis results, statistics, etc.
    return render_template('dashboard.html')

@main_bp.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for file analysis."""
    # Check if inference pipeline is available
    if not inference_pipeline:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate a unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save the file
        file.save(file_path)
        logger.info(f"File saved: {file_path}")
        
        try:
            # Process the file
            result = inference_pipeline.process_pcap_file(file_path)
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@main_bp.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Route to serve uploaded files."""
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)

@main_bp.route('/model_info', methods=['GET'])
def model_info():
    """Route to display model information."""
    if not model_loader:
        flash('Model not loaded. Please check logs for errors.', 'error')
        return redirect(url_for('main.index'))
    
    try:
        model_info = model_loader.get_model_info()
        return render_template('model_info.html', model_info=model_info)
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        flash(f'Error getting model info: {str(e)}', 'error')
        return redirect(url_for('main.index'))

@main_bp.route('/live_analysis', methods=['GET', 'POST'])
def live_analysis():
    """Route for live traffic analysis."""
    if request.method == 'POST':
        # Check if inference pipeline is available
        if not inference_pipeline:
            flash('Model not loaded. Please check logs for errors.', 'error')
            return redirect(url_for('main.live_analysis'))
        
        # Get the uploaded file
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(url_for('main.live_analysis'))
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(url_for('main.live_analysis'))
        
        if file and allowed_file(file.filename):
            # Generate a unique filename
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Save the file
            file.save(file_path)
            logger.info(f"File saved: {file_path}")
            
            try:
                # Get window size from form
                window_size = int(request.form.get('window_size', 1000))
                
                # Process the file with sliding window
                result = inference_pipeline.analyze_live_traffic(file_path, window_size)
                
                return render_template('live_results.html', result=result)
                
            except Exception as e:
                logger.error(f"Error processing file: {str(e)}")
                flash(f'Error processing file: {str(e)}', 'error')
                return redirect(url_for('main.live_analysis'))
        else:
            flash('Invalid file type. Please upload a PCAP file.', 'error')
            return redirect(url_for('main.live_analysis'))
    
    return render_template('live_analysis.html')

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    allowed_extensions = current_app.config.get('ALLOWED_EXTENSIONS', {'pcap', 'pcapng'})
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions