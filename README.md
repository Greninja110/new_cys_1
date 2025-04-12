# Network Traffic Analyzer

A machine learning-based tool to detect DDoS attacks in network traffic.

## Overview

This project provides a complete pipeline for analyzing network traffic to detect DDoS attacks. It includes:

1. Processing of PCAP and parquet network traffic files
2. Feature extraction for network traffic analysis
3. Machine learning model for DDoS detection
4. Web interface for uploading and analyzing network captures

## System Requirements

- Python 3.12.8
- Windows 11
- 32GB DDR4 RAM
- NVIDIA RTX 3050 4GB GPU

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/network_traffic_analyzer.git
cd network_traffic_analyzer
```

2. Create a virtual environment:
```
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Set up data directories:
```
mkdir -p data/raw/CIC-IDS2017
mkdir -p data/raw/CIC-DDoS2019
mkdir -p data/processed/training
mkdir -p data/processed/testing
mkdir -p data/uploads
mkdir -p data/models
```

5. Place your dataset files:
   - Put CIC-IDS2017 files in `data/raw/CIC-IDS2017/`
   - Put CIC-DDoS2019 files in `data/raw/CIC-DDoS2019/`

## Usage

### Training the Model

To train the DDoS detection model:

```
python main.py
```

This will:
1. Process the raw dataset files
2. Extract relevant network traffic features
3. Train a machine learning model
4. Save the trained model to `data/models/`

### Running the Web Application

To start the web interface:

```
python run.py
```

Then navigate to http://127.0.0.1:5000 in your browser to access the application.

## Project Structure

```
network_traffic_analyzer/
│
├── data/                               # Data storage directory
│   ├── raw/                            # Original parquet/pcap files
│   ├── processed/                      # Converted and preprocessed CSV files
│   ├── uploads/                        # User uploaded PCAP files storage
│   └── models/                         # Trained model files
│
├── src/                                # Source code
│   ├── data/                           # Data processing modules
│   ├── features/                       # Feature engineering
│   ├── models/                         # ML model implementations
│   ├── utils/                          # Utility functions
│   ├── pipeline/                       # End-to-end pipelines
│   └── web/                            # Web application components
│
├── templates/                          # HTML templates for Flask
├── static/                             # Static web assets
├── notebooks/                          # Jupyter notebooks
├── tests/                              # Unit and integration tests
├── logs/                               # Log files directory
├── configs/                            # Configuration files
│
├── requirements.txt                    # Project dependencies
├── setup.py                            # Package setup file
├── .gitignore                          # Git ignore file
├── README.md                           # Project documentation
├── main.py                             # Entry point for training the model
└── run.py                              # Entry point for running the Flask app
```

## Datasets

This project uses the following datasets:

1. CIC-IDS2017: Contains both normal traffic and various attack types
2. CIC-DDoS2019: Contains various DDoS attack patterns

## Features

- Convert PCAP files to structured CSV data
- Extract relevant network traffic features
- Classify traffic as normal or DDoS attack
- Visualize analysis results
- Web interface for easy usage

## License

MIT License

## Contributors

Your Name - youremail@example.com