"""
Form definitions for the network traffic analyzer web application.
"""

from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import StringField, SubmitField, SelectField, IntegerField
from wtforms.validators import DataRequired, Optional, NumberRange

class UploadForm(FlaskForm):
    """Form for uploading PCAP files."""
    file = FileField('Upload PCAP File', 
                    validators=[
                        FileRequired(),
                        FileAllowed(['pcap', 'pcapng'], 'PCAP files only!')
                    ])
    submit = SubmitField('Analyze')

class LiveAnalysisForm(FlaskForm):
    """Form for live traffic analysis."""
    file = FileField('Upload PCAP File',
                    validators=[
                        FileRequired(),
                        FileAllowed(['pcap', 'pcapng'], 'PCAP files only!')
                    ])
    window_size = IntegerField('Window Size (packets)',
                             validators=[
                                 DataRequired(),
                                 NumberRange(min=100, max=10000)
                             ],
                             default=1000)
    submit = SubmitField('Start Analysis')

class BatchAnalysisForm(FlaskForm):
    """Form for batch analysis of multiple files."""
    files = FileField('Upload PCAP Files',
                     validators=[
                         FileRequired(),
                         FileAllowed(['pcap', 'pcapng'], 'PCAP files only!')
                     ])
    submit = SubmitField('Analyze Batch')