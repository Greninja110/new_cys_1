/* upload.js */
// static/js/upload.js
document.addEventListener('DOMContentLoaded', function() {
    // File upload progress simulation
    const fileUpload = document.getElementById('file_upload');
    const uploadForm = fileUpload ? fileUpload.closest('form') : null;
    const progressBar = document.getElementById('upload-progress');
    
    if (uploadForm && progressBar) {
        uploadForm.addEventListener('submit', function(e) {
            const file = fileUpload.files[0];
            if (file) {
                // Show progress bar
                progressBar.classList.remove('d-none');
                const bar = progressBar.querySelector('.progress-bar');
                
                // Simulate upload progress
                let progress = 0;
                const interval = setInterval(function() {
                    progress += 5;
                    bar.style.width = progress + '%';
                    bar.setAttribute('aria-valuenow', progress);
                    
                    if (progress >= 100) {
                        clearInterval(interval);
                    }
                }, 100);
            }
        });
    }
    
    // File name display
    if (fileUpload) {
        fileUpload.addEventListener('change', function() {
            const fileName = this.files[0]?.name || 'Choose file';
            const label = document.querySelector('.custom-file-label');
            if (label) {
                label.textContent = fileName;
            }
        });
    }
});

