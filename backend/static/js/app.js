// backend/static/js/app.js

class XRayUploader {
    constructor() {
        this.initElements();
        this.initState();
        this.init();
    }

    initElements() {
        // Form elements
        this.fileInput = document.getElementById('file');
        this.form = document.getElementById('upload-form');
        this.dropzone = document.getElementById('dropzone');
        
        // Info elements
        this.fileInfo = document.getElementById('file-info');
        this.fileName = document.getElementById('file-name');
        this.fileSize = document.getElementById('file-size');
        this.clearBtn = document.getElementById('clear-btn');
        
        // Button elements
        this.submitBtn = document.getElementById('submit-btn');
        this.resetBtn = document.getElementById('reset-btn');
        this.btnText = this.submitBtn.querySelector('.btn-text');
        this.btnLoader = this.submitBtn.querySelector('.btn-loader');
        
        // Preview elements
        this.preview = document.getElementById('preview');
        this.previewImg = document.getElementById('preview-img');
        
        // Result elements
        this.resultBox = document.getElementById('result');
        this.labelEl = document.getElementById('pred-label');
        this.barEl = document.getElementById('pred-bar');
        this.probEl = document.getElementById('pred-prob');
        
        // Progress elements
        this.uploadProgress = document.getElementById('upload-progress');
        this.uploadBar = document.getElementById('upload-bar');
        this.uploadPercent = document.getElementById('upload-percent');
        
        // Action buttons
        this.downloadBtn = document.getElementById('download-report');
        this.analyzeAnotherBtn = document.getElementById('analyze-another');
        
        // Notification container
        this.notificationContainer = document.getElementById('notification-container');
    }

    initState() {
        this.currentFile = null;
        this.isUploading = false;
        this.uploadStartTime = null;
    }

    init() {
        this.setupDragAndDrop();
        this.setupFileInput();
        this.setupFormSubmission();
        this.setupButtons();
        this.setupKeyboardShortcuts();
    }

    setupDragAndDrop() {
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            this.dropzone.addEventListener(eventName, this.preventDefaults, false);
            document.body.addEventListener(eventName, this.preventDefaults, false);
        });

        // Highlight drop area when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            this.dropzone.addEventListener(eventName, () => this.highlight(), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            this.dropzone.addEventListener(eventName, () => this.unhighlight(), false);
        });

        // Handle dropped files
        this.dropzone.addEventListener('drop', (e) => this.handleDrop(e), false);
        this.dropzone.addEventListener('click', () => {
            if (!this.isUploading) {
                this.fileInput.click();
            }
        });
    }

    setupFileInput() {
        this.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelect(e.target.files[0]);
            }
        });
    }

    setupFormSubmission() {
        this.form.addEventListener('submit', (e) => this.handleSubmit(e));
    }

    setupButtons() {
        this.clearBtn.addEventListener('click', () => this.clearFile());
        this.resetBtn.addEventListener('click', () => this.reset());
        this.downloadBtn.addEventListener('click', () => this.downloadReport());
        this.analyzeAnotherBtn.addEventListener('click', () => this.reset());
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.hideAllNotifications();
            }
            if (e.ctrlKey && e.key === 'r') {
                e.preventDefault();
                this.reset();
            }
        });
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    highlight() {
        if (!this.isUploading) {
            this.dropzone.classList.add('hover');
        }
    }

    unhighlight() {
        this.dropzone.classList.remove('hover');
    }

    handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length > 0 && !this.isUploading) {
            this.fileInput.files = files;
            this.handleFileSelect(files[0]);
        }
    }

    handleFileSelect(file) {
        // Validate file type
        if (!this.isValidImageFile(file)) {
            this.showNotification('Please select a valid image file (PNG, JPG, JPEG)', 'error');
            return;
        }

        // Validate file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            this.showNotification('File size must be less than 10MB', 'error');
            return;
        }

        this.currentFile = file;
        this.showFileInfo(file);
        this.showPreview(file);
        this.hideResult();
        this.dropzone.classList.add('has-file');
    }

    isValidImageFile(file) {
        const validTypes = ['image/png', 'image/jpeg', 'image/jpg'];
        return validTypes.includes(file.type);
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    showFileInfo(file) {
        this.fileName.textContent = file.name;
        this.fileSize.textContent = this.formatFileSize(file.size);
        this.fileInfo.classList.add('show');
    }

    hideFileInfo() {
        this.fileInfo.classList.remove('show');
    }

    showPreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            this.previewImg.src = e.target.result;
            this.preview.classList.add('show');
        };
        reader.readAsDataURL(file);
    }

    hidePreview() {
        this.preview.classList.remove('show');
        this.previewImg.src = '';
    }

    clearFile() {
        this.fileInput.value = '';
        this.currentFile = null;
        this.hideFileInfo();
        this.hidePreview();
        this.dropzone.classList.remove('has-file');
    }

    async handleSubmit(e) {
        e.preventDefault();
        
        if (!this.currentFile) {
            this.showNotification('Please select an image file first', 'warning');
            return;
        }

        if (this.isUploading) {
            return;
        }

        const formData = new FormData();
        formData.append('file', this.currentFile);

        this.setLoadingState(true);
        this.uploadStartTime = Date.now();

        try {
            const response = await this.uploadWithProgress('/predict', formData);
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.showResult(data);
            
            const processingTime = Date.now() - this.uploadStartTime;
            this.trackAnalytics('prediction_complete', {
                fileSize: this.currentFile.size,
                fileType: this.currentFile.type,
                processingTime,
                result: data.label,
                confidence: data.prob
            });

        } catch (error) {
            console.error('Prediction error:', error);
            this.showNotification(
                error.message || 'Failed to analyze image. Please try again.',
                'error'
            );
        } finally {
            this.setLoadingState(false);
            this.hideUploadProgress();
        }
    }

    async uploadWithProgress(url, formData) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            
            xhr.upload.onprogress = (e) => {
                if (e.lengthComputable) {
                    const percentComplete = Math.round((e.loaded / e.total) * 100);
                    this.updateUploadProgress(percentComplete);
                }
            };
            
            xhr.onload = () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    resolve({
                        ok: true,
                        status: xhr.status,
                        json: () => Promise.resolve(JSON.parse(xhr.responseText))
                    });
                } else {
                    resolve({
                        ok: false,
                        status: xhr.status,
                        json: () => Promise.resolve(JSON.parse(xhr.responseText))
                    });
                }
            };
            
            xhr.onerror = () => reject(new Error('Network error'));
            
            xhr.open('POST', url);
            xhr.send(formData);
            
            this.showUploadProgress();
        });
    }

    showUploadProgress() {
        this.uploadProgress.classList.add('show');
        this.updateUploadProgress(0);
    }

    hideUploadProgress() {
        this.uploadProgress.classList.remove('show');
    }

    updateUploadProgress(percent) {
        const bar = this.uploadBar.querySelector('.bar-fill');
        if (bar) {
            bar.style.width = `${percent}%`;
        }
        this.uploadPercent.textContent = `${percent}%`;
    }

    setLoadingState(loading) {
        this.isUploading = loading;
        this.submitBtn.disabled = loading;
        
        if (loading) {
            this.submitBtn.classList.add('loading');
            this.btnText.textContent = 'Analyzing...';
        } else {
            this.submitBtn.classList.remove('loading');
            this.btnText.textContent = 'Analyze Image';
        }
        
        // Disable other interactive elements
        this.dropzone.style.pointerEvents = loading ? 'none' : '';
        this.clearBtn.disabled = loading;
        this.resetBtn.disabled = loading;
    }

    showResult(data) {
        const label = data.label || 'Unknown';
        const confidence = Math.round((data.prob || 0) * 100);
        
        // Update result elements
        this.labelEl.textContent = label;
        this.probEl.textContent = `${confidence}%`;
        
        // Update progress bar
        const barFill = this.barEl.querySelector('.bar-fill');
        if (barFill) {
            setTimeout(() => {
                barFill.style.width = `${confidence}%`;
            }, 100);
        }
        this.barEl.setAttribute('aria-valuenow', confidence.toString());
        
        // Set color based on prediction
        const isPneumonia = label.toLowerCase().includes('pneumonia');
        const isLowConfidence = confidence < 70;
        
        let barClass = 'bar';
        if (isLowConfidence) {
            barClass += ' warning';
        } else if (isPneumonia) {
            barClass += ' danger';
        } else {
            barClass += ' success';
        }
        this.barEl.className = barClass;
        
        // Show result with animation
        this.resultBox.classList.add('show');
        
        // Show success notification
        this.showNotification(
            `Analysis complete: ${label} (${confidence}% confidence)`,
            'success'
        );

        // Log for debugging
        console.log('Prediction result:', { label, confidence, timestamp: new Date().toISOString() });
    }

    hideResult() {
        this.resultBox.classList.remove('show');
    }

    downloadReport() {
        if (!this.currentFile) return;
        
        const reportData = {
            fileName: this.currentFile.name,
            fileSize: this.formatFileSize(this.currentFile.size),
            analysisDate: new Date().toLocaleString(),
            prediction: this.labelEl.textContent,
            confidence: this.probEl.textContent,
            modelVersion: '1.0.0'
        };
        
        const reportText = `
Chest X-ray Analysis Report
==========================

File Information:
- Filename: ${reportData.fileName}
- File Size: ${reportData.fileSize}
- Analysis Date: ${reportData.analysisDate}

Analysis Results:
- Prediction: ${reportData.prediction}
- Confidence: ${reportData.confidence}
- Model Version: ${reportData.modelVersion}

Disclaimer:
This analysis is for educational and research purposes only.
Please consult with qualified medical professionals for clinical decisions.
        `.trim();
        
        const blob = new Blob([reportText], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `chest-xray-report-${Date.now()}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        this.showNotification('Report downloaded successfully', 'success');
    }

    reset() {
        this.clearFile();
        this.hideResult();
        this.hideUploadProgress();
        this.setLoadingState(false);
        this.showNotification('Interface reset', 'success');
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        this.notificationContainer.appendChild(notification);
        
        // Trigger animation
        setTimeout(() => {
            notification.classList.add('show');
        }, 50);
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            this.hideNotification(notification);
        }, 5000);
        
        // Click to dismiss
        notification.addEventListener('click', () => {
            this.hideNotification(notification);
        });
    }

    hideNotification(notification) {
        notification.classList.remove('show');
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }

    hideAllNotifications() {
        const notifications = this.notificationContainer.querySelectorAll('.notification');
        notifications.forEach(notification => {
            this.hideNotification(notification);
        });
    }

    trackAnalytics(event, data) {
        // Log to console for development
        console.log('Analytics Event:', event, data);
        
        // Add your analytics service here
        // Example: gtag('event', event, data);
        // Example: analytics.track(event, data);
    }
}

// Utility functions
class ImageUtils {
    static async resizeImage(file, maxWidth = 800, maxHeight = 800, quality = 0.9) {
        return new Promise((resolve) => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();

            img.onload = () => {
                const { width, height } = ImageUtils.calculateDimensions(
                    img.width, img.height, maxWidth, maxHeight
                );

                canvas.width = width;
                canvas.height = height;

                // Enable image smoothing
                ctx.imageSmoothingEnabled = true;
                ctx.imageSmoothingQuality = 'high';
                
                ctx.drawImage(img, 0, 0, width, height);
                
                canvas.toBlob(resolve, file.type, quality);
            };

            img.src = URL.createObjectURL(file);
        });
    }

    static calculateDimensions(srcWidth, srcHeight, maxWidth, maxHeight) {
        let { width, height } = { width: srcWidth, height: srcHeight };

        if (width > height) {
            if (width > maxWidth) {
                height = (height * maxWidth) / width;
                width = maxWidth;
            }
        } else {
            if (height > maxHeight) {
                width = (width * maxHeight) / height;
                height = maxHeight;
            }
        }

        return { width: Math.round(width), height: Math.round(height) };
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const uploader = new XRayUploader();
    
    // Make it globally accessible for debugging
    window.xrayUploader = uploader;
    
    console.log('Chest X-ray Uploader initialized successfully');
});

// Export for testing (if using modules)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { XRayUploader, ImageUtils };
}
