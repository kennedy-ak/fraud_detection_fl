// Dashboard JavaScript functionality

// Auto-refresh functionality
class DashboardManager {
    constructor() {
        this.refreshInterval = 30000; // 30 seconds
        this.charts = {};
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.startAutoRefresh();
        this.setupPolling(); // Changed from setupWebSocket
    }
    
    setupEventListeners() {
        // File upload drag and drop
        const uploadArea = document.querySelector('.file-upload-area');
        if (uploadArea) {
            uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
            uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
            uploadArea.addEventListener('drop', this.handleFileDrop.bind(this));
        }
        
        // Training session refresh buttons
        document.querySelectorAll('.refresh-session').forEach(btn => {
            btn.addEventListener('click', this.refreshTrainingSession.bind(this));
        });
    }
    
    handleDragOver(e) {
        e.preventDefault();
        e.target.classList.add('dragover');
    }
    
    handleDragLeave(e) {
        e.preventDefault();
        e.target.classList.remove('dragover');
    }
    
    handleFileDrop(e) {
        e.preventDefault();
        e.target.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const fileInput = document.querySelector('input[type="file"]');
            if (fileInput) {
                fileInput.files = files;
                this.displayFileInfo(files[0]);
            }
        }
    }
    
    displayFileInfo(file) {
        const info = document.createElement('div');
        info.className = 'alert alert-info mt-2';
        info.innerHTML = `
            <strong>File selected:</strong> ${file.name} 
            (${(file.size / 1024 / 1024).toFixed(2)} MB)
        `;
        
        const form = document.querySelector('#uploadForm');
        if (form) {
            const existing = form.querySelector('.file-info');
            if (existing) existing.remove();
            
            info.classList.add('file-info');
            form.appendChild(info);
        }
    }
    
    refreshTrainingSession(e) {
        const sessionId = e.target.dataset.sessionId;
        if (sessionId) {
            this.fetchTrainingProgress(sessionId);
        }
    }
    
    async fetchTrainingProgress(sessionId) {
        try {
            const response = await fetch(`/models/api/training-status/${sessionId}/`);
            const data = await response.json();
            this.updateTrainingUI(sessionId, data);
        } catch (error) {
            console.error('Error fetching training progress:', error);
        }
    }
    
    updateTrainingUI(sessionId, data) {
        const row = document.querySelector(`[data-session-id="${sessionId}"]`);
        if (row) {
            // Update status badge
            const statusCell = row.querySelector('.status-cell');
            if (statusCell) {
                statusCell.innerHTML = this.getStatusBadge(data.status);
            }
            
            // Update metrics
            const accuracyCell = row.querySelector('.accuracy-cell');
            if (accuracyCell && data.accuracy) {
                accuracyCell.textContent = data.accuracy.toFixed(4);
            }
            
            const lossCell = row.querySelector('.loss-cell');
            if (lossCell && data.loss) {
                lossCell.textContent = data.loss.toFixed(4);
            }
        }
    }
    
    getStatusBadge(status) {
        const badges = {
            'completed': '<span class="badge bg-success"><i class="fas fa-check"></i> Completed</span>',
            'training': '<span class="badge bg-primary"><i class="fas fa-spinner fa-spin"></i> Training</span>',
            'failed': '<span class="badge bg-danger"><i class="fas fa-times"></i> Failed</span>',
            'pending': '<span class="badge bg-secondary"><i class="fas fa-clock"></i> Pending</span>'
        };
        return badges[status] || badges['pending'];
    }
    
    startAutoRefresh() {
        setInterval(() => {
            if (document.visibilityState === 'visible') {
                this.refreshDashboardData();
            }
        }, this.refreshInterval);
    }
    
    async refreshDashboardData() {
        try {
            const response = await fetch('/accounts/api/training-progress/');
            const data = await response.json();
            this.updateDashboardCharts(data);
        } catch (error) {
            console.error('Error refreshing dashboard:', error);
        }
    }
    
    updateDashboardCharts(data) {
        // Update progress chart if it exists
        if (this.charts.progress) {
            const sessions = data.sessions.filter(s => s.accuracy !== null);
            const labels = sessions.map(s => new Date(s.created_at).toLocaleDateString());
            const accuracies = sessions.map(s => s.accuracy);
            
            this.charts.progress.data.labels = labels;
            this.charts.progress.data.datasets[0].data = accuracies;
            this.charts.progress.update();
        }
    }
    
    setupPolling() {
        // Use polling instead of WebSocket for real-time updates
        setInterval(() => {
            if (document.visibilityState === 'visible') {
                this.refreshDashboardData();
            }
        }, this.refreshInterval);
    }
    
    showAggregationNotification(data) {
        const notification = document.createElement('div');
        notification.className = 'alert alert-success alert-dismissible fade show position-fixed';
        notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; max-width: 300px;';
        notification.innerHTML = `
            <strong>Federated Learning Update!</strong><br>
            Global model accuracy: ${data.accuracy.toFixed(4)}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 10 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 10000);
    }
    
    // Utility methods
    formatBytes(bytes, decimals = 2) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }
    
    showLoadingSpinner(element) {
        element.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
    }
    
    hideLoadingSpinner(element, originalText) {
        element.innerHTML = originalText;
    }
}

// Form validation
class FormValidator {
    static validateDatasetUpload(form) {
        const fileInput = form.querySelector('input[type="file"]');
        const datasetName = form.querySelector('input[name="dataset_name"]');
        
        let isValid = true;
        const errors = [];
        
        // Check if file is selected
        if (!fileInput.files.length) {
            errors.push('Please select a CSV file to upload');
            isValid = false;
        } else {
            const file = fileInput.files[0];
            
            // Check file type
            if (!file.name.toLowerCase().endsWith('.csv')) {
                errors.push('Please upload a CSV file');
                isValid = false;
            }

            // Check file size (500MB limit)
            if (file.size > 500 * 1024 * 1024) {
                errors.push('File size must be less than 500MB');
                isValid = false;
            }
        }
        
        // Check dataset name
        if (!datasetName.value.trim()) {
            errors.push('Please provide a dataset name');
            isValid = false;
        }
        
        // Display errors
        if (!isValid) {
            FormValidator.showErrors(form, errors);
        }
        
        return isValid;
    }
    
    static showErrors(form, errors) {
        // Remove existing error alerts
        form.querySelectorAll('.alert-danger').forEach(alert => alert.remove());
        
        // Create new error alert
        const errorAlert = document.createElement('div');
        errorAlert.className = 'alert alert-danger';
        errorAlert.innerHTML = `
            <strong>Please fix the following errors:</strong>
            <ul class="mb-0">
                ${errors.map(error => `<li>${error}</li>`).join('')}
            </ul>
        `;
        
        form.insertBefore(errorAlert, form.firstChild);
    }
}

// Progress tracking
class ProgressTracker {
    constructor(sessionId) {
        this.sessionId = sessionId;
        this.pollInterval = 2000; // 2 seconds
        this.maxPolls = 300; // 10 minutes max
        this.pollCount = 0;
    }
    
    start() {
        this.poll();
    }
    
    async poll() {
        if (this.pollCount >= this.maxPolls) {
            console.warn('Maximum polling attempts reached');
            return;
        }
        
        try {
            const response = await fetch(`/models/api/training-status/${this.sessionId}/`);
            const data = await response.json();
            
            this.updateProgress(data);
            
            if (data.status === 'completed' || data.status === 'failed') {
                this.onComplete(data);
            } else {
                this.pollCount++;
                setTimeout(() => this.poll(), this.pollInterval);
            }
        } catch (error) {
            console.error('Polling error:', error);
            setTimeout(() => this.poll(), this.pollInterval * 2); // Back off on error
        }
    }
    
    updateProgress(data) {
        // Update progress bar
        const progressBar = document.querySelector('.training-progress');
        if (progressBar) {
            const percentage = this.calculateProgress(data);
            progressBar.style.width = `${percentage}%`;
            progressBar.textContent = `${percentage}%`;
        }
        
        // Update status
        const statusElement = document.querySelector('.training-status');
        if (statusElement) {
            statusElement.innerHTML = this.getStatusHTML(data.status);
        }
        
        // Update metrics
        if (data.accuracy) {
            const accuracyElement = document.querySelector('.current-accuracy');
            if (accuracyElement) {
                accuracyElement.textContent = data.accuracy.toFixed(4);
            }
        }
    }
    
    calculateProgress(data) {
        switch (data.status) {
            case 'pending': return 0;
            case 'training': return 50;
            case 'completed': return 100;
            case 'failed': return 0;
            default: return 0;
        }
    }
    
    getStatusHTML(status) {
        const statusConfig = {
            'pending': { class: 'secondary', icon: 'clock', text: 'Pending' },
            'training': { class: 'primary', icon: 'spinner fa-spin', text: 'Training' },
            'completed': { class: 'success', icon: 'check', text: 'Completed' },
            'failed': { class: 'danger', icon: 'times', text: 'Failed' }
        };
        
        const config = statusConfig[status] || statusConfig['pending'];
        return `<span class="badge bg-${config.class}">
            <i class="fas fa-${config.icon}"></i> ${config.text}
        </span>`;
    }
    
    onComplete(data) {
        console.log('Training completed:', data);
        
        // Show completion notification
        const notification = document.createElement('div');
        notification.className = 'alert alert-success alert-dismissible fade show position-fixed';
        notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; max-width: 400px;';
        notification.innerHTML = `
            <strong>Training ${data.status}!</strong><br>
            ${data.status === 'completed' ? `Final accuracy: ${data.accuracy ? data.accuracy.toFixed(4) : 'N/A'}` : ''}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(notification);
        
        // Optionally redirect to dashboard after a delay
        if (data.status === 'completed') {
            setTimeout(() => {
                window.location.href = '/accounts/';
            }, 5000);
        }
    }
}

// Real-time Training Status Manager
class TrainingStatusManager {
    constructor(sessionId) {
        this.sessionId = sessionId;
        this.pollInterval = 3000; // 3 seconds
        this.isPolling = false;
        this.maxRetries = 100; // Maximum number of polling attempts
        this.retryCount = 0;
    }
    
    start() {
        if (!this.isPolling) {
            this.isPolling = true;
            this.poll();
        }
    }
    
    stop() {
        this.isPolling = false;
    }
    
    async poll() {
        if (!this.isPolling || this.retryCount >= this.maxRetries) {
            return;
        }
        
        try {
            const response = await fetch(`/models/api/training-status/${this.sessionId}/`);
            
            if (response.ok) {
                const data = await response.json();
                this.handleStatusUpdate(data);
                
                // Continue polling if still training
                if (data.status === 'training' && this.isPolling) {
                    this.retryCount++;
                    setTimeout(() => this.poll(), this.pollInterval);
                } else if (data.status === 'completed' || data.status === 'failed') {
                    this.handleCompletion(data);
                    this.stop();
                }
            } else {
                console.warn('Failed to fetch training status:', response.status);
                this.retryCount++;
                if (this.isPolling) {
                    setTimeout(() => this.poll(), this.pollInterval * 2); // Back off on error
                }
            }
        } catch (error) {
            console.error('Error polling training status:', error);
            this.retryCount++;
            if (this.isPolling) {
                setTimeout(() => this.poll(), this.pollInterval * 2); // Back off on error
            }
        }
    }
    
    handleStatusUpdate(data) {
        // Trigger custom event for status update
        const event = new CustomEvent('trainingStatusUpdate', {
            detail: data
        });
        document.dispatchEvent(event);
    }
    
    handleCompletion(data) {
        // Trigger custom event for completion
        const event = new CustomEvent('trainingCompleted', {
            detail: data
        });
        document.dispatchEvent(event);
    }
}

// Enhanced Notification System
class NotificationManager {
    constructor() {
        this.notifications = [];
        this.maxNotifications = 3;
    }
    
    show(message, type = 'info', duration = 5000, persistent = false) {
        const notification = this.createNotification(message, type, duration, persistent);
        this.addNotification(notification);
        
        if (!persistent && duration > 0) {
            setTimeout(() => {
                this.removeNotification(notification);
            }, duration);
        }
        
        return notification;
    }
    
    createNotification(message, type, duration, persistent) {
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        notification.style.cssText = `
            top: ${20 + (this.notifications.length * 80)}px; 
            right: 20px; 
            z-index: 9999; 
            max-width: 400px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        `;
        
        const closeButton = persistent ? '' : '<button type="button" class="btn-close" data-bs-dismiss="alert"></button>';
        
        notification.innerHTML = `
            ${message}
            ${closeButton}
        `;
        
        // Add close event listener
        if (!persistent) {
            const closeBtn = notification.querySelector('.btn-close');
            if (closeBtn) {
                closeBtn.addEventListener('click', () => {
                    this.removeNotification(notification);
                });
            }
        }
        
        return notification;
    }
    
    addNotification(notification) {
        document.body.appendChild(notification);
        this.notifications.push(notification);
        
        // Remove oldest notification if we exceed the limit
        if (this.notifications.length > this.maxNotifications) {
            this.removeNotification(this.notifications[0]);
        }
    }
    
    removeNotification(notification) {
        const index = this.notifications.indexOf(notification);
        if (index > -1) {
            this.notifications.splice(index, 1);
            if (notification.parentNode) {
                notification.remove();
            }
            
            // Reposition remaining notifications
            this.repositionNotifications();
        }
    }
    
    repositionNotifications() {
        this.notifications.forEach((notification, index) => {
            notification.style.top = `${20 + (index * 80)}px`;
        });
    }
    
    clear() {
        this.notifications.forEach(notification => {
            if (notification.parentNode) {
                notification.remove();
            }
        });
        this.notifications = [];
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize dashboard manager
    window.dashboardManager = new DashboardManager();
    
    // Initialize notification manager
    window.notificationManager = new NotificationManager();
    
    // Setup form validation
    const uploadForm = document.querySelector('#uploadForm');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            if (!FormValidator.validateDatasetUpload(this)) {
                e.preventDefault();
            }
        });
    }
    
    // Initialize progress tracking if on training page
    const sessionId = document.querySelector('[data-session-id]')?.dataset.sessionId;
    if (sessionId) {
        window.progressTracker = new ProgressTracker(sessionId);
        window.progressTracker.start();
        
        // Also initialize training status manager for real-time updates
        window.trainingStatusManager = new TrainingStatusManager(sessionId);
        window.trainingStatusManager.start();
    }
    
    // Setup tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Setup popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function(popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Listen for training status updates
    document.addEventListener('trainingStatusUpdate', function(event) {
        const data = event.detail;
        console.log('Training status updated:', data);
        
        // Update any dashboard elements that need real-time updates
        if (window.dashboardManager) {
            window.dashboardManager.updateTrainingUI(data.session_id, data);
        }
    });
    
    // Listen for training completion
    document.addEventListener('trainingCompleted', function(event) {
        const data = event.detail;
        console.log('Training completed:', data);
        
        if (data.status === 'completed') {
            window.notificationManager.show(
                `<strong>Training Completed!</strong><br>Final accuracy: ${data.accuracy ? data.accuracy.toFixed(4) : 'N/A'}`,
                'success',
                8000
            );
        } else if (data.status === 'failed') {
            window.notificationManager.show(
                `<strong>Training Failed!</strong><br>Please check the logs for more information.`,
                'danger',
                8000
            );
        }
    });
});

// Export for use in other scripts
window.DashboardManager = DashboardManager;
window.FormValidator = FormValidator;
window.ProgressTracker = ProgressTracker;
window.TrainingStatusManager = TrainingStatusManager;
window.NotificationManager = NotificationManager;
            