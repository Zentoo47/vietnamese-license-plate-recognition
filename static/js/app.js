// License Plate Detection App - JavaScript Client

const API_BASE = '/api';

// File upload handling
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

function initializeApp() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');

    if (!dropzone || !fileInput) return;

    // Click to upload
    dropzone.addEventListener('click', () => fileInput.click());

    // Drag and drop
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            uploadImage(files[0]);
        }
    });

    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            uploadImage(e.target.files[0]);
        }
    });
}

async function uploadImage(file) {
    // Validate
    if (!file.type.startsWith('image/')) {
        showError('Vui lòng chọn file ảnh!');
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        showError('File quá lớn! Vui lòng chọn file nhỏ hơn 10MB.');
        return;
    }

    // Show loading
    showLoading(true);
    hideError();

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE}/detect`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok && data.success) {
            displayResults(data);
        } else {
            showError(data.detail || 'Có lỗi xảy ra khi nhận diện!');
        }
    } catch (error) {
        showError('Lỗi kết nối: ' + error.message);
    } finally {
        showLoading(false);
    }
}

function displayResults(data) {
    document.getElementById('results').style.display = 'block';

    if (data.results && data.results.length > 0) {
        const result = data.results[0];

        // Update plate text
        const plateText = document.getElementById('plate-text');
        if (plateText) plateText.textContent = result.plate_text;

        // Update vehicle type
        const vehicleType = document.getElementById('vehicle-type');
        if (vehicleType) vehicleType.textContent = result.vehicle_type;

        // Update accuracy
        const accuracy = document.getElementById('accuracy');
        if (accuracy) accuracy.textContent = (result.confidence * 100).toFixed(1) + '%';

        // Update confidence bar
        const confidenceBar = document.getElementById('confidence-bar');
        if (confidenceBar) confidenceBar.style.width = (result.confidence * 100) + '%';

        // Update process time
        const processTime = document.getElementById('process-time');
        if (processTime) processTime.textContent = (data.total_time * 1000).toFixed(0) + 'ms';

        // Update total plates
        const totalPlates = document.getElementById('total-plates');
        if (totalPlates) totalPlates.textContent = data.total_plates;

        // Update image
        if (data.image_result) {
            const resultImage = document.getElementById('result-image');
            const imagePreview = document.getElementById('image-preview');
            if (resultImage) {
                resultImage.src = 'data:image/jpeg;base64,' + data.image_result;
            }
            if (imagePreview) imagePreview.style.display = 'block';
        }
    } else {
        const plateText = document.getElementById('plate-text');
        if (plateText) plateText.textContent = 'Không tìm thấy biển số';
    }

    // Hide upload area
    const uploadArea = document.getElementById('upload-area');
    if (uploadArea) uploadArea.style.display = 'none';
}

function resetForm() {
    document.getElementById('results').style.display = 'none';
    const uploadArea = document.getElementById('upload-area');
    if (uploadArea) uploadArea.style.display = 'block';

    const fileInput = document.getElementById('file-input');
    if (fileInput) fileInput.value = '';

    hideError();
}

function showLoading(show) {
    const loading = document.getElementById('loading');
    const uploadArea = document.getElementById('upload-area');

    if (loading) {
        loading.classList.toggle('active', show);
    }
    if (uploadArea && show) {
        uploadArea.style.display = 'none';
    }
}

function showError(message) {
    const errorMessage = document.getElementById('error-message');
    const errorText = document.getElementById('error-text');

    if (errorMessage) errorMessage.style.display = 'block';
    if (errorText) errorText.textContent = message;
}

function hideError() {
    const errorMessage = document.getElementById('error-message');
    if (errorMessage) errorMessage.style.display = 'none';
}

// Tab switching
function switchTab(tabName) {
    // Update buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.closest('.tab-btn').classList.add('active');

    // Update content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(tabName + '-tab').classList.add('active');

    // Load data if needed
    if (tabName === 'history') {
        loadHistory();
        loadStats();
    }
}

// Load history
async function loadHistory() {
    try {
        const response = await fetch(`${API_BASE}/history?limit=50`);
        const data = await response.json();

        const tbody = document.getElementById('history-body');
        if (!tbody) return;

        if (!data || data.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="text-center text-gray-500 py-8">Chưa có dữ liệu</td></tr>';
            return;
        }

        tbody.innerHTML = data.map(item => `
            <tr>
                <td>${item.id}</td>
                <td class="font-semibold">${item.plate_text}</td>
                <td><span class="badge badge-success">${item.vehicle_type || '-'}</span></td>
                <td>${(item.confidence * 100).toFixed(1)}%</td>
                <td>${item.processing_time ? (item.processing_time * 1000).toFixed(0) + 'ms' : '-'}</td>
                <td class="text-gray-500">${new Date(item.created_at).toLocaleString('vi-VN')}</td>
            </tr>
        `).join('');
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

// Load stats
async function loadStats() {
    try {
        const response = await fetch(`${API_BASE}/stats`);
        const data = await response.json();

        const totalEl = document.getElementById('stat-total');
        const avgConfEl = document.getElementById('stat-avg-conf');
        const avgTimeEl = document.getElementById('stat-avg-time');

        if (totalEl) totalEl.textContent = data.total_detections || 0;
        if (avgConfEl) avgConfEl.textContent = ((data.avg_confidence || 0) * 100).toFixed(1) + '%';
        if (avgTimeEl) avgTimeEl.textContent = ((data.avg_processing_time || 0) * 1000).toFixed(0) + 'ms';
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// Export functions for global access
window.uploadImage = uploadImage;
window.resetForm = resetForm;
window.switchTab = switchTab;
window.loadHistory = loadHistory;
window.loadStats = loadStats;
