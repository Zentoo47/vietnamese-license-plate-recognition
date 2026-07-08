// License Plate Detection App — JavaScript Client

document.addEventListener('DOMContentLoaded', () => {
    // ---- Image dropzone ----
    const fileInput = document.getElementById('file-input');
    const dropzone = document.getElementById('dropzone');

    if (dropzone && fileInput) {
        dropzone.addEventListener('click', () => fileInput.click());

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
                if (files[0].type.startsWith('image/')) {
                    handleFile(files[0]);
                } else {
                    showError('Vui lòng chọn file ảnh cho tab này!');
                }
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });
    }

    // ---- Video dropzone ----
    const videoFileInput = document.getElementById('video-file-input');
    const videoDropzone = document.getElementById('video-dropzone');

    if (videoDropzone && videoFileInput) {
        videoDropzone.addEventListener('click', (event) => {
            if (event.target.tagName !== 'BUTTON') {
                videoFileInput.click();
            }
        });

        videoDropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            videoDropzone.classList.add('dragover');
        });

        videoDropzone.addEventListener('dragleave', () => {
            videoDropzone.classList.remove('dragover');
        });

        videoDropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            videoDropzone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                if (files[0].type.startsWith('video/') || files[0].type === 'application/octet-stream') {
                    handleVideoFile(files[0]);
                } else {
                    showVideoError('Vui lòng chọn file video cho tab này!');
                }
            }
        });

        videoFileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleVideoFile(e.target.files[0]);
            }
        });
    }
});

// ============================================================
// Image Upload
// ============================================================

async function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showError('Vui lòng chọn file ảnh!');
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        showError('File quá lớn! Vui lòng chọn file nhỏ hơn 10MB.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    document.getElementById('upload-area').style.display = 'none';
    document.getElementById('loading').classList.add('active');
    document.getElementById('error-message').style.display = 'none';
    document.getElementById('results').style.display = 'none';

    try {
        const response = await fetch('/api/detect', {
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
        document.getElementById('loading').classList.remove('active');
    }
}

function displayResults(data) {
    document.getElementById('results').style.display = 'block';

    if (data.results && data.results.length > 0) {
        const result = data.results[0];

        document.getElementById('plate-text').textContent = result.plate_text;
        document.getElementById('vehicle-type').textContent = result.vehicle_type;
        document.getElementById('accuracy').textContent = (result.confidence * 100).toFixed(1) + '%';
        document.getElementById('confidence-bar').style.width = (result.confidence * 100) + '%';
        document.getElementById('process-time').textContent = (data.total_time * 1000).toFixed(0) + 'ms';
        document.getElementById('total-plates').textContent = data.total_plates;

        if (data.image_result) {
            document.getElementById('result-image').src = 'data:image/jpeg;base64,' + data.image_result;
            document.getElementById('image-preview').style.display = 'block';
        } else {
            document.getElementById('image-preview').style.display = 'none';
        }
    } else {
        document.getElementById('plate-text').textContent = 'Không tìm thấy biển số';
        document.getElementById('vehicle-type').textContent = '-';
        document.getElementById('image-preview').style.display = 'none';
    }
}

function showError(message) {
    document.getElementById('error-text').textContent = message;
    document.getElementById('error-message').style.display = 'block';
    document.getElementById('upload-area').style.display = 'block';
    document.getElementById('results').style.display = 'none';
}

function resetForm() {
    document.getElementById('results').style.display = 'none';
    document.getElementById('upload-area').style.display = 'block';
    document.getElementById('file-input').value = '';
    document.getElementById('error-message').style.display = 'none';
    document.getElementById('image-preview').style.display = 'none';
}

// ============================================================
// Video Upload
// ============================================================

async function handleVideoFile(file) {
    const allowedExtensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm'];
    const fileName = (file.name || '').toLowerCase();
    const hasAllowedExtension = allowedExtensions.some(ext => fileName.endsWith(ext));

    if (!file.type.startsWith('video/') && file.type !== 'application/octet-stream' && !hasAllowedExtension) {
        showVideoError('Vui lòng chọn file video!');
        return;
    }

    if (file.size > 500 * 1024 * 1024) {
        showVideoError('File quá lớn! Vui lòng chọn file nhỏ hơn 500MB.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    document.getElementById('video-upload-area').style.display = 'none';
    document.getElementById('video-results').style.display = 'none';
    document.getElementById('video-error-message').style.display = 'none';
    document.getElementById('video-loading-title').textContent = 'Đang tải video lên...';
    document.getElementById('video-loading-detail').textContent = 'Vui lòng chờ, video lớn có thể mất vài phút.';
    document.getElementById('video-loading').classList.add('active');

    try {
        const response = await fetch('/api/detect-video', {
            method: 'POST',
            body: formData
        });

        document.getElementById('video-loading-title').textContent = 'Đang xử lý nhận diện video...';
        document.getElementById('video-loading-detail').textContent = 'Hệ thống đang vẽ kết quả lên video.';

        const data = await response.json();

        if (response.ok && data.success) {
            displayVideoResults(data);
        } else {
            showVideoError(data.detail || 'Có lỗi xảy ra khi nhận diện video!');
        }
    } catch (error) {
        showVideoError('Lỗi kết nối: ' + error.message);
    } finally {
        document.getElementById('video-loading').classList.remove('active');
    }
}

function displayVideoResults(data) {
    document.getElementById('video-results').style.display = 'block';
    document.getElementById('video-total-plates').textContent = data.total_plates || 0;
    document.getElementById('video-process-time').textContent = Number(data.total_time || 0).toFixed(2) + ' s';

    if (data.video_result) {
        const player = document.getElementById('result-video-player');
        player.src = data.video_result + '?t=' + Date.now();
        player.load();
    }

    const uniquePlateList = document.getElementById('unique-plate-list');
    uniquePlateList.innerHTML = '';

    if (data.results && data.results.length > 0) {
        data.results.forEach(result => {
            const plateItem = document.createElement('div');
            plateItem.className = 'bg-blue-50 p-3 rounded-lg shadow-sm text-center';
            plateItem.innerHTML = `
                <p class="font-bold text-lg text-blue-800">${escapeHtml(result.plate_text)}</p>
                <p class="text-sm text-gray-600">Độ chính xác: ${(result.confidence * 100).toFixed(1)}%</p>
                <p class="text-xs text-gray-500">Loại xe: ${escapeHtml(result.vehicle_type)}</p>
            `;
            uniquePlateList.appendChild(plateItem);
        });
    } else {
        uniquePlateList.innerHTML = '<p class="col-span-full text-gray-600">Không tìm thấy biển số trong video.</p>';
    }
}

function showVideoError(message) {
    document.getElementById('video-error-text').textContent = message;
    document.getElementById('video-error-message').style.display = 'block';
    document.getElementById('video-upload-area').style.display = 'block';
    document.getElementById('video-results').style.display = 'none';
}

function resetVideoForm() {
    document.getElementById('video-results').style.display = 'none';
    document.getElementById('video-upload-area').style.display = 'block';
    document.getElementById('video-file-input').value = '';
    document.getElementById('video-error-message').style.display = 'none';
    document.getElementById('result-video-player').src = '';
}

// ============================================================
// History & Stats
// ============================================================

async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();
        const tbody = document.getElementById('history-body');
        tbody.innerHTML = '';

        if (!Array.isArray(data) || data.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="text-center text-gray-500 py-8">Chưa có dữ liệu</td></tr>';
            return;
        }

        data.forEach(item => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${item.id || '-'}</td>
                <td class="font-semibold">${item.plate_text || '-'}</td>
                <td>${item.vehicle_type || '-'}</td>
                <td>${item.confidence ? (item.confidence * 100).toFixed(1) + '%' : '-'}</td>
                <td>${item.processing_time ? (item.processing_time * 1000).toFixed(0) + 'ms' : '-'}</td>
                <td>${item.created_at ? new Date(item.created_at).toLocaleString('vi-VN') : '-'}</td>
            `;
            tbody.appendChild(row);
        });
    } catch (error) {
        document.getElementById('history-body').innerHTML =
            '<tr><td colspan="6" class="text-center text-red-500 py-8">Không tải được lịch sử</td></tr>';
    }
}

async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        document.getElementById('stat-total').textContent = data.total_detections || 0;
        document.getElementById('stat-avg-conf').textContent = ((data.avg_confidence || 0) * 100).toFixed(1) + '%';
        document.getElementById('stat-avg-time').textContent = ((data.avg_processing_time || 0) * 1000).toFixed(0) + 'ms';
    } catch (error) {
        console.error('Không tải được thống kê:', error);
    }
}

// ============================================================
// Tab Switching
// ============================================================

function switchTab(tabName) {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelectorAll('.tab-btn').forEach(btn => {
        if (btn.getAttribute('onclick') === `switchTab('${tabName}')`) {
            btn.classList.add('active');
        }
    });

    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(tabName + '-tab').classList.add('active');

    if (tabName === 'history') {
        loadHistory();
        loadStats();
    }
}

// ============================================================
// Utilities
// ============================================================

function escapeHtml(value) {
    return String(value).replace(/[&<>'"/]/g, (char) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        "'": '&#39;',
        '"': '&quot;',
        '/': '&#x2F;'
    }[char]));
}
