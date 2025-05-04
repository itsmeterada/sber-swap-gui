let sourcePath = null;
let targetPath = null;
let socket = io();

// Check models
checkModels();

function checkModels() {
    fetch('/check_models')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            const modelCheck = document.getElementById('model-check');
            if (data.all_present) {
                modelCheck.className = 'alert alert-success';
                modelCheck.textContent = 'All models are present!';
            } else {
                modelCheck.className = 'alert alert-warning';
                modelCheck.innerHTML = `
                    Missing model files: ${data.missing_files.join(', ')}<br>
                    <button class="btn btn-primary mt-2" onclick="downloadModels()">Download Missing Models</button>
                `;
            }
        })
        .catch(error => {
            const modelCheck = document.getElementById('model-check');
            modelCheck.className = 'alert alert-danger';
            modelCheck.textContent = 'Error checking models: ' + error;
        });
}

function downloadModels() {
    const modelCheck = document.getElementById('model-check');
    modelCheck.className = 'alert alert-info';
    modelCheck.innerHTML = `
        <div>Downloading models... <span id="download-progress">0%</span></div>
        <div class="progress mt-2">
            <div id="download-progress-bar" class="progress-bar" role="progressbar" style="width: 0%">0%</div>
        </div>
    `;
    
    fetch('/download_models', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            showModelDownloadError(data.error);
        }
    })
    .catch(error => {
        showModelDownloadError('Error starting download: ' + error);
    });
}

// Socket handlers for download progress
socket.on('download_progress', function(data) {
    document.getElementById('download-progress').textContent = `${data.percentage}% (${data.filename})`;
    const progressBar = document.getElementById('download-progress-bar');
    progressBar.style.width = data.percentage + '%';
    progressBar.textContent = data.percentage + '%';
});

socket.on('download_complete', function(data) {
    const modelCheck = document.getElementById('model-check');
    modelCheck.className = 'alert alert-success';
    modelCheck.textContent = data.message;
});

socket.on('download_error', function(data) {
    showModelDownloadError(data.error);
});

function showModelDownloadError(error) {
    const modelCheck = document.getElementById('model-check');
    modelCheck.className = 'alert alert-danger';
    modelCheck.innerHTML = `Download failed: ${error}<br>
        <button class="btn btn-warning mt-2" onclick="checkModels()">Check Again</button>`;
}

// File upload handlers
document.getElementById('source-file').addEventListener('change', function(e) {
    handleFilePreview(e.target.files[0], 'source');
});

document.getElementById('target-file').addEventListener('change', function(e) {
    handleFilePreview(e.target.files[0], 'target');
});

// Drag and drop handlers
['source', 'target'].forEach(type => {
    const area = document.querySelector(`#${type}-preview`).parentElement;
    area.addEventListener('dragover', (e) => e.preventDefault());
    area.addEventListener('drop', (e) => {
        e.preventDefault();
        if (e.dataTransfer.files.length) {
            handleFilePreview(e.dataTransfer.files[0], type);
            // Update file input to match dropped file
            const input = document.getElementById(`${type}-file`);
            const dt = new DataTransfer();
            dt.items.add(e.dataTransfer.files[0]);
            input.files = dt.files;
        }
    });
});

function handleFilePreview(file, type) {
    // ファイルタイプチェック
    const allowedTypes = ['image/jpeg', 'image/png', 'image/gif'];
    if (!allowedTypes.includes(file.type)) {
        showStatus(`Invalid file type. Allowed types: ${allowedTypes.join(', ')}`, 'danger');
        return;
    }
    
    // プレビュー表示
    const reader = new FileReader();
    reader.onload = function(e) {
        const preview = document.getElementById(type + '-preview');
        const text = document.getElementById(type + '-text');
        preview.src = e.target.result;
        preview.style.display = 'block';
        text.style.display = 'none';
    };
    reader.readAsDataURL(file);
    
    // ファイルアップロード
    uploadFile(file, type);
}

function uploadFile(file, type) {
    const formData = new FormData();
    // 1つのリクエストで1つのファイルのみを送信
    formData.append(type, file);
    
    // 詳細なデバッグ情報を追加
    console.log(`Uploading ${type} file:`, file.name, file.type, file.size);
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        console.log('Response status:', response.status);
        if (!response.ok) {
            return response.json().then(err => Promise.reject(err));
        }
        return response.json();
    })
    .then(data => {
        console.log('Upload successful:', data);
        if (type === 'source') sourcePath = data.source_path;
        if (type === 'target') targetPath = data.target_path;
        updateProcessButton();
    })
    .catch(error => {
        console.error('Upload error:', error);
        showStatus(`Error uploading file: ${error.error || error.message || error}`, 'danger');
    });
}

function updateProcessButton() {
    const btn = document.getElementById('process-btn');
    btn.disabled = !(sourcePath && targetPath);
}

document.getElementById('process-btn').addEventListener('click', function() {
    const useSr = document.getElementById('use-sr').checked;
    const iou = parseFloat(document.getElementById('iou-slider').value);
    
    // Reset result display
    document.getElementById('result-image').style.display = 'none';
    document.getElementById('download-btn-container').style.display = 'none';
    
    fetch('/process', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            source_path: sourcePath,
            target_path: targetPath,
            use_sr: useSr,
            iou: iou
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Process failed');
        }
        return response.json();
    })
    .then(data => {
        if (data.status === 'started') {
            showProgress(true);
            showStatus('Processing...', 'info');
        } else {
            showStatus(data.error, 'danger');
        }
    })
    .catch(error => {
        showStatus('Error starting process: ' + error, 'danger');
        showProgress(false);
    });
});

// Socket.IO handlers
socket.on('progress_update', function(data) {
    updateProgress(data.progress);
});

socket.on('processing_complete', function(data) {
    showStatus('Processing complete!', 'success');
    showResult(data.output_path);
    showProgress(false);
});

socket.on('processing_error', function(data) {
    showStatus('Error: ' + data.error, 'danger');
    showProgress(false);
});

function showProgress(show) {
    document.querySelector('.progress').style.display = show ? 'block' : 'none';
    if (!show) updateProgress(0);
}

function updateProgress(percent) {
    const bar = document.getElementById('progress-bar');
    bar.style.width = percent + '%';
    bar.textContent = percent + '%';
}

function showStatus(message, type) {
    const status = document.getElementById('status');
    status.className = 'alert alert-' + type;
    status.textContent = message;
    status.style.display = 'block';
}

function showResult(imagePath) {
    const result = document.getElementById('result-image');
    const downloadBtn = document.getElementById('download-btn');
    const container = document.getElementById('download-btn-container');
    
    result.src = imagePath;
    result.style.display = 'block';
    downloadBtn.href = imagePath;
    container.style.display = 'block';
}

document.getElementById('iou-slider').addEventListener('input', function() {
    document.getElementById('iou-value').textContent = this.value;
});