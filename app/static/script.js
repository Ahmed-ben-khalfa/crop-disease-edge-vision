const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadPlaceholder = document.getElementById('upload-placeholder');
const imagePreview = document.getElementById('image-preview');
const analyzeBtn = document.getElementById('analyze-btn');
const resetBtn = document.getElementById('reset-btn');

const loadingOverlay = document.getElementById('loading-overlay');
const emptyState = document.getElementById('empty-state');
const resultsContent = document.getElementById('results-content');
const predictionsList = document.getElementById('predictions-list');

const oodWarning = document.getElementById('ood-warning');
const multiWarning = document.getElementById('multi-warning');
const severityBadge = document.getElementById('severity-badge');
const severityBar = document.getElementById('severity-bar');
const severityRatio = document.getElementById('severity-ratio');

let selectedFile = null;

// Event Listeners for Drag & Drop
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Veuillez uploader une image valide.');
        return;
    }
    
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        imagePreview.classList.remove('hidden');
        uploadPlaceholder.classList.add('hidden');
        analyzeBtn.disabled = false;
        resetBtn.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
}

function resetApp() {
    selectedFile = null;
    fileInput.value = '';
    imagePreview.src = '';
    imagePreview.classList.add('hidden');
    uploadPlaceholder.classList.remove('hidden');
    analyzeBtn.disabled = true;
    resetBtn.classList.add('hidden');
    
    emptyState.classList.remove('hidden');
    resultsContent.classList.add('hidden');
    oodWarning.classList.add('hidden');
    multiWarning.classList.add('hidden');
}

async function analyzeImage() {
    if (!selectedFile) return;

    // Show loading
    loadingOverlay.classList.remove('hidden');
    emptyState.classList.add('hidden');
    resultsContent.classList.add('hidden');
    analyzeBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Erreur lors de la prédiction');
        }

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        alert('Erreur: ' + error.message);
        emptyState.classList.remove('hidden');
    } finally {
        loadingOverlay.classList.add('hidden');
        analyzeBtn.disabled = false;
    }
}

function displayResults(data) {
    resultsContent.classList.remove('hidden');
    
    // OOD Handling
    if (data.is_ood || data.is_non_leaf) {
        oodWarning.classList.remove('hidden');
        const reason = data.is_non_leaf ? 
            "L'attention spatiale est trop faible, ce n'est probablement pas une feuille." :
            `Image atypique (Distance: ${data.ood_distance.toFixed(1)} > Seuil: ${data.ood_threshold.toFixed(1)}).`;
        document.getElementById('ood-text').textContent = reason;
    } else {
        oodWarning.classList.add('hidden');
    }

    // Multi-disease Handling
    if (data.is_multi_disease) {
        multiWarning.classList.remove('hidden');
        document.getElementById('multi-text').textContent = 
            `Possibilité de multiples maladies : ${data.multi_diseases.join(', ')}`;
    } else {
        multiWarning.classList.add('hidden');
    }

    // Severity
    const ratio = data.severity.ratio * 100;
    severityRatio.textContent = `${ratio.toFixed(1)}%`;
    severityBar.style.width = `${ratio}%`;
    severityBadge.textContent = data.severity.description;
    
    severityBadge.className = 'severity-badge'; // reset
    severityBadge.classList.add(`level-${data.severity.level}`);
    
    if (data.severity.level === 0) severityBar.style.background = 'var(--success-gradient)';
    else if (data.severity.level === 1) severityBar.style.background = 'var(--warning-gradient)';
    else severityBar.style.background = 'var(--danger-gradient)';

    // Predictions List
    predictionsList.innerHTML = '';
    data.predictions.forEach((pred, index) => {
        const div = document.createElement('div');
        div.className = `prediction-item ${index === 0 ? 'top-prediction' : ''}`;
        
        const formatName = pred.label.replace(/___/g, ' - ').replace(/_/g, ' ');
        const percent = (pred.probability * 100).toFixed(1);
        
        div.innerHTML = `
            <span class="pred-label">${formatName}</span>
            <span class="pred-prob">${percent}%</span>
        `;
        predictionsList.appendChild(div);
    });
}
