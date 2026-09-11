document.addEventListener('DOMContentLoaded', () => {
    // Tab Switching
    const tabs = document.querySelectorAll('.nav-links li');
    const sections = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            sections.forEach(s => s.classList.remove('active'));

            tab.classList.add('active');
            const target = tab.getAttribute('data-tab');
            document.getElementById(target).classList.add('active');
        });
    });

    // File Upload handling
    const fileInput = document.getElementById('dataset-upload');
    const targetSelectContainer = document.getElementById('target-select-container');
    const confirmUploadBtn = document.getElementById('confirm-upload-btn');
    const targetColInput = document.getElementById('target-col-input');
    const strategySelect = document.getElementById('strategy-select');
    const uploadStatus = document.getElementById('upload-status');

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            targetSelectContainer.style.display = 'block';
            uploadStatus.textContent = `Selected: ${e.target.files[0].name}. Enter target column.`;
            uploadStatus.style.color = 'var(--text-muted)';
        }
    });

    confirmUploadBtn.addEventListener('click', async () => {
        const file = fileInput.files[0];
        const targetCol = targetColInput.value.trim();
        const strategy = strategySelect.value;
        
        if (!targetCol) {
            uploadStatus.textContent = "Please enter a target column.";
            uploadStatus.style.color = "var(--error)";
            return;
        }

        const formData = new FormData();
        formData.append('file', file);
        formData.append('target_col', targetCol);
        formData.append('strategy', strategy);

        uploadStatus.textContent = "Uploading and running AutoML... This may take a few minutes.";
        uploadStatus.style.color = "var(--accent)";
        confirmUploadBtn.disabled = true;

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (response.ok) {
                uploadStatus.textContent = "Model optimized and ready!";
                uploadStatus.style.color = "var(--success)";
                setTimeout(() => window.location.reload(), 1500);
            } else {
                uploadStatus.textContent = `Error: ${data.detail}`;
                uploadStatus.style.color = "var(--error)";
            }
        } catch (error) {
            uploadStatus.textContent = "Upload failed.";
            uploadStatus.style.color = "var(--error)";
        } finally {
            confirmUploadBtn.disabled = false;
        }
    });

    // Load Dashboard Stats
    async function loadStats() {
        try {
            const response = await fetch('/api/stats');
            if (response.ok) {
                const data = await response.json();
                
                const metrics = data.metrics || {};
                const leaderboard = metrics.leaderboard || [];
                const winner = leaderboard.length > 0 ? leaderboard[0] : null;

                document.getElementById('model-acc').textContent = metrics.accuracy ? (metrics.accuracy * 100).toFixed(2) + '%' : '--';
                document.getElementById('model-roc').textContent = metrics.roc_auc ? metrics.roc_auc.toFixed(3) : '--';
                document.getElementById('model-f1').textContent = metrics.f1 ? metrics.f1.toFixed(3) : '--';
                document.getElementById('target-var').textContent = data.target_col;
                
                if (winner) {
                    document.getElementById('winning-model-name').textContent = `Winning Model: ${winner.model}`;
                } else {
                    document.getElementById('winning-model-name').textContent = `Model Loaded`;
                }

                // Render Feature Importance Chart
                const features = Object.keys(data.top_features);
                const importances = Object.values(data.top_features);
                
                const ctx = document.getElementById('featureChart').getContext('2d');
                new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: features,
                        datasets: [{
                            label: 'Importance Score',
                            data: importances,
                            backgroundColor: '#2563eb',
                            borderRadius: 4
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        indexAxis: 'y',
                        plugins: {
                            legend: { display: false }
                        },
                        scales: {
                            x: { grid: { color: '#f1f5f9' }, ticks: { color: '#64748b' } },
                            y: { grid: { display: false }, ticks: { color: '#64748b' } }
                        }
                    }
                });
                
                // Populate Leaderboard Table
                const tbody = document.querySelector('#leaderboard-table tbody');
                tbody.innerHTML = '';
                leaderboard.forEach((entry, idx) => {
                    const tr = document.createElement('tr');
                    const timeStr = entry.time_sec !== undefined ? entry.time_sec.toFixed(2) : '--';
                    tr.innerHTML = `
                        <td>${idx === 0 ? '🏆 ' : ''}${entry.model}</td>
                        <td>${(entry.accuracy * 100).toFixed(2)}%</td>
                        <td>${timeStr}s</td>
                    `;
                    tbody.appendChild(tr);
                });
            }
        } catch (e) {
            console.error("Failed to load stats", e);
        }
    }

    // Load Schema and populate forms/dropdowns
    async function loadSchema() {
        try {
            const response = await fetch('/api/schema');
            if (response.ok) {
                const data = await response.json();
                const schema = data.schema;
                const targetCol = data.target_col;
                
                // 1. Populate Dynamic Insights Dropdowns
                const insightVar1 = document.getElementById('insight-var-1');
                const insightVar2 = document.getElementById('insight-var-2');
                insightVar1.innerHTML = '';
                insightVar2.innerHTML = '';
                
                schema.forEach(details => {
                    const col = details.name;
                    if (col !== targetCol) {
                        insightVar1.innerHTML += `<option value="${col}">${col}</option>`;
                        insightVar2.innerHTML += `<option value="${col}">${col}</option>`;
                    }
                });
                
                if (insightVar2.options.length > 1) {
                    insightVar2.selectedIndex = 1; // Select different defaults
                }
                
                // 2. Populate Inference Form
                const grid = document.getElementById('dynamic-form-grid');
                grid.innerHTML = '';
                
                schema.forEach(details => {
                    const col = details.name;
                    if (col === targetCol) return;
                    
                    const div = document.createElement('div');
                    div.className = 'form-group';
                    div.innerHTML = `<label>${col}</label>`;
                    
                    if (details.type === 'categorical') {
                        const select = document.createElement('select');
                        select.id = `input-${col}`;
                        details.options.forEach(val => {
                            const option = document.createElement('option');
                            option.value = val;
                            option.textContent = val;
                            select.appendChild(option);
                        });
                        div.appendChild(select);
                    } else {
                        const input = document.createElement('input');
                        input.type = 'number';
                        input.step = 'any';
                        input.id = `input-${col}`;
                        input.placeholder = `e.g. 0.0`;
                        input.value = 0.0;
                        div.appendChild(input);
                    }
                    grid.appendChild(div);
                });
            }
        } catch (e) {
            console.error("Failed to load schema", e);
            document.getElementById('dynamic-form-grid').innerHTML = '<p style="color:red;">Error loading schema.</p>';
        }
    }

    // Handle Dynamic Insights Generation
    let insightChartInstance = null;
    document.getElementById('generate-insight-btn').addEventListener('click', async () => {
        const var1 = document.getElementById('insight-var-1').value;
        const var2 = document.getElementById('insight-var-2').value;
        
        if (!var1 || !var2) return;
        
        const loadingDiv = document.getElementById('insight-loading');
        const resultsDiv = document.getElementById('insight-results');
        const textContent = document.getElementById('insight-text-content');
        
        loadingDiv.style.display = 'block';
        resultsDiv.style.display = 'none';
        
        try {
            const response = await fetch('/api/insights', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ var1, var2 })
            });
            
            const data = await response.json();
            loadingDiv.style.display = 'none';
            
            if (response.ok) {
                resultsDiv.style.display = 'block';
                textContent.innerHTML = data.insight_html;
                document.getElementById('insight-chart-title').textContent = `${var1} vs ${var2}`;
                
                // Render Chart
                if (insightChartInstance) {
                    insightChartInstance.destroy();
                }
                const ctx = document.getElementById('insightChart').getContext('2d');
                insightChartInstance = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: data.chart_data.labels,
                        datasets: [{
                            label: 'Average/Count',
                            data: data.chart_data.values,
                            backgroundColor: '#10b981',
                            borderRadius: 4
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: {
                            x: { grid: { display: false }, ticks: { color: '#64748b' } },
                            y: { grid: { color: '#f1f5f9' }, ticks: { color: '#64748b' } }
                        }
                    }
                });
                
            } else {
                alert(`Error generating insight: ${data.detail}`);
            }
        } catch (error) {
            loadingDiv.style.display = 'none';
            alert("Network error generating insights.");
        }
    });

    // Handle Prediction
    document.getElementById('predict-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const responseSchema = await fetch('/api/schema');
        const schemaData = await responseSchema.json();
        
        const customer_data = {};
        for (const [col, details] of Object.entries(schemaData.schema)) {
            if (col === schemaData.target_col) continue;
            
            const el = document.getElementById(`input-${col}`);
            if (el) {
                let val = el.value;
                if (details.type === 'numerical') {
                    val = parseFloat(val);
                    if (isNaN(val)) val = details.mean; // fallback to mean if empty
                }
                customer_data[col] = val;
            }
        }
        
        const resultDiv = document.getElementById('prediction-result');
        resultDiv.style.display = 'block';
        resultDiv.innerHTML = '<span class="spinner"></span> Analyzing profile...';
        
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ customer_data })
            });
            const data = await response.json();
            
            if (response.ok) {
                // Determine risk level based on probability or label
                let predText = data.prediction;
                let riskClass = "var(--text-main)";
                let riskBg = "#eff6ff";
                let riskBorder = "var(--accent)";
                let proba = data.confidence ? (data.confidence * 100).toFixed(1) : null;
                
                // Usually Churn=1 or "Yes" means high risk
                if (predText === 1 || String(predText).toLowerCase() === 'yes' || String(predText).toLowerCase() === 'churn') {
                    predText = `🚨 High Risk of Positive Outcome`;
                    riskBg = "#fef2f2";
                    riskBorder = "#ef4444";
                    riskClass = "#b91c1c";
                } else if (predText === 0 || String(predText).toLowerCase() === 'no') {
                    predText = `✅ Low Risk (Negative Outcome)`;
                    riskBg = "#ecfdf5";
                    riskBorder = "#10b981";
                    riskClass = "#047857";
                } else {
                    predText = `Prediction: ${predText}`;
                }
                
                const probaText = proba ? `<div style="font-size: 13px; color: var(--text-muted); margin-top: 4px;">Model Confidence: ${proba}%</div>` : "";
                
                resultDiv.innerHTML = `<div style="color: ${riskClass}; font-weight: 600;">${predText}</div>${probaText}`;
                resultDiv.style.background = riskBg;
                resultDiv.style.border = `1px solid ${riskBorder}`;
            } else {
                resultDiv.innerHTML = `<strong>Error:</strong> ${data.detail}`;
                resultDiv.style.background = '#fef2f2';
                resultDiv.style.border = '1px solid var(--error)';
            }
        } catch (error) {
            resultDiv.innerHTML = `<strong>Error:</strong> Failed to connect to inference engine.`;
            resultDiv.style.background = '#fef2f2';
            resultDiv.style.border = '1px solid var(--error)';
        }
    });

    // Chat functionality
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const chatHistory = document.getElementById('chat-history');
    let chatContext = [];

    async function sendMessage() {
        const text = chatInput.value.trim();
        if (!text) return;

        chatHistory.innerHTML += `
            <div class="message user">
                <div class="bubble">${text}</div>
                <div class="avatar">U</div>
            </div>
        `;
        chatInput.value = '';
        chatHistory.scrollTop = chatHistory.scrollHeight;
        
        const tempId = 'loading-' + Date.now();
        chatHistory.innerHTML += `
            <div class="message system" id="${tempId}">
                <div class="avatar">🤖</div>
                <div class="bubble"><span class="spinner"></span> Thinking...</div>
            </div>
        `;
        chatHistory.scrollTop = chatHistory.scrollHeight;

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text, history: chatContext })
            });

            const data = await response.json();
            document.getElementById(tempId).remove();

            if (response.ok) {
                chatHistory.innerHTML += `
                    <div class="message system">
                        <div class="avatar">🤖</div>
                        <div class="bubble">${data.reply.replace(/\n/g, '<br>')}</div>
                    </div>
                `;
                chatContext.push({ role: 'user', content: text });
                chatContext.push({ role: 'assistant', content: data.reply });
            } else {
                chatHistory.innerHTML += `
                    <div class="message system">
                        <div class="avatar">⚠️</div>
                        <div class="bubble" style="color:var(--error)">${data.detail}</div>
                    </div>
                `;
            }
        } catch (error) {
            document.getElementById(tempId).remove();
            chatHistory.innerHTML += `
                <div class="message system">
                    <div class="avatar">⚠️</div>
                    <div class="bubble" style="color:var(--error)">Network error.</div>
                </div>
            `;
        }
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    sendBtn.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

    // Initialize
    loadStats();
    loadSchema();
});
