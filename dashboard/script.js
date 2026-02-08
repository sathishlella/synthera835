// SynthERA-835 Dashboard JavaScript
// Interactive functionality for the generator dashboard

// State
let claims = [];
let stats = { claims_generated: 0, claims_denied: 0, claims_paid: 0, claims_partial: 0 };
let currentPage = 'dashboard';
let currentTablePage = 1;
const claimsPerPage = 20;

// Load claims data from CSV
async function loadClaimsData() {
    // Try output folders in order of priority
    const paths = [
        '/api/claims',                            // API endpoint (Flask server)
        '/synthera835_output/claims.csv',         // Flask server static path
        '../synthera835_output/claims.csv',       // Static server (from dashboard folder)
        '../../synthera835_output/claims.csv',    // Alternate path
        '../synthera835_output_10k/claims.csv',   // 10k dataset
        '../../synthera835_output_10k/claims.csv' // Alternate path for 10k
    ];

    for (const path of paths) {
        try {
            const response = await fetch(path);
            if (response.ok) {
                console.log('Loaded claims from:', path);
                return await response.text();
            }
        } catch (e) {
            // Continue to next path
        }
    }

    console.error('Error loading claims data: No valid path found');
    showNotification('Could not load claims data. Using sample data.', 'warning');
    return null;
}

// Parse CSV to array of objects
function parseCSV(csvText) {
    const lines = csvText.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
    const data = [];

    for (let i = 1; i < lines.length; i++) {
        const values = parseCSVLine(lines[i]);
        if (values.length === headers.length) {
            const row = {};
            headers.forEach((header, index) => {
                row[header] = values[index];
            });
            data.push(row);
        }
    }
    return data;
}

// Parse a single CSV line (handles quoted values)
function parseCSVLine(line) {
    const values = [];
    let current = '';
    let inQuotes = false;

    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        if (char === '"') {
            inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
            values.push(current.trim());
            current = '';
        } else {
            current += char;
        }
    }
    values.push(current.trim());
    return values;
}

// Calculate stats from claims data
// Note: Each claim can have multiple service lines (1-4 per claim)
// So we need to count unique claim_ids, not rows
function calculateStats(claimsData) {
    // Group by claim_id to get unique claims
    const claimsById = {};
    claimsData.forEach(row => {
        const claimId = row.claim_id;
        if (!claimsById[claimId]) {
            claimsById[claimId] = {
                claim_id: claimId,
                status: (row.claim_status || row.status || '').toLowerCase(),
                lines: []
            };
        }
        claimsById[claimId].lines.push(row);
    });

    const uniqueClaims = Object.values(claimsById);
    const total = uniqueClaims.length;
    let denied = 0;
    let paid = 0;
    let partial = 0;

    uniqueClaims.forEach(claim => {
        const status = claim.status;
        if (status === 'denied' || status === 'full_denial') {
            denied++;
        } else if (status === 'paid' || status === 'full_payment') {
            paid++;
        } else if (status === 'partial' || status === 'partial_denial') {
            partial++;
        }
    });

    return {
        claims_generated: total,
        claims_denied: denied + partial,
        claims_paid: paid,
        claims_partial: partial,
        total_lines: claimsData.length  // For reference
    };
}

// Initialize dashboard with real data
async function initializeDashboard() {
    showNotification('Loading claims data...', 'info');

    const csvText = await loadClaimsData();

    if (csvText) {
        claims = parseCSV(csvText);
        stats = calculateStats(claims);
        updateStats(stats);
        updateClaimsTable(claims);
        updateCharts(claims);
        showNotification(`Loaded ${claims.length.toLocaleString()} claims successfully!`, 'success');
    } else {
        // Use default sample data if loading fails
        stats = { claims_generated: 10000, claims_denied: 2215, claims_paid: 7785, claims_partial: 547 };
        updateStats(stats);
    }
}

// Update stats display
function updateStats(newStats) {
    stats = newStats;
    document.getElementById('totalClaims').textContent = stats.claims_generated.toLocaleString();
    document.getElementById('deniedClaims').textContent = stats.claims_denied.toLocaleString();
    document.getElementById('paidClaims').textContent = stats.claims_paid.toLocaleString();

    const recoverable = Math.round(stats.claims_denied * 0.6);
    document.getElementById('recoverableClaims').textContent = recoverable.toLocaleString();

    const rate = stats.claims_generated > 0
        ? Math.round((stats.claims_denied / stats.claims_generated) * 100)
        : 0;
    document.getElementById('denialRate').textContent = rate + '%';
}

// Update claims table with actual data
// Groups service lines by claim_id to show unique claims
function updateClaimsTable(claimsData) {
    const tbody = document.getElementById('claimsTableBody');
    if (!tbody || !claimsData || claimsData.length === 0) return;

    // Group by claim_id to get unique claims
    const claimsById = {};
    claimsData.forEach(row => {
        const claimId = row.claim_id || row.id || 'N/A';
        if (!claimsById[claimId]) {
            claimsById[claimId] = {
                claim_id: claimId,
                date: row.service_date || row.date || row.date_of_service || row.claim_date || 'N/A',
                cpt_codes: [],
                total_charge: 0,
                total_paid: 0,
                status: (row.claim_status || row.status || 'unknown').toLowerCase(),
                carc: row.primary_carc || row.carc_code || row.adjustment_reason || '-',
                is_recoverable: row.is_recoverable === 'True' || row.is_recoverable === true
            };
        }
        // Aggregate values from all service lines
        const cpt = row.primary_cpt || row.cpt_code || row.procedure_code;
        if (cpt && !claimsById[claimId].cpt_codes.includes(cpt)) {
            claimsById[claimId].cpt_codes.push(cpt);
        }
        claimsById[claimId].total_charge += parseFloat(row.total_charge || row.charge_amount || 0);
        claimsById[claimId].total_paid += parseFloat(row.total_paid || row.paid_amount || 0);
    });

    const uniqueClaims = Object.values(claimsById);

    // Calculate pagination based on unique claims
    const start = (currentTablePage - 1) * claimsPerPage;
    const end = Math.min(start + claimsPerPage, uniqueClaims.length);
    const pageData = uniqueClaims.slice(start, end);

    // Build table rows
    let html = '';
    pageData.forEach(claim => {
        const claimId = claim.claim_id;
        const date = claim.date;
        const cptCode = claim.cpt_codes.join(', ') || 'N/A';
        const charge = claim.total_charge;
        const paid = claim.total_paid;
        const status = claim.status;
        const carc = claim.carc;
        const recoverable = claim.is_recoverable;

        let statusClass = 'paid';
        let statusText = 'Paid';
        if (status.includes('denied') || status === 'full_denial') {
            statusClass = 'denied';
            statusText = 'Denied';
        } else if (status.includes('partial')) {
            statusClass = 'partial';
            statusText = 'Partial';
        }

        let recoveryHtml = '-';
        if (statusClass === 'denied' || statusClass === 'partial') {
            recoveryHtml = recoverable
                ? '<span class="recovery-badge yes">Resubmit</span>'
                : '<span class="recovery-badge no">Write Off</span>';
        }

        html += `
            <tr>
                <td><code>${claimId.substring(0, 13)}</code></td>
                <td>${date}</td>
                <td>${cptCode}</td>
                <td>$${charge.toFixed(2)}</td>
                <td>$${paid.toFixed(2)}</td>
                <td><span class="status-badge ${statusClass}">${statusText}</span></td>
                <td>${carc}</td>
                <td>${recoveryHtml}</td>
            </tr>
        `;
    });

    tbody.innerHTML = html;

    // Update footer with unique claims count
    const footer = document.querySelector('.table-footer span');
    if (footer) {
        footer.textContent = `Showing ${start + 1}-${end} of ${uniqueClaims.length.toLocaleString()} claims`;
    }

    // Update pagination based on unique claims
    updatePagination(uniqueClaims.length);
}

// Update pagination buttons
function updatePagination(totalClaims) {
    const totalPages = Math.ceil(totalClaims / claimsPerPage);
    const pagination = document.querySelector('.pagination');
    if (!pagination) return;

    let html = `<button class="btn-page" ${currentTablePage === 1 ? 'disabled' : ''} onclick="changePage(${currentTablePage - 1})">&lt;</button>`;

    // Show first page
    html += `<button class="btn-page ${currentTablePage === 1 ? 'active' : ''}" onclick="changePage(1)">1</button>`;

    // Show ellipsis if needed
    if (currentTablePage > 3) {
        html += `<button class="btn-page" disabled>...</button>`;
    }

    // Show current page and neighbors
    for (let i = Math.max(2, currentTablePage - 1); i <= Math.min(totalPages - 1, currentTablePage + 1); i++) {
        if (i > 1 && i < totalPages) {
            html += `<button class="btn-page ${currentTablePage === i ? 'active' : ''}" onclick="changePage(${i})">${i}</button>`;
        }
    }

    // Show ellipsis if needed
    if (currentTablePage < totalPages - 2) {
        html += `<button class="btn-page" disabled>...</button>`;
    }

    // Show last page
    if (totalPages > 1) {
        html += `<button class="btn-page ${currentTablePage === totalPages ? 'active' : ''}" onclick="changePage(${totalPages})">${totalPages}</button>`;
    }

    html += `<button class="btn-page" ${currentTablePage === totalPages ? 'disabled' : ''} onclick="changePage(${currentTablePage + 1})">&gt;</button>`;

    pagination.innerHTML = html;
}

// Change table page
function changePage(page) {
    if (page < 1) return;
    const totalPages = Math.ceil(claims.length / claimsPerPage);
    if (page > totalPages) return;

    currentTablePage = page;
    updateClaimsTable(claims);
}

// Update charts with real data
function updateCharts(claimsData) {
    // Calculate CARC distribution
    const carcCounts = {};
    claimsData.forEach(claim => {
        const carc = claim.primary_carc || claim.carc_code;
        if (carc && carc !== '-') {
            carcCounts[carc] = (carcCounts[carc] || 0) + 1;
        }
    });

    // Sort by count and get top 5
    const sortedCarcs = Object.entries(carcCounts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5);

    const total = Object.values(carcCounts).reduce((a, b) => a + b, 0);

    // Update CARC chart
    const carcChart = document.getElementById('carcChart');
    if (carcChart && sortedCarcs.length > 0) {
        let html = '';
        sortedCarcs.forEach(([code, count]) => {
            const pct = total > 0 ? Math.round((count / total) * 100) : 0;
            html += `
                <div class="bar-item">
                    <span class="bar-label">${code}</span>
                    <div class="bar-track">
                        <div class="bar-fill" style="width: ${pct}%"></div>
                    </div>
                    <span class="bar-value">${pct}%</span>
                </div>
            `;
        });
        carcChart.innerHTML = html;
    }

    // Calculate denial category distribution
    const categoryCounts = {};
    claimsData.forEach(claim => {
        const category = claim.denial_category || claim.category;
        const status = (claim.claim_status || claim.status || '').toLowerCase();
        if (category && (status.includes('denied') || status.includes('partial'))) {
            categoryCounts[category] = (categoryCounts[category] || 0) + 1;
        }
    });

    // Update donut center with actual denied count
    const donutValue = document.querySelector('.donut-value');
    if (donutValue) {
        donutValue.textContent = stats.claims_denied.toLocaleString();
    }
}

// Navigation handler
function navigateTo(page, element, event) {
    event.preventDefault();

    // Update active state
    document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
    element.classList.add('active');

    currentPage = page;

    // Show notification for page change
    const pageNames = {
        'dashboard': 'Dashboard',
        'generate': 'Generate Claims',
        'analytics': 'Analytics',
        'export': 'Export Data',
        'settings': 'Settings'
    };

    // Update header title
    document.querySelector('.header-left h1').textContent = pageNames[page] || 'Dashboard';

    // Handle specific page actions
    if (page === 'generate') {
        generateClaims(); // Open the modal
    } else if (page === 'export') {
        downloadCSV(); // Trigger CSV download
    } else if (page === 'analytics') {
        showNotification('Analytics view - displaying current data', 'info');
    } else if (page === 'settings') {
        showNotification('Settings panel coming soon!', 'info');
    }

    // Close sidebar on mobile after navigation
    if (window.innerWidth <= 768) {
        document.querySelector('.sidebar').classList.remove('active');
    }

    console.log('Navigated to:', page);
}

// Toggle sidebar on mobile
function toggleSidebar() {
    document.querySelector('.sidebar').classList.toggle('active');
}

// Open generate modal
function generateClaims() {
    document.getElementById('generatorModal').classList.add('active');
}

// Close modal
function closeModal() {
    document.getElementById('generatorModal').classList.remove('active');
}

// Start generation - calls the backend API to run Python generator
async function startGeneration() {
    const numClaims = document.getElementById('numClaims').value;
    const denialRate = document.getElementById('denialRateSlider').value;
    const seed = document.getElementById('randomSeed').value;

    closeModal();
    showNotification(`🔄 Generating ${numClaims} claims with ${denialRate}% denial rate...`, 'info');

    // Show estimated stats while generating
    updateStats({
        claims_generated: parseInt(numClaims),
        claims_denied: Math.round(numClaims * denialRate / 100),
        claims_paid: Math.round(numClaims * (100 - denialRate - 2) / 100),
        claims_partial: Math.round(numClaims * 0.02)
    });

    try {
        // Call the backend API to run the Python generator
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                num_claims: parseInt(numClaims),
                denial_rate: parseFloat(denialRate) / 100,
                seed: seed ? parseInt(seed) : null
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Generation failed');
        }

        // Poll for completion
        showNotification('⏳ Generator running... please wait', 'info');
        await pollGenerationStatus();

    } catch (error) {
        console.error('Generation error:', error);
        showNotification(`❌ ${error.message}. Server may not be running.`, 'error');

        // Fallback: Still reload whatever data exists
        setTimeout(async () => {
            await reloadDashboardData();
        }, 1000);
    }
}

// Poll the generation status until complete
async function pollGenerationStatus() {
    const maxAttempts = 60; // 60 seconds max
    let attempts = 0;

    while (attempts < maxAttempts) {
        try {
            const response = await fetch('/api/status');
            const status = await response.json();

            if (!status.running) {
                if (status.error) {
                    showNotification(`❌ Generation failed: ${status.error}`, 'error');
                } else {
                    showNotification('✓ Generation complete! Loading data...', 'success');
                    await reloadDashboardData();
                }
                return;
            }

            // Still running, update progress
            if (status.message) {
                showNotification(`⏳ ${status.message}`, 'info');
            }

        } catch (error) {
            // If polling fails, might be network issue, continue trying
            console.warn('Polling error:', error);
        }

        await new Promise(resolve => setTimeout(resolve, 1000));
        attempts++;
    }

    showNotification('⚠️ Generation taking too long, check server logs', 'warning');
    await reloadDashboardData();
}

// Reload dashboard data from CSV
async function reloadDashboardData() {
    showNotification('Loading claims data...', 'info');

    const csvText = await loadClaimsData();

    if (csvText) {
        claims = parseCSV(csvText);
        stats = calculateStats(claims);
        currentTablePage = 1;
        updateStats(stats);
        updateClaimsTable(claims);
        updateCharts(claims);
        showNotification(`Loaded ${claims.length.toLocaleString()} claims successfully!`, 'success');
    } else {
        showNotification('Could not reload claims data', 'warning');
    }
}

// Show notification toast
function showNotification(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = message;

    let bgColor = '#3b82f6'; // blue for info
    if (type === 'success') bgColor = '#10b981';
    else if (type === 'warning') bgColor = '#f59e0b';
    else if (type === 'error') bgColor = '#ef4444';

    toast.style.cssText = `
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        padding: 1rem 1.5rem;
        background: ${bgColor};
        color: white;
        border-radius: 8px;
        font-weight: 500;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        z-index: 9999;
        animation: slideIn 0.3s ease;
    `;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease forwards';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Filter table
function filterTable() {
    const search = document.getElementById('searchInput').value.toLowerCase();
    const status = document.getElementById('statusFilter').value;

    let filtered = claims;

    if (search) {
        filtered = filtered.filter(claim => {
            const claimId = (claim.claim_id || claim.id || '').toLowerCase();
            const cpt = (claim.primary_cpt || claim.cpt_code || '').toLowerCase();
            const carc = (claim.primary_carc || claim.carc_code || '').toLowerCase();
            return claimId.includes(search) || cpt.includes(search) || carc.includes(search);
        });
    }

    if (status !== 'all') {
        filtered = filtered.filter(claim => {
            const claimStatus = (claim.claim_status || claim.status || '').toLowerCase();
            if (status === 'paid') return !claimStatus.includes('denied') && !claimStatus.includes('partial');
            if (status === 'denied') return claimStatus.includes('denied') || claimStatus === 'full_denial';
            if (status === 'partial') return claimStatus.includes('partial');
            return true;
        });
    }

    currentTablePage = 1;

    // Store filtered data for export
    window.filteredClaims = filtered;

    // Update stats cards to reflect filtered data
    const filteredStats = calculateStats(filtered);
    updateStats(filteredStats);

    // Update charts with filtered data
    updateCharts(filtered);

    // Update table
    updateClaimsTable(filtered);
}

// DOMContentLoaded handler
document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('searchInput');
    const statusFilter = document.getElementById('statusFilter');

    if (searchInput) {
        searchInput.addEventListener('input', filterTable);
    }
    if (statusFilter) {
        statusFilter.addEventListener('change', filterTable);
    }

    // Close sidebar when clicking outside on mobile
    document.addEventListener('click', (e) => {
        const sidebar = document.querySelector('.sidebar');
        const menuToggle = document.querySelector('.menu-toggle');
        if (window.innerWidth <= 768 &&
            !sidebar.contains(e.target) &&
            !menuToggle.contains(e.target)) {
            sidebar.classList.remove('active');
        }
    });

    // ESC to close modal
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeModal();
    });

    // Initialize dashboard with real data
    initializeDashboard();
});

// Add animation keyframes
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);

// Animate bars on load
setTimeout(() => {
    document.querySelectorAll('.bar-fill').forEach((bar, i) => {
        const width = bar.style.width;
        bar.style.width = '0';
        setTimeout(() => {
            bar.style.transition = 'width 0.8s ease';
            bar.style.width = width;
        }, i * 100);
    });
}, 500);

// Download claims as CSV
// Aggregates service lines by claim_id to export unique claims
async function downloadCSV() {
    if (!claims || claims.length === 0) {
        showNotification('No claims data to download', 'warning');
        return;
    }

    // Get current filter state
    const search = document.getElementById('searchInput')?.value.toLowerCase() || '';
    const status = document.getElementById('statusFilter')?.value || 'all';

    let dataToExport = claims;

    // Apply filters if any
    if (search) {
        dataToExport = dataToExport.filter(claim => {
            const claimId = (claim.claim_id || claim.id || '').toLowerCase();
            const cpt = (claim.primary_cpt || claim.cpt_code || '').toLowerCase();
            const carc = (claim.primary_carc || claim.carc_code || '').toLowerCase();
            return claimId.includes(search) || cpt.includes(search) || carc.includes(search);
        });
    }

    if (status !== 'all') {
        dataToExport = dataToExport.filter(claim => {
            const claimStatus = (claim.claim_status || claim.status || '').toLowerCase();
            if (status === 'paid') return !claimStatus.includes('denied') && !claimStatus.includes('partial');
            if (status === 'denied') return claimStatus.includes('denied') || claimStatus === 'full_denial';
            if (status === 'partial') return claimStatus.includes('partial');
            return true;
        });
    }

    if (dataToExport.length === 0) {
        showNotification('No matching claims to download', 'warning');
        return;
    }

    // Aggregate by claim_id to export unique claims
    const claimsById = {};
    dataToExport.forEach(row => {
        const claimId = row.claim_id || row.id || 'N/A';
        if (!claimsById[claimId]) {
            claimsById[claimId] = {
                claim_id: claimId,
                patient_id: row.patient_id || '',
                provider_npi: row.provider_npi || '',
                payer_id: row.payer_id || '',
                date_of_service: row.date_of_service || row.service_date || row.date || '',
                cpt_codes: [],
                total_charge: 0,
                total_paid: 0,
                claim_status: row.claim_status || row.status || '',
                carc_code: row.carc_code || row.primary_carc || '',
                rarc_code: row.rarc_code || '',
                denial_category: row.denial_category || '',
                is_recoverable: row.is_recoverable || '',
                recovery_action: row.recovery_action || ''
            };
        }
        // Aggregate values from all service lines
        const cpt = row.procedure_code || row.cpt_code || row.primary_cpt;
        if (cpt && !claimsById[claimId].cpt_codes.includes(cpt)) {
            claimsById[claimId].cpt_codes.push(cpt);
        }
        claimsById[claimId].total_charge += parseFloat(row.charge_amount || row.total_charge || 0);
        claimsById[claimId].total_paid += parseFloat(row.paid_amount || row.total_paid || 0);
    });

    const uniqueClaims = Object.values(claimsById).map(claim => ({
        claim_id: claim.claim_id,
        patient_id: claim.patient_id,
        provider_npi: claim.provider_npi,
        payer_id: claim.payer_id,
        date_of_service: claim.date_of_service,
        procedure_codes: claim.cpt_codes.join('; '),
        total_charge: claim.total_charge.toFixed(2),
        total_paid: claim.total_paid.toFixed(2),
        claim_status: claim.claim_status,
        carc_code: claim.carc_code,
        rarc_code: claim.rarc_code,
        denial_category: claim.denial_category,
        is_recoverable: claim.is_recoverable,
        recovery_action: claim.recovery_action
    }));

    // Get headers from aggregated data
    const headers = Object.keys(uniqueClaims[0]);

    // Build CSV content
    let csv = headers.join(',') + '\n';
    uniqueClaims.forEach(row => {
        csv += headers.map(header => {
            let value = row[header] || '';
            // Escape quotes and wrap in quotes if contains comma
            if (typeof value === 'string' && (value.includes(',') || value.includes('"') || value.includes('\n') || value.includes(';'))) {
                value = '"' + value.replace(/"/g, '""') + '"';
            }
            return value;
        }).join(',') + '\n';
    });

    // Create and trigger download with proper filename
    const filename = `synthera835_claims_${new Date().toISOString().split('T')[0]}.csv`;

    // Add BOM for Excel compatibility
    const BOM = '\uFEFF';
    const csvContent = BOM + csv;

    // Try using File System Access API (modern browsers) for proper filename control
    if (window.showSaveFilePicker) {
        try {
            const handle = await window.showSaveFilePicker({
                suggestedName: filename,
                types: [{
                    description: 'CSV Files',
                    accept: { 'text/csv': ['.csv'] }
                }]
            });
            const writable = await handle.createWritable();
            await writable.write(csvContent);
            await writable.close();
            showNotification(`Downloaded ${uniqueClaims.length.toLocaleString()} claims as CSV`, 'success');
            return;
        } catch (err) {
            // User cancelled or API failed, fall through to blob method
            if (err.name === 'AbortError') {
                showNotification('Download cancelled', 'info');
                return;
            }
        }
    }

    // Fallback: Use blob approach
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    showNotification(`Downloaded ${uniqueClaims.length.toLocaleString()} claims as CSV`, 'success');
}

// Download claims as JSON
// Aggregates service lines by claim_id to export unique claims
async function downloadJSON() {
    if (!claims || claims.length === 0) {
        showNotification('No claims data to download', 'warning');
        return;
    }

    // Get current filter state
    const search = document.getElementById('searchInput')?.value.toLowerCase() || '';
    const status = document.getElementById('statusFilter')?.value || 'all';

    let dataToExport = claims;

    // Apply filters if any
    if (search) {
        dataToExport = dataToExport.filter(claim => {
            const claimId = (claim.claim_id || claim.id || '').toLowerCase();
            const cpt = (claim.primary_cpt || claim.cpt_code || '').toLowerCase();
            const carc = (claim.primary_carc || claim.carc_code || '').toLowerCase();
            return claimId.includes(search) || cpt.includes(search) || carc.includes(search);
        });
    }

    if (status !== 'all') {
        dataToExport = dataToExport.filter(claim => {
            const claimStatus = (claim.claim_status || claim.status || '').toLowerCase();
            if (status === 'paid') return !claimStatus.includes('denied') && !claimStatus.includes('partial');
            if (status === 'denied') return claimStatus.includes('denied') || claimStatus === 'full_denial';
            if (status === 'partial') return claimStatus.includes('partial');
            return true;
        });
    }

    if (dataToExport.length === 0) {
        showNotification('No matching claims to download', 'warning');
        return;
    }

    // Aggregate by claim_id to export unique claims
    const claimsById = {};
    dataToExport.forEach(row => {
        const claimId = row.claim_id || row.id || 'N/A';
        if (!claimsById[claimId]) {
            claimsById[claimId] = {
                claim_id: claimId,
                patient_id: row.patient_id || '',
                provider_npi: row.provider_npi || '',
                payer_id: row.payer_id || '',
                date_of_service: row.date_of_service || row.service_date || row.date || '',
                service_lines: [],
                total_charge: 0,
                total_paid: 0,
                claim_status: row.claim_status || row.status || '',
                carc_code: row.carc_code || row.primary_carc || '',
                rarc_code: row.rarc_code || '',
                denial_category: row.denial_category || '',
                is_recoverable: row.is_recoverable || '',
                recovery_action: row.recovery_action || ''
            };
        }
        // Add each service line with its details
        claimsById[claimId].service_lines.push({
            procedure_code: row.procedure_code || row.cpt_code || row.primary_cpt || '',
            charge_amount: parseFloat(row.charge_amount || row.total_charge || 0),
            paid_amount: parseFloat(row.paid_amount || row.total_paid || 0)
        });
        claimsById[claimId].total_charge += parseFloat(row.charge_amount || row.total_charge || 0);
        claimsById[claimId].total_paid += parseFloat(row.paid_amount || row.total_paid || 0);
    });

    const uniqueClaims = Object.values(claimsById);

    // Create JSON with metadata
    const jsonData = {
        metadata: {
            exportDate: new Date().toISOString(),
            totalClaims: uniqueClaims.length,
            filters: {
                search: search || null,
                status: status
            },
            generator: 'SynthERA-835'
        },
        claims: uniqueClaims
    };

    const filename = `synthera835_claims_${new Date().toISOString().split('T')[0]}.json`;
    const jsonString = JSON.stringify(jsonData, null, 2);

    // Try using File System Access API for proper filename control
    if (window.showSaveFilePicker) {
        try {
            const handle = await window.showSaveFilePicker({
                suggestedName: filename,
                types: [{
                    description: 'JSON Files',
                    accept: { 'application/json': ['.json'] }
                }]
            });
            const writable = await handle.createWritable();
            await writable.write(jsonString);
            await writable.close();
            showNotification(`Downloaded ${uniqueClaims.length.toLocaleString()} claims as JSON`, 'success');
            return;
        } catch (err) {
            if (err.name === 'AbortError') {
                showNotification('Download cancelled', 'info');
                return;
            }
        }
    }

    // Fallback: Use blob approach
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    showNotification(`Downloaded ${uniqueClaims.length.toLocaleString()} claims as JSON`, 'success');
}

console.log('SynthERA-835 Dashboard loaded');
