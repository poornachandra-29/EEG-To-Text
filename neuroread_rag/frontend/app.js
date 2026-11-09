/**
 * NeuroRead AI Frontend Client Application.
 * Connects to REST API endpoints, renders interactive charts, and manages RAG search.
 */

document.addEventListener("DOMContentLoaded", () => {
    initApp();
});

let reportData = null;
let participantData = null;

async function initApp() {
    setupNavigation();
    setupDownloadButtons();
    setupResearchSearch();
    setupZuCoLauncher();

    try {
        // Fetch report and participant data in parallel
        const [repRes, partRes] = await Promise.all([
            fetch("/api/report"),
            fetch("/api/participant")
        ]);

        if (!repRes.ok || !partRes.ok) {
            throw new Error("Failed to load pipeline data from server API.");
        }

        reportData = await repRes.json();
        participantData = await partRes.json();

        renderHeaderInfo();
        renderExecutiveSummary();
        renderSentenceBreakdownSection();
        renderFocusAttentionSection();
        renderCognitiveWorkloadSection();
        renderMetricAssessmentSection();
        renderResearchComparisonTable();
        renderPersonalizedRecommendations();
        renderCompleteReportView();
        triggerDefaultSearch();

    } catch (err) {
        console.error("Initialization error:", err);
        document.getElementById("executive-summary-text").innerHTML = 
            `<div style="color:#dc2626;">Error connecting to API server: ${err.message}. Make sure server.py is running.</div>`;
    }
}

/* Navigation Switching */
function setupNavigation() {
    const navItems = document.querySelectorAll(".nav-item");
    const sections = document.querySelectorAll(".dashboard-section");

    navItems.forEach(item => {
        item.addEventListener("click", (e) => {
            e.preventDefault();
            navItems.forEach(n => n.classList.remove("active"));
            sections.forEach(s => s.classList.remove("active"));

            item.classList.add("active");
            const targetId = item.getAttribute("data-section");
            const targetSec = document.getElementById(targetId);
            if (targetSec) {
                targetSec.classList.add("active");
                window.scrollTo({ top: 0, behavior: "smooth" });
            }
        });
    });
}

/* Header & Downloads */
async function renderHeaderInfo() {
    const pid = reportData.metadata.participant_id;
    const total = reportData.metadata.total_sentences_analyzed;
    const headerName = document.getElementById("header-patient-name");
    const sidebarName = document.getElementById("sidebar-patient-name");
    const totalDisplay = document.getElementById("total-sentences-display");
    const switchSelect = document.getElementById("switch-patient-select");
    
    if (headerName) headerName.innerText = pid;
    if (sidebarName) sidebarName.innerText = pid;
    if (totalDisplay) totalDisplay.innerText = `${total} Sentences`;

    // Fetch available participants for the homepage switcher
    if (switchSelect) {
        try {
            const res = await fetch("/api/participants");
            const data = await res.json();
            switchSelect.innerHTML = "";
            data.available.forEach(p => {
                const opt = document.createElement("option");
                opt.value = p;
                opt.innerText = p;
                if (p === pid) opt.selected = true;
                switchSelect.appendChild(opt);
            });

            switchSelect.onchange = async () => {
                const newPid = switchSelect.value;
                if (newPid && newPid !== pid) {
                    if (totalDisplay) totalDisplay.innerText = "Switching...";
                    await fetch(`/api/switch_participant?id=${newPid}`);
                    window.location.reload();
                }
            };
        } catch (e) {
            switchSelect.innerHTML = `<option value="${pid}">${pid}</option>`;
        }
    }
}

function setupDownloadButtons() {
    document.getElementById("btn-download-md").addEventListener("click", () => {
        window.location.href = "/api/download/report.md";
    });
    document.getElementById("btn-download-json").addEventListener("click", () => {
        window.location.href = "/api/download/report.json";
    });
    document.getElementById("btn-print-report").addEventListener("click", () => {
        // Switch to full-report view before printing
        const fullReportNav = document.querySelector('[data-section="full-report"]');
        if (fullReportNav) fullReportNav.click();
        setTimeout(() => window.print(), 300);
    });
}

/* 1. Executive Summary Rendering */
function renderExecutiveSummary() {
    document.getElementById("executive-summary-text").innerText = 
        reportData.section_1_executive_summary;

    // Render Learning Efficiency Score (LES) Hero Card
    const les = reportData.learning_efficiency || {};
    const fatigue = reportData.session_fatigue_curve || {};
    const scoreVal = document.getElementById("les-score-value");
    const tierBadge = document.getElementById("les-tier-badge");
    const flowVal = document.getElementById("les-flow-val");
    const cadenceVal = document.getElementById("les-cadence-val");
    const pomoVal = document.getElementById("les-pomodoro-val");

    if (scoreVal && les.score !== undefined) {
        scoreVal.innerText = `${les.score}%`;
    }
    if (tierBadge && les.tier) {
        tierBadge.innerText = les.tier;
        if (les.color) tierBadge.style.background = les.color;
    }
    if (flowVal) {
        flowVal.innerText = `${les.flow_factor || les.flow_percentage || 0}%`;
    }
    if (cadenceVal) {
        cadenceVal.innerText = `${les.cadence_stability_factor || 75}%`;
    }
    if (pomoVal) {
        pomoVal.innerText = `${fatigue.recommended_study_block_minutes || 25} min`;
    }

    const grid = document.getElementById("summary-cards-grid");
    grid.innerHTML = "";

    const coreKeys = [
        { id: "EEI", label: "EEG Engagement Index", unit: "" },
        { id: "FMT", label: "Frontal Midline Theta", unit: "µV²" },
        { id: "GD", label: "Gaze Duration", unit: "ms" },
        { id: "TRT", label: "Total Reading Time", unit: "ms" },
        { id: "PupilSize", label: "Pupil Diameter", unit: "units" },
        { id: "CWI", label: "Cognitive Workload Index", unit: "z" },
        { id: "SAI", label: "Sustained Attention Index", unit: "z" }
    ];

    coreKeys.forEach(k => {
        const m = participantData.metrics[k.id];
        if (!m) return;

        const badgeClass = m.deviation_category === "Within Typical Range" 
            ? "badge-typical" 
            : (m.deviation_category.includes("Elevated") ? "badge-elevated" : "badge-below");

        const card = document.createElement("div");
        card.className = "metric-card";
        card.innerHTML = `
            <div class="metric-header">
                <span class="metric-title">${k.label}</span>
                <span class="metric-badge ${badgeClass}">${m.deviation_category}</span>
            </div>
            <div class="metric-val">${m.mean} <span style="font-size:14px; font-weight:normal; color:#64748b;">${k.unit}</span></div>
            <div class="metric-sub">Session Baseline: ${m.session_baseline} (${m.relative_deviation_pct >= 0 ? '+' : ''}${m.relative_deviation_pct}%)</div>
        `;
        grid.appendChild(card);
    });
}

/* 3-Pillar Formatter: Plain English translation without jargon */
function format3PillarText(summaryRaw) {
    if (!summaryRaw) return "";
    let data = summaryRaw;
    if (typeof summaryRaw === "string") {
        try {
            data = JSON.parse(summaryRaw);
        } catch (e) {
            return `<p style="line-height:1.6; color:#334155;">${summaryRaw}</p>`;
        }
    }
    if (data && data.what_is_happening) {
        return `
            <div style="background:#f8fafc; border-left:4px solid #3b82f6; padding:12px 16px; border-radius:6px; margin-bottom:14px;">
                <p style="margin:0 0 8px 0; font-size:13px; line-height:1.5;"><strong style="color:#dc2626;">🔴 What Is Happening:</strong> ${data.what_is_happening}</p>
                <p style="margin:0 0 8px 0; font-size:13px; line-height:1.5;"><strong style="color:#2563eb;">📊 What You Need:</strong> ${data.what_you_need}</p>
                <p style="margin:0; font-size:13px; line-height:1.5;"><strong style="color:#16a34a;">✅ How To Improve:</strong> ${data.how_to_improve}</p>
            </div>
        `;
    }
    return `<p style="line-height:1.6; color:#334155;">${summaryRaw}</p>`;
}

/* Sentence-Level Breakdown: Flow vs. Comprehension Difficulty */
function renderSentenceBreakdownSection() {
    const s6b = reportData.section_6b_sentence_comparison;
    if (!s6b) return;

    // Contrast Insight text
    const insightEl = document.getElementById("contrast-insight-text");
    if (insightEl && s6b.contrast_insight) {
        insightEl.innerText = s6b.contrast_insight;
    }

    // Truncate helper
    const trunc = (txt, lim = 55) => (txt && txt.length > lim ? txt.substring(0, lim) + "..." : txt);

    // Flow table
    const flowContainer = document.getElementById("flow-sentences-container");
    if (flowContainer && s6b.flow_sentences) {
        let html = `<table style="width:100%; border-collapse:collapse; font-size:12.5px;">
            <thead><tr style="background:#dcfce7; color:#14532d; text-align:left;">
                <th style="padding:6px 8px;">#</th>
                <th style="padding:6px 8px;">Sentence</th>
                <th style="padding:6px 8px;">Words</th>
                <th style="padding:6px 8px;">Fix</th>
                <th style="padding:6px 8px;">EEI</th>
                <th style="padding:6px 8px;">Grade</th>
                <th style="padding:6px 8px;">Why Smooth</th>
            </tr></thead><tbody>`;
        s6b.flow_sentences.forEach(s => {
            html += `<tr style="border-bottom:1px solid #bbf7d0;">
                <td style="padding:6px 8px; font-weight:700;">#${s.rank}</td>
                <td style="padding:6px 8px; font-style:italic;">"${trunc(s.sentence_text)}"</td>
                <td style="padding:6px 8px;">${s.word_count}</td>
                <td style="padding:6px 8px; font-weight:600;">${s.fixation_count}</td>
                <td style="padding:6px 8px; color:#15803d; font-weight:700;">${s.eei}</td>
                <td style="padding:6px 8px; color:#475569;">${s.grade_level ? 'Gr ' + s.grade_level : '-'}</td>
                <td style="padding:6px 8px; color:#166534; font-size:11.5px;">${s.why_smooth}</td>
            </tr>`;
        });
        html += `</tbody></table>`;
        flowContainer.innerHTML = html;
    }

    // Breakdown table
    const breakContainer = document.getElementById("breakdown-sentences-container");
    if (breakContainer && s6b.breakdown_sentences) {
        let html = `<table style="width:100%; border-collapse:collapse; font-size:12.5px;">
            <thead><tr style="background:#fee2e2; color:#7f1d1d; text-align:left;">
                <th style="padding:6px 8px;">#</th>
                <th style="padding:6px 8px;">Sentence</th>
                <th style="padding:6px 8px;">Words</th>
                <th style="padding:6px 8px;">Fix</th>
                <th style="padding:6px 8px;">EEI</th>
                <th style="padding:6px 8px;">Grade</th>
                <th style="padding:6px 8px;">Linguistic Root Cause</th>
            </tr></thead><tbody>`;
        s6b.breakdown_sentences.forEach(s => {
            const rc = s.linguistic_root_cause || s.likely_difficulty || "Comprehension Friction";
            html += `<tr style="border-bottom:1px solid #fecaca;">
                <td style="padding:6px 8px; font-weight:700;">#${s.rank}</td>
                <td style="padding:6px 8px; font-style:italic;">"${trunc(s.sentence_text)}"</td>
                <td style="padding:6px 8px;">${s.word_count}</td>
                <td style="padding:6px 8px; font-weight:700; color:#b91c1c;">${s.fixation_count}</td>
                <td style="padding:6px 8px; color:#b91c1c; font-weight:600;">${s.eei}</td>
                <td style="padding:6px 8px; color:#475569;">${s.grade_level ? 'Gr ' + s.grade_level : '-'}</td>
                <td style="padding:6px 8px; color:#991b1b; font-size:11.5px; font-weight:600;">${rc}</td>
            </tr>`;
        });
        html += `</tbody></table>`;
        breakContainer.innerHTML = html;
    }

    // Personal Optimal Zone
    const optContainer = document.getElementById("optimal-zone-container");
    if (optContainer) {
        optContainer.innerHTML = `
            <div style="background:#eff6ff; padding:12px 16px; border-radius:6px; flex:1; min-width:180px;">
                <div style="font-size:12px; color:#1e40af; font-weight:600;">Optimal Sentence Length</div>
                <div style="font-size:18px; font-weight:700; color:#1e3a8a;">10 &ndash; 24 words</div>
                <div style="font-size:11px; color:#64748b; margin-top:2px;">Longer sentences trigger refixation clusters</div>
            </div>
            <div style="background:#eff6ff; padding:12px 16px; border-radius:6px; flex:1; min-width:180px;">
                <div style="font-size:12px; color:#1e40af; font-weight:600;">Target Fixation Density</div>
                <div style="font-size:18px; font-weight:700; color:#1e3a8a;">&lt; 0.09 fix/word</div>
                <div style="font-size:11px; color:#64748b; margin-top:2px;">You average 0.07 on effortless sentences</div>
            </div>
            <div style="background:#eff6ff; padding:12px 16px; border-radius:6px; flex:1; min-width:180px;">
                <div style="font-size:12px; color:#1e40af; font-weight:600;">Recommended Break Interval</div>
                <div style="font-size:18px; font-weight:700; color:#1e3a8a;">Every 20&ndash;25 mins</div>
                <div style="font-size:11px; color:#64748b; margin-top:2px;">Ultradian pauses prevent theta-alpha fatigue drift</div>
            </div>
        `;
    }
}

/* 2. Focus & Attention Section */
function renderFocusAttentionSection() {
    const s2 = reportData.section_2_focus_attention;
    const statsContainer = document.getElementById("focus-stats-container");

    statsContainer.innerHTML = `
        ${format3PillarText(s2.summary)}
        <div style="display:flex; flex-direction:column; gap:10px; margin-top:12px;">
            <div style="background:#f1f5f9; padding:12px; border-radius:6px;">
                <strong>Sustained Attention Evidence Proxy (SAI):</strong> 
                <span style="font-family:monospace; font-size:15px; color:#2563eb;">${s2.sai_mean > 0 ? '+' : ''}${s2.sai_mean}</span>
                <div style="font-size:12px; color:#64748b; margin-top:2px;">Category: ${s2.sai_category} relative to participant baseline</div>
            </div>
            <div style="background:#f1f5f9; padding:12px; border-radius:6px;">
                <strong>Attention-Instability Evidence Proxy (MWLI_proxy):</strong> 
                <span style="font-family:monospace; font-size:15px; color:#d97706;">${s2.mwli_mean > 0 ? '+' : ''}${s2.mwli_mean}</span>
                <div style="font-size:12px; color:#64748b; margin-top:2px;">Derived from inverse EEI and reading dwell variability</div>
            </div>
        </div>
    `;

    // Render SVG Bar Chart of Cognitive Patterns
    renderPatternChart(s2.patterns_distribution);
}

function renderPatternChart(patterns) {
    const container = document.getElementById("pattern-chart-container");
    if (!patterns || patterns.length === 0) return;

    const width = 450;
    const barHeight = 28;
    const gap = 12;
    const height = patterns.length * (barHeight + gap) + 20;

    let svg = `<svg viewBox="0 0 ${width} ${height}" style="width:100%; height:auto;">`;

    patterns.forEach((p, idx) => {
        const y = idx * (barHeight + gap) + 10;
        const maxBarW = 200;
        const barW = Math.max(8, (p.percentage / 100) * maxBarW);
        const color = p.pattern_name.includes("Focused") ? "#16a34a" 
                    : p.pattern_name.includes("Workload") ? "#d97706"
                    : p.pattern_name.includes("Fatigue") ? "#dc2626"
                    : p.pattern_name.includes("Instability") ? "#9333ea" : "#64748b";

        svg += `
            <text x="0" y="${y + 18}" class="svg-bar-label">${p.pattern_name}</text>
            <rect x="220" y="${y}" width="${barW}" height="${barHeight}" rx="4" fill="${color}" opacity="0.85" />
            <text x="${220 + barW + 8}" y="${y + 19}" class="svg-bar-val">${p.percentage}% (${p.sentence_count})</text>
        `;
    });

    svg += `</svg>`;
    container.innerHTML = svg;
}

/* 3. Cognitive Workload Section */
function renderCognitiveWorkloadSection() {
    const s5 = reportData.section_5_cognitive_workload;
    const banner = document.getElementById("workload-overview-banner");
    banner.innerHTML = `
        <h4 style="font-size:15px; margin-bottom:8px; color:#1e40af;">Workload Profile Synthesis (3-Pillar Assessment)</h4>
        ${format3PillarText(s5.summary)}
        <div style="margin-top:10px; font-size:13px;">
            <strong>Composite Workload Index (CWI):</strong> <span style="font-family:monospace; font-weight:700;">${s5.cwi_mean > 0 ? '+' : ''}${s5.cwi_mean} robust-z</span> 
            &nbsp;|&nbsp; <strong>Relative Shift:</strong> ${s5.cwi_deviation >= 0 ? '+' : ''}${s5.cwi_deviation}%
        </div>
    `;

    const factorsList = document.getElementById("workload-factors-list");
    factorsList.innerHTML = "";
    s5.contributing_factors.forEach(f => {
        const div = document.createElement("div");
        div.style = "padding:8px 12px; background:#f8fafc; border-left:3px solid #2563eb; margin-bottom:8px; font-size:13px;";
        div.innerText = f;
        factorsList.appendChild(div);
    });

    const clustersList = document.getElementById("cluster-profiles-list");
    clustersList.innerHTML = "";
    reportData.section_6_sentence_observations.clusters.forEach(c => {
        const div = document.createElement("div");
        div.style = "padding:10px 14px; background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; margin-bottom:8px;";
        div.innerHTML = `
            <div style="display:flex; justify-content:space-between; font-weight:600; font-size:13px;">
                <span>${c.cluster_name} (${c.percentage}%)</span>
                <span style="color:#2563eb;">Conf: ${(c.mean_membership_probability * 100).toFixed(0)}%</span>
            </div>
            <div style="font-size:12px; color:#64748b; margin-top:2px;">Dominant Pattern: ${c.dominant_pattern}</div>
        `;
        clustersList.appendChild(div);
    });
}

/* 4. Metric-wise Scientific Assessment */
function renderMetricAssessmentSection() {
    const grid = document.getElementById("detailed-metrics-grid");
    const tabs = document.querySelectorAll("#metric-filter-tabs .tab-btn");

    function renderMetrics(filter) {
        grid.innerHTML = "";
        const allInterps = {
            ...reportData.section_3_eeg_analysis.metrics,
            ...reportData.section_4_eye_tracking_analysis.metrics,
            ...getCompositeMetrics()
        };

        Object.keys(allInterps).forEach(k => {
            const m = allInterps[k];
            if (filter !== "ALL" && m.category !== filter) return;

            const pm = m.participant_measurement;
            const pb = m.personal_baseline_comparison;
            const statusClass = pb.deviation_category === "Within Typical Range" ? "badge-typical" : "badge-elevated";

            const card = document.createElement("div");
            card.className = "detailed-card";
            card.innerHTML = `
                <div class="title-row">
                    <h4>${m.metric_name} (${m.metric_id})</h4>
                    <span class="metric-badge ${statusClass}">${pb.deviation_category}</span>
                </div>
                <div style="font-size:12px; color:#64748b;">${m.scientific_definition}</div>
                <div class="stats-row">
                    <div><strong>Mean:</strong> ${pm.mean} ${pm.unit}</div>
                    <div><strong>Baseline:</strong> ${pb.session_baseline}</div>
                    <div><strong>Shift:</strong> ${pb.relative_deviation_pct >= 0 ? '+' : ''}${pb.relative_deviation_pct}%</div>
                </div>
                <div class="compat-box">
                    <strong>Literature Compatibility:</strong> ${m.research_compatibility}
                </div>
            `;
            grid.appendChild(card);
        });
    }

    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            tabs.forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            renderMetrics(tab.getAttribute("data-filter"));
        });
    });

    renderMetrics("ALL");
}

function getCompositeMetrics() {
    const composites = {};
    const fullInterps = reportData.section_7_research_comparison;
    fullInterps.forEach(c => {
        if (["REI", "CWI", "SAI", "MWLI_proxy", "VSI_proxy"].includes(c.metric_id)) {
            composites[c.metric_id] = {
                metric_id: c.metric_id,
                metric_name: c.metric_name,
                category: "Composite Proxy",
                scientific_definition: "Multimodal exploratory composite index derived from spectral EEG and eye-tracking signals.",
                participant_measurement: { mean: c.participant_value, unit: "" },
                personal_baseline_comparison: { session_baseline: c.session_baseline, relative_deviation_pct: c.relative_deviation, deviation_category: c.status },
                research_compatibility: c.scientific_alignment
            };
        }
    });
    return composites;
}

/* 5. Research Evidence Comparison Table */
function renderResearchComparisonTable() {
    const tbody = document.getElementById("comparison-table-body");
    tbody.innerHTML = "";

    reportData.section_7_research_comparison.forEach(item => {
        const tr = document.createElement("tr");
        const statusClass = item.status === "Within Typical Range" ? "badge-typical" : "badge-elevated";
        const doiLink = item.doi ? `<a href="https://doi.org/${item.doi}" target="_blank" style="color:#2563eb;">${item.doi}</a>` : "Peer-Reviewed";

        tr.innerHTML = `
            <td><strong>${item.metric_name}</strong> (${item.metric_id})</td>
            <td style="font-family:monospace;">${item.participant_value}</td>
            <td style="font-family:monospace;">${item.session_baseline}</td>
            <td><span class="metric-badge ${statusClass}">${item.status}</span></td>
            <td>${item.key_reference} <br><small>${doiLink}</small></td>
            <td style="font-size:12px; color:#475569; max-width:320px;">${item.scientific_alignment}</td>
        `;
        tbody.appendChild(tr);
    });
}

/* 6. Research Explorer */
function setupResearchSearch() {
    const searchBtn = document.getElementById("btn-research-search");
    const searchInput = document.getElementById("research-search-input");
    const metricSelect = document.getElementById("research-metric-select");

    async function executeSearch() {
        const q = searchInput.value.trim();
        const m = metricSelect.value;
        const container = document.getElementById("search-results-container");

        container.innerHTML = `<div style="text-align:center; padding:20px; color:#64748b;">Searching literature vector store...</div>`;

        try {
            const url = `/api/search?q=${encodeURIComponent(q)}&metric=${encodeURIComponent(m)}`;
            const res = await fetch(url);
            const data = await res.json();

            container.innerHTML = "";
            if (!data.results || data.results.length === 0) {
                container.innerHTML = `<div style="padding:20px; color:#64748b;">No literature found matching "${q}".</div>`;
                return;
            }

            data.results.forEach(p => {
                const card = document.createElement("div");
                card.className = "paper-card";
                card.innerHTML = `
                    <h4>${p.paper_title}</h4>
                    <div class="paper-meta">
                        <strong>${p.authors} (${p.year})</strong> &nbsp;|&nbsp; 
                        <span>DOI: <a href="https://doi.org/${p.doi}" target="_blank" style="color:#2563eb;">${p.doi}</a></span> &nbsp;|&nbsp;
                        <span style="color:#16a34a; font-weight:600;">Relevance: ${(p.relevance_score * 100).toFixed(0)}%</span>
                    </div>
                    <div class="paper-snippet">${p.content}</div>
                `;
                container.appendChild(card);
            });
        } catch (e) {
            container.innerHTML = `<div style="color:#dc2626;">Search failed: ${e.message}</div>`;
        }
    }

    searchBtn.addEventListener("click", executeSearch);
    searchInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") executeSearch();
    });
}

function triggerDefaultSearch() {
    document.getElementById("research-search-input").value = "Frontal midline theta cognitive workload";
    document.getElementById("btn-research-search").click();
}

/* 7. Personalized Recommendations */
function renderPersonalizedRecommendations() {
    const container = document.getElementById("recommendations-container");
    container.innerHTML = "";

    reportData.section_8_recommendations.forEach(r => {
        const card = document.createElement("div");
        card.className = "rec-card-full";
        const citationsHtml = r.supporting_citations.map(c => 
            `<a href="https://doi.org/${c.doi}" target="_blank" style="color:#2563eb; text-decoration:underline;">${c.citation_text}</a>`
        ).join(", ");

        card.innerHTML = `
            <h3>${r.title} <span class="rec-tag">${r.category}</span></h3>
            <p style="margin-top:8px; font-size:13.5px;"><strong>Observed Basis:</strong> ${r.observed_basis}</p>
            <p style="margin-top:6px; font-size:13.5px; color:#1e40af; background:#eff6ff; padding:10px; border-radius:6px;">
                <strong>Actionable Strategy:</strong> ${r.actionable_guidance}
            </p>
            <p style="margin-top:8px; font-size:13px; color:#475569;"><strong>Scientific Rationale:</strong> ${r.scientific_rationale}</p>
            <p style="margin-top:8px; font-size:12px; color:#64748b;"><strong>Supporting Citations:</strong> ${citationsHtml}</p>
        `;
        container.appendChild(card);
    });
}

/* 8. Complete Report View */
function renderCompleteReportView() {
    const container = document.getElementById("printable-report-container");
    container.innerHTML = `
        <div style="border-bottom:2px solid #e2e8f0; padding-bottom:14px; margin-bottom:20px;">
            <h1 style="font-size:22px; margin-bottom:6px;">NeuroRead AI: Research-Grounded Assessment Report</h1>
            <div style="font-size:13px; color:#64748b;">Participant: ${reportData.metadata.participant_id} | Protocol: ZuCo Simultaneous EEG & Eye Tracking</div>
        </div>
        <div style="margin-bottom:18px;">
            <h3 style="color:#1e3a8a;">1. Executive Summary</h3>
            <p style="font-size:13.5px; line-height:1.6;">${reportData.section_1_executive_summary}</p>
        </div>
        <div style="margin-bottom:18px;">
            <h3 style="color:#1e3a8a;">2. Focus & Attention</h3>
            ${format3PillarText(reportData.section_2_focus_attention.summary)}
        </div>
        <div style="margin-bottom:18px;">
            <h3 style="color:#1e3a8a;">3. Spectral EEG Analysis</h3>
            ${format3PillarText(reportData.section_3_eeg_analysis.summary)}
        </div>
        <div style="margin-bottom:18px;">
            <h3 style="color:#1e3a8a;">4. Eye-Tracking Analysis</h3>
            ${format3PillarText(reportData.section_4_eye_tracking_analysis.summary)}
        </div>
        <div style="margin-bottom:18px;">
            <h3 style="color:#1e3a8a;">5. Cognitive Workload Evidence</h3>
            ${format3PillarText(reportData.section_5_cognitive_workload.summary)}
        </div>
        <div style="margin-bottom:18px;">
            <h3 style="color:#1e3a8a;">6. Scientific Limitations & Guardrails</h3>
            <ul style="font-size:13px; color:#475569; padding-left:20px;">
                ${reportData.section_9_scientific_limitations.map(l => `<li>${l}</li>`).join("")}
            </ul>
        </div>
        <div>
            <h3 style="color:#1e3a8a;">7. Scientific References</h3>
            <ol style="font-size:12.5px; color:#334155; padding-left:20px;">
                ${reportData.section_10_references.map(r => `<li><strong>${r.authors} (${r.year})</strong>. <em>${r.title}</em>. ${r.doi ? `doi:${r.doi}` : ""}</li>`).join("")}
            </ol>
        </div>
    `;
}

/* ZuCo Dataset Task & Patient Pipeline Launcher */
let zucoTasksData = [];

async function setupZuCoLauncher() {
    const taskSelect = document.getElementById("zuco-task-select");
    const patientSelect = document.getElementById("zuco-patient-select");
    const runBtn = document.getElementById("btn-run-zuco-pipeline");
    const progressBanner = document.getElementById("pipeline-progress-banner");
    const progressText = document.getElementById("pipeline-progress-text");
    const progressSub = document.getElementById("pipeline-progress-sub");

    if (!taskSelect || !patientSelect || !runBtn) return;

    try {
        const res = await fetch("/api/zuco/tasks");
        const data = await res.json();
        zucoTasksData = data.tasks || [];

        taskSelect.innerHTML = "";
        zucoTasksData.forEach((t, idx) => {
            const opt = document.createElement("option");
            opt.value = t.task_id;
            opt.innerText = `${t.task_name} (${t.patient_count} students)`;
            if (idx === 0) opt.selected = true;
            taskSelect.appendChild(opt);
        });

        function populatePatients(taskId) {
            patientSelect.innerHTML = "";
            const taskObj = zucoTasksData.find(t => t.task_id === taskId);
            if (!taskObj || !taskObj.patients) return;

            taskObj.patients.forEach(p => {
                const opt = document.createElement("option");
                opt.value = p.filename;
                opt.innerText = `${p.filename} (Student ${p.patient_code})`;
                if (reportData && reportData.metadata && reportData.metadata.participant_id === p.patient_id) {
                    opt.selected = true;
                }
                patientSelect.appendChild(opt);
            });
        }

        taskSelect.onchange = () => {
            populatePatients(taskSelect.value);
        };

        if (zucoTasksData.length > 0) {
            populatePatients(zucoTasksData[0].task_id);
        }

        // Run Pipeline on Selected Task and Student .mat file
        runBtn.onclick = async () => {
            const selectedTask = taskSelect.value;
            const selectedPatient = patientSelect.value;
            if (!selectedTask || !selectedPatient) {
                alert("Please select both study material and a student .mat file.");
                return;
            }

            // Disable button and show progress
            runBtn.disabled = true;
            runBtn.style.opacity = "0.6";
            progressBanner.style.display = "block";
            progressText.innerText = `Processing ${selectedPatient} from ${selectedTask}...`;
            progressSub.innerText = "Step 1/4: Reading .mat file & extracting sentence-level EEG / eye-tracking features...";

            let progressTimer = setInterval(() => {
                if (progressSub.innerText.includes("Step 1/4")) {
                    progressSub.innerText = "Step 2/4: Applying trained RobustScaler & GMM cluster predictions...";
                } else if (progressSub.innerText.includes("Step 2/4")) {
                    progressSub.innerText = "Step 3/4: Generating sentence-level CSV & running deterministic analysis...";
                } else if (progressSub.innerText.includes("Step 3/4")) {
                    progressSub.innerText = "Step 4/4: Querying literature vector store & synthesizing grounded report...";
                }
            }, 3000);

            try {
                const response = await fetch("/api/pipeline/run", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        task: selectedTask,
                        patient_mat: selectedPatient
                    })
                });

                clearInterval(progressTimer);
                const result = await response.json();

                if (!response.ok || result.error) {
                    throw new Error(result.error || "Pipeline run failed on server.");
                }

                progressText.innerText = `✓ Pipeline completed successfully for ${result.participant_id}! (${result.sentences_count} sentences)`;
                progressSub.innerText = "Updating dashboard with new participant results...";

                // Fetch new data and re-render dashboard
                const [repRes, partRes] = await Promise.all([
                    fetch("/api/report"),
                    fetch("/api/participant")
                ]);

                reportData = await repRes.json();
                participantData = await partRes.json();

                renderHeaderInfo();
                renderExecutiveSummary();
                renderFocusAttentionSection();
                renderCognitiveWorkloadSection();
                renderMetricAssessmentSection();
                renderResearchComparisonTable();
                renderPersonalizedRecommendations();
                renderCompleteReportView();

                setTimeout(() => {
                    progressBanner.style.display = "none";
                    runBtn.disabled = false;
                    runBtn.style.opacity = "1";
                }, 2500);

            } catch (err) {
                clearInterval(progressTimer);
                progressText.innerText = "❌ Pipeline run encountered an error:";
                progressSub.innerText = err.message;
                runBtn.disabled = false;
                runBtn.style.opacity = "1";
            }
        };

    } catch (e) {
        console.error("Failed to load ZuCo tasks:", e);
    }
}
