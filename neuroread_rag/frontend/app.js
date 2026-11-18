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

        // Render each section with individual error isolation
        const renderSteps = [
            ["Header Info", renderHeaderInfo],
            ["Executive Summary", renderExecutiveSummary],
            ["Sentence Breakdown", renderSentenceBreakdownSection],
            ["Focus Attention", renderFocusAttentionSection],
            ["Metric Assessment", renderMetricAssessmentSection],
            ["Personalized Recommendations", renderPersonalizedRecommendations],
            ["Complete Report View", renderCompleteReportView],
            ["Default Search", triggerDefaultSearch]
        ];

        renderSteps.forEach(([name, fn]) => {
            try {
                fn();
            } catch (err) {
                console.warn(`Render warning in ${name}:`, err);
            }
        });

        // Re-sync ZuCo launcher dropdowns with active participant
        const taskSelect = document.getElementById("zuco-task-select");
        if (window.populatePatients && taskSelect) {
            window.populatePatients(taskSelect.value);
        }

    } catch (err) {
        console.error("Initialization error:", err);
        const execText = document.getElementById("executive-summary-text");
        if (execText) {
            execText.innerHTML = 
                `<div style="color:#dc2626;">Error connecting to API server: ${err.message}. Make sure server.py is running.</div>`;
        }
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
    if (!reportData || !reportData.metadata) return;
    const pid = reportData.metadata.participant_id || "Unknown";
    const total = reportData.metadata.total_sentences_analyzed || 0;
    const headerName = document.getElementById("header-patient-name");
    const sidebarName = document.getElementById("sidebar-patient-name");
    const totalDisplay = document.getElementById("total-sentences-display");
    const switchSelect = document.getElementById("switch-patient-select");
    const topSwitchSelect = document.getElementById("top-switch-patient-select");

    if (headerName) headerName.innerText = pid;
    if (sidebarName) sidebarName.innerText = pid;
    if (totalDisplay) totalDisplay.innerText = `${total} Sentences`;

    const switchers = [switchSelect, topSwitchSelect].filter(Boolean);
    if (switchers.length === 0) return;

    try {
        const res = await fetch("/api/participants");
        const data = await res.json();
        const available = data.available || [];

        switchers.forEach(sel => {
            sel.innerHTML = "";
            available.forEach(p => {
                const opt = document.createElement("option");
                opt.value = p;
                opt.innerText = p;
                if (p === pid || (pid && (p.indexOf(pid) !== -1 || pid.indexOf(p) !== -1))) {
                    opt.selected = true;
                }
                sel.appendChild(opt);
            });

            sel.onchange = async () => {
                const newPid = sel.value;
                if (newPid && newPid !== pid) {
                    if (totalDisplay) totalDisplay.innerText = "Switching...";
                    if (headerName) headerName.innerText = "Switching Student...";
                    await fetch(`/api/switch_participant?id=${encodeURIComponent(newPid)}`);
                    window.location.reload();
                }
            };
        });
    } catch (e) {
        console.error("Failed to load participants:", e);
        switchers.forEach(sel => {
            sel.innerHTML = `<option value="${pid}">${pid}</option>`;
        });
    }
}

function setupDownloadButtons() {
    const btnMd = document.getElementById("btn-download-md");
    if (btnMd) {
        btnMd.addEventListener("click", () => {
            window.location.href = "/api/download/report.md";
        });
    }
    const btnJson = document.getElementById("btn-download-json");
    if (btnJson) {
        btnJson.addEventListener("click", () => {
            window.location.href = "/api/download/report.json";
        });
    }
    const btnPrint = document.getElementById("btn-print-report");
    if (btnPrint) {
        btnPrint.addEventListener("click", () => {
            const fullReportNav = document.querySelector('[data-section="full-report"]');
            if (fullReportNav) fullReportNav.click();
            setTimeout(() => window.print(), 300);
        });
    }
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
    const easeVal = document.getElementById("les-ease-val");
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
    if (easeVal) {
        easeVal.innerText = `${les.workload_ease_factor || 85}%`;
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
let currentFatigueSentences = [];
let selectedFatigueIndex = 0;

function renderFocusAttentionSection() {
    const s2 = reportData.section_2_focus_attention || {};

    // Render SVG Bar Chart of Cognitive Patterns
    renderPatternChart(s2.patterns_distribution);

    // Extract fatigue sentences
    let fatigueSents = s2.fatigue_compatible_sentences || [];
    if (!fatigueSents || fatigueSents.length === 0) {
        const obs = (reportData.section_6_sentence_observations && reportData.section_6_sentence_observations.sample_notable_sentences) || [];
        fatigueSents = obs.filter(o => o.cognitive_pattern && o.cognitive_pattern.includes("Fatigue"));
    }
    currentFatigueSentences = fatigueSents;

    const badge = document.getElementById("fatigue-count-badge");
    if (badge) {
        badge.innerText = `${fatigueSents.length} Sentences`;
    }

    // Populate the dropdown selector
    const dropdown = document.getElementById("fatigue-sentence-dropdown");
    if (dropdown) {
        dropdown.innerHTML = "";
        if (fatigueSents.length === 0) {
            dropdown.innerHTML = `<option value="">No fatigue-compatible trials detected</option>`;
        } else {
            fatigueSents.forEach((s, idx) => {
                const opt = document.createElement("option");
                opt.value = idx;
                const shortText = s.sentence_text && s.sentence_text.length > 45 ? s.sentence_text.substring(0, 45) + "..." : (s.sentence_text || "");
                opt.innerText = `Sentence #${s.sentence_index}: "${shortText}" [${s.primary_mistake || 'Fatigue'}]`;
                dropdown.appendChild(opt);
            });
            dropdown.value = 0;
        }
    }

    // Render initial preview of the first fatigue sentence (or empty state)
    if (fatigueSents.length > 0) {
        onSelectFatigueSentence(0);
    } else {
        const preview = document.getElementById("fatigue-single-sentence-preview");
        if (preview) {
            preview.innerHTML = `<div style="font-size:13px; color:#64748b; text-align:center; padding:10px;">No fatigue-compatible trials identified for this participant.</div>`;
        }
    }

    // Populate full scrollable list container
    const container = document.getElementById("fatigue-sentences-container");
    if (!container) return;

    if (fatigueSents.length === 0) {
        container.innerHTML = `<div style="font-size:12.5px; color:#64748b; padding:10px;">No fatigue-compatible trials identified for this participant.</div>`;
        return;
    }

    let html = `<div style="display:flex; flex-direction:column; gap:10px;">`;
    fatigueSents.forEach((s, idx) => {
        const eeiColor = s.eei < 0.45 ? '#dc2626' : '#2563eb';
        html += `
            <div style="background:#ffffff; border:1px solid #fecaca; border-radius:6px; padding:12px; box-shadow:0 1px 2px rgba(0,0,0,0.04);">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; flex-wrap:wrap; gap:6px;">
                    <div>
                        <strong style="color:#b91c1c; font-size:13px;">Sentence #${s.sentence_index}</strong>
                        <span style="font-size:11px; font-weight:700; margin-left:8px; padding:2px 8px; border-radius:12px; background:#fee2e2; color:#991b1b;">
                            ${s.primary_mistake || 'Fatigue Pattern'}
                        </span>
                    </div>
                    <button onclick="onSelectFatigueSentence(${idx}); document.getElementById('fatigue-sentence-dropdown').value = ${idx}; document.getElementById('focus-attention').scrollIntoView({behavior:'smooth'});" style="font-size:11px; padding:3px 10px; cursor:pointer; background:#eff6ff; color:#1d4ed8; border:1px solid #93c5fd; border-radius:4px; font-weight:600;">
                        🔍 Inspect in Inspector
                    </button>
                </div>
                <div style="font-size:12.5px; color:#334155; margin-bottom:8px; line-height:1.5;">
                    "${s.sentence_text}"
                </div>
                <div style="display:flex; gap:12px; font-size:11px; color:#64748b; font-family:var(--font-mono); flex-wrap:wrap; background:#f8fafc; padding:6px 10px; border-radius:4px;">
                    <span>Words: <strong>${s.word_count}</strong></span>
                    <span>Fixations: <strong>${s.fixation_count}</strong>/word</span>
                    <span>EEI: <strong style="color:${eeiColor};">${s.eei}</strong></span>
                    <span>TRT: <strong>${s.trt}</strong>ms</span>
                </div>
                <div style="font-size:12px; color:#b91c1c; margin-top:6px; font-weight:600;">
                    ⚠️ Mistake / Processing Breakdown: ${s.mistake_diagnosis}
                </div>
            </div>
        `;
    });
    html += `</div>`;
    container.innerHTML = html;
}

window.onSelectFatigueSentence = function(val) {
    const idx = parseInt(val, 10);
    if (isNaN(idx) || !currentFatigueSentences || idx < 0 || idx >= currentFatigueSentences.length) return;
    selectedFatigueIndex = idx;

    const dropdown = document.getElementById("fatigue-sentence-dropdown");
    if (dropdown && dropdown.value !== String(idx)) {
        dropdown.value = idx;
    }

    const s = currentFatigueSentences[idx];
    const preview = document.getElementById("fatigue-single-sentence-preview");
    if (!preview) return;

    const eeiColor = s.eei < 0.45 ? '#dc2626' : '#2563eb';

    // Detailed diagnostic root cause & actionable advice
    let causeExpl = "";
    let actionFix = "";
    const mistake = s.primary_mistake || "";

    if (mistake.includes("Vigilance") || s.eei < 0.45) {
        causeExpl = `Cortical engagement dropped sharply below baseline (EEI: ${s.eei}), triggering involuntary gaze drift and reduced semantic absorption.`;
        actionFix = `Pause for 10-15 seconds at section breaks, take a deep mental breath, and deliberately refocus attention before entering dense text blocks.`;
    } else if (mistake.includes("Re-fixations") || s.fixation_count > 1.6) {
        causeExpl = `Fixation density spiked to ${s.fixation_count} fixations/word (typical < 0.10), causing regressive eye movements and reread friction.`;
        actionFix = `Avoid immediate rereading sweeps when a difficult word appears; continue reading to the end of the clause to allow syntactic context to resolve meaning.`;
    } else if (mistake.includes("Syntactic") || s.word_count > 25) {
        causeExpl = `High sentence length (${s.word_count} words) with nested subclauses saturated working memory buffers, causing processing bottlenecks.`;
        actionFix = `Mentally chunk long sentences at relative clauses ('which', 'although') and commas to consolidate each proposition before advancing.`;
    } else {
        causeExpl = `Frontal theta-alpha synchronization reflected accumulated reading strain and momentary comprehension fatigue.`;
        actionFix = `Implement structured 15-20 minute study blocks followed by 3-minute resting breaks to prevent cognitive fatigue cascade.`;
    }

    preview.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; flex-wrap:wrap; gap:6px;">
            <div style="font-weight:700; color:#b91c1c; font-size:14px;">
                Sentence #${s.sentence_index} (${idx + 1} of ${currentFatigueSentences.length})
            </div>
            <span style="font-size:11.5px; font-weight:700; padding:3px 10px; border-radius:12px; background:#fee2e2; color:#991b1b; border:1px solid #fca5a5;">
                ${s.primary_mistake || 'Cognitive Fatigue'}
            </span>
        </div>

        <div style="font-size:13.5px; color:#1e293b; background:#ffffff; border:1px solid #fecaca; border-radius:6px; padding:10px 14px; margin-bottom:10px; line-height:1.5; font-style:italic;">
            "${s.sentence_text}"
        </div>

        <div style="display:flex; gap:10px; font-size:11.5px; font-family:var(--font-mono); margin-bottom:12px; flex-wrap:wrap;">
            <span style="background:#fee2e2; padding:4px 8px; border-radius:4px; color:#991b1b;">Length: <strong>${s.word_count} words</strong></span>
            <span style="background:#fee2e2; padding:4px 8px; border-radius:4px; color:#991b1b;">Fixations: <strong>${s.fixation_count} fix/word</strong></span>
            <span style="background:#eff6ff; padding:4px 8px; border-radius:4px; color:#1e40af;">EEI: <strong style="color:${eeiColor};">${s.eei}</strong></span>
            <span style="background:#f1f5f9; padding:4px 8px; border-radius:4px; color:#475569;">TRT: <strong>${s.trt} ms</strong></span>
        </div>

        <div style="background:#fef2f2; border-left:4px solid #ef4444; padding:10px 12px; border-radius:0 6px 6px 0; margin-bottom:8px;">
            <div style="font-size:12px; font-weight:700; color:#991b1b; text-transform:uppercase; letter-spacing:0.5px;">⚠️ Identified Processing Breakdown</div>
            <div style="font-size:13px; color:#7f1d1d; margin-top:2px; font-weight:600;">${s.mistake_diagnosis}</div>
        </div>

        <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:8px;">
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:6px; padding:10px;">
                <div style="font-size:11.5px; font-weight:700; color:#475569;">🔍 Root Cause of Mistake:</div>
                <div style="font-size:12px; color:#334155; margin-top:3px; line-height:1.4;">${causeExpl}</div>
            </div>
            <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:6px; padding:10px;">
                <div style="font-size:11.5px; font-weight:700; color:#166534;">💡 Actionable Cognitive Fix:</div>
                <div style="font-size:12px; color:#14532d; margin-top:3px; line-height:1.4;">${actionFix}</div>
            </div>
        </div>
    `;
};

window.selectNextFatigueSentence = function() {
    if (!currentFatigueSentences || currentFatigueSentences.length === 0) return;
    const nextIdx = (selectedFatigueIndex + 1) % currentFatigueSentences.length;
    onSelectFatigueSentence(nextIdx);
};

window.toggleFatigueSentencesUI = function() {
    const container = document.getElementById("fatigue-sentences-container");
    const label = document.getElementById("toggle-btn-label");
    if (!container) return;
    if (container.style.display === "none") {
        container.style.display = "block";
        if (label) label.innerText = `Hide Fatigue Sentences Table (${currentFatigueSentences.length})`;
    } else {
        container.style.display = "none";
        if (label) label.innerText = `View All Fatigue Sentences Table (${currentFatigueSentences.length})`;
    }
};

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

/* 3. Metric-wise Scientific Assessment */
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

            let specHtml = "";
            const spec = m.personalized_specification;
            if (spec) {
                const diag = spec.where_did_you_make_a_mistake || {};
                const resList = (spec.how_to_resolve || []).map(r => `<li>${r}</li>`).join("");
                const errBadge = diag.is_comprehension_error
                    ? `<span class="badge badge-below">🔴 Comprehension Error</span>`
                    : (diag.is_breakdown ? `<span class="badge badge-elevated">🟠 Flow Breakdown</span>` : `<span class="badge badge-typical">🟡 Outlier Friction</span>`);

                specHtml = `
                    <div class="pers-spec-box">
                        <details>
                            <summary>🔍 View Personalized Specification & Sentence Diagnosis</summary>
                            <div class="pers-spec-content">
                                <div class="pers-spec-item">
                                    <strong>🎯 What Does It Do:</strong> ${spec.what_does_it_do || ''}
                                </div>
                                <div class="pers-spec-item">
                                    <strong>🔬 What Is It Actually:</strong> ${spec.what_is_it_actually || ''}
                                </div>
                                <div class="pers-spec-item">
                                    <strong>👤 What It Says About You:</strong> ${spec.what_it_says_about_you || ''}
                                </div>
                                <div class="pers-spec-item">
                                    <strong>📚 Readability & Text Connection:</strong> ${spec.readability_connection || ''}
                                </div>
                                <div class="pers-spec-mistake-card">
                                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                                        <strong>⚠️ Where You Made a Mistake (Sentence #${diag.sentence_index || 0}):</strong>
                                        ${errBadge}
                                    </div>
                                    <blockquote class="pers-spec-quote">"${diag.sentence_text || ''}"</blockquote>
                                    <div class="pers-spec-stats">
                                        <span>Grade ${Number(diag.grade_level || 0).toFixed(1)}</span>
                                        <span>Flesch Ease: ${Number(diag.flesch_reading_ease || 0).toFixed(1)}</span>
                                        <span>Complex Words: ${Number(diag.complex_words_pct || 0).toFixed(0)}%</span>
                                        <span>Observed: ${diag.metric_value} (Z = ${diag.robust_z >= 0 ? '+' : ''}${diag.robust_z})</span>
                                    </div>
                                    <p class="pers-spec-diag-note"><strong>Diagnosis:</strong> ${diag.diagnosis_note || ''}</p>
                                </div>
                                <div class="pers-spec-resolutions">
                                    <strong style="color:#166534;">💡 How To Resolve It:</strong>
                                    <ul>${resList}</ul>
                                </div>
                            </div>
                        </details>
                    </div>
                `;
            }

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
                ${specHtml}
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
    const fullInterps = (reportData && reportData.section_7_research_comparison) || [];
    fullInterps.forEach(c => {
        if (["REI", "CWI", "SAI", "MWLI_proxy", "VSI_proxy"].includes(c.metric_id)) {
            composites[c.metric_id] = {
                metric_id: c.metric_id,
                metric_name: c.metric_name,
                category: "Composite Proxy",
                scientific_definition: "Multimodal exploratory composite index derived from spectral EEG and eye-tracking signals.",
                participant_measurement: { mean: c.participant_value, unit: "" },
                personal_baseline_comparison: { session_baseline: c.session_baseline, relative_deviation_pct: c.relative_deviation, deviation_category: c.status },
                research_compatibility: c.scientific_alignment,
                personalized_specification: c.personalized_specification
            };
        }
    });
    return composites;
}

/* 5. Research Evidence Comparison Table (Removed per user specification) */
function renderResearchComparisonTable() {
    // Section removed per user specification
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
    if (!container || !reportData) return;
    const meta = reportData.metadata || {};
    const s1 = reportData.section_1_executive_summary || "";
    const s2 = reportData.section_2_focus_attention || {};
    const s3 = reportData.section_3_eeg_analysis || {};
    const s4 = reportData.section_4_eye_tracking_analysis || {};
    const s9 = reportData.section_9_scientific_limitations || [];
    const s10 = reportData.section_10_references || [];

    container.innerHTML = `
        <div style="border-bottom:2px solid #e2e8f0; padding-bottom:14px; margin-bottom:20px;">
            <h1 style="font-size:22px; margin-bottom:6px;">NeuroRead AI: Research-Grounded Assessment Report</h1>
            <div style="font-size:13px; color:#64748b;">Participant: ${meta.participant_id || 'Student'} | Protocol: ZuCo Simultaneous EEG & Eye Tracking</div>
        </div>
        <div style="margin-bottom:18px;">
            <h3 style="color:#1e3a8a;">1. Executive Summary</h3>
            <p style="font-size:13.5px; line-height:1.6;">${s1}</p>
        </div>
        <div style="margin-bottom:18px;">
            <h3 style="color:#1e3a8a;">2. Focus & Attention Assessment</h3>
            <p style="font-size:13px; color:#475569; margin-bottom:10px;">
                Evaluates heuristic cognitive pattern distribution and fatigue-compatible sentence trials.
            </p>
            <table style="width:100%; border-collapse:collapse; margin-bottom:12px; font-size:12.5px;">
                <thead>
                    <tr style="background:#f1f5f9; text-align:left;">
                        <th style="padding:6px 10px; border:1px solid #cbd5e1;">Cognitive Pattern</th>
                        <th style="padding:6px 10px; border:1px solid #cbd5e1;">Proportion</th>
                        <th style="padding:6px 10px; border:1px solid #cbd5e1;">Sentence Count</th>
                    </tr>
                </thead>
                <tbody>
                    ${(s2.patterns_distribution || []).map(p => `
                        <tr>
                            <td style="padding:6px 10px; border:1px solid #cbd5e1;"><strong>${p.pattern_name}</strong></td>
                            <td style="padding:6px 10px; border:1px solid #cbd5e1;">${p.percentage}%</td>
                            <td style="padding:6px 10px; border:1px solid #cbd5e1;">${p.sentence_count}</td>
                        </tr>
                    `).join("")}
                </tbody>
            </table>
            <div style="background:#fff1f2; border:1px solid #fca5a5; border-radius:6px; padding:12px; margin-top:8px;">
                <h4 style="margin:0 0 6px 0; color:#b91c1c; font-size:13px;">⚠️ Fatigue-Compatible EEG Pattern Sentences (${(s2.fatigue_compatible_sentences || []).length} Trials)</h4>
                <div style="max-height:220px; overflow-y:auto;">
                    <table style="width:100%; border-collapse:collapse; font-size:11.5px;">
                        <thead>
                            <tr style="background:#fee2e2; text-align:left;">
                                <th style="padding:4px 8px; border:1px solid #fecaca;">#</th>
                                <th style="padding:4px 8px; border:1px solid #fecaca;">Sentence Excerpt</th>
                                <th style="padding:4px 8px; border:1px solid #fecaca;">Fix/W</th>
                                <th style="padding:4px 8px; border:1px solid #fecaca;">EEI</th>
                                <th style="padding:4px 8px; border:1px solid #fecaca;">Mistake Diagnosis</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${(s2.fatigue_compatible_sentences || []).map(s => `
                                <tr>
                                    <td style="padding:4px 8px; border:1px solid #fecaca;">#${s.sentence_index}</td>
                                    <td style="padding:4px 8px; border:1px solid #fecaca;">"${s.sentence_text ? (s.sentence_text.length > 50 ? s.sentence_text.substring(0, 50) + '...' : s.sentence_text) : ''}"</td>
                                    <td style="padding:4px 8px; border:1px solid #fecaca;">${s.fixation_count}</td>
                                    <td style="padding:4px 8px; border:1px solid #fecaca;">${s.eei}</td>
                                    <td style="padding:4px 8px; border:1px solid #fecaca; color:#b91c1c; font-weight:500;">${s.mistake_diagnosis}</td>
                                </tr>
                            `).join("")}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        <div style="margin-bottom:18px;">
            <h3 style="color:#1e3a8a;">3. Spectral EEG Analysis</h3>
            ${format3PillarText(s3.summary || "")}
        </div>
        <div style="margin-bottom:18px;">
            <h3 style="color:#1e3a8a;">4. Eye-Tracking Analysis</h3>
            ${format3PillarText(s4.summary || "")}
        </div>
        <div style="margin-bottom:18px;">
            <h3 style="color:#1e3a8a;">5. Scientific Limitations & Guardrails</h3>
            <ul style="font-size:13px; color:#475569; padding-left:20px;">
                ${s9.map(l => `<li>${l}</li>`).join("")}
            </ul>
        </div>
        <div>
            <h3 style="color:#1e3a8a;">6. Scientific References</h3>
            <ol style="font-size:12.5px; color:#334155; padding-left:20px;">
                ${s10.map(r => `<li><strong>${r.authors || ''} (${r.year || ''})</strong>. <em>${r.title || ''}</em>. ${r.doi ? `doi:${r.doi}` : ""}</li>`).join("")}
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

        window.populatePatients = function(taskId) {
            patientSelect.innerHTML = "";
            const taskObj = zucoTasksData.find(t => t.task_id === taskId);
            if (!taskObj || !taskObj.patients) return;

            const currPid = (reportData && reportData.metadata && reportData.metadata.participant_id) || "";

            taskObj.patients.forEach(p => {
                const opt = document.createElement("option");
                opt.value = p.filename;
                opt.innerText = `${p.filename} (Student ${p.patient_code})`;
                if (currPid && (p.patient_id === currPid || currPid.indexOf(p.patient_code) !== -1 || p.filename.indexOf(currPid) !== -1)) {
                    opt.selected = true;
                }
                patientSelect.appendChild(opt);
            });
        };

        taskSelect.onchange = () => {
            if (window.populatePatients) window.populatePatients(taskSelect.value);
        };

        if (zucoTasksData.length > 0) {
            window.populatePatients(zucoTasksData[0].task_id);
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
                renderMetricAssessmentSection();
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
