/* charts.js — Chart.js wrappers matching dc.html design */

const FONT_MONO = "'IBM Plex Mono', ui-monospace, monospace";
const FONT_SANS = "'IBM Plex Sans', -apple-system, sans-serif";
const BORDER_COLOR = '#E4E4DF';

// Active chart registry — keyed by canvas id so we can destroy on re-render
const _charts = {};

function destroyChart(id) {
  if (_charts[id]) { _charts[id].destroy(); delete _charts[id]; }
}

// ── Resilience Radar ──────────────────────────────────────────────────────────
function renderRadar(canvasId, grades, dotColors) {
  destroyChart(canvasId);
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  const axisKeys = ['immunity','utility','safety','honesty','lever'];
  const axisLabels = ['Immunity','Utility','Safety','Honesty','Lever'];
  const colors = dotColors || ['#2F6FDB','#7C5CD6','#2F9E6B','#D39A2F','#D0533F','#9A9A93'];

  const datasets = Object.entries(grades).map(([model, g], idx) => {
    const sub = g.subscores || {};
    const data = axisKeys.map(k => Math.round((sub[k] || 0) * 100));
    const color = colors[idx % colors.length];
    return {
      label: model,
      data,
      borderColor: color,
      backgroundColor: color + '22',
      borderWidth: 2,
      pointBackgroundColor: color,
      pointRadius: 3,
    };
  });

  _charts[canvasId] = new Chart(canvas, {
    type: 'radar',
    data: { labels: axisLabels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      scales: {
        r: {
          min: 0, max: 100,
          ticks: { stepSize: 25, font: { family: FONT_MONO, size: 9 }, color: '#9A9A93' },
          grid: { color: BORDER_COLOR },
          angleLines: { color: BORDER_COLOR },
          pointLabels: { font: { family: FONT_MONO, size: 10 }, color: '#1A1A19' },
        }
      },
      plugins: {
        legend: { display: true, labels: { font: { family: FONT_MONO, size: 10 }, usePointStyle: true } },
      },
    },
  });
}

// ── Live Outcome Donut ────────────────────────────────────────────────────────
function renderDonut(canvasId, outcomes) {
  destroyChart(canvasId);
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  const labels = Object.keys(outcomes);
  const data   = Object.values(outcomes);
  const colors = labels.map(l => OUTCOME_COLORS[l] || '#9A9A93');

  _charts[canvasId] = new Chart(canvas, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data, backgroundColor: colors,
        borderColor: '#fff', borderWidth: 2, hoverBorderWidth: 2,
      }],
    },
    options: {
      cutout: '55%',
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => ` ${ctx.label}: ${ctx.parsed}` } },
      },
    },
  });
}

// ── ΔImmunity Heatmap (using a custom grid, not Chart.js) ────────────────────
function renderDeltaHeatmap(container, deltaData, refModel) {
  if (!deltaData || !Object.keys(deltaData).length) {
    container.innerHTML = '<p style="color:var(--muted);font-family:var(--font-mono);font-size:11px">No ΔImmunity data for ' + refModel + '</p>';
    return;
  }
  const attacks  = Object.keys(deltaData).sort();
  const defenses = [...new Set(attacks.flatMap(a => Object.keys(deltaData[a])))].sort();

  const scaleColor = (v) => {
    // v in [-1, 1] -> red (negative) to green (positive)
    if (v >= 0) {
      const g = Math.round(v * 100);
      return `rgba(21,128,61,${(v * 0.8 + 0.05).toFixed(2)})`;
    } else {
      return `rgba(185,28,28,${(Math.abs(v) * 0.8 + 0.05).toFixed(2)})`;
    }
  };

  const headers = defenses.map(d => `<th style="font-family:var(--font-mono);font-size:9px;padding:5px 8px;white-space:nowrap;border-bottom:1px solid var(--border)">${d}</th>`).join('');
  const rows = attacks.map(atk => {
    const cells = defenses.map(def => {
      const cell = deltaData[atk]?.[def] || {};
      const delta = cell.delta || 0;
      const bg = scaleColor(delta);
      const tc = Math.abs(delta) > 0.35 ? '#fff' : '#1A1A19';
      return `<td style="text-align:center;padding:6px 8px;background:${bg};border:1px solid #F4F4F0">
        <span style="font-family:var(--font-mono);font-size:10px;color:${tc};font-weight:500">${delta >= 0 ? '+' : ''}${Math.round(delta*100)}%</span>
      </td>`;
    }).join('');
    return `<tr>
      <td style="font-family:var(--font-mono);font-size:9.5px;color:#3A3A37;padding:6px 12px 6px 0;white-space:nowrap;border-right:1px solid var(--border)">${atk}</td>
      ${cells}
    </tr>`;
  }).join('');

  container.innerHTML = `
<table style="border-collapse:collapse;width:100%;overflow-x:auto">
  <thead><tr><th style="border-bottom:1px solid var(--border)"></th>${headers}</tr></thead>
  <tbody>${rows}</tbody>
</table>`;
}

// ── Confusion matrix heatmap ──────────────────────────────────────────────────
function renderConfusionMatrix(container, confusion, categories) {
  const maxVal = Math.max(...categories.flatMap(h => categories.map(m => confusion[h]?.[m] || 0)), 1);
  const headers = categories.map(c => `<th style="font-family:var(--font-mono);font-size:8.5px;padding:4px 6px;white-space:nowrap;border-bottom:1px solid var(--border)">${c.replace(/_/g,' ')}</th>`).join('');
  const rows = categories.map(h => {
    const cells = categories.map(m => {
      const v = confusion[h]?.[m] || 0;
      const intensity = v / maxVal;
      const bg = `rgba(47,111,219,${intensity.toFixed(2)})`;
      const tc = intensity > 0.5 ? '#fff' : '#1A1A19';
      return `<td style="text-align:center;padding:6px 8px;background:${bg};border:1px solid #F4F4F0">
        <span style="font-family:var(--font-mono);font-size:10px;color:${tc}">${v}</span>
      </td>`;
    }).join('');
    return `<tr>
      <td style="font-family:var(--font-mono);font-size:9px;color:#3A3A37;padding:5px 10px 5px 0;white-space:nowrap;border-right:1px solid var(--border)">${h.replace(/_/g,' ')}</td>
      ${cells}
    </tr>`;
  }).join('');
  container.innerHTML = `
<div style="overflow-x:auto">
<table style="border-collapse:collapse;min-width:max-content">
  <thead><tr><th style="border-bottom:1px solid var(--border)"></th>${headers}</tr></thead>
  <tbody>${rows}</tbody>
</table>
</div>`;
}
