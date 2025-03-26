const ctx = document.getElementById('graph');
const wellSelector = document.getElementById('well-selector');

function fetchAndUpdateProduction(api_uwi) {
    fetch(`/get_production/?api_uwi=${api_uwi}`)
      .then(res => res.json())
      .then(json => {
        if (!json.data || json.data.length === 0) {
          console.error("No production data found for well:", api_uwi);
          return;
        }
  
        chart.data.datasets[1].data = json.data;
  
        // Update forecast origin to last historical data point
        const lastPoint = json.data[json.data.length - 1];
        currentX = 0;
        currentY = lastPoint.y;
        sendUpdate(); // triggers updateChart
      })
      .catch(err => {
        console.error("Error fetching production data:", err);
      });
  }  
  
// Load FAST_EDIT wells
fetch('/api/fast_edit_wells/')
  .then(res => res.json())
  .then(json => {
    if (!json.wells) throw new Error("No wells returned");
    wellSelector.innerHTML = '';
    json.wells.forEach(well => {
      const opt = document.createElement('option');
      opt.value = well.api;
      opt.textContent = `${well.name} (${well.trajectory})`;
      wellSelector.appendChild(opt);
    });

    // Auto-load first well
    fetchAndUpdateProduction(json.wells[0].api);
  })
  .catch(err => {
    wellSelector.innerHTML = '<option>Error loading wells</option>';
    console.error(err);
  });

wellSelector.addEventListener('change', (e) => {
  const selectedAPI = e.target.value;
  fetchAndUpdateProduction(selectedAPI);
});

let originX = 0;
let originY = 0;
let slope = 1;

const lineData = {
    datasets: [
        {
            label: 'Slope Line',
            data: [],
            borderColor: 'blue',
            showLine: true,
            fill: false,
            pointRadius: 0
        },
        {
            label: 'Oil Production (bbl)',
            data: [],
            borderColor: 'green',
            backgroundColor: 'green',
            showLine: true,
            fill: false,
            pointRadius: 3
          }         
    ]
};

const config = {
    type: 'scatter',
    data: lineData,
    options: {
        responsive: true,
        maintainAspectRatio: false,  // allow full height use
        animation: false,
        scales: {
            x: {
                type: 'time',
                time: {
                    unit: 'month'
                },
                title: {
                    display: true,
                    text: 'Date'
                }
            },
            y: {
                type: 'logarithmic',
                min: 1,
                title: {
                    display: true,
                    text: 'Production (bbl, log scale)'
                },
                ticks: {
                    callback: (value) => value.toLocaleString()
                }
            }
        }
    }
};

const chart = new Chart(ctx, config);

let q_i = 1000; // placeholder, will come from Y joystick
let D_i = 0.15;
let b = 0.5;

function arpsHyperbolic(t, q_i, D_i, b) {
    return q_i / Math.pow(1 + b * D_i * t, 1 / b);
}

function updateChart(x, y, m) {
    q_i = y;
    // Get the last date from historical data (or define your own start date)
    const baseDate = luxon.DateTime.fromISO(chart.data.datasets[1].data[0].x);
    const startDate = baseDate.plus({ months: Math.round(currentX) });
   

    const forecast = [];
    for (let i = 0; i <= 24; i++) {
        const date = startDate.plus({ months: i }).toISODate();
        const q = Math.max(1, arpsHyperbolic(i, q_i, currentDi, currentB));
        forecast.push({ x: date, y: q });
    }    

    chart.data.datasets[0].data = forecast;
    chart.update();

    // ✅ Update stats
    const statsBox = document.getElementById('stats-box');
    statsBox.innerHTML = `
        qᵢ: ${q_i.toFixed(1)}<br>
        Dᵢ: ${currentDi.toFixed(3)}<br>
        b: ${currentB.toFixed(3)}
    `;
}