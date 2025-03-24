const ctx = document.getElementById('graph');

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
            data: [
                { x: '2020-01-01', y: 154.0 },
                { x: '2020-02-01', y: 143.0 },
                { x: '2020-03-01', y: 89.0 },
                { x: '2020-04-01', y: 38.0 },
                { x: '2020-05-01', y: 39.0 },
                { x: '2020-06-01', y: 35.0 },
                { x: '2020-07-01', y: 5.0 },
                { x: '2020-08-01', y: 57.0 },
                { x: '2020-09-01', y: 50.0 },
                { x: '2020-10-01', y: 53.0 },
                { x: '2020-11-01', y: 74.0 },
                { x: '2020-12-01', y: 65.0 },
                { x: '2021-01-01', y: 69.0 },
                { x: '2021-02-01', y: 106.0 },
                { x: '2021-03-01', y: 98.0 },
                { x: '2021-04-01', y: 108.0 },
                { x: '2021-05-01', y: 89.0 },
                { x: '2021-06-01', y: 47.0 },
                { x: '2021-07-01', y: 98.0 },
                { x: '2021-08-01', y: 50.0 },
                { x: '2021-09-01', y: 102.0 },
                { x: '2021-10-01', y: 83.0 },
                { x: '2021-11-01', y: 56.0 },
                { x: '2021-12-01', y: 20.0 },
                { x: '2022-01-01', y: 24.0 },
                { x: '2022-02-01', y: 62.0 },
                { x: '2022-03-01', y: 131.0 },
                { x: '2022-04-01', y: 64.0 },
                { x: '2022-05-01', y: 106.0 },
                { x: '2022-06-01', y: 89.0 },
                { x: '2022-07-01', y: 79.0 },
                { x: '2022-08-01', y: 78.0 },
                { x: '2022-09-01', y: 80.0 },
                { x: '2022-10-01', y: 74.0 },
                { x: '2022-11-01', y: 60.0 },
                { x: '2022-12-01', y: 50.0 },
                { x: '2023-01-01', y: 95.0 },
                { x: '2023-02-01', y: 69.0 },
                { x: '2023-03-01', y: 59.0 },
                { x: '2023-05-01', y: 124.0 },
                { x: '2023-06-01', y: 51.0 },
                { x: '2023-07-01', y: 22.0 },
                { x: '2023-08-01', y: 18.0 },
                { x: '2023-09-01', y: 5.0 },
                { x: '2023-10-01', y: 6.0 },
                { x: '2023-11-01', y: 9.0 },
                { x: '2023-12-01', y: 10.0 },
                { x: '2024-02-01', y: 33.0 },
                { x: '2024-03-01', y: 154.0 },
                { x: '2024-04-01', y: 84.0 },
                { x: '2024-05-01', y: 86.0 },
                { x: '2024-06-01', y: 64.0 },
                { x: '2024-07-01', y: 68.0 },
                { x: '2024-08-01', y: 48.0 },
                { x: '2024-09-01', y: 61.0 },
                { x: '2024-10-01', y: 66.0 },
                { x: '2024-11-01', y: 60.0 },
                { x: '2024-12-01', y: 42.0 }
              ],
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
