let currentX = 0;
let currentY = 0;
let currentDi = 0.15;
let currentB = 0.5;


let lastTouchOrigin = null;
let lastTouchSlope = null;

function sendUpdate(live = false) {
    updateChart(currentX, currentY, null); // no slope now

    if (live) {
        fetch(`/update/?x=${currentX.toFixed(2)}&y=${currentY.toFixed(2)}&Di=${currentDi.toFixed(3)}&b=${currentB.toFixed(3)}`);
    }
}

// Touch zone for moving origin
const originZone = document.getElementById('touch-origin');
originZone.addEventListener('touchstart', (e) => {
    lastTouchOrigin = e.touches[0];
});

originZone.addEventListener('touchend', () => {
    lastTouchOrigin = null;
    sendUpdate(true); // 👈 sync with backend here
});
originZone.addEventListener('touchmove', (e) => {
    const touch = e.touches[0];
    if (!lastTouchOrigin) return;

    const dx = (touch.clientX - lastTouchOrigin.clientX) * 1;
    const dy = -(touch.clientY - lastTouchOrigin.clientY) * 5;

    currentX += dx;
    currentY += dy;
    
    currentX = Math.max(-24, currentX);  // Limit to 2 years before data

    lastTouchOrigin = touch;
    sendUpdate();  // 👈 just update chart, no fetch
});

// Touch zone for adjusting slope
const slopeZone = document.getElementById('touch-slope');
slopeZone.addEventListener('touchstart', (e) => {
    lastTouchSlope = e.touches[0];
});

slopeZone.addEventListener('touchend', () => {
    lastTouchSlope = null;
    sendUpdate(true); // 👈 sync with backend here
});

slopeZone.addEventListener('touchmove', (e) => {
    const touch = e.touches[0];
    if (!lastTouchSlope) return;

    const dx = (touch.clientX - lastTouchSlope.clientX) * 0.005; // b-factor
    const dy = -(touch.clientY - lastTouchSlope.clientY) * 0.05; // Di

    currentB += dx;
    currentDi += dy;

    // clamp values
    currentB = Math.max(0.0, Math.min(1.5, currentB));
    currentDi = Math.max(0.01, Math.min(1.0, currentDi));

    lastTouchSlope = touch;
    sendUpdate();
});


// window.addEventListener('load', () => {
//     currentX = 0;
//     currentY = 0;
//     currentM = 1;
//     sendUpdate();
// });
