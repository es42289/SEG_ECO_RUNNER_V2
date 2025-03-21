# example.py
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

# Initialize position
if "x" not in st.session_state:
    st.session_state.x = 0
if "y" not in st.session_state:
    st.session_state.y = 0

st.title("Joystick-Controlled Point on Graph")

# HTML + JS joystick (nipplejs)
components.html(
    """
    <div id="zone_joystick" style="width:200px;height:200px;position:relative;margin:auto;"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/nipplejs/0.9.0/nipplejs.min.js"></script>
    <script>
        const zone = document.getElementById('zone_joystick');
        const joystick = nipplejs.create({
            zone: zone,
            mode: 'static',
            position: { left: '50%', top: '50%' },
            color: 'blue'
        });

        let lastSent = 0;
        let throttleTime = 100; // ms

        function sendDelta(dx, dy) {
            const now = new Date().getTime();
            if (now - lastSent > throttleTime) {
                const msg = {dx: dx, dy: dy};
                window.parent.postMessage(msg, "*");
                lastSent = now;
            }
        }

        joystick.on('move', function(evt, data) {
            let dx = Math.round(data.vector.x);
            let dy = Math.round(data.vector.y);
            sendDelta(dx, dy);
        });

        joystick.on('end', function() {
            sendDelta(0, 0);
        });
    </script>
    """,
    height=250,
)

# Receive message from JS (requires streamlit-webrtc or other websocket-like system)
delta = st.experimental_get_query_params()
dx = int(delta.get("dx", [0])[0])
dy = int(delta.get("dy", [0])[0])

# Apply changes (this part needs to be driven by the front-end ideally, or replaced with buttons for real-time feedback)
st.session_state.x += dx
st.session_state.y += dy

# Plot point
fig, ax = plt.subplots()
ax.plot(st.session_state.x, st.session_state.y, "ro")
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.grid(True)
st.pyplot(fig)
