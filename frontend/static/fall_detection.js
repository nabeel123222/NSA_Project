// =========================================================
// NSA — FALL DETECTION FRONTEND
// =========================================================

// -------------------------------
// ELEMENTS
// -------------------------------

const fallStatus =
    document.getElementById("fallStatus");

const fallDescription =
    document.getElementById("fallDescription");

const fallProbability =
    document.getElementById("fallProbability");

const normalProbability =
    document.getElementById("normalProbability");

const fallConfidenceBar =
    document.getElementById("fallConfidenceBar");

const friendRate =
    document.getElementById("friendRate");

const youRate =
    document.getElementById("youRate");

const fallIcon =
    document.getElementById("fallIcon");

const fallLive =
    document.getElementById("fallLive");


// =========================================================
// UPDATE FALL UI
// =========================================================

let lastFallAlert = false;

// =========================================================
// EMERGENCY ALERT SOUND
// =========================================================

let alertAudioContext = null;
let alertInterval = null;


function playAlertBeep() {

    if (!alertAudioContext) {
        alertAudioContext =
            new (window.AudioContext ||
                window.webkitAudioContext)();
    }

    const oscillator =
        alertAudioContext.createOscillator();

    const gain =
        alertAudioContext.createGain();

    oscillator.type = "sine";

    oscillator.frequency.setValueAtTime(
        880,
        alertAudioContext.currentTime
    );

    gain.gain.setValueAtTime(
        0.0001,
        alertAudioContext.currentTime
    );

    gain.gain.exponentialRampToValueAtTime(
        0.25,
        alertAudioContext.currentTime + 0.03
    );

    gain.gain.exponentialRampToValueAtTime(
        0.0001,
        alertAudioContext.currentTime + 0.35
    );

    oscillator.connect(gain);
    gain.connect(alertAudioContext.destination);

    oscillator.start();

    oscillator.stop(
        alertAudioContext.currentTime + 0.35
    );
}


function startAlertSound() {

    if (alertInterval) {
        return;
    }

    playAlertBeep();

    alertInterval = setInterval(() => {
        playAlertBeep();
    }, 1000);
}


function stopAlertSound() {

    if (alertInterval) {
        clearInterval(alertInterval);
        alertInterval = null;
    }
}

// =========================================================
// EMERGENCY FALL ALERT
// =========================================================

const fallAlertOverlay =
    document.getElementById("fallAlertOverlay");

const fallAlertTime =
    document.getElementById("fallAlertTime");

const acknowledgeFallBtn =
    document.getElementById("acknowledgeFallBtn");

const testAlertSoundBtn =
    document.getElementById("testAlertSoundBtn");


if (testAlertSoundBtn) {

    testAlertSoundBtn.addEventListener(
        "click",
        () => {
            playAlertBeep();
        }
    );

}

let fallAlertActive = false;


// Show emergency alert

function showEmergencyFallAlert() {

    if (fallAlertActive) {
        return;
    }

    fallAlertActive = true;

    const now = new Date();

    fallAlertTime.textContent =
        now.toLocaleTimeString();

    fallAlertOverlay.classList.add("active");

    startAlertSound();

}


// Hide emergency alert

async function acknowledgeFallAlert() {

    try {

        const response = await fetch(
            "/api/alerts/acknowledge",
            {
                method: "POST"
            }
        );

        const data = await response.json();

        if (data.success) {

            console.log(
                "Alert acknowledged:",
                data.alert_id
            );

        } else {

            console.warn(
                "Alert acknowledgement failed:",
                data.message
            );

        }

    } catch (error) {

        console.error(
            "Alert acknowledgement error:",
            error
        );

    }

    fallAlertActive = false;

    fallAlertOverlay.classList.remove("active");

    stopAlertSound();
}


// Acknowledge button

if (acknowledgeFallBtn) {

    acknowledgeFallBtn.addEventListener(
        "click",
        acknowledgeFallAlert
    );

}

function updateFallUI(data) {

    const state =
        data.fall_state || "NORMAL";

    const fall =
        Number(data.fall_probability || 0);

    const normal =
        Number(data.normal_probability || 0);

    const friend =
        Number(data.friend_rate || 0);

    const you =
        Number(data.you_rate || 0);


    const fallPercent =
        Math.round(fall * 100);

    const normalPercent =
        Math.round(normal * 100);


    // -------------------------------
    // TEXT
    // -------------------------------

    fallStatus.textContent =
        state;

    fallProbability.textContent =
        fallPercent + "%";

    normalProbability.textContent =
        normalPercent + "%";

    friendRate.textContent =
        friend.toFixed(1) + " Hz";

    youRate.textContent =
        you.toFixed(1) + " Hz";


    // -------------------------------
    // PROGRESS BAR
    // -------------------------------

    fallConfidenceBar.style.width =
        fallPercent + "%";


    // =====================================================
    // FALL DETECTED
    // =====================================================

    if (state === "FALL DETECTED") {

    showEmergencyFallAlert();

        fallStatus.style.color =
            "#d62828";

        fallDescription.textContent =
            "Fall-like CSI pattern detected.";

        fallIcon.className =
            "fall-icon alert";

        fallIcon.textContent =
            "!";

        fallConfidenceBar.style.background =
            "#d62828";

        fallLive.innerHTML =
            "<span></span> ALERT";

        fallLive.style.color =
            "#d62828";

        fallLive.style.background =
            "#fff0f0";

    }


    // =====================================================
    // NORMAL
    // =====================================================

    else {

        fallStatus.style.color =
            "#22a447";

        fallDescription.textContent =
            "No fall pattern detected.";

        fallIcon.className =
            "fall-icon";

        fallIcon.textContent =
            "✓";

        fallConfidenceBar.style.background =
            "#22a447";

        fallLive.innerHTML =
            "<span></span> LIVE";

        fallLive.style.color =
            "#22a447";

        fallLive.style.background =
            "#eaf8ee";

    }

}


// =========================================================
// GET FALL STATUS
// =========================================================

async function updateFallStatus() {

    try {

        const response =
            await fetch(
                "/api/fall/status",
                {
                    cache: "no-store"
                }
            );


        if (!response.ok) {

            throw new Error(
                "Fall status API returned " +
                response.status
            );

        }


        const data =
            await response.json();


        updateFallUI(data);

    }


    catch (error) {

        console.error(
            "Fall detection status error:",
            error
        );

        fallStatus.textContent =
            "OFFLINE";

        fallStatus.style.color =
            "#888888";

        fallDescription.textContent =
            "Waiting for fall detection backend.";

        fallLive.innerHTML =
            "<span></span> OFFLINE";

        fallLive.style.color =
            "#888888";

        fallLive.style.background =
            "#f2f2f2";

    }

}


// =========================================================
// START POLLING
// =========================================================

// Update immediately
updateFallStatus();

// Update every 1 second
setInterval(
    updateFallStatus,
    1000
);
