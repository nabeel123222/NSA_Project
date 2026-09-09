// =========================================================
// NSA — HUMAN DETECTION FRONTEND
// =========================================================

// -------------------------------
// ELEMENTS
// -------------------------------

const monitorBtn = document.getElementById("monitorBtn");
const stopBtn = document.getElementById("stopBtn");

const humanStatus = document.getElementById("humanStatus");
const detectionDescription =
    document.getElementById("detectionDescription");

const detectionIcon =
    document.getElementById("detectionIcon");

const confidence =
    document.getElementById("confidence");

const confidenceBar =
    document.getElementById("confidenceBar");

const presentProbability =
    document.getElementById("presentProbability");

const emptyProbability =
    document.getElementById("emptyProbability");

const rawPrediction =
    document.getElementById("rawPrediction");

const sampleRate =
    document.getElementById("sampleRate");

const signalRate =
    document.getElementById("signalRate");

const espStatus =
    document.getElementById("espStatus");

const roomStatus =
    document.getElementById("roomStatus");

const logBox =
    document.getElementById("logBox");

const logStatus =
    document.getElementById("logStatus");

const liveIndicator =
    document.getElementById("liveIndicator");


// -------------------------------
// VARIABLES
// -------------------------------

let monitoring = false;

let statusTimer = null;

let lastState = "";


// =========================================================
// TIME
// =========================================================

function getCurrentTime() {

    const now = new Date();

    return now.toLocaleTimeString(
        [],
        {
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit"
        }
    );
}


// =========================================================
// LOG
// =========================================================

function addLog(message) {

    const entry =
        document.createElement("div");

    entry.className = "log-entry";

    entry.innerHTML =
        `
        <span class="log-time">
            ${getCurrentTime()}
        </span>

        <span>
            ${message}
        </span>
        `;

    logBox.appendChild(entry);

    logBox.scrollTop =
        logBox.scrollHeight;

    // Keep log small
    while (logBox.children.length > 30) {

        logBox.removeChild(
            logBox.firstChild
        );

    }
}


// =========================================================
// UI STATE
// =========================================================

function setDetectionState(
    state,
    confidenceValue
) {

    const percentage =
        Math.round(
            confidenceValue * 100
        );


    // -----------------------------------------
    // PRESENT
    // -----------------------------------------

    if (state === "PRESENT") {

        humanStatus.innerHTML =
            "PERSON DETECTED";

        humanStatus.style.color =
            "#22a447";

        detectionDescription.innerHTML =
            "A person is detected inside the CSI sensing zone.";

        detectionIcon.className =
            "detection-icon";

        detectionIcon.innerHTML =
            "<span>●</span>";

        confidenceBar.style.width =
            percentage + "%";

        confidenceBar.style.background =
            "#22a447";

        liveIndicator.innerHTML =
            "<span></span> LIVE";

        liveIndicator.style.color =
            "#22a447";

        liveIndicator.style.background =
            "#eaf8ee";

        logStatus.innerHTML =
            "Person Detected";

        logStatus.style.color =
            "#22a447";

    }


    // -----------------------------------------
    // EMPTY
    // -----------------------------------------

    else if (state === "EMPTY") {

        humanStatus.innerHTML =
            "ROOM EMPTY";

        humanStatus.style.color =
            "#777777";

        detectionDescription.innerHTML =
            "No person detected inside the CSI sensing zone.";

        detectionIcon.className =
            "detection-icon empty";

        detectionIcon.innerHTML =
            "<span>○</span>";

        confidenceBar.style.width =
            percentage + "%";

        confidenceBar.style.background =
            "#888888";

        liveIndicator.innerHTML =
            "<span></span> LIVE";

        liveIndicator.style.color =
            "#777777";

        liveIndicator.style.background =
            "#f2f2f2";

        logStatus.innerHTML =
            "Room Empty";

        logStatus.style.color =
            "#777777";

    }


    // -----------------------------------------
    // STARTING / STOPPED
    // -----------------------------------------

    else {

        humanStatus.innerHTML =
            state;

        humanStatus.style.color =
            "#c98b00";

        detectionDescription.innerHTML =
            "Waiting for CSI data from the ESP32.";

        detectionIcon.className =
            "detection-icon warning";

        detectionIcon.innerHTML =
            "<span>◌</span>";

        confidenceBar.style.width =
            "0%";

        liveIndicator.innerHTML =
            "<span></span> WAITING";

        liveIndicator.style.color =
            "#c98b00";

        liveIndicator.style.background =
            "#fff7df";

        logStatus.innerHTML =
            "Waiting";

        logStatus.style.color =
            "#c98b00";

    }
}


// =========================================================
// UPDATE LIVE STATUS
// =========================================================

async function updateStatus() {

    try {

        const response =
            await fetch(
                "/api/detection/status",
                {
                    cache: "no-store"
                }
            );


        if (!response.ok) {

            throw new Error(
                "Status API returned " +
                response.status
            );

        }


        const data =
            await response.json();


        // -----------------------------------------
        // BASIC INFORMATION
        // -----------------------------------------

        const state =
            data.state || "STOPPED";

        const present =
            Number(
                data.present_probability || 0
            );

        const empty =
            Number(
                data.empty_probability || 0
            );

        const confidenceValue =
            Number(
                data.confidence || 0
            );

        const rate =
            Number(
                data.sample_rate || 0
            );


        // -----------------------------------------
        // ESP32 STATUS
        // -----------------------------------------

        if (
            state === "PRESENT" ||
            state === "EMPTY" ||
            state === "STARTING"
        ) {

            espStatus.innerHTML =
                "Connected";

            espStatus.style.color =
                "#22a447";

        }


        // -----------------------------------------
        // ROOM / DETECTION ZONE
        // -----------------------------------------

        if (state === "PRESENT") {

            roomStatus.innerHTML =
                "Person Detected";

            roomStatus.style.color =
                "#22a447";

        }

        else if (state === "EMPTY") {

            roomStatus.innerHTML =
                "Monitoring";

            roomStatus.style.color =
                "#777777";

        }

        else if (state === "STARTING") {

            roomStatus.innerHTML =
                "Starting...";

            roomStatus.style.color =
                "#c98b00";

        }

        else {

            roomStatus.innerHTML =
                "Ready";

        }


        // -----------------------------------------
        // CONFIDENCE
        // -----------------------------------------

        confidence.innerHTML =
            Math.round(
                confidenceValue * 100
            ) + "%";


        confidenceBar.style.width =
            Math.round(
                confidenceValue * 100
            ) + "%";


        // -----------------------------------------
        // PROBABILITIES
        // -----------------------------------------

        presentProbability.innerHTML =
            Math.round(
                present * 100
            ) + "%";

        emptyProbability.innerHTML =
            Math.round(
                empty * 100
            ) + "%";


        // -----------------------------------------
        // RAW PREDICTION
        // -----------------------------------------

        rawPrediction.innerHTML =
            data.raw_prediction || "UNKNOWN";


        if (
            data.raw_prediction === "PRESENT"
        ) {

            rawPrediction.style.color =
                "#22a447";

        }

        else if (
            data.raw_prediction === "EMPTY"
        ) {

            rawPrediction.style.color =
                "#777777";

        }

        else {

            rawPrediction.style.color =
                "#c98b00";

        }


        // -----------------------------------------
        // SAMPLE RATE
        // -----------------------------------------

        if (rate > 0) {

            const roundedRate =
                Math.round(rate);

            sampleRate.innerHTML =
                roundedRate + " Hz";

            signalRate.innerHTML =
                roundedRate + " Hz";

        }

        else {

            sampleRate.innerHTML =
                "0 Hz";

            signalRate.innerHTML =
                "0 Hz";

        }


        // -----------------------------------------
        // DETECTION STATE
        // -----------------------------------------

        setDetectionState(
            state,
            confidenceValue
        );


        // -----------------------------------------
        // LOG ONLY WHEN STATE CHANGES
        // -----------------------------------------

        if (
            lastState !== "" &&
            lastState !== state
        ) {

            if (state === "PRESENT") {

                addLog(
                    "Person detected inside the CSI sensing zone."
                );

            }

            else if (state === "EMPTY") {

                addLog(
                    "No person detected. Sensing zone is empty."
                );

            }

            else if (state === "STOPPED") {

                addLog(
                    "Human detection service stopped."
                );

            }

        }


        lastState = state;


    }
    catch (error) {

        console.error(
            "Status update error:",
            error
        );

        espStatus.innerHTML =
            "Connection Error";

        espStatus.style.color =
            "#d64545";

        humanStatus.innerHTML =
            "CONNECTION ERROR";

        humanStatus.style.color =
            "#d64545";

        detectionDescription.innerHTML =
            "Unable to communicate with the Flask detection service.";

    }

}


// =========================================================
// START MONITORING
// =========================================================

monitorBtn.onclick =
    async function () {

        if (monitoring) {

            return;

        }


        try {

            monitorBtn.disabled =
                true;

            monitorBtn.innerHTML =
                "Starting...";


            const response =
                await fetch(
                    "/api/detection/start"
                );


            const data =
                await response.json();


            if (
                data.status === "success" ||
                data.status === "already_running"
            ) {

                monitoring = true;


                monitorBtn.innerHTML =
                    "● Monitoring Active";


                stopBtn.disabled =
                    false;


                logStatus.innerHTML =
                    "Starting";


                addLog(
                    "Human detection monitoring started."
                );


                // Immediately check
                await updateStatus();


                // Clear old timer
                if (statusTimer !== null) {

                    clearInterval(
                        statusTimer
                    );

                }


                // Update every 500 ms
                statusTimer =
                    setInterval(
                        updateStatus,
                        500
                    );

            }

            else {

                throw new Error(
                    data.message ||
                    "Unable to start detector"
                );

            }


        }
        catch (error) {

            console.error(
                "Start error:",
                error
            );

            addLog(
                "ERROR: Unable to start human detection."
            );

            monitorBtn.disabled =
                false;

            monitorBtn.innerHTML =
                "▶ Start Monitoring";

        }

    };


// =========================================================
// STOP MONITORING
// =========================================================

stopBtn.onclick =
    async function () {

        try {

            stopBtn.disabled =
                true;


            await fetch(
                "/api/detection/stop"
            );


            monitoring = false;


            if (statusTimer !== null) {

                clearInterval(
                    statusTimer
                );

                statusTimer = null;

            }


            monitorBtn.disabled =
                false;

            monitorBtn.innerHTML =
                "▶ Start Monitoring";


            stopBtn.disabled =
                true;


            addLog(
                "Human detection monitoring stopped."
            );


            await updateStatus();

        }
        catch (error) {

            console.error(
                "Stop error:",
                error
            );

            stopBtn.disabled =
                false;

        }

    };


// =========================================================
// INITIAL STATUS
// =========================================================

stopBtn.disabled = true;

updateStatus();