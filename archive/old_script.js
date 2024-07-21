// let mediaRecorder;
// let audioChunks = [];
// const socket = new WebSocket('ws://127.0.0.1:65432', [], {
//     headers: {
//         'Origin': window.location.origin
//     }
// });

// document.getElementById('startRecording').addEventListener('click', startRecording);
// document.getElementById('stopRecording').addEventListener('click', stopRecording);

// socket.onopen = () => {
//     console.log('WebSocket connection established');
// };

// socket.onmessage = (event) => {
//     console.log("Received transcription:", event.data);
//     document.getElementById('transcription').textContent = `Transcription: ${event.data}`;
// };

// socket.onerror = (error) => {
//     console.error('WebSocket error:', error);
// };

// async function startRecording() {
//     const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
//     mediaRecorder = new MediaRecorder(stream);

//     mediaRecorder.ondataavailable = (event) => {
//         audioChunks.push(event.data);
//         if (audioChunks.length >= 4) {  // Send every 2 seconds (4 chunks of 0.5 seconds each)
//             sendAudioData();
//         }
//     };

//     mediaRecorder.start(500);  // Capture in 0.5-second chunks
//     document.getElementById('startRecording').disabled = true;
//     document.getElementById('stopRecording').disabled = false;
//     document.getElementById('status').textContent = 'Recording...';
// }

// function stopRecording() {
//     mediaRecorder.stop();
//     document.getElementById('startRecording').disabled = false;
//     document.getElementById('stopRecording').disabled = true;
//     document.getElementById('status').textContent = 'Recording stopped';
// }

// function sendAudioData() {
//     const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
//     audioBlob.arrayBuffer().then(arrayBuffer => {
//         const audioContext = new (window.AudioContext || window.webkitAudioContext)();
//         audioContext.decodeAudioData(arrayBuffer).then(audioBuffer => {
//             const pcmData = audioBuffer.getChannelData(0);
//             const floatArray = new Float32Array(pcmData);
//             socket.send(floatArray.buffer);
//         });
//     });
//     audioChunks = [];
// }


let mediaRecorder;
let audioChunks = [];
const socket = new WebSocket('ws://127.0.0.1:65432');

document.getElementById('startRecording').addEventListener('click', startRecording);
document.getElementById('stopRecording').addEventListener('click', stopRecording);

socket.onopen = () => {
    console.log('WebSocket connection established');
};

socket.onmessage = (event) => {
    console.log("Received transcription:", event.data);
    document.getElementById('transcription').textContent = `Transcription: ${event.data}`;
};

socket.onerror = (error) => {
    console.error('WebSocket error:', error);
};

async function startRecording() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
        if (audioChunks.length >= 4) {  // Send every 2 seconds (4 chunks of 0.5 seconds each)
            sendAudioData();
        }
    };

    mediaRecorder.start(500);  // Capture in 0.5-second chunks
    document.getElementById('startRecording').disabled = true;
    document.getElementById('stopRecording').disabled = false;
    document.getElementById('status').textContent = 'Recording...';
}

function stopRecording() {
    mediaRecorder.stop();
    document.getElementById('startRecording').disabled = false;
    document.getElementById('stopRecording').disabled = true;
    document.getElementById('status').textContent = 'Recording stopped';
}

function sendAudioData() {
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    audioBlob.arrayBuffer().then(arrayBuffer => {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        audioContext.decodeAudioData(arrayBuffer).then(audioBuffer => {
            const pcmData = audioBuffer.getChannelData(0);
            const floatArray = new Float32Array(pcmData);
            socket.send(floatArray.buffer);
        });
    });
    audioChunks = [];
}