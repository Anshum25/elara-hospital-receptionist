import React, { useState, useEffect, useRef } from 'react';
import './index.css';

// Using the fast FastAPI server backend
const WS_URL = 'ws://localhost:8000/ws/chat';

function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  
  // Captions
  const [userTranscript, setUserTranscript] = useState('');
  const [agentResponse, setAgentResponse] = useState('Hello! I\'m Elara, your AI Hospital Receptionist. How can I help you today?');
  
  // Refs
  const wsRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const audioPlayerRef = useRef(null);

  const connectWebSocket = () => {
    wsRef.current = new WebSocket(WS_URL);

    wsRef.current.onopen = () => {
      console.log('✅ Connected to Hospital AI Agent');
      setIsConnected(true);
    };

    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      switch (data.type) {
        case 'transcript':
          // The AI heard us!
          setUserTranscript(data.message);
          setAgentResponse('...'); // Thinking
          break;
          
        case 'response':
          // The AI generated text
          setAgentResponse(data.message);
          setIsProcessing(false);
          break;

        case 'audio_response': {
          // The AI generated voice! Decode Base64 audio and play it
          const audioBytes = atob(data.data);
          const arrayBuffer = new ArrayBuffer(audioBytes.length);
          const uint8Array = new Uint8Array(arrayBuffer);
          for (let i = 0; i < audioBytes.length; i++) {
            uint8Array[i] = audioBytes.charCodeAt(i);
          }
          const blob = new Blob([uint8Array], { type: 'audio/wav' });
          const audioUrl = URL.createObjectURL(blob);
          
          if (audioPlayerRef.current) {
            audioPlayerRef.current.src = audioUrl;
            audioPlayerRef.current.play().catch(e => console.error("Playback failed:", e));
          }
          break;
        }

        case 'error':
          console.error('Server error:', data.message);
          setAgentResponse("Sorry, I had trouble processing that.");
          setIsProcessing(false);
          break;
          
        case 'pong':
          // heartbeat
          break;
          
        default:
          break;
      }
    };

    wsRef.current.onclose = () => {
      console.log('❌ Disconnected from Server');
      setIsConnected(false);
      // Try to reconnect after 3 seconds
      setTimeout(connectWebSocket, 3000);
    };
  };

  useEffect(() => {
    // 1. Establish WebSocket connection to FastAPI
    connectWebSocket();
    
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  // Keep connection alive
  useEffect(() => {
    if (!isConnected) return;
    const interval = setInterval(() => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000);
    return () => clearInterval(interval);
  }, [isConnected]);

  // Handle Recording Audio
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = async () => {
        setIsProcessing(true);
        setUserTranscript(''); // Clear old text while waiting
        setAgentResponse('Listening...');
        
        // Stop audio playback if agent is currently speaking
        if (audioPlayerRef.current) {
          audioPlayerRef.current.pause();
        }

        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        
        // Convert Blob to Base64 to send via WebSocket easily
        const reader = new FileReader();
        reader.readAsDataURL(audioBlob);
        reader.onloadend = () => {
          const base64Audio = reader.result; // data:audio/webm;base64,....
          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({
              type: 'audio',
              data: base64Audio
            }));
          }
        };
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      console.error('Error accessing microphone:', err);
      alert('Please allow microphone access to talk to the AI.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      // Stop all tracks to turn off the red dot in the browser tab
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
    }
  };

  const toggleRecording = () => {
    if (!isConnected) return;
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  return (
    <div className="app-container">
      {/* Hidden Audio Player for the TTS response */}
      <audio ref={audioPlayerRef} className="audio-player" />
      
      {/* Top Navigation */}
      <nav className="navbar">
        <div className="logo-area">
          <div className="logo-text">
            <h1>MCare</h1>
            <p>Intelligence Center</p>
          </div>
        </div>
        
        <div className="status-badge">
          <div className={`status-dot ${isConnected ? 'connected' : 'disconnected'}`}></div>
          {isConnected ? 'System Online' : 'Connecting to Core...'}
        </div>
      </nav>

      {/* Main Interactive Center */}
      <main className="main-content">
        <div className={`agent-interface ${isRecording ? 'recording' : ''}`}>
          
          {/* Background Avatar */}
          <div className="avatar-container">
            {/* Make sure to copy medbot_avatar.png into the public folder! */}
            <img 
              src="/medbot_avatar.png" 
              alt="Medical AI Assistant" 
              className="avatar-image" 
              onError={(e) => { e.target.style.display = 'none'; }}
            />
          </div>
          
          <div className="gradient-overlay"></div>

          {/* Interactive UI Overlays */}
          <div className="call-overlay">
            
            <div className="nameplate">
              <h2>Elara</h2>
              <p>AI RECEPTIONIST</p>
            </div>

            <div className="bottom-section">
              
              <div className="captions-container">
                {userTranscript && (
                  <div className="user-transcript" key={userTranscript}>
                    "{userTranscript}"
                  </div>
                )}
                <div className="agent-response" key={agentResponse}>
                  {agentResponse}
                </div>
              </div>
              
              {isConnected && !isRecording && !isProcessing && (
                <div className="instruction-text">Hold microphone to speak</div>
              )}
              {isRecording && <div className="instruction-text" style={{color: '#ef4444'}}>Recording... Click to stop.</div>}

              <div className={`controls ${isProcessing ? 'processing' : ''}`}>
                <button 
                  className={`mic-btn ${isRecording ? 'recording' : ''}`} 
                  onClick={toggleRecording}
                  disabled={isProcessing || !isConnected}
                  aria-label="Toggle Microphone"
                >
                  <span className="mic-icon">
                    {isRecording ? '⏹' : '🎤'}
                  </span>
                  <div className="spinner"></div>
                </button>
              </div>

            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
