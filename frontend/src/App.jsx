import React, { useState, useEffect, useRef, useCallback } from 'react';
import './index.css';

const WS_URL = 'ws://localhost:8000/ws/chat';
const DEFAULT_GREETING = "Hello! I'm Elara, your AI Hospital Receptionist. How can I help you today?";

function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [isMicEnabled, setIsMicEnabled] = useState(true);
  const [isListening, setIsListening] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);

  // Captions
  const [userTranscript, setUserTranscript] = useState('');
  const [interimTranscript, setInterimTranscript] = useState('');
  const [agentResponse, setAgentResponse] = useState(DEFAULT_GREETING);

  // Refs
  const wsRef = useRef(null);
  const audioPlayerRef = useRef(null);
  const recognitionRef = useRef(null);
  const shouldListenRef = useRef(false);
  const isProcessingRef = useRef(false);
  const isSpeakingRef = useRef(false);
  const pauseTimerRef = useRef(null);
  const pendingUtteranceRef = useRef('');
  const needsGestureStartRef = useRef(false);
  const requestCounterRef = useRef(0);
  const latestRequestIdRef = useRef(0);
  const didPlayGreetingRef = useRef(false);
  const startListeningRef = useRef(() => {});
  const awaitingGreetingAudioRef = useRef(false);
  const greetingStartTimeoutRef = useRef(null);
  const micPermissionRequestedRef = useRef(false);
  const recognitionStartingRef = useRef(false);
  const pendingAudioUrlRef = useRef('');
  const pendingAudioRequestIdRef = useRef(0);
  const [isAudioBlocked, setIsAudioBlocked] = useState(false);
  const [needsManualVoiceStart, setNeedsManualVoiceStart] = useState(false);

  const clearPauseTimer = useCallback(() => {
    if (pauseTimerRef.current) {
      clearTimeout(pauseTimerRef.current);
      pauseTimerRef.current = null;
    }
  }, []);

  const recoverListening = useCallback((delayMs = 250) => {
    setIsListening(false);
    recognitionRef.current = null;
    if (isMicEnabled && shouldListenRef.current && !isSpeakingRef.current) {
      setTimeout(() => {
        if (isMicEnabled && shouldListenRef.current && !isSpeakingRef.current) {
          startListeningRef.current();
        }
      }, delayMs);
    }
  }, [isMicEnabled]);

  const stopRecognitionOnly = useCallback(() => {
    recognitionStartingRef.current = false;
    if (recognitionRef.current) {
      try { recognitionRef.current.stop(); } catch (e) { /* ignore */ }
      recognitionRef.current = null;
    }
    setIsListening(false);
  }, []);

  const stopAgentSpeech = useCallback(() => {
    if (audioPlayerRef.current) {
      try {
        audioPlayerRef.current.pause();
        audioPlayerRef.current.currentTime = 0;
        audioPlayerRef.current.removeAttribute('src');
        audioPlayerRef.current.load();
      } catch (e) {
        // ignore media state errors
      }
    }
    setIsSpeaking(false);
    isSpeakingRef.current = false;
  }, []);

  const playAudioUrl = useCallback((audioUrl, requestId) => {
    if (!audioPlayerRef.current) return;

    setIsSpeaking(true);
    isSpeakingRef.current = true;
    audioPlayerRef.current.src = audioUrl;
    audioPlayerRef.current.play().then(() => {
      setIsAudioBlocked(false);
      pendingAudioUrlRef.current = '';
      pendingAudioRequestIdRef.current = 0;
    }).catch((e) => {
      if (e?.name === 'NotAllowedError') {
        setIsAudioBlocked(true);
        pendingAudioUrlRef.current = audioUrl;
        pendingAudioRequestIdRef.current = requestId || latestRequestIdRef.current;
        // Don't block mic forever when autoplay is restricted.
        awaitingGreetingAudioRef.current = false;
        if (isMicEnabled) {
          shouldListenRef.current = true;
          startListeningRef.current();
        }
      } else {
        console.error('Playback failed:', e);
      }
      setIsSpeaking(false);
      isSpeakingRef.current = false;
      setIsProcessing(false);
      isProcessingRef.current = false;
      awaitingGreetingAudioRef.current = false;
      if (isMicEnabled) {
        shouldListenRef.current = true;
        startListening();
      }
    });
  }, []);

  const playWelcomeGreeting = useCallback(() => {
    if (didPlayGreetingRef.current) return;
    didPlayGreetingRef.current = true;

    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return;
    }

    const requestId = ++requestCounterRef.current;
    latestRequestIdRef.current = requestId;
    awaitingGreetingAudioRef.current = true;
    setIsProcessing(true);
    isProcessingRef.current = true;
    shouldListenRef.current = true;

    wsRef.current.send(JSON.stringify({
      type: 'greeting',
      message: DEFAULT_GREETING,
      request_id: requestId
    }));

    // Safety: if greeting audio doesn't arrive, start listening anyway.
    if (greetingStartTimeoutRef.current) {
      clearTimeout(greetingStartTimeoutRef.current);
    }
    greetingStartTimeoutRef.current = setTimeout(() => {
      if (awaitingGreetingAudioRef.current && isMicEnabled && !isListening && !isSpeakingRef.current) {
        awaitingGreetingAudioRef.current = false;
        setIsProcessing(false);
        isProcessingRef.current = false;
        startListeningRef.current();
      }
    }, 2500);
  }, [isMicEnabled]);

  // ─── Send text to backend via WebSocket ───
  const sendMessage = useCallback((text) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      const requestId = ++requestCounterRef.current;
      latestRequestIdRef.current = requestId;
      setIsProcessing(true);
      isProcessingRef.current = true;
      setAgentResponse('...');
      setInterimTranscript('');

      wsRef.current.send(JSON.stringify({
        type: 'text',
        message: text,
        request_id: requestId
      }));
    }
  }, []);

  // ─── Speech Recognition (always-on, auto-pause detection) ───
  const startListening = useCallback(() => {
    if (!isMicEnabled || isSpeakingRef.current) return;
    if (recognitionRef.current || recognitionStartingRef.current) return;

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      console.error('Speech Recognition not supported');
      return;
    }

    // Create fresh instance each time for reliability
    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
      recognitionStartingRef.current = false;
      needsGestureStartRef.current = false;
      setNeedsManualVoiceStart(false);
      setIsListening(true);
    };

    recognition.onresult = (event) => {
      let interim = '';
      let finalText = '';

      for (let i = 0; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalText += `${transcript} `;
        } else {
          interim += `${transcript} `;
        }
      }

      const cleanInterim = interim.trim();
      const cleanFinal = finalText.trim();
      const combinedUtterance = `${cleanFinal} ${cleanInterim}`.trim();

      if (combinedUtterance && isSpeakingRef.current) {
        stopAgentSpeech();
      }

      setInterimTranscript(cleanInterim);
      if (cleanFinal) {
        setUserTranscript(cleanFinal);
      }

      if (combinedUtterance) {
        pendingUtteranceRef.current = combinedUtterance;
        clearPauseTimer();
        pauseTimerRef.current = setTimeout(() => {
          const textToSend = pendingUtteranceRef.current.trim();
          if (!textToSend || !shouldListenRef.current) return;

          setUserTranscript(textToSend);
          setInterimTranscript('');
          pendingUtteranceRef.current = '';
          sendMessage(textToSend);
        }, 900);
      }
    };

    recognition.onerror = (event) => {
      recognitionStartingRef.current = false;
      if (event.error === 'not-allowed') {
        needsGestureStartRef.current = true;
        setNeedsManualVoiceStart(true);
        alert('Please allow microphone access to talk to the AI.');
        return;
      }
      if (event.error === 'aborted') {
        // Treat as recoverable so recognition never gets stuck.
        recoverListening(120);
        return;
      }
      if (event.error === 'no-speech') {
        // Silence timeout can happen frequently in continuous mode.
        recoverListening(180);
        return;
      }
      if (event.error === 'network') {
        // Browser speech service hiccup: recover automatically.
        recoverListening(600);
        return;
      }
      // Any unknown speech error: try to recover instead of deadlocking.
      recoverListening(400);
      console.error('Speech error:', event.error);
    };

    recognition.onend = () => {
      recognitionStartingRef.current = false;
      recognitionRef.current = null;
      setIsListening(false);
      // Auto-restart if we should still be listening
      if (shouldListenRef.current && isMicEnabled && !isSpeakingRef.current) {
        setTimeout(() => {
          if (shouldListenRef.current && isMicEnabled) {
            startListening();
          }
        }, 150);
      }
    };

    recognitionRef.current = recognition;
    shouldListenRef.current = true;
    recognitionStartingRef.current = true;

    try {
      recognition.start();
    } catch (e) {
      recognitionStartingRef.current = false;
      needsGestureStartRef.current = true;
      setNeedsManualVoiceStart(true);
      console.log('Recognition start error:', e);
    }
  }, [clearPauseTimer, isMicEnabled, recoverListening, sendMessage, stopAgentSpeech]);

  useEffect(() => {
    startListeningRef.current = startListening;
  }, [startListening]);

  const stopListening = useCallback(() => {
    clearPauseTimer();
    pendingUtteranceRef.current = '';
    shouldListenRef.current = false;
    stopRecognitionOnly();
  }, [clearPauseTimer, stopRecognitionOnly]);

  const handleManualVoiceStart = useCallback(async () => {
    try {
      if (navigator.mediaDevices?.getUserMedia) {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        stream.getTracks().forEach((track) => track.stop());
      }
      needsGestureStartRef.current = false;
      setNeedsManualVoiceStart(false);
      shouldListenRef.current = true;
      startListeningRef.current();
    } catch (error) {
      needsGestureStartRef.current = true;
      setNeedsManualVoiceStart(true);
      console.error('Microphone permission failed:', error);
      alert('Microphone access is blocked. Please allow mic in browser site settings.');
    }
  }, []);

  const ensureMicPermission = useCallback(async () => {
    if (micPermissionRequestedRef.current) return;
    micPermissionRequestedRef.current = true;
    try {
      if (navigator.mediaDevices?.getUserMedia) {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        stream.getTracks().forEach((track) => track.stop());
      }
      setNeedsManualVoiceStart(false);
    } catch (error) {
      setNeedsManualVoiceStart(true);
      needsGestureStartRef.current = true;
      console.error('Initial microphone permission failed:', error);
    }
  }, []);

  // ─── WebSocket Connection ───
  const connectWebSocket = useCallback(() => {
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('✅ Connected to Hospital AI Agent');
      setIsConnected(true);
      // Speak welcome line first, then listen.
      playWelcomeGreeting();
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case 'transcript':
          setUserTranscript(data.message);
          setAgentResponse('...');
          break;

        case 'response':
          if (data.request_id && data.request_id !== latestRequestIdRef.current) {
            break;
          }
          setAgentResponse(data.message);
          if (awaitingGreetingAudioRef.current) {
            break;
          }
          setIsProcessing(false);
          isProcessingRef.current = false;
          // If no audio response follows, resume listening after a short delay
          setTimeout(() => {
            if (isMicEnabled && !isSpeakingRef.current) {
              startListening();
            }
          }, 800);
          break;

        case 'audio_response': {
          if (data.request_id && data.request_id !== latestRequestIdRef.current) {
            break;
          }
          // Prevent feedback loop: hard-disable recognition while AI audio is playing.
          shouldListenRef.current = false;
          setIsListening(false);
          setIsSpeaking(true);
          isSpeakingRef.current = true;
          stopRecognitionOnly();
          clearPauseTimer();
          pendingUtteranceRef.current = '';
          setInterimTranscript('');

          // Decode and play the AI's voice
          const audioBytes = atob(data.data);
          const arrayBuffer = new ArrayBuffer(audioBytes.length);
          const uint8Array = new Uint8Array(arrayBuffer);
          for (let i = 0; i < audioBytes.length; i++) {
            uint8Array[i] = audioBytes.charCodeAt(i);
          }
          const blob = new Blob([uint8Array], { type: 'audio/wav' });
          const audioUrl = URL.createObjectURL(blob);

          if (audioPlayerRef.current) {
            if (pendingAudioUrlRef.current && pendingAudioUrlRef.current !== audioUrl) {
              URL.revokeObjectURL(pendingAudioUrlRef.current);
            }
            playAudioUrl(audioUrl, data.request_id);
          }
          break;
        }

        case 'error':
          console.error('Server error:', data.message);
          setAgentResponse("Sorry, I had trouble processing that.");
          setIsProcessing(false);
          isProcessingRef.current = false;
          setTimeout(() => {
            if (isMicEnabled) startListening();
          }, 500);
          break;

        case 'pong':
          break;

        default:
          break;
      }
    };

    ws.onclose = () => {
      console.log('❌ Disconnected');
      setIsConnected(false);
      stopListening();
      setTimeout(() => connectWebSocket(), 3000);
    };
  }, [clearPauseTimer, isMicEnabled, playAudioUrl, playWelcomeGreeting, startListening, stopListening, stopRecognitionOnly]);

  // ─── Handle audio playback finishing → resume listening ───
  useEffect(() => {
    const audio = audioPlayerRef.current;
    if (!audio) return;

    const handleEnded = () => {
      const finishedUrl = audio.currentSrc;
      if (finishedUrl && finishedUrl.startsWith('blob:')) {
        URL.revokeObjectURL(finishedUrl);
      }
      setIsSpeaking(false);
      isSpeakingRef.current = false;
      setIsProcessing(false);
      isProcessingRef.current = false;
      awaitingGreetingAudioRef.current = false;
      if (greetingStartTimeoutRef.current) {
        clearTimeout(greetingStartTimeoutRef.current);
        greetingStartTimeoutRef.current = null;
      }
      // Resume listening after AI finishes speaking
      setTimeout(() => {
        if (isMicEnabled) {
          shouldListenRef.current = true;
          startListening();
        }
      }, 400);
    };

    audio.addEventListener('ended', handleEnded);
    return () => audio.removeEventListener('ended', handleEnded);
  }, [isMicEnabled, startListening]);

  useEffect(() => {
    if (!isConnected) return;
    if (isSpeakingRef.current) return;
    if (isMicEnabled) {
      shouldListenRef.current = true;
      startListening();
    } else {
      stopListening();
    }
  }, [isConnected, isMicEnabled, startListening, stopListening]);

  useEffect(() => {
    if (!isConnected || !isMicEnabled) return;
    ensureMicPermission();
  }, [ensureMicPermission, isConnected, isMicEnabled]);

  // Some browsers require a user gesture after refresh before starting recognition.
  useEffect(() => {
    if (!isConnected || !isMicEnabled || isListening) return undefined;

    const startFromGesture = () => {
      needsGestureStartRef.current = false;
      if (isListening || isSpeakingRef.current) return;
      shouldListenRef.current = true;
      startListening();
    };

    window.addEventListener('pointerdown', startFromGesture);
    window.addEventListener('keydown', startFromGesture);
    window.addEventListener('touchstart', startFromGesture);

    return () => {
      window.removeEventListener('pointerdown', startFromGesture);
      window.removeEventListener('keydown', startFromGesture);
      window.removeEventListener('touchstart', startFromGesture);
    };
  }, [isConnected, isListening, isMicEnabled, startListening]);

  // Unlock autoplay on first user interaction and replay pending audio.
  useEffect(() => {
    if (!isAudioBlocked) return undefined;

    const unlockAudio = () => {
      const pendingUrl = pendingAudioUrlRef.current;
      const pendingRequestId = pendingAudioRequestIdRef.current;
      if (!pendingUrl) {
        setIsAudioBlocked(false);
        return;
      }
      if (pendingRequestId && pendingRequestId !== latestRequestIdRef.current) {
        URL.revokeObjectURL(pendingUrl);
        pendingAudioUrlRef.current = '';
        pendingAudioRequestIdRef.current = 0;
        setIsAudioBlocked(false);
        return;
      }
      playAudioUrl(pendingUrl, pendingRequestId);
    };

    window.addEventListener('pointerdown', unlockAudio);
    window.addEventListener('keydown', unlockAudio);
    window.addEventListener('touchstart', unlockAudio);

    return () => {
      window.removeEventListener('pointerdown', unlockAudio);
      window.removeEventListener('keydown', unlockAudio);
      window.removeEventListener('touchstart', unlockAudio);
    };
  }, [isAudioBlocked, playAudioUrl]);

  // Retry listening when tab becomes visible again after refresh/background.
  useEffect(() => {
    const handleVisibility = () => {
      if (
        document.visibilityState === 'visible' &&
        isConnected &&
        isMicEnabled &&
        !isListening &&
        !isSpeakingRef.current
      ) {
        startListening();
      }
    };

    document.addEventListener('visibilitychange', handleVisibility);
    return () => document.removeEventListener('visibilitychange', handleVisibility);
  }, [isConnected, isListening, isMicEnabled, startListening]);

  // Safety watchdog: recover listening if browser recognition silently stops.
  useEffect(() => {
    if (!isConnected || !isMicEnabled) return undefined;
    const interval = setInterval(() => {
      if (
        shouldListenRef.current &&
        !isListening &&
        !isSpeakingRef.current &&
        !recognitionRef.current
      ) {
        startListening();
      }
    }, 1200);
    return () => clearInterval(interval);
  }, [isConnected, isListening, isMicEnabled, startListening]);

  // ─── Boot up ───
  useEffect(() => {
    connectWebSocket();
    return () => {
      stopListening();
      stopAgentSpeech();
      if (greetingStartTimeoutRef.current) {
        clearTimeout(greetingStartTimeoutRef.current);
      }
      if (pendingAudioUrlRef.current) {
        URL.revokeObjectURL(pendingAudioUrlRef.current);
      }
      if (wsRef.current) wsRef.current.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ─── Heartbeat ───
  useEffect(() => {
    if (!isConnected) return;
    const interval = setInterval(() => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000);
    return () => clearInterval(interval);
  }, [isConnected]);

  // ─── Status label ───
  const getStatusText = () => {
    if (needsManualVoiceStart) return 'Tap Start Voice to enable microphone';
    if (isAudioBlocked) return 'Tap anywhere to enable audio';
    if (!isMicEnabled) return 'Mic is off';
    if (isSpeaking) return 'Elara is speaking…';
    if (isProcessing) return 'Thinking…';
    if (isListening) return 'Listening… speak anytime';
    if (isConnected) return 'Ready';
    return 'Connecting…';
  };

  const getStatusClass = () => {
    if (isSpeaking) return 'speaking';
    if (isProcessing) return 'processing';
    if (isListening) return 'active';
    return '';
  };

  return (
    <div className="app-container">
      {/* Hidden Audio Player */}
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
          {isConnected ? 'System Online' : 'Connecting to Core…'}
        </div>
      </nav>

      {/* Main Content */}
      <main className="main-content">
        <div className={`agent-interface ${isListening ? 'listening' : ''}`}>

          {/* Static Agent Image */}
          <div className="agent-image-container">
            <img
              src="/elara_receptionist.png"
              alt="Hospital reception desk"
              className="agent-image"
            />
          </div>

          <div className="gradient-overlay"></div>

          {/* Overlays */}
          <div className="call-overlay">
            <div className="nameplate">
              <h2>Elara</h2>
              <p>AI RECEPTIONIST</p>
            </div>

            <div className="bottom-section">
              {/* Captions */}
              <div className="captions-container">
                {(userTranscript || interimTranscript) && (
                  <div className="user-transcript" key={interimTranscript || userTranscript}>
                    &ldquo;{interimTranscript || userTranscript}&rdquo;
                  </div>
                )}
                <div className="agent-response" key={agentResponse}>
                  {agentResponse}
                </div>
              </div>

              <button
                type="button"
                className={`mic-toggle ${isMicEnabled ? 'on' : 'off'}`}
                onClick={() => setIsMicEnabled(prev => !prev)}
              >
                {isMicEnabled ? 'Turn Mic Off' : 'Turn Mic On'}
              </button>

              {isMicEnabled && (!isListening || needsManualVoiceStart) && (
                <button
                  type="button"
                  className="mic-toggle on"
                  onClick={handleManualVoiceStart}
                >
                  Start Voice
                </button>
              )}

              {/* Live Listening Indicator */}
              <div className={`live-indicator ${getStatusClass()}`}>
                <div className="wave-bars">
                  <span></span>
                  <span></span>
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
                <span className="status-label">{getStatusText()}</span>
              </div>

            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
