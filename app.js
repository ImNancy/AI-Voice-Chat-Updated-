class VoiceChatApp {
  constructor() {
      this.isRecording = false;
      this.isProcessing = false;
      this.recognition = null;
      this.speechSynthesis = window.speechSynthesis;
      this.chatMessages = [];

      this.initializeElements();
      this.initializeSpeechRecognition();
      this.attachEventListeners();
      this.checkServerConnection();
  }

  initializeElements() {
      this.recordBtn = document.getElementById('recordBtn');
      this.micIcon = document.getElementById('micIcon');
      this.btnText = document.getElementById('btnText');
      this.statusIndicator = document.getElementById('statusIndicator');
      this.statusText = document.getElementById('statusText');
      this.chatMessages = document.getElementById('chatMessages');
      this.loadingOverlay = document.getElementById('loadingOverlay');
      this.errorToast = document.getElementById('errorToast');
      this.errorMessage = document.getElementById('errorMessage');
      this.closeToast = document.getElementById('closeToast');
      this.clearBtn = document.getElementById('clearBtn');
      this.audioVisualizer = document.getElementById('audioVisualizer');
      this.connectionStatus = document.getElementById('connectionStatus');
      this.chatInput = document.getElementById('chatInput');
      this.sendBtn = document.getElementById('sendBtn');
  }

  initializeSpeechRecognition() {
      if ('webkitSpeechRecognition' in window) {
          this.recognition = new webkitSpeechRecognition();
      } else if ('SpeechRecognition' in window) {
          this.recognition = new SpeechRecognition();
      } else {
          this.showError('Speech recognition is not supported in this browser. Please use Chrome, Edge, or Safari.');
          return;
      }

      this.recognition.continuous = false;
      this.recognition.interimResults = true;
      this.recognition.lang = 'en-US';
      this.recognition.maxAlternatives = 1;

      this.recognition.onstart = () => {
          this.onRecordingStart();
      };

      this.recognition.onresult = (event) => {
          let transcript = '';
          for (let i = event.resultIndex; i < event.results.length; i++) {
              if (event.results[i].isFinal) {
                  transcript += event.results[i][0].transcript;
              }
          }
          if (transcript.trim()) {
              this.onSpeechResult(transcript.trim());
          }
      };

      this.recognition.onerror = (event) => {
          console.error('Speech recognition error:', event.error);
          this.onRecordingEnd();

          let errorMsg = 'Speech recognition failed. ';
          switch(event.error) {
              case 'no-speech':
                  errorMsg += 'No speech detected. Try speaking closer to your microphone or use the text input below.';
                  break;
              case 'audio-capture':
                  errorMsg += 'Microphone access denied or not available. Please check your microphone settings.';
                  break;
              case 'not-allowed':
                  errorMsg += 'Microphone permission denied. Please allow microphone access in your browser settings.';
                  break;
              case 'network':
                  errorMsg += 'Network error occurred. Please check your internet connection.';
                  break;
              case 'aborted':
                  // Don't show error for user-initiated stops
                  return;
              default:
                  errorMsg += `Error: ${event.error}. You can still use the text input below.`;
          }
          this.showError(errorMsg);
      };

      this.recognition.onend = () => {
          this.onRecordingEnd();
      };
  }

  attachEventListeners() {
      this.recordBtn.addEventListener('click', () => {
          this.toggleRecording();
      });

      this.clearBtn.addEventListener('click', () => {
          this.clearChat();
      });

      this.closeToast.addEventListener('click', () => {
          this.hideError();
      });

      this.sendBtn.addEventListener('click', () => {
          this.sendTextMessage();
      });

      this.chatInput.addEventListener('keypress', (e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              this.sendTextMessage();
          }
      });

      // Hide error toast after 5 seconds
      document.addEventListener('click', (e) => {
          if (this.errorToast.classList.contains('active') && !this.errorToast.contains(e.target)) {
              this.hideError();
          }
      });
  }

  async checkServerConnection() {
      try {
          const response = await fetch('/api/health');
          const data = await response.json();

          if (data.status === 'healthy') {
              this.updateConnectionStatus('connected', 'Connected');
              if (!data.groq_configured) {
                  this.showError('Groq API key not configured. Please set the GROQ_API_KEY environment variable.');
              }
          } else {
              this.updateConnectionStatus('disconnected', 'Server Error');
          }
      } catch (error) {
          console.error('Connection check failed:', error);
          this.updateConnectionStatus('disconnected', 'Disconnected');
      }
  }

  updateConnectionStatus(status, text) {
      this.connectionStatus.className = `connection-status ${status}`;
      this.connectionStatus.querySelector('span').textContent = text;
  }

  toggleRecording() {
      if (this.isProcessing) {
          return;
      }

      if (this.isRecording) {
          this.stopRecording();
      } else {
          this.startRecording();
      }
  }

  startRecording() {
      if (!this.recognition) {
          this.showError('Speech recognition not available');
          return;
      }

      try {
          this.recognition.start();
      } catch (error) {
          console.error('Failed to start recording:', error);
          this.showError('Failed to start recording. Please try again.');
      }
  }

  stopRecording() {
      if (this.recognition && this.isRecording) {
          this.recognition.stop();
      }
  }

  onRecordingStart() {
      this.isRecording = true;
      this.recordBtn.classList.add('recording');
      this.micIcon.className = 'fas fa-stop';
      this.btnText.textContent = 'Listening...';
      this.audioVisualizer.classList.add('active');
      this.updateStatus('listening', 'Listening...');
  }

  onRecordingEnd() {
      this.isRecording = false;
      this.recordBtn.classList.remove('recording');
      this.micIcon.className = 'fas fa-microphone';
      this.btnText.textContent = 'Tap to Speak';
      this.audioVisualizer.classList.remove('active');

      if (!this.isProcessing) {
          this.updateStatus('ready', 'Ready');
      }
  }

  async onSpeechResult(transcript) {
      if (!transcript.trim()) {
          this.showError('No speech detected. Please try again.');
          return;
      }

      this.addMessage('user', transcript);
      this.isProcessing = true;
      this.showLoading();
      this.updateStatus('processing', 'Processing...');

      try {
          const response = await fetch('/api/chat', {
              method: 'POST',
              headers: {
                  'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                  message: transcript
              })
          });

          const data = await response.json();

          if (data.success && data.response) {
              this.addMessage('ai', data.response);
              this.speakResponse(data.response);
          } else {
              throw new Error(data.error || 'Failed to get AI response');
          }
      } catch (error) {
          console.error('Chat API error:', error);
          this.showError(`Failed to get AI response: ${error.message}`);
          this.addMessage('ai', 'Sorry, I encountered an error processing your request. Please try again.');
      } finally {
          this.hideLoading();
          this.isProcessing = false;
          this.updateStatus('ready', 'Ready');
      }
  }

  speakResponse(text) {
      if (this.speechSynthesis.speaking) {
          this.speechSynthesis.cancel();
      }

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.9;
      utterance.pitch = 1;
      utterance.volume = 0.8;

      // Try to use a more natural voice if available
      const voices = this.speechSynthesis.getVoices();
      const preferredVoice = voices.find(voice => 
          voice.name.includes('Google') || 
          voice.name.includes('Microsoft') ||
          voice.lang.startsWith('en')
      );

      if (preferredVoice) {
          utterance.voice = preferredVoice;
      }

      utterance.onstart = () => {
          this.updateStatus('speaking', 'Speaking...');
      };

      utterance.onend = () => {
          this.updateStatus('ready', 'Ready');
      };

      utterance.onerror = (event) => {
          console.error('Speech synthesis error:', event);
          this.updateStatus('ready', 'Ready');
      };

      this.speechSynthesis.speak(utterance);
  }

  addMessage(sender, content) {
      const messageDiv = document.createElement('div');
      messageDiv.className = `message ${sender}`;

      const avatar = document.createElement('div');
      avatar.className = `${sender}-avatar`;
      avatar.innerHTML = sender === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';

      const messageContent = document.createElement('div');
      messageContent.className = 'message-content';
      const paragraph = document.createElement('p');
      paragraph.textContent = content;
      messageContent.appendChild(paragraph);

      messageDiv.appendChild(avatar);
      messageDiv.appendChild(messageContent);

      this.chatMessages.appendChild(messageDiv);
      this.scrollToBottom();
  }

  clearChat() {
      // Keep only the welcome message
      const welcomeMessage = this.chatMessages.querySelector('.welcome-message');
      this.chatMessages.innerHTML = '';
      if (welcomeMessage) {
          this.chatMessages.appendChild(welcomeMessage);
      }

      // Cancel any ongoing speech
      if (this.speechSynthesis.speaking) {
          this.speechSynthesis.cancel();
      }

      this.updateStatus('ready', 'Ready');
  }

  updateStatus(type, text) {
      this.statusIndicator.className = `status-indicator ${type}`;
      this.statusText.textContent = text;
  }

  showLoading() {
      this.loadingOverlay.classList.add('active');
  }

  hideLoading() {
      this.loadingOverlay.classList.remove('active');
  }

  showError(message) {
      this.errorMessage.textContent = message;
      this.errorToast.classList.add('active');

      // Auto-hide after 8 seconds
      setTimeout(() => {
          this.hideError();
      }, 8000);
  }

  hideError() {
      this.errorToast.classList.remove('active');
  }

  scrollToBottom() {
      this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
  }

  async sendTextMessage() {
      const message = this.chatInput.value.trim();
      if (!message) {
          return;
      }

      // Clear input and disable send button
      this.chatInput.value = '';
      this.sendBtn.disabled = true;

      // Add user message to chat
      this.addMessage('user', message);
      this.isProcessing = true;
      this.showLoading();
      this.updateStatus('processing', 'Processing...');

      try {
          const response = await fetch('/api/chat', {
              method: 'POST',
              headers: {
                  'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                  message: message
              })
          });

          const data = await response.json();

          if (data.success && data.response) {
              this.addMessage('ai', data.response);
              this.speakResponse(data.response);
          } else {
              throw new Error(data.error || 'Failed to get AI response');
          }
      } catch (error) {
          console.error('Chat API error:', error);
          this.showError(`Failed to get AI response: ${error.message}`);
          this.addMessage('ai', 'Sorry, I encountered an error processing your request. Please try again.');
      } finally {
          this.hideLoading();
          this.isProcessing = false;
          this.updateStatus('ready', 'Ready');
          this.sendBtn.disabled = false;
          this.chatInput.focus();
      }
  }
}

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  // Load voices for speech synthesis
  if (window.speechSynthesis) {
      window.speechSynthesis.onvoiceschanged = () => {
          // Voices loaded
      };
  }

  // Initialize the voice chat app
  window.voiceChatApp = new VoiceChatApp();
});

// Handle page visibility change to manage speech synthesis
document.addEventListener('visibilitychange', () => {
  if (document.hidden && window.speechSynthesis && window.speechSynthesis.speaking) {
      window.speechSynthesis.pause();
  } else if (!document.hidden && window.speechSynthesis && window.speechSynthesis.paused) {
      window.speechSynthesis.resume();
  }
});
