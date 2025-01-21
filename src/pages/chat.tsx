import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Button } from "@/components/ui/button"
import { ArrowLeft, Send, Upload, Bot, User, Loader2, Mic, MicOff, AlertCircle, MessageSquare, Trash2, User2, ArrowDown, Copy, Check } from 'lucide-react'
import { useNavigate, useLocation } from 'react-router-dom'
import { cn } from '@/lib/utils'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { queryDocuments, voiceToText } from '@/lib/api'
import '@/styles/chat.css'

interface Message {
  role: 'user' | 'assistant'
  content: string
  sources?: any[]
}

function convertToWav(audioChunks: Blob[]): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const audioContext = new AudioContext();
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    const fileReader = new FileReader();

    fileReader.onload = async (event) => {
      try {
        const arrayBuffer = event.target?.result as ArrayBuffer;
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        
        // Convert to WAV
        const wavBuffer = audioBufferToWav(audioBuffer);
        const wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });
        resolve(wavBlob);
      } catch (error) {
        reject(error);
      }
    };

    fileReader.onerror = (error) => reject(error);
    fileReader.readAsArrayBuffer(audioBlob);
  });
}

// WAV conversion utility
function audioBufferToWav(buffer: AudioBuffer): ArrayBuffer {
  const numChannels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const format = 1; // PCM
  const bitDepth = 16;
  
  const bytesPerSample = bitDepth / 8;
  const blockAlign = numChannels * bytesPerSample;
  
  const data = new Float32Array(buffer.length * numChannels);
  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < buffer.length; i++) {
      data[i * numChannels + channel] = channelData[i];
    }
  }
  
  const dataLength = data.length * bytesPerSample;
  const bufferLength = 44 + dataLength;
  const arrayBuffer = new ArrayBuffer(bufferLength);
  const view = new DataView(arrayBuffer);
  
  // WAV header
  writeString(view, 0, 'RIFF');
  view.setUint32(4, bufferLength - 8, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, format, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitDepth, true);
  writeString(view, 36, 'data');
  view.setUint32(40, dataLength, true);
  
  // Write audio data
  let offset = 44;
  for (let i = 0; i < data.length; i++) {
    const sample = Math.max(-1, Math.min(1, data[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
    offset += 2;
  }
  
  return arrayBuffer;
}

function interleave(buffer: AudioBuffer): Float32Array {
  const numChannels = buffer.numberOfChannels;
  const length = buffer.length * numChannels;
  const result = new Float32Array(length);
  
  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < buffer.length; i++) {
      result[i * numChannels + channel] = channelData[i];
    }
  }
  
  return result;
}

function floatTo16BitPCM(view: DataView, offset: number, input: Float32Array): void {
  for (let i = 0; i < input.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, input[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
}

function writeString(view: DataView, offset: number, string: string): void {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

// Custom component for rendering text content
const CustomParagraph = ({ children }: { children: React.ReactNode }) => {
  if (typeof children === 'string') {
    // Split text by newlines while preserving empty lines
    const lines = children.split(/\n/);
    
    return (
      <div className="space-y-2">
        {lines.map((line, index) => {
          if (!line.trim()) {
            return <div key={index} className="h-4" />; // Empty line spacing
          }
          
          const hasArabic = /[\u0600-\u06FF]/.test(line);
          return (
            <p
              key={index}
              className={cn(
                "text-[15px] leading-relaxed",
                hasArabic && "text-right",
                "whitespace-pre-wrap break-words"
              )}
              style={{ 
                direction: hasArabic ? 'rtl' : 'ltr',
                wordSpacing: '0.05em',
                lineHeight: '1.8'
              }}
            >
              {line}
            </p>
          );
        })}
      </div>
    );
  }
  return <p className="mb-2 last:mb-0">{children}</p>;
};

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isRecording, setIsRecording] = useState(false)
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null)
  const [recordingError, setRecordingError] = useState<string | null>(null)
  const [selectedLanguage] = useState('ar-SA')
  const [isLoading, setIsLoading] = useState(false)
  const [isScrolling, setIsScrolling] = useState(false)
  const [showScrollButton, setShowScrollButton] = useState(false)
  const [isAtBottom, setIsAtBottom] = useState(true)
  const [copiedMessageId, setCopiedMessageId] = useState<number | null>(null)
  const navigate = useNavigate()
  const location = useLocation()
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const chatContainerRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const scrollTimeout = useRef<NodeJS.Timeout>()

  // Enhanced scroll to bottom with smooth animation
  const scrollToBottom = (behavior: ScrollBehavior = 'smooth') => {
    if (chatContainerRef.current) {
      const { scrollHeight } = chatContainerRef.current
      chatContainerRef.current.scrollTo({
        top: scrollHeight,
        behavior,
      })
    }
  }

  // Check if scrolled to bottom
  const checkIfAtBottom = () => {
    if (chatContainerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = chatContainerRef.current
      const isBottom = Math.abs(scrollHeight - scrollTop - clientHeight) < 10
      setIsAtBottom(isBottom)
      setShowScrollButton(!isBottom)
    }
  }

  // Handle scroll events with debounce
  const handleScroll = () => {
    if (!chatContainerRef.current) return
    
    checkIfAtBottom()
    setIsScrolling(true)

    if (scrollTimeout.current) {
      clearTimeout(scrollTimeout.current)
    }

    scrollTimeout.current = setTimeout(() => {
      setIsScrolling(false)
    }, 150)
  }

  // Clean up scroll timeout
  useEffect(() => {
    return () => {
      if (scrollTimeout.current) {
        clearTimeout(scrollTimeout.current)
      }
    }
  }, [])

  // Set up scroll event listener
  useEffect(() => {
    const container = chatContainerRef.current
    if (container) {
      container.addEventListener('scroll', handleScroll)
      checkIfAtBottom() // Initial check
    }
    return () => {
      if (container) {
        container.removeEventListener('scroll', handleScroll)
      }
    }
  }, [])

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (messages.length > 0) {
      const lastMessage = messages[messages.length - 1]
      if (lastMessage.role === 'user' || isAtBottom) {
        scrollToBottom()
      }
    }
  }, [messages, isAtBottom])

  // Focus input on mount and after sending message
  useEffect(() => {
    if (!isLoading && inputRef.current) {
      inputRef.current.focus()
    }
  }, [isLoading])

  const clearSession = async () => {
    try {
      await fetch('/clear-session', {
        method: 'POST',
      })
      setMessages([])
    } catch (error) {
      console.error('Failed to clear session:', error)
    }
  }

  const handleNewMessage = (newMessage: Message) => {
    setMessages(prev => [...prev, newMessage])
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage = input.trim()
    setInput('')
    handleNewMessage({ role: 'user', content: userMessage })
    setIsLoading(true)

    try {
      const result = await queryDocuments(userMessage)
      if (result.error) {
        throw new Error(result.error)
      }

      handleNewMessage({ 
        role: 'assistant', 
        content: result.data.response,
        sources: result.data.sources 
      })
    } catch (error) {
      console.error('Error:', error)
      handleNewMessage({ 
        role: 'assistant', 
        content: error instanceof Error ? error.message : 'Sorry, I encountered an error processing your request.' 
      })
    } finally {
      setIsLoading(false)
    }
  }

  const MAX_RETRIES = 3;
  const RETRY_DELAY = 1000; // 1 second

  const startRecording = async () => {
    try {
      setRecordingError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          channelCount: 1,
          sampleRate: 16000
        } 
      });

      const options = {
        mimeType: 'audio/webm;codecs=opus',
        audioBitsPerSecond: 16000
      };

      const recorder = new MediaRecorder(stream, options);
      const audioChunks: BlobPart[] = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunks.push(e.data);
          console.log('Received audio chunk:', e.data.size, 'bytes');
        }
      };

      recorder.onstop = async () => {
        let retries = 0;
        let success = false;

        while (retries < MAX_RETRIES && !success) {
          try {
            console.log(`Voice recognition attempt ${retries + 1}/${MAX_RETRIES}`);
            
            if (audioChunks.length === 0) {
              throw new Error('No audio data recorded');
            }

            // Create WebM blob from chunks
            const webmBlob = new Blob(audioChunks, { type: 'audio/webm' });
            
            // Convert WebM to WAV
            const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
            const arrayBuffer = await webmBlob.arrayBuffer();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            
            // Create WAV file
            const wavBuffer = audioBufferToWav(audioBuffer);
            const wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });
            
            const text = await handleVoiceToText(wavBlob);
            if (text) {
              console.log('Voice recognition successful:', text);
              setInput(text);
              setRecordingError(null);
              success = true;
              break;
            }
          } catch (error) {
            console.error(`Voice recognition attempt ${retries + 1} failed:`, error);
            retries++;
            
            if (retries < MAX_RETRIES) {
              await new Promise(resolve => setTimeout(resolve, RETRY_DELAY));
            } else {
              setRecordingError(
                error instanceof Error 
                  ? error.message 
                  : 'Failed to process voice input after multiple attempts. Please try again.'
              );
            }
          }
        }

        stream.getTracks().forEach(track => track.stop());
      };

      console.log('Starting recording...');
      recorder.start(200);
      setMediaRecorder(recorder);
      setIsRecording(true);

    } catch (error) {
      console.error('Error accessing microphone:', error);
      setRecordingError(
        error instanceof Error 
          ? error.message 
          : 'Failed to access microphone. Please make sure you have granted microphone permissions.'
      );
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    if (mediaRecorder && isRecording) {
      console.log('Stopping recording...');
      mediaRecorder.stop()
      setIsRecording(false)
      setMediaRecorder(null)
    }
  }

  // Optimized WAV conversion
  async function convertToWavOptimized(audioBlob: Blob): Promise<Blob> {
    return new Promise((resolve, reject) => {
      const fileReader = new FileReader();
      
      fileReader.onload = async (event) => {
        try {
          const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
          const arrayBuffer = event.target?.result as ArrayBuffer;
          
          // Decode audio with optimized settings
          const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
          
          // Convert to mono if stereo
          const numberOfChannels = 1;
          const sampleRate = 16000; // Reduced sample rate
          const length = audioBuffer.length;
          
          // Create offline context for faster processing
          const offlineContext = new OfflineAudioContext(numberOfChannels, length, sampleRate);
          const source = offlineContext.createBufferSource();
          source.buffer = audioBuffer;
          source.connect(offlineContext.destination);
          source.start();
          
          const renderedBuffer = await offlineContext.startRendering();
          const wavBuffer = audioBufferToWav(renderedBuffer);
          const wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });
          
          resolve(wavBlob);
        } catch (error) {
          reject(error);
        }
      };
      
      fileReader.onerror = reject;
      fileReader.readAsArrayBuffer(audioBlob);
    });
  }

  const handleVoiceToText = async (audioBlob: Blob) => {
    try {
      const result = await voiceToText(audioBlob)
      if (result.error) {
        throw new Error(result.error)
      }
      return result.data.text
    } catch (error) {
      console.error('Voice to text error:', error)
      throw error
    }
  }

  const handleVoiceRecognition = async () => {
    if (!mediaRecorder) return;
    
    try {
      setIsRecording(false);
      setIsLoading(true);
      
      const audioBlob = new Blob([], { type: 'audio/webm' });
      if (!audioBlob) {
        console.error('No audio data available');
        return;
      }
      
      // Create form data with audio and language
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.wav');
      formData.append('language', 'ar-AR'); // Set Arabic as the language
      
      const response = await fetch('/api/voice-to-text', {
        method: 'POST',
        body: formData,
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to convert speech to text');
      }
      
      if (data.success && data.text) {
        setInput(data.text);
      }
    } catch (error) {
      console.error('Voice recognition error:', error);
      // toast({
      //   title: 'خطأ في التعرف على الصوت',
      //   description: error instanceof Error ? error.message : 'حدث خطأ غير متوقع',
      //   variant: 'destructive',
      // });
    } finally {
      setIsLoading(false);
    }
  };

  // Copy message content to clipboard
  const copyMessage = async (messageContent: string, messageId: number) => {
    try {
      await navigator.clipboard.writeText(messageContent);
      setCopiedMessageId(messageId);
      setTimeout(() => setCopiedMessageId(null), 2000); // Reset after 2 seconds
    } catch (err) {
      console.error('Failed to copy text:', err);
    }
  };

  // Custom component for rendering markdown code blocks
  const CodeBlock = ({ inline, className, children, ...props }: any) => {
    const match = /language-(\w+)/.exec(className || '')
    return !inline && match ? (
      <SyntaxHighlighter
        style={vscDarkPlus}
        language={match[1]}
        PreTag="div"
        className="rounded-lg my-2"
        {...props}
      >
        {String(children).replace(/\n$/, '')}
      </SyntaxHighlighter>
    ) : (
      <code className="bg-primary/10 rounded px-1" {...props}>
        {children}
      </code>
    )
  }

  // Custom component for rendering markdown pre blocks
  const PreBlock = ({ children, ...props }: any) => (
    <pre className="rounded-lg my-2" {...props}>
      {children}
    </pre>
  )

  // Add styles for the recording button
  const recordingButtonClass = `
    p-2.5 rounded-lg transition-all duration-200 flex items-center justify-center
    ${isRecording 
      ? 'bg-red-500 hover:bg-red-600 animate-pulse' 
      : 'bg-blue-500 hover:bg-blue-600'
    }
    text-white shadow-lg hover:shadow-xl transform hover:scale-105 active:scale-95
  `

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="flex flex-col h-screen bg-gradient-to-br from-background via-background/95 to-primary/5"
    >
      {/* Header */}
      <motion.header 
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="fixed top-0 left-0 right-0 z-10 bg-background/60 backdrop-blur-xl border-b border-border/20 shadow-sm"
      >
        <div className="max-w-5xl mx-auto px-4 py-3 flex justify-between items-center">
          <motion.div 
            className="flex items-center gap-3"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <div className="relative">
              <MessageSquare className="h-6 w-6 text-primary" />
              <motion.div
                animate={{ 
                  scale: [1, 1.2, 1],
                  opacity: [0.5, 0.8, 0.5]
                }}
                transition={{ 
                  duration: 2,
                  repeat: Infinity,
                  ease: "easeInOut"
                }}
                className="absolute -inset-1 bg-primary/20 rounded-full blur-sm"
              />
            </div>
            <h1 className="text-xl font-semibold bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
              FileWhiz Chat
            </h1>
          </motion.div>
          <div className="flex items-center gap-2">
            <Button
              onClick={() => navigate('/upload')}
              variant="ghost"
              size="sm"
              className="gap-2 text-muted-foreground hover:text-foreground relative overflow-hidden group"
            >
              <Upload className="h-4 w-4 transition-transform group-hover:scale-110" />
              <span className="hidden sm:inline relative">
                <span className="relative z-10">Upload Files</span>
                <motion.span
                  className="absolute inset-0 bg-primary/10 rounded-md -z-10"
                  initial={{ scale: 0 }}
                  whileHover={{ scale: 1 }}
                  transition={{ type: "spring", stiffness: 300, damping: 20 }}
                />
              </span>
            </Button>
            <Button
              onClick={clearSession}
              variant="ghost"
              size="icon"
              className="text-destructive/60 hover:text-destructive relative group"
            >
              <Trash2 className="h-5 w-5 transition-transform group-hover:scale-110" />
              <motion.span
                className="absolute inset-0 bg-destructive/10 rounded-md -z-10"
                initial={{ scale: 0 }}
                whileHover={{ scale: 1 }}
                transition={{ type: "spring", stiffness: 300, damping: 20 }}
              />
            </Button>
          </div>
        </div>
      </motion.header>

      {/* Chat Container */}
      <div 
        ref={chatContainerRef}
        className="flex-1 overflow-y-auto pt-16 pb-20 scroll-smooth"
        style={{
          scrollbarWidth: 'thin',
          scrollbarColor: 'rgba(var(--primary) / 0.3) transparent',
        }}
      >
        <div className="max-w-5xl mx-auto px-4 sm:px-6 md:px-8">
          <div className="space-y-4 md:space-y-6 py-4">
            {messages.length === 0 ? (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3, type: "spring" }}
                className="text-center py-12 sm:py-16 md:py-20"
              >
                <motion.div 
                  className="w-20 h-20 mx-auto mb-6 rounded-3xl bg-gradient-to-br from-primary/20 to-primary/5 flex items-center justify-center relative overflow-hidden group"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <MessageSquare className="h-10 w-10 text-primary/60 transition-transform group-hover:scale-110" />
                  <motion.div
                    animate={{ 
                      scale: [1, 1.2, 1],
                      opacity: [0.5, 0.8, 0.5]
                    }}
                    transition={{ 
                      duration: 3,
                      repeat: Infinity,
                      ease: "easeInOut"
                    }}
                    className="absolute -inset-1 bg-primary/5"
                  />
                </motion.div>
                <motion.h2 
                  className="text-xl font-semibold mb-2 bg-gradient-to-r from-foreground/80 to-foreground/60 bg-clip-text text-transparent"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.4 }}
                >
                  Welcome to FileWhiz Chat!
                </motion.h2>
                <motion.p 
                  className="text-muted-foreground max-w-sm mx-auto"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.5 }}
                >
                  Ask questions about your documents or use voice commands to interact.
                </motion.p>
              </motion.div>
            ) : (
              <div className="space-y-4 md:space-y-6">
                {messages.map((message, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 20, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    transition={{ 
                      delay: Math.min(index * 0.1, 0.5), // Cap the delay
                      type: "spring",
                      stiffness: 300,
                      damping: 25 
                    }}
                    className={`flex gap-3 ${
                      message.role === 'user' ? 'justify-end' : 'justify-start'
                    } group`}
                  >
                    <motion.div 
                      className={`flex gap-3 max-w-[85%] sm:max-w-[75%] md:max-w-[65%] ${
                        message.role === 'user' ? 'flex-row-reverse' : 'flex-row'
                      }`}
                      whileHover={{ scale: 1.01 }}
                      transition={{ type: "spring", stiffness: 300, damping: 25 }}
                    >
                      <motion.div 
                        className={`w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0 relative overflow-hidden
                          ${message.role === 'user' 
                            ? 'bg-gradient-to-br from-primary to-primary/80 text-primary-foreground' 
                            : 'bg-gradient-to-br from-primary/20 to-primary/5 text-primary'
                          }`}
                        whileHover={{ scale: 1.1 }}
                        whileTap={{ scale: 0.9 }}
                      >
                        {message.role === 'user' ? (
                          <User2 className="h-5 w-5" />
                        ) : (
                          <Bot className="h-5 w-5" />
                        )}
                        <motion.div
                          className="absolute inset-0 bg-white/10"
                          animate={{ 
                            scale: [1, 1.2, 1],
                            opacity: [0.1, 0.2, 0.1]
                          }}
                          transition={{ 
                            duration: 3,
                            repeat: Infinity,
                            ease: "easeInOut"
                          }}
                        />
                      </motion.div>
                      <div className={`rounded-2xl px-4 py-3 relative ${
                        message.role === 'user'
                          ? 'bg-gradient-to-br from-primary to-primary/90 text-primary-foreground'
                          : 'bg-gradient-to-br from-card to-card/95 dark:from-gray-800/90 dark:to-gray-800/70 border border-border/30'
                        } shadow-lg hover:shadow-xl transition-all duration-300`}
                      >
                        <motion.div
                          className="absolute inset-0 bg-white/5 rounded-2xl"
                          animate={{ 
                            scale: [1, 1.02, 1],
                            opacity: [0.1, 0.15, 0.1]
                          }}
                          transition={{ 
                            duration: 4,
                            repeat: Infinity,
                            ease: "easeInOut"
                          }}
                        />
                        <div className="relative">
                          <div className="flex justify-between items-start gap-2">
                            <div className="flex-1">
                              <ReactMarkdown
                                components={{
                                  code: CodeBlock,
                                  p: CustomParagraph,
                                  pre: ({ children }) => (
                                    <pre className="relative my-4 overflow-x-auto rounded-lg bg-muted/50 p-4">
                                      {children}
                                    </pre>
                                  ),
                                }}
                                className={cn(
                                  "prose prose-sm max-w-none",
                                  "prose-p:my-1",
                                  "prose-pre:my-4",
                                  message.role === 'user' ? "prose-invert" : "prose-neutral dark:prose-invert"
                                )}
                              >
                                {message.content}
                              </ReactMarkdown>
                            </div>
                            <motion.button
                              initial={{ opacity: 0 }}
                              animate={{ opacity: 1 }}
                              whileHover={{ scale: 1.1 }}
                              whileTap={{ scale: 0.9 }}
                              onClick={() => copyMessage(message.content, index)}
                              className={`flex-shrink-0 p-1.5 rounded-lg mt-0.5
                                ${message.role === 'user' 
                                  ? 'bg-primary-foreground/10 hover:bg-primary-foreground/20 text-primary-foreground' 
                                  : 'bg-primary/10 hover:bg-primary/20 text-primary'
                                } opacity-0 group-hover:opacity-100 transition-all duration-200`}
                              title="Copy message"
                            >
                              {copiedMessageId === index ? (
                                <Check className="h-3.5 w-3.5" />
                              ) : (
                                <Copy className="h-3.5 w-3.5" />
                              )}
                            </motion.button>
                          </div>
                        </div>
                        {message.sources && message.sources.length > 0 && (
                          <div className="mt-3 pt-2 border-t border-border/40">
                            <p className="text-xs font-medium text-muted-foreground mb-1">Sources:</p>
                            <div className="space-y-1">
                              {message.sources.map((source, idx) => (
                                <div
                                  key={idx}
                                  className="text-xs px-2 py-1 rounded-lg bg-background/50 text-muted-foreground"
                                >
                                  {source.source}
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </motion.div>
                  </motion.div>
                ))}
              </div>
            )}
            {/* Scroll to bottom button */}
            <motion.button
              initial={false}
              animate={{
                opacity: showScrollButton ? 1 : 0,
                y: showScrollButton ? 0 : 20,
                pointerEvents: showScrollButton ? 'auto' : 'none',
              }}
              transition={{ duration: 0.2 }}
              onClick={() => scrollToBottom('smooth')}
              className="fixed bottom-24 right-4 sm:right-6 md:right-8 z-20 p-3 rounded-full 
                bg-primary/90 text-primary-foreground shadow-lg hover:shadow-xl 
                transform hover:scale-110 active:scale-95 transition-all duration-200"
            >
              <motion.div
                animate={{ y: [0, 3, 0] }}
                transition={{ 
                  duration: 1.5,
                  repeat: Infinity,
                  ease: "easeInOut"
                }}
              >
                <ArrowDown className="w-5 h-5" />
              </motion.div>
            </motion.button>
            <div ref={messagesEndRef} className="h-px" />
          </div>
        </div>
      </div>

      {/* Input Area */}
      <motion.div 
        initial={false}
        animate={{
          y: isScrolling ? 10 : 0,
          opacity: isScrolling ? 0.9 : 1,
        }}
        transition={{ 
          type: "spring",
          stiffness: 300,
          damping: 30
        }}
        className="fixed bottom-0 left-0 right-0 bg-background/80 backdrop-blur-xl border-t border-border/20 shadow-lg transform-gpu"
      >
        <div className="max-w-5xl mx-auto px-4 sm:px-6 md:px-8 py-4">
          <form onSubmit={handleSubmit} className="flex items-center gap-2">
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={isRecording ? stopRecording : startRecording}
              className={cn(
                "flex-shrink-0 rounded-xl hover:bg-primary/10 transition-all duration-200",
                isRecording && "bg-destructive/10 text-destructive hover:bg-destructive/20"
              )}
            >
              <motion.div
                initial={false}
                animate={{ scale: isRecording ? [1, 1.1, 1] : 1 }}
                transition={{
                  duration: 2,
                  repeat: isRecording ? Infinity : 0,
                  ease: "easeInOut"
                }}
              >
                {isRecording ? (
                  <div className="relative">
                    <MicOff className="h-5 w-5" />
                    <motion.div
                      animate={{ 
                        scale: [1, 1.2, 1],
                        opacity: [0.6, 1, 0.6]
                      }}
                      transition={{ 
                        duration: 1.5,
                        repeat: Infinity,
                        ease: "easeInOut"
                      }}
                      className="absolute -top-1 -right-1 w-2 h-2 bg-destructive rounded-full"
                    />
                  </div>
                ) : (
                  <Mic className="h-5 w-5" />
                )}
              </motion.div>
            </Button>
            <div className="relative flex-1 min-w-0">
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type your message..."
                className="w-full px-4 py-2.5 bg-background/50 border border-border/30 
                  focus:border-primary/30 rounded-xl placeholder:text-muted-foreground/50 
                  focus:outline-none focus:ring-2 focus:ring-primary/20 shadow-sm 
                  transition-all duration-300 text-[15px] leading-relaxed"
                disabled={isLoading}
              />
              <Button
                type="submit"
                size="icon"
                disabled={!input.trim() || isLoading}
                className={cn(
                  "absolute right-2 top-1/2 -translate-y-1/2 rounded-lg transition-all duration-300",
                  !input.trim() || isLoading 
                    ? "opacity-50 cursor-not-allowed" 
                    : "hover:bg-primary/90 hover:shadow-md"
                )}
              >
                <motion.div
                  initial={false}
                  animate={isLoading ? { rotate: 360 } : { rotate: 0 }}
                  transition={isLoading ? {
                    duration: 1,
                    repeat: Infinity,
                    ease: "linear"
                  } : {
                    type: "spring",
                    stiffness: 200,
                    damping: 10
                  }}
                >
                  {isLoading ? (
                    <Loader2 className="h-4 w-4" />
                  ) : (
                    <Send className="h-4 w-4" />
                  )}
                </motion.div>
              </Button>
            </div>
          </form>
          {recordingError && (
            <motion.div
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 5 }}
              transition={{
                type: "spring",
                stiffness: 500,
                damping: 30
              }}
              className="mt-2 flex items-center gap-2 text-sm text-destructive"
            >
              <AlertCircle className="h-4 w-4" />
              <span>{recordingError}</span>
            </motion.div>
          )}
        </div>
      </motion.div>
    </motion.div>
  )
}