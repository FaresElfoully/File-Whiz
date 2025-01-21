import { useState, useCallback, useEffect } from 'react'
import { Upload, FileType, X, AlertCircle, CheckCircle2, Loader2, MessageSquare, FileSpreadsheet, FileText, Presentation, FileCode, Files } from 'lucide-react'
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { cn } from "@/lib/utils"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Label } from "@/components/ui/label"
import { motion, AnimatePresence } from 'framer-motion'
import { useNavigate, useLocation } from 'react-router-dom'
import { uploadFiles, resetSession, getProcessedFiles } from '@/lib/api'

type FileStatus = 'idle' | 'uploading' | 'error' | 'success'
type FileTypeOption = 'xlsx' | 'pdf' | 'pptx' | 'doc' | 'mixed'

interface FileUpload {
  file: File
  progress: number
  status: FileStatus
  error?: string
  size: string
}

const FILE_TYPE_CONFIGS: Record<FileTypeOption, { 
  label: string, 
  accept: string[], 
  description: string,
  icon: React.ElementType,
  gradient: string
}> = {
  xlsx: {
    label: "Excel Files",
    accept: ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'],
    description: "Upload Excel spreadsheets for data analysis",
    icon: FileSpreadsheet,
    gradient: "from-emerald-500/20 to-emerald-600/20"
  },
  pdf: {
    label: "PDF Documents",
    accept: ['application/pdf'],
    description: "Upload PDF documents for text extraction",
    icon: FileText,
    gradient: "from-rose-500/20 to-rose-600/20"
  },
  pptx: {
    label: "PowerPoint",
    accept: ['application/vnd.openxmlformats-officedocument.presentationml.presentation'],
    description: "Upload presentations for content analysis",
    icon: Presentation,
    gradient: "from-amber-500/20 to-amber-600/20"
  },
  doc: {
    label: "Word Documents",
    accept: ['application/vnd.openxmlformats-officedocument.wordprocessingml.document'],
    description: "Upload Word documents for text processing",
    icon: FileCode,
    gradient: "from-blue-500/20 to-blue-600/20"
  },
  mixed: {
    label: "Mixed Files",
    accept: [
      'application/pdf',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'application/vnd.openxmlformats-officedocument.presentationml.presentation',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    ],
    description: "Upload multiple file types together",
    icon: Files,
    gradient: "from-[hsl(var(--violet-gradient-1))/0.2] to-[hsl(var(--violet-gradient-2))/0.2]"
  }
}

// Add file size formatter
const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
};

export default function UploadPage() {
  const [files, setFiles] = useState<FileUpload[]>([])
  const [isDragging, setIsDragging] = useState(false)
  const [selectedFileType, setSelectedFileType] = useState<FileTypeOption | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [processingComplete, setProcessingComplete] = useState(false)
  const [isResetting, setIsResetting] = useState(false);
  const navigate = useNavigate()
  const location = useLocation()

  const startNewSession = async () => {
    if (isResetting) return;
    setIsResetting(true);
    try {
      const result = await resetSession()
      if (result.error) {
        throw new Error(result.error)
      }

      // Clear files but keep the file type selection
      setFiles([])
      setProcessingComplete(false)
      setIsProcessing(false)
    } catch (error) {
      console.error('Error starting new session:', error)
      alert('Failed to start new session. Please try again.')
    } finally {
      setIsResetting(false);
    }
  }

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(false)
    
    if (!selectedFileType) {
      return // Prevent upload if no file type selected
    }
    
    const droppedFiles = Array.from(e.dataTransfer.files)
    handleFiles(droppedFiles)
  }, [selectedFileType])

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (!selectedFileType) {
      return // Prevent upload if no file type selected
    }
    
    const selectedFiles = Array.from(e.target.files || [])
    handleFiles(selectedFiles)
  }, [selectedFileType])

  const handleFiles = async (newFiles: File[]) => {
    const maxSize = 16 * 1024 * 1024; // 16MB
    const acceptedTypes = FILE_TYPE_CONFIGS[selectedFileType!].accept;
    
    // Validate files
    const invalidFiles = newFiles.filter(file => {
      const isValidType = acceptedTypes.includes(file.type);
      const isValidSize = file.size <= maxSize;
      if (!isValidType) {
        console.warn(`Invalid file type: ${file.type} for ${file.name}`);
      }
      if (!isValidSize) {
        console.warn(`File too large: ${formatFileSize(file.size)} for ${file.name}`);
      }
      return !isValidType || !isValidSize;
    });
    
    // Filter valid files
    const validFiles = newFiles.filter(file => 
      acceptedTypes.includes(file.type) && file.size <= maxSize
    );
    
    if (invalidFiles.length > 0) {
      const messages = invalidFiles.map(file => {
        if (!acceptedTypes.includes(file.type)) {
          return `${file.name}: Invalid file type`;
        }
        if (file.size > maxSize) {
          return `${file.name}: File too large (${formatFileSize(file.size)})`;
        }
        return `${file.name}: Unknown error`;
      });
      alert(`Some files were not accepted:\n${messages.join('\n')}`);
    }
    
    // Add valid files to state
    const fileUploads: FileUpload[] = validFiles.map(file => ({
      file,
      progress: 0,
      status: 'idle',
      size: formatFileSize(file.size)
    }));

    setFiles(prev => [...prev, ...fileUploads])
  }

  const handleAreaClick = () => {
    if (!selectedFileType) {
      alert("Please select a file type before uploading")
      return
    }
    document.getElementById('file-upload')?.click()
  }

  const removeFile = async (fileToRemove: FileUpload) => {
    try {
      // Remove file from local state first
      setFiles(prev => prev.filter(f => f.file !== fileToRemove.file))

      // Call backend to remove file from uploads folder
      const response = await fetch('/remove-file', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          filename: fileToRemove.file.name,
          fileType: fileToRemove.file.type
        })
      });

      if (!response.ok) {
        throw new Error('Failed to remove file from server');
      }
    } catch (error) {
      console.error('Error removing file:', error);
      // Restore the file in the UI if server deletion failed
      setFiles(prev => [...prev, fileToRemove]);
      alert('Failed to remove file. Please try again.');
    }
  }

  const handleUpload = async () => {
    if (files.length === 0) {
      alert('Please add some files first');
      return;
    }

    setIsProcessing(true);
    try {
      // Update all files to uploading state
      setFiles(prev => prev.map(f => ({
        ...f,
        status: 'uploading' as FileStatus,
        progress: 0
      })));

      const filesToUpload = files.map(f => f.file);
      const result = await uploadFiles(filesToUpload);
      
      if (result.error) {
        throw new Error(result.error);
      }

      // Wait for backend processing
      let processingComplete = false;
      while (!processingComplete) {
        const statusResult = await getProcessedFiles();
        if (statusResult.error) {
          throw new Error(statusResult.error);
        }

        const processedFiles = statusResult.data.files || [];
        if (processedFiles.length >= files.length) {
          processingComplete = true;
          break;
        }

        // Update progress for each file
        setFiles(prev => prev.map(f => {
          const processedFile = processedFiles.find(pf => pf.name === f.file.name);
          return {
            ...f,
            status: processedFile ? 'success' : 'uploading' as FileStatus,
            progress: processedFile ? 100 : 50
          };
        }));

        // Wait a bit before checking again
        await new Promise(resolve => setTimeout(resolve, 1000));
      }

      // All files processed successfully
      setFiles(prev => prev.map(f => ({
        ...f,
        status: 'success' as FileStatus,
        progress: 100
      })));

      setProcessingComplete(true);
      navigate('/chat');
    } catch (error: any) {
      console.error('Upload error:', error);
      
      // Update all files to error state
      setFiles(prev => prev.map(f => ({
        ...f,
        status: 'error' as FileStatus,
        progress: 0,
        error: error.message
      })));

      alert(error.message || 'Failed to upload files. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="min-h-screen bg-background pt-20"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Top Navigation Bar */}
        <motion.div 
          className="flex justify-between items-center mb-12 bg-card p-4 rounded-2xl border border-border"
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          <div className="flex gap-4">
            <Button 
              onClick={handleUpload}
              disabled={files.length === 0 || isProcessing}
              className={cn(
                "flex items-center gap-2 min-w-[150px] violet-gradient-bg",
                "hover:opacity-90 transition-opacity",
                "disabled:opacity-50"
              )}
            >
              {isProcessing ? (
                <motion.div className="flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Processing...
                </motion.div>
              ) : (
                <motion.div className="flex items-center gap-2">
                  <FileType className="h-4 w-4" />
                  Process Files
                </motion.div>
              )}
            </Button>

            {processingComplete && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
              >
                <Button
                  onClick={() => navigate('/chat')}
                  className="flex items-center gap-2 violet-gradient-bg hover:opacity-90"
                >
                  <MessageSquare className="h-4 w-4" />
                  Start Chatting
                </Button>
              </motion.div>
            )}
          </div>

          <Button 
            onClick={startNewSession}
            variant="outline"
            className={cn(
              "flex items-center gap-2 border-primary/20 hover:bg-primary/10",
              isResetting && "opacity-50 cursor-not-allowed"
            )}
            disabled={isResetting}
          >
            <FileType className="h-4 w-4" />
            Start New Session
          </Button>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2 space-y-8">
            {/* File Type Selection Grid */}
            <motion.div
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.3 }}
              className="bg-card rounded-2xl p-6 shadow-xl border border-border"
            >
              <h2 className="text-xl font-medium mb-6 text-primary">
                Select File Type
              </h2>
              <RadioGroup
                value={selectedFileType || ""}
                onValueChange={(value) => setSelectedFileType(value as FileTypeOption)}
                className="grid grid-cols-2 md:grid-cols-3 gap-6"
              >
                {Object.entries(FILE_TYPE_CONFIGS).map(([type, config], index) => {
                  const Icon = config.icon
                  return (
                    <motion.div 
                      key={type}
                      initial={{ scale: 0.9, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      transition={{ delay: index * 0.1 }}
                      className="relative group"
                    >
                      <Label
                        htmlFor={type}
                        className="block cursor-pointer"
                      >
                        <div className={cn(
                          "absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500",
                          "bg-gradient-to-br",
                          config.gradient
                        )} />
                        <div className={cn(
                          "relative p-6 rounded-2xl border border-border hover:border-primary/30 transition-all duration-300 bg-card group-hover:translate-y-[-2px]",
                          selectedFileType === type && "border-primary/50 bg-primary/5"
                        )}>
                          <RadioGroupItem 
                            value={type} 
                            id={type} 
                            className="peer sr-only" // Hide the radio button visually but keep it accessible
                          />
                          <div className="space-y-4">
                            <div className={cn(
                              "w-12 h-12 rounded-xl flex items-center justify-center transition-colors duration-300",
                              "bg-gradient-to-br",
                              config.gradient,
                              "group-hover:scale-110 transform transition-transform duration-300"
                            )}>
                              <Icon className="w-6 h-6 text-foreground" />
                            </div>
                            <div className="space-y-1">
                              <p className={cn(
                                "block font-medium text-foreground",
                                selectedFileType === type && "text-primary"
                              )}>
                                {config.label}
                              </p>
                              <p className="text-xs text-muted-foreground leading-relaxed">
                                {config.description}
                              </p>
                            </div>
                          </div>
                        </div>
                      </Label>
                    </motion.div>
                  )
                })}
              </RadioGroup>
            </motion.div>

            {/* Upload Area */}
            <motion.div
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.4 }}
              onClick={handleAreaClick}
              className={cn(
                "relative border-2 border-dashed rounded-2xl p-12 transition-all duration-300",
                "bg-card shadow-xl",
                isDragging ? "border-primary bg-primary/5" : "border-border",
                !selectedFileType && "opacity-50 cursor-not-allowed",
                files.length > 0 && "border-solid",
                selectedFileType && "hover:border-primary hover:bg-primary/5 cursor-pointer"
              )}
              onDragOver={(e) => {
                e.preventDefault()
                if (selectedFileType) setIsDragging(true)
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
            >
              <div className="relative text-center">
                <motion.div
                  animate={{
                    scale: isDragging ? 1.1 : 1,
                    rotate: isDragging ? 180 : 0
                  }}
                  transition={{ type: "spring", stiffness: 260, damping: 20 }}
                  className="bg-primary/10 rounded-full p-4 w-20 h-20 mx-auto"
                >
                  <Upload className="w-full h-full text-primary" />
                </motion.div>
                <div className="mt-6">
                  <span className="block text-sm font-medium text-foreground">
                    {selectedFileType 
                      ? `Drag your ${FILE_TYPE_CONFIGS[selectedFileType].label} here or click to browse`
                      : "Please select a file type above"}
                  </span>
                  <input
                    id="file-upload"
                    type="file"
                    className="hidden"
                    multiple
                    accept={selectedFileType ? FILE_TYPE_CONFIGS[selectedFileType].accept.join(',') : ''}
                    onChange={handleFileInput}
                    disabled={!selectedFileType}
                  />
                </div>
                {selectedFileType && (
                  <p className="mt-2 text-xs text-muted-foreground">
                    Maximum file size: 16MB
                  </p>
                )}
              </div>
            </motion.div>

            {/* File List */}
            <AnimatePresence mode="popLayout">
              {files.length > 0 && (
                <motion.div 
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  className="space-y-4"
                >
                  {files.map((fileUpload, index) => (
                    <motion.div
                      key={fileUpload.file.name}
                      initial={{ x: -20, opacity: 0 }}
                      animate={{ x: 0, opacity: 1 }}
                      exit={{ x: -20, opacity: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="relative bg-[hsl(var(--violet-1))] rounded-xl p-4 shadow-lg border border-[hsl(var(--violet-gradient-1)/0.1)] hover:border-[hsl(var(--violet-gradient-2)/0.2)] transition-all duration-300"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-4">
                          <FileType className="h-8 w-8 text-primary" />
                          <div>
                            <p className="text-sm font-medium">{fileUpload.file.name}</p>
                            <p className="text-xs text-muted-foreground">
                              {fileUpload.size}
                            </p>
                          </div>
                        </div>
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => removeFile(fileUpload)}
                          className="hover:text-destructive transition-colors"
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      </div>
                      <div className="mt-4">
                        <Progress 
                          value={fileUpload.progress} 
                          className="h-1"
                        />
                      </div>
                      <div className="mt-2 flex items-center space-x-2">
                        <AnimatePresence mode="wait">
                          {fileUpload.status === 'uploading' && (
                            <motion.p
                              initial={{ opacity: 0 }}
                              animate={{ opacity: 1 }}
                              exit={{ opacity: 0 }}
                              className="text-xs text-muted-foreground flex items-center gap-2"
                            >
                              <Loader2 className="h-3 w-3 animate-spin" />
                              Uploading... {fileUpload.progress}%
                            </motion.p>
                          )}
                          {fileUpload.status === 'error' && (
                            <motion.div
                              initial={{ opacity: 0 }}
                              animate={{ opacity: 1 }}
                              exit={{ opacity: 0 }}
                              className="flex items-center space-x-1 text-destructive"
                            >
                              <AlertCircle className="h-4 w-4" />
                              <p className="text-xs">{fileUpload.error}</p>
                            </motion.div>
                          )}
                          {fileUpload.status === 'success' && (
                            <motion.div
                              initial={{ opacity: 0 }}
                              animate={{ opacity: 1 }}
                              exit={{ opacity: 0 }}
                              className="flex items-center space-x-1 text-primary"
                            >
                              <CheckCircle2 className="h-4 w-4" />
                              <p className="text-xs">Upload complete</p>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>
                    </motion.div>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Quick Tips Sidebar */}
          <motion.div 
            initial={{ x: 20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="lg:col-span-1"
          >
            <div className="bg-card rounded-2xl p-6 shadow-xl border border-border sticky top-24">
              <h3 className="text-lg font-medium text-primary">
                Quick Tips
              </h3>
              <ul className="mt-6 space-y-4">
                {selectedFileType ? (
                  <>
                    <motion.li 
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="flex items-start space-x-3 text-sm text-muted-foreground"
                    >
                      <span>• Selected type: {FILE_TYPE_CONFIGS[selectedFileType].label}</span>
                    </motion.li>
                    <motion.li 
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.1 }}
                      className="flex items-start space-x-3 text-sm text-muted-foreground"
                    >
                      <span>• Maximum file size: 16MB</span>
                    </motion.li>
                    <motion.li 
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.2 }}
                      className="flex items-start space-x-3 text-sm text-muted-foreground"
                    >
                      <span>• Drag and drop multiple files</span>
                    </motion.li>
                  </>
                ) : (
                  <motion.li 
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="text-sm text-muted-foreground"
                  >
                    Please select a file type to begin uploading
                  </motion.li>
                )}
              </ul>
            </div>
          </motion.div>
        </div>
      </div>
    </motion.div>
  )
} 