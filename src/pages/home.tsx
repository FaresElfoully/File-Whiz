import { motion, useScroll, useTransform, useInView } from 'framer-motion'
import { ArrowRight, FileText, Zap, Shield, Brain, CheckCircle, MousePointerClick, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useNavigate } from 'react-router-dom'
import { useRef, useState } from 'react'
import { cn } from '@/lib/utils'
import { clearSession } from '@/lib/api'

export default function HomePage() {
  const navigate = useNavigate()
  const { scrollY } = useScroll()
  const featuresRef = useRef(null)
  const stepsRef = useRef(null)
  const isFeaturesInView = useInView(featuresRef, { once: true, margin: "-100px" })
  const isStepsInView = useInView(stepsRef, { once: true, margin: "-100px" })
  const [isStarting, setIsStarting] = useState(false)

  // Parallax effect for hero section
  const y = useTransform(scrollY, [0, 500], [0, 150])
  const opacity = useTransform(scrollY, [0, 200], [1, 0])

  const handleGetStarted = async () => {
    if (isStarting) return;
    setIsStarting(true);
    
    try {
      // Clear session before starting
      const result = await clearSession();
      
      if (result.error) {
        console.error('Failed to clear session:', result.error);
        // Show error toast if we have UI components for it
        // For now, proceed anyway
      }
      
      // Navigate to upload page
      navigate('/upload');
    } catch (error) {
      console.error('Error clearing session:', error);
      // Navigate anyway - the upload page will handle session initialization
      navigate('/upload');
    } finally {
      setIsStarting(false);
    }
  }

  return (
    <div className="relative">
      {/* Hero Section with Parallax */}
      <section className="relative min-h-[90vh] flex items-center justify-center overflow-hidden">
        {/* Animated Background Elements */}
        <motion.div 
          className="absolute inset-0 -z-10"
          style={{ y, opacity }}
        >
          <div className="absolute inset-x-0 -top-40 transform-gpu overflow-hidden blur-3xl sm:-top-80">
            <div className="relative left-[calc(50%-11rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 rotate-[30deg] bg-gradient-to-tr from-[hsl(var(--violet-gradient-1))] to-[hsl(var(--violet-gradient-3))] opacity-30 sm:left-[calc(50%-30rem)] sm:w-[72.1875rem]" />
          </div>
          {/* Additional floating elements */}
          <motion.div
            animate={{
              y: [0, 20, 0],
              rotate: [0, 5, 0],
            }}
            transition={{
              duration: 8,
              repeat: Infinity,
              ease: "easeInOut"
            }}
            className="absolute right-[10%] top-[20%] w-24 h-24 bg-primary/10 rounded-full blur-xl"
          />
          <motion.div
            animate={{
              y: [0, -30, 0],
              rotate: [0, -5, 0],
            }}
            transition={{
              duration: 10,
              repeat: Infinity,
              ease: "easeInOut"
            }}
            className="absolute left-[15%] bottom-[30%] w-32 h-32 bg-violet-500/10 rounded-full blur-xl"
          />
        </motion.div>

        <div className="container mx-auto px-4 relative">
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center max-w-3xl mx-auto"
          >
            <motion.div
              initial={{ scale: 0.5, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.5 }}
              className="mb-6"
            >
              <img 
                src="/public/logo.png"
                alt="Logo" 
                className="w-20 h-20 mx-auto dark:invert dark:brightness-200 transition-all duration-200"
              />
            </motion.div>

            <motion.h1 
              className="text-5xl md:text-7xl font-bold tracking-tight mb-6"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-[hsl(var(--violet-gradient-1))] via-[hsl(var(--violet-gradient-2))] to-[hsl(var(--violet-gradient-3))]">
                Transform Your Documents
              </span>
              <br />
              <span className="text-foreground">
                with AI-Powered Analysis
              </span>
            </motion.h1>
            
            <motion.p 
              className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
            >
              Upload your documents and let our advanced AI analyze, summarize, and extract insights in seconds. 
              Support for multiple formats and languages.
            </motion.p>

            <motion.div 
              className="flex flex-col sm:flex-row gap-4 justify-center"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
            >
              <Button 
                size="lg" 
                onClick={handleGetStarted}
                disabled={isStarting}
                className={cn(
                  "bg-primary hover:bg-primary/90 group relative overflow-hidden",
                  isStarting && "opacity-70 cursor-not-allowed"
                )}
              >
                <span className="relative z-10">
                  {isStarting ? 'Starting...' : 'Get Started'}
                </span>
                {!isStarting && (
                  <motion.div
                    className="absolute inset-0 bg-white/20"
                    initial={{ x: "-100%" }}
                    whileHover={{ x: "100%" }}
                    transition={{ duration: 0.5 }}
                  />
                )}
                {isStarting ? (
                  <Loader2 className="ml-2 h-4 w-4 animate-spin" />
                ) : (
                  <ArrowRight className="ml-2 h-4 w-4 relative z-10 group-hover:translate-x-1 transition-transform" />
                )}
              </Button>
              <Button 
                size="lg" 
                variant="outline"
                onClick={() => navigate('/chat')}
                className="group"
              >
                Try Demo
                <MousePointerClick className="ml-2 h-4 w-4 group-hover:scale-110 transition-transform" />
              </Button>
            </motion.div>

            {/* Scroll Indicator */}
            <motion.div 
              className="absolute bottom-8 left-1/2 -translate-x-1/2"
              animate={{
                y: [0, 10, 0],
              }}
              transition={{
                duration: 1.5,
                repeat: Infinity,
                ease: "easeInOut"
              }}
            >
              <div className="w-6 h-10 rounded-full border-2 border-muted-foreground/20 flex items-start justify-center p-2">
                <motion.div
                  className="w-1 h-1 rounded-full bg-primary"
                  animate={{
                    y: [0, 12, 0],
                  }}
                  transition={{
                    duration: 1.5,
                    repeat: Infinity,
                    ease: "easeInOut"
                  }}
                />
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Features Section with Hover Effects */}
      <section ref={featuresRef} className="py-20 bg-card/50">
        <div className="container mx-auto px-4">
          <motion.div 
            className="text-center mb-16"
            initial={{ opacity: 0, y: 20 }}
            animate={isFeaturesInView ? { opacity: 1, y: 0 } : {}}
          >
            <h2 className="text-4xl font-bold mb-4">Powerful Features</h2>
            <p className="text-muted-foreground text-lg">Everything you need to analyze your documents effectively</p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                animate={isFeaturesInView ? { opacity: 1, y: 0 } : {}}
                transition={{ delay: index * 0.2 }}
                whileHover={{ y: -5, transition: { duration: 0.2 } }}
                className="relative p-6 rounded-2xl border bg-card hover:shadow-lg transition-all duration-300 group"
              >
                <div className={`w-12 h-12 rounded-lg flex items-center justify-center mb-4 ${feature.bgColor} group-hover:scale-110 transition-transform duration-300`}>
                  <feature.icon className={`w-6 h-6 ${feature.iconColor}`} />
                </div>
                <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                <p className="text-muted-foreground">{feature.description}</p>
                {/* Feature highlights */}
                <ul className="mt-4 space-y-2">
                  {feature.highlights.map((highlight, i) => (
                    <motion.li
                      key={i}
                      initial={{ opacity: 0, x: -10 }}
                      animate={isFeaturesInView ? { opacity: 1, x: 0 } : {}}
                      transition={{ delay: index * 0.2 + i * 0.1 }}
                      className="flex items-center text-sm text-muted-foreground"
                    >
                      <CheckCircle className="w-4 h-4 mr-2 text-primary" />
                      {highlight}
                    </motion.li>
                  ))}
                </ul>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section with Interactive Steps */}
      <section ref={stepsRef} className="py-20">
        <div className="container mx-auto px-4">
          <motion.div 
            className="text-center mb-16"
            initial={{ opacity: 0, y: 20 }}
            animate={isStepsInView ? { opacity: 1, y: 0 } : {}}
          >
            <h2 className="text-4xl font-bold mb-4">How It Works</h2>
            <p className="text-muted-foreground text-lg">Simple steps to get started with our document analysis</p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8 relative">
            {/* Connection Lines */}
            <div className="hidden md:block absolute top-1/2 left-1/4 right-1/4 h-0.5 bg-gradient-to-r from-primary/50 to-primary/50 -translate-y-1/2" />
            
            {steps.map((step, index) => (
              <motion.div
                key={step.title}
                initial={{ opacity: 0, y: 20 }}
                animate={isStepsInView ? { opacity: 1, y: 0 } : {}}
                transition={{ delay: index * 0.2 }}
                whileHover={{ scale: 1.05 }}
                className="text-center relative"
              >
                <motion.div 
                  className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mx-auto mb-6 relative"
                  whileHover={{ rotate: 360 }}
                  transition={{ duration: 0.5 }}
                >
                  <span className="text-2xl font-bold text-primary">{index + 1}</span>
                  <div className="absolute inset-0 border-2 border-primary rounded-full opacity-20 animate-ping" />
                </motion.div>
                <h3 className="text-2xl font-semibold mb-3">{step.title}</h3>
                <p className="text-muted-foreground">{step.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Enhanced CTA Section */}
      <section className="py-20 bg-card/50 relative overflow-hidden">
        <motion.div
          className="absolute inset-0 opacity-30"
          animate={{
            backgroundPosition: ['0% 0%', '100% 100%'],
          }}
          transition={{
            duration: 20,
            repeat: Infinity,
            repeatType: 'reverse',
          }}
          style={{
            backgroundImage: 'radial-gradient(circle at center, hsl(var(--violet-gradient-1)) 0%, transparent 50%)',
          }}
        />
        
        <div className="container mx-auto px-4 relative">
          <motion.div 
            className="text-center max-w-3xl mx-auto"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl font-bold mb-4">Ready to Get Started?</h2>
            <p className="text-muted-foreground text-lg mb-8">
              Join thousands of users who are already transforming their document analysis workflow
            </p>
            <Button 
              size="lg" 
              onClick={handleGetStarted}
              disabled={isStarting}
              className={cn(
                "bg-primary hover:bg-primary/90 group relative overflow-hidden",
                isStarting && "opacity-70 cursor-not-allowed"
              )}
            >
              <span className="relative z-10">
                {isStarting ? 'Starting...' : 'Try It Now'}
              </span>
              {!isStarting && (
                <motion.div
                  className="absolute inset-0 bg-white/20"
                  initial={{ x: "-100%" }}
                  whileHover={{ x: "100%" }}
                  transition={{ duration: 0.5 }}
                />
              )}
              {isStarting ? (
                <Loader2 className="ml-2 h-4 w-4 animate-spin" />
              ) : (
                <ArrowRight className="ml-2 h-4 w-4 relative z-10 group-hover:translate-x-1 transition-transform" />
              )}
            </Button>
          </motion.div>
        </div>
      </section>
    </div>
  )
}

const features = [
  {
    title: 'Smart Analysis',
    description: 'Advanced AI algorithms analyze your documents for key insights and patterns',
    icon: Brain,
    bgColor: 'bg-primary/10',
    iconColor: 'text-primary',
    highlights: [
      'Natural Language Processing',
      'Pattern Recognition',
      'Semantic Analysis'
    ]
  },
  {
    title: 'Multiple Formats',
    description: 'Support for PDF, DOCX, TXT and more file formats',
    icon: FileText,
    bgColor: 'bg-violet-500/10',
    iconColor: 'text-violet-500',
    highlights: [
      'PDF Documents',
      'Word Documents',
      'Text Files'
    ]
  },
  {
    title: 'Instant Results',
    description: 'Get analysis results in seconds with our optimized processing',
    icon: Zap,
    bgColor: 'bg-amber-500/10',
    iconColor: 'text-amber-500',
    highlights: [
      'Real-time Processing',
      'Quick Summaries',
      'Instant Insights'
    ]
  }
]

const steps = [
  {
    title: 'Upload Documents',
    description: 'Simply drag and drop your documents or click to upload'
  },
  {
    title: 'AI Analysis',
    description: 'Our AI processes and analyzes your documents instantly'
  },
  {
    title: 'Get Insights',
    description: 'View detailed analysis and insights from your documents'
  }
] 