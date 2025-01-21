import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { ArrowRight, FileText, Sparkles } from 'lucide-react'
import { Button } from "@/components/ui/button"

export default function HeroBanner() {
  const navigate = useNavigate()

  return (
    <div className="relative isolate overflow-hidden">
      {/* Gradient Background */}
      <div className="absolute inset-x-0 -top-40 -z-10 transform-gpu overflow-hidden blur-3xl sm:-top-80">
        <div className="relative left-[calc(50%-11rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 rotate-[30deg] bg-gradient-to-tr from-[hsl(var(--violet-gradient-1))] to-[hsl(var(--violet-gradient-3))] opacity-30 sm:left-[calc(50%-30rem)] sm:w-[72.1875rem]" />
      </div>
      
      <div className="mx-auto max-w-7xl px-6 pb-24 pt-16 sm:pt-32 lg:px-8 lg:pt-32">
        <div className="mx-auto max-w-2xl gap-x-14 lg:mx-0 lg:flex lg:max-w-none lg:items-center">
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="w-full max-w-xl lg:shrink-0 xl:max-w-2xl"
          >
            <motion.h1 
              className="text-4xl font-bold tracking-tight sm:text-6xl"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              <span className="violet-gradient-text">Transform Your Documents</span>
              <br />
              <span className="bg-gradient-to-r from-[hsl(var(--violet-gradient-2))] to-[hsl(var(--violet-gradient-3))] bg-clip-text text-transparent">
                with AI-Powered Analysis
              </span>
            </motion.h1>
            <motion.p 
              className="relative mt-6 text-lg leading-8 text-muted-foreground sm:max-w-md lg:max-w-none"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
            >
              Upload any document and get instant insights. Our advanced AI analyzes your files in seconds,
              providing you with valuable information and actionable insights.
            </motion.p>
            <motion.div 
              className="mt-10 flex items-center gap-x-6"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
            >
              <Button
                onClick={() => navigate('/upload')}
                size="lg"
                className="rounded-full px-8 hover:scale-105 transition-all duration-300 violet-gradient-bg text-white shadow-lg group"
              >
                <span>Get Started</span>
                <ArrowRight className="ml-2 h-4 w-4 group-hover:translate-x-1 transition-transform" />
              </Button>
              <Button
                variant="ghost"
                size="lg"
                className="rounded-full group hover:bg-[hsl(var(--violet-gradient-1)/0.1)] transition-all duration-300"
                onClick={() => navigate('/about')}
              >
                Learn More
                <ArrowRight className="ml-2 h-4 w-4 group-hover:translate-x-1 transition-transform opacity-50 group-hover:opacity-100" />
              </Button>
            </motion.div>
          </motion.div>
          <motion.div 
            className="mt-14 flex justify-end gap-8 sm:-mt-44 sm:justify-start sm:pl-20 lg:mt-0 lg:pl-0"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.5 }}
          >
            <div className="ml-auto w-44 flex-none space-y-8 pt-32 sm:ml-0 sm:pt-80 lg:order-last lg:pt-36 xl:order-none xl:pt-80">
              <div className="relative">
                <motion.div 
                  animate={{ 
                    rotate: [0, 5, -5, 0],
                    scale: [1, 1.02, 0.98, 1]
                  }}
                  transition={{ 
                    duration: 5,
                    repeat: Infinity,
                    ease: "easeInOut"
                  }}
                  className="overflow-hidden rounded-2xl bg-gradient-to-br from-[hsl(var(--violet-gradient-1)/0.2)] via-[hsl(var(--violet-gradient-2)/0.2)] to-[hsl(var(--violet-gradient-3)/0.2)] p-8 backdrop-blur-sm border border-[hsl(var(--violet-gradient-1)/0.2)]"
                >
                  <div className="relative">
                    <FileText className="h-16 w-16 text-[hsl(var(--violet-gradient-2))]" />
                    <motion.div
                      animate={{
                        scale: [1, 1.2, 1],
                        opacity: [0.5, 1, 0.5]
                      }}
                      transition={{
                        duration: 2,
                        repeat: Infinity,
                        ease: "easeInOut"
                      }}
                      className="absolute -top-2 -right-2"
                    >
                      <Sparkles className="h-6 w-6 text-[hsl(var(--violet-gradient-1))]" />
                    </motion.div>
                  </div>
                </motion.div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
      
      {/* Bottom Gradient */}
      <div className="absolute inset-x-0 top-[calc(100%-13rem)] -z-10 transform-gpu overflow-hidden blur-3xl sm:top-[calc(100%-30rem)]">
        <div className="relative left-[calc(50%+3rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 bg-gradient-to-tr from-[hsl(var(--violet-gradient-1))] to-[hsl(var(--violet-gradient-3))] opacity-30 sm:left-[calc(50%+36rem)] sm:w-[72.1875rem]" />
      </div>
    </div>
  )
}

