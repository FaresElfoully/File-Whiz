import { motion } from 'framer-motion'
import { Card } from "@/components/ui/card"
import { FileType, Zap, Globe, ArrowUpRight } from 'lucide-react'

const features = [
  {
    title: "Multi-Format Support",
    description: "Process various document types including PDF, DOCX, and PPTX with ease.",
    icon: FileType,
    gradient: "from-[hsl(var(--violet-gradient-1))] via-[hsl(var(--violet-gradient-2))] to-[hsl(var(--violet-gradient-3))]",
    iconColor: "text-[hsl(var(--violet-gradient-2))]",
    delay: 0.2
  },
  {
    title: "Real-Time Analysis",
    description: "Get instant insights as you upload your documents, saving you valuable time.",
    icon: Zap,
    gradient: "from-[hsl(var(--violet-gradient-1))] via-[hsl(var(--violet-gradient-2))] to-[hsl(var(--violet-gradient-3))]",
    iconColor: "text-[hsl(var(--violet-gradient-2))]",
    delay: 0.4
  },
  {
    title: "Multilingual Capabilities",
    description: "Break language barriers with support for multiple languages, including Arabic.",
    icon: Globe,
    gradient: "from-green-500/10 via-green-500/5 to-transparent",
    iconColor: "text-green-500",
    delay: 0.6
  },
]

export default function FeatureHighlights() {
  return (
    <motion.section 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="py-24"
    >
      <motion.div 
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="text-center mb-16"
      >
        <h2 className="text-3xl font-bold bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-transparent">
          Powerful Features
        </h2>
        <p className="mt-4 text-muted-foreground">
          Everything you need to analyze and understand your documents
        </p>
      </motion.div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        {features.map((feature, index) => (
          <motion.div
            key={feature.title}
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: feature.delay }}
            whileHover={{ y: -5 }}
            className="relative group"
          >
            <div className={`absolute inset-0 rounded-3xl bg-gradient-to-r ${feature.gradient} opacity-0 group-hover:opacity-100 transition-opacity duration-500`} />
            <Card className="relative p-6 backdrop-blur-sm border-primary/10 hover:border-primary/20 transition-colors duration-500">
              <div className="relative z-10 flex flex-col items-center text-center">
                <div className={`p-3 rounded-2xl ${feature.gradient} mb-4 group-hover:scale-110 transition-transform duration-500`}>
                  <feature.icon className={`w-8 h-8 ${feature.iconColor}`} />
                </div>
                <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                <p className="text-muted-foreground">{feature.description}</p>
              </div>
            </Card>
          </motion.div>
        ))}
      </div>
    </motion.section>
  )
}

