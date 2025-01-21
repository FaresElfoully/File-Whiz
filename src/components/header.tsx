import { motion } from 'framer-motion'
import { Link, useLocation } from 'react-router-dom'
import { ModeToggle } from './mode-toggle'
import { useTheme } from './theme-provider'

export default function Header() {
  const location = useLocation()
  const { theme } = useTheme()

  return (
    <header className="fixed top-0 left-0 right-0 z-50">
      <div className="glass border-b border-border/40 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo and Brand */}
            <motion.div 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-center"
            >
              <Link 
                to="/" 
                className="flex items-center space-x-2 hover:opacity-80 transition-opacity"
              >
                <img 
                  src="/public/logo-dark.png"
                  alt="FileWhiz Logo"
                  className="h-8 w-8"
                />
                <span className="font-bold text-lg text-foreground bg-clip-text text-transparent bg-gradient-to-r from-primary to-primary/80">
                  FileWhiz
                </span>
              </Link>
            </motion.div>

            {/* Navigation */}
            <motion.nav 
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="hidden md:flex items-center space-x-6"
            >
              <Link 
                to="/" 
                className={`text-sm font-medium transition-colors hover:text-primary ${
                  location.pathname === '/' ? 'text-primary' : 'text-muted-foreground'
                }`}
              >
                Home
              </Link>
              <Link 
                to="/upload" 
                className={`text-sm font-medium transition-colors hover:text-primary ${
                  location.pathname === '/upload' ? 'text-primary' : 'text-muted-foreground'
                }`}
              >
                Upload
              </Link>
              <Link 
                to="/chat" 
                className={`text-sm font-medium transition-colors hover:text-primary ${
                  location.pathname === '/chat' ? 'text-primary' : 'text-muted-foreground'
                }`}
              >
                Chat
              </Link>
            </motion.nav>

            {/* Theme Toggle */}
            <motion.div 
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
            >
              <ModeToggle />
            </motion.div>
          </div>
        </div>
      </div>
    </header>
  )
}

