import { Facebook, Twitter, Linkedin, Github } from 'lucide-react'

export default function Footer() {
  return (
    <footer className="border-t border-border/40 bg-background">
      <div className="mx-auto max-w-7xl px-6 py-12 md:flex md:items-center md:justify-between lg:px-8">
        <div className="flex justify-center space-x-6 md:order-2">
          <a href="#" className="text-muted-foreground/60 hover:text-foreground transition-colors">
            <span className="sr-only">Facebook</span>
            <Facebook className="h-5 w-5" />
          </a>
          <a href="#" className="text-muted-foreground/60 hover:text-foreground transition-colors">
            <span className="sr-only">Twitter</span>
            <Twitter className="h-5 w-5" />
          </a>
          <a href="#" className="text-muted-foreground/60 hover:text-foreground transition-colors">
            <span className="sr-only">LinkedIn</span>
            <Linkedin className="h-5 w-5" />
          </a>
          <a href="#" className="text-muted-foreground/60 hover:text-foreground transition-colors">
            <span className="sr-only">GitHub</span>
            <Github className="h-5 w-5" />
          </a>
        </div>
        <div className="mt-8 md:order-1 md:mt-0">
          <p className="text-center text-sm leading-5 text-muted-foreground">
            &copy; {new Date().getFullYear()} FileWhiz. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  )
}

