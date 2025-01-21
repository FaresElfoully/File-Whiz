import { ThemeProvider } from "@/components/theme-provider"
import { Routes, Route } from "react-router-dom"
import Header from "@/components/header"
import Footer from "@/components/footer"
import HomePage from "@/pages/home"
import UploadPage from "@/pages/upload"
import ChatPage from "@/pages/chat"

function App() {
  return (
    <ThemeProvider defaultTheme="dark" storageKey="ui-theme">
      <div className="flex flex-col min-h-screen bg-background">
        <Header />
        <main className="flex-grow pt-16">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/chat" element={<ChatPage />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </ThemeProvider>
  )
}

export default App 