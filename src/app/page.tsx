import Header from "@/components/header"
import HeroBanner from "@/components/hero-banner"
import FeatureHighlights from "@/components/feature-highlights"
import Footer from "@/components/footer"

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen">
      <Header />
      <main className="flex-grow">
        <HeroBanner />
        <FeatureHighlights />
      </main>
      <Footer />
    </div>
  )
}

