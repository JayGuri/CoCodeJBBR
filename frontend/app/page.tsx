import { Card, CardContent } from "@/components/ui/card"
import { Brain, Lock, Search, Sparkles, Mail, Phone, Clock } from "lucide-react"
import { Button } from "@/components/ui/button"
import Link from "next/link"

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen">
      {/* Hero Section */}
      <section id="hero" className="w-full py-12 md:py-24 lg:py-32 xl:py-48">
        <div className="container px-4 md:px-6">
          <div className="flex flex-col items-center space-y-4 text-center">
            <p className="subtitle">Explore our Quizzing Platform</p>
            <h1 className="hero-heading">
              Discover the
              <br />
              Power of Learning
            </h1>
            <p className="subtitle max-w-[700px]">
              Welcome to our Quizzing Platform, where we transform the way you acquire knowledge. Designed with
              accessibility in mind
            </p>
            <div className="flex space-x-4">
              <Link
                href="/flashcards"
                className="bg-primary hover:bg-primary/90 inline-flex h-12 items-center justify-center rounded-full px-8 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              >
                Generate Flashcards
              </Link>
              <Link
                href="/quiz-options"
                className="bg-secondary hover:bg-secondary/90 text-secondary-foreground inline-flex h-12 items-center justify-center rounded-full px-8 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              >
                Take a Quiz
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Illustration Section */}
      <section className="w-full py-12 md:py-24 lg:py-32 bg-background">
        <div className="container px-4 md:px-6">
          <div className="flex flex-col items-center justify-center space-y-4 text-center">
            <h2 className="text-3xl font-bold tracking-tighter md:text-4xl/tight">Visualize Your Learning Journey</h2>
            <p className="mx-auto max-w-[700px] text-muted-foreground md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed">
              See how our platform can transform your educational experience
            </p>
          </div>
          <div className="mt-12 flex justify-center">
            <img
              src="/illustration.png"
              alt="Learning Platform Illustration"
              className="w-full max-w-3xl h-auto object-cover rounded-lg shadow-lg"
            />
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="about" className="w-full py-12 md:py-24 lg:py-32 bg-muted/50">
        <div className="container px-4 md:px-6">
          <div className="flex flex-col items-center justify-center space-y-4 text-center">
            <div className="space-y-2">
              <h2 className="text-3xl font-bold tracking-tighter md:text-4xl/tight">Personalized Quizzes</h2>
              <p className="mx-auto max-w-[700px] text-muted-foreground md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed">
                Tailored to your needs, our quizzes adapt to your progress, ensuring you master the content at your own
                pace
              </p>
            </div>
          </div>
          <div className="mx-auto grid max-w-5xl items-center gap-6 py-12 lg:grid-cols-2 lg:gap-12">
            {/* Unlock Your Potential (Green on hover) */}
            <Card className="transform transition-all hover:scale-105 hover:shadow-lg group">
              <CardContent className="flex flex-col items-center space-y-4 p-6 transition-all group-hover:bg-[#34b98a]">
                <div className="rounded-full bg-primary/10 p-3 transition-all group-hover:bg-white/10">
                  <Brain className="h-12 w-12 text-primary group-hover:text-white" />
                </div>
                <h3 className="text-xl font-bold group-hover:text-white">Unlock Your Potential</h3>
                <p className="text-muted-foreground text-center group-hover:text-white">
                  By providing personalized feedback and adjusting the difficulty based on your performance
                </p>
              </CardContent>
            </Card>

            {/* Inclusive Design (Yellow on hover) */}
            <Card className="transform transition-all hover:scale-105 hover:shadow-lg group">
              <CardContent className="flex flex-col items-center space-y-4 p-6 transition-all group-hover:bg-[#FACC15]">
                <div className="rounded-full bg-primary/10 p-3 transition-all group-hover:bg-black/10">
                  <Lock className="h-12 w-12 text-primary group-hover:text-black" />
                </div>
                <h3 className="text-xl font-bold group-hover:text-black">Inclusive Design</h3>
                <p className="text-muted-foreground text-center group-hover:text-black">
                  Our platform is designed with accessibility in mind, ensuring that individuals with diverse needs can
                  fully engage
                </p>
              </CardContent>
            </Card>

            {/* Seamless Exploration (Green on hover) */}
            <Card className="transform transition-all hover:scale-105 hover:shadow-lg group">
              <CardContent className="flex flex-col items-center space-y-4 p-6 transition-all group-hover:bg-[#34b98a]">
                <div className="rounded-full bg-primary/10 p-3 transition-all group-hover:bg-white/10">
                  <Search className="h-12 w-12 text-primary group-hover:text-white" />
                </div>
                <h3 className="text-xl font-bold group-hover:text-white">Seamless Exploration</h3>
                <p className="text-muted-foreground text-center group-hover:text-white">
                  Navigating our platform is a breeze, allowing you to effortlessly access the resources and tools you
                  need
                </p>
              </CardContent>
            </Card>

            {/* Comprehensive Support (Yellow on hover) */}
            <Card className="transform transition-all hover:scale-105 hover:shadow-lg group">
              <CardContent className="flex flex-col items-center space-y-4 p-6 transition-all group-hover:bg-[#FACC15]">
                <div className="rounded-full bg-primary/10 p-3 transition-all group-hover:bg-black/10">
                  <Sparkles className="h-12 w-12 text-primary group-hover:text-black" />
                </div>
                <h3 className="text-xl font-bold group-hover:text-black">Comprehensive Support</h3>
                <p className="text-muted-foreground text-center group-hover:text-black">
                  Our platform offers a comprehensive suite of features and support to enhance your learning experience
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Quizzes Section */}
      <section id="quizzes" className="w-full py-12 md:py-24 lg:py-32">
        <div className="container px-4 md:px-6">
          <div className="flex flex-col items-center justify-center space-y-4 text-center">
            <h2 className="text-3xl font-bold tracking-tighter md:text-4xl/tight">Start Your Learning Journey</h2>
            <p className="mx-auto max-w-[700px] text-muted-foreground md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed">
              Choose from our pre-made quizzes or create your own by uploading a PDF.
            </p>
            <div className="flex space-x-4">
              <Link
                href="/flashcards"
                className="bg-primary hover:bg-primary/90 inline-flex h-12 items-center justify-center rounded-full px-8 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              >
                Explore Flashcards
              </Link>
              <Link
                href="/quiz-options"
                className="bg-secondary hover:bg-secondary/90 text-secondary-foreground inline-flex h-12 items-center justify-center rounded-full px-8 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              >
                Explore Quizzes
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Contact Section */}
      <section id="contact" className="w-full py-12 md:py-24 lg:py-32 bg-gradient-to-b from-background to-muted">
        <div className="container px-4 md:px-6">
          <div className="flex flex-col items-center justify-center space-y-4 text-center">
            <div className="space-y-2">
              <h2 className="text-3xl font-bold tracking-tighter md:text-4xl/tight">Contact Us</h2>
              <p className="mx-auto max-w-[700px] text-muted-foreground md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed">
                Get in touch with our team for support or inquiries
              </p>
            </div>
            <div className="mx-auto w-full max-w-4xl grid gap-8 md:grid-cols-3 mt-8">
              <Card className="transform transition-all hover:scale-105 hover:shadow-lg">
                <CardContent className="flex flex-col items-center space-y-4 p-6">
                  <div className="rounded-full bg-primary/10 p-3">
                    <Mail className="h-6 w-6 text-primary" />
                  </div>
                  <h3 className="text-lg font-semibold">Email Us</h3>
                  <p className="text-muted-foreground">quizzeasyy@gmail.com</p>
                  <Button variant="outline" asChild>
                    <a href="mailto:quizzeasyy@gmail.com">Send Email</a>
                  </Button>
                </CardContent>
              </Card>
              <Card className="transform transition-all hover:scale-105 hover:shadow-lg">
                <CardContent className="flex flex-col items-center space-y-4 p-6">
                  <div className="rounded-full bg-primary/10 p-3">
                    <Phone className="h-6 w-6 text-primary" />
                  </div>
                  <h3 className="text-lg font-semibold">Call Us</h3>
                  <p className="text-muted-foreground">+91 8879854245</p>
                  <Button variant="outline" asChild>
                    <a href="tel:+15551234567">Call Now</a>
                  </Button>
                </CardContent>
              </Card>
              <Card className="transform transition-all hover:scale-105 hover:shadow-lg">
                <CardContent className="flex flex-col items-center space-y-4 p-6">
                  <div className="rounded-full bg-primary/10 p-3">
                    <Clock className="h-6 w-6 text-primary" />
                  </div>
                  <h3 className="text-lg font-semibold">Business Hours</h3>
                  <p className="text-muted-foreground">Monday - Friday</p>
                  <p className="text-muted-foreground">9AM - 5PM IST</p>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}