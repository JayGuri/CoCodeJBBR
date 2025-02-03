import { Card, CardContent } from "@/components/ui/card"
import { Brain, Lock, Search, Sparkles } from "lucide-react"
import Link from "next/link"

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen">
      {/* Hero Section */}
      <section className="w-full py-12 md:py-24 lg:py-32 xl:py-48">
        <div className="container px-4 md:px-6">
          <div className="flex flex-col items-center space-y-4 text-center">
            <div className="space-y-2">
              <h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl lg:text-6xl/none">
                Discover the Power of Learning
              </h1>
              <p className="mx-auto max-w-[700px] text-muted-foreground md:text-xl">
                Welcome to our Quizzing Platform, where we transform the way you acquire knowledge. Designed with
                accessibility in mind
              </p>
            </div>
            <Link
              href="/quiz"
              className="inline-flex h-10 items-center justify-center rounded-md bg-primary px-8 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              Get Started
            </Link>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="w-full py-12 md:py-24 lg:py-32 bg-muted" id="about">
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
            <Card>
              <CardContent className="flex flex-col items-center space-y-4 p-6">
                <Brain className="h-12 w-12 text-primary" />
                <h3 className="text-xl font-bold">Unlock Your Potential</h3>
                <p className="text-muted-foreground text-center">
                  By providing personalized feedback and adjusting the difficulty based on your performance
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="flex flex-col items-center space-y-4 p-6">
                <Lock className="h-12 w-12 text-primary" />
                <h3 className="text-xl font-bold">Inclusive Design</h3>
                <p className="text-muted-foreground text-center">
                  Our platform is designed with accessibility in mind, ensuring that individuals with diverse needs can
                  fully engage
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="flex flex-col items-center space-y-4 p-6">
                <Search className="h-12 w-12 text-primary" />
                <h3 className="text-xl font-bold">Seamless Exploration</h3>
                <p className="text-muted-foreground text-center">
                  Navigating our platform is a breeze, allowing you to effortlessly access the resources and tools you
                  need
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="flex flex-col items-center space-y-4 p-6">
                <Sparkles className="h-12 w-12 text-primary" />
                <h3 className="text-xl font-bold">Comprehensive Support</h3>
                <p className="text-muted-foreground text-center">
                  Our platform offers a comprehensive suite of features and support
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Contact Section */}
      <section className="w-full py-12 md:py-24 lg:py-32" id="contact">
        <div className="container px-4 md:px-6">
          <div className="flex flex-col items-center justify-center space-y-4 text-center">
            <div className="space-y-2">
              <h2 className="text-3xl font-bold tracking-tighter md:text-4xl/tight">Contact Us</h2>
              <p className="mx-auto max-w-[700px] text-muted-foreground md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed">
                Get in touch with our team for support or inquiries
              </p>
            </div>
            <div className="mx-auto w-full max-w-sm space-y-4">
              <div className="space-y-2 text-left">
                <p>Email: support@quizeasy.com</p>
                <p>Phone: +1 (555) 123-4567</p>
                <p>Hours: Monday - Friday, 9AM - 5PM EST</p>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}

