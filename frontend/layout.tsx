import { ThemeProvider } from "./components/theme-provider"
import { ColorModeToggle } from "./components/color-mode-toggle"
import type React from "react"

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <head>
        <title>QuizEasy - Accessible Learning Platform</title>
        <meta
          name="description"
          content="Transform the way you acquire knowledge with our accessible quizzing platform"
        />
      </head>
      <body>
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
          <div className="relative flex min-h-screen flex-col">
            <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
              <div className="container flex h-14 items-center">
                <div className="mr-4 flex">
                  <a className="mr-6 flex items-center space-x-2" href="/">
                    <span className="font-bold">QuizEasy</span>
                  </a>
                </div>
                <nav className="flex items-center space-x-6 text-sm font-medium">
                  <a href="/">Home</a>
                  <a href="#about">About</a>
                  <a href="#quizzes">Quizzes</a>
                  <a href="#contact">Contact</a>
                </nav>
                <div className="flex flex-1 items-center justify-end space-x-4">
                  <ColorModeToggle />
                  <nav className="flex items-center space-x-2">
                    <a
                      href="/get-started"
                      className="bg-primary text-primary-foreground hover:bg-primary/90 inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 h-9 px-4 py-2"
                    >
                      Get Started
                    </a>
                  </nav>
                </div>
              </div>
            </header>
            <main className="flex-1">{children}</main>
            <footer className="border-t py-6 md:py-0">
              <div className="container flex flex-col items-center justify-between gap-4 md:h-24 md:flex-row">
                <p className="text-center text-sm leading-loose text-muted-foreground md:text-left">
                  Built with accessibility in mind. Contact us at support@quizeasy.com
                </p>
              </div>
            </footer>
          </div>
        </ThemeProvider>
      </body>
    </html>
  )
}

