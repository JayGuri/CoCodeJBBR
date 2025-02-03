import Link from "next/link"
import { ThemeProvider } from "../components/theme-provider"
import { ThemeToggle } from "../components/theme-toggle"
import type React from "react"
import "./globals.css"

export const metadata = {
  title: "QuizEasy - Accessible Learning Platform",
  description: "Transform the way you acquire knowledge with our accessible quizzing platform",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body suppressHydrationWarning>
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
          <div className="relative flex min-h-screen flex-col">
            <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
              <div className="container flex h-14 items-center">
                <div className="mr-4 flex">
                  <Link className="mr-6 flex items-center space-x-2" href="/">
                    <span className="font-bold">QuizEasy</span>
                  </Link>
                </div>
                <nav className="flex items-center space-x-6 text-sm font-medium">
                  <Link href="/">Home</Link>
                  <Link href="/#about">About</Link>
                  <Link href="/quiz-options">Quizzes</Link>
                  <Link href="/flashcards">Flashcards</Link>
                  <Link href="/#contact">Contact</Link>
                </nav>
                <div className="flex flex-1 items-center justify-end space-x-4">
                  <ThemeToggle />
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