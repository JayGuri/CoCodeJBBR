"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Input } from "../../components/ui/input"
import { Label } from "../../components/ui/label"
import { RadioGroup, RadioGroupItem } from "../../components/ui/radio-group"
import { LoadingSpinner } from "../../components/loading-spinner"
import { generateContent } from "../../services/api"

export default function QuizOptionsPage() {
  const [file, setFile] = useState<File | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [numItems, setNumItems] = useState(10)
  const [topic, setTopic] = useState("")
  const [mode, setMode] = useState<"flashcard" | "quiz">("quiz")
  const router = useRouter()

  // Handle file selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
    }
  }

  // Handle form submission for custom content generation
  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (file) {
      setIsLoading(true)
      try {
        const data = await generateContent(file, mode, numItems, topic)
        // Store the received data in localStorage
        localStorage.setItem("generatedContent", JSON.stringify(data))
        localStorage.setItem("contentMode", "custom") // Set mode to custom
        router.push(mode === "quiz" ? "/quiz" : "/flashcards")
      } catch (error) {
        console.error("Error generating content:", error)
        // Handle error (e.g., show error message to user)
      } finally {
        setIsLoading(false)
      }
    }
  }

  // Handle click for demo content
  const handleDemoClick = (demoMode: "quiz" | "flashcard") => {
    localStorage.setItem("contentMode", "demo") // Set mode to demo
    router.push(demoMode === "quiz" ? "/quiz" : "/flashcards")
  }

  return (
    <div className="container mx-auto py-10">
      <h1 className="text-4xl md:text-5xl font-extrabold text-center mb-10 bg-clip-text text-transparent bg-gradient-to-r from-primary to-primary/80">
        Choose Your Learning Experience
      </h1>
      <div className="grid md:grid-cols-2 gap-6">
        {/* Demo Content Card */}
        <Card className="w-full">
          <CardHeader>
            <CardTitle>Try a Demo</CardTitle>
            <CardDescription>Experience our format with a sample on Photosynthesis</CardDescription>
          </CardHeader>
          <CardContent>
            <p>This contains information about the process of photosynthesis in plants.</p>
          </CardContent>
          <CardFooter className="flex gap-4">
            <Button onClick={() => handleDemoClick("quiz")}>Start Demo Quiz</Button>
            <Button variant="outline" onClick={() => handleDemoClick("flashcard")}>
              View Demo Flashcards
            </Button>
          </CardFooter>
        </Card>

        {/* Custom Content Card */}
        <Card className="w-full">
          <CardHeader>
            <CardTitle>Upload Your Own Content</CardTitle>
            <CardDescription>Generate a quiz or flashcards from your PDF</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit}>
              <div className="space-y-4">
                {/* File upload input */}
                <div>
                  <Label htmlFor="pdf-upload">Upload PDF</Label>
                  <Input id="pdf-upload" type="file" accept=".pdf" className="mt-2" onChange={handleFileChange} />
                </div>
                {/* Mode selection (Quiz or Flashcards) */}
                <RadioGroup defaultValue="quiz" onValueChange={(value) => setMode(value as "quiz" | "flashcard")}>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="quiz" id="quiz" />
                    <Label htmlFor="quiz">Quiz</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="flashcard" id="flashcard" />
                    <Label htmlFor="flashcard">Flashcards</Label>
                  </div>
                </RadioGroup>
                {/* Number of items input */}
                <div>
                  <Label htmlFor="num-items">{mode === "quiz" ? "Number of Questions" : "Number of Flashcards"}</Label>
                  <Input
                    id="num-items"
                    type="number"
                    min="1"
                    max="50"
                    value={numItems}
                    onChange={(e) => setNumItems(Number(e.target.value))}
                    className="mt-2"
                  />
                </div>
                {/* Optional topic input */}
                <div>
                  <Label htmlFor="topic">Specific Topic (optional)</Label>
                  <Input
                    id="topic"
                    type="text"
                    value={topic}
                    onChange={(e) => setTopic(e.target.value)}
                    placeholder="Enter a specific topic"
                    className="mt-2"
                  />
                </div>
                {/* Submit button */}
                <Button type="submit" className="w-full" disabled={!file || isLoading}>
                  {isLoading ? "Processing..." : `Generate ${mode === "quiz" ? "Quiz" : "Flashcards"}`}
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>
      </div>
      {isLoading && <LoadingSpinner />}
    </div>
  )
}

