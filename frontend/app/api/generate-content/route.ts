import { type NextRequest, NextResponse } from "next/server"

// Mock function to generate content
function generateMockContent(mode: string, numItems: number, topic?: string) {
  const content = []
  for (let i = 0; i < numItems; i++) {
    if (mode === "quiz") {
      content.push({
        id: i + 1,
        question: `Sample question ${i + 1}${topic ? ` about ${topic}` : ""}?`,
        options: ["Option A", "Option B", "Option C", "Option D"],
        correctAnswer: Math.floor(Math.random() * 4),
      })
    } else {
      content.push({
        id: i + 1,
        front: `Front of flashcard ${i + 1}${topic ? ` about ${topic}` : ""}`,
        back: `Back of flashcard ${i + 1}${topic ? ` about ${topic}` : ""}`,
      })
    }
  }
  return content
}

export async function POST(request: NextRequest) {
  const formData = await request.formData()
  const file = formData.get("file") as File
  const mode = formData.get("mode") as string
  const numItems = Number.parseInt(formData.get("numItems") as string, 10)
  const topic = formData.get("topic") as string | undefined

  // Here you would typically process the PDF file
  // For this example, we'll just use the filename
  console.log(`Processing file: ${file.name}`)

  // Generate mock content
  const content = generateMockContent(mode, numItems, topic)

  return NextResponse.json({
    type: mode,
    content: content,
  })
}

