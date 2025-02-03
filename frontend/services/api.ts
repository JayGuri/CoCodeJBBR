// Define types for our API responses
interface FlashcardData {
  id: number
  front: string
  back: string
}

interface QuizQuestion {
  id: number
  question: string
  options: string[]
  correctAnswer: number
}

interface GenerateContentResponse {
  type: "flashcard" | "quiz"
  content: FlashcardData[] | QuizQuestion[]
}

export async function generateContent(
  file: File,
  mode: "flashcard" | "quiz",
  numItems: number,
  topic?: string,
): Promise<GenerateContentResponse> {
  const formData = new FormData()
  formData.append("file", file)
  formData.append("mode", mode)
  formData.append("numItems", numItems.toString())
  if (topic) formData.append("topic", topic)

  const response = await fetch("/api/generate-content", {
    method: "POST",
    body: formData,
  })

  if (!response.ok) {
    throw new Error("Failed to generate content")
  }

  return response.json()
}

export async function getNextQuizQuestion(previousAnswer?: boolean): Promise<QuizQuestion> {
  const response = await fetch("/api/next-question", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ previousAnswer }),
  })

  if (!response.ok) {
    throw new Error("Failed to get next question")
  }

  return response.json()
}

