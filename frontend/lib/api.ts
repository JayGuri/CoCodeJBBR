export interface Flashcard {
  topic: string
  question: string
  answer: string
}

export interface UploadResponse {
  session_id: string
  flashcard: Flashcard | null
}

export interface NextFlashcardResponse {
  session_id: string
  flashcard: Flashcard | null
}

export const uploadPDF = async (file: File, numCards: number, specificTopic?: string): Promise<UploadResponse> => {
  const formData = new FormData()
  formData.append("file", file)
  formData.append("num_cards", numCards.toString())
  if (specificTopic) {
    formData.append("specific_topic", specificTopic)
  }

  try {
    const response = await fetch("http://localhost:8000/upload", {
      method: "POST",
      body: formData,
    })

    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || "Failed to upload PDF")
    }

    const data: UploadResponse = await response.json()
    console.log("Upload response:", data) // Log the response for debugging
    return data
  } catch (error) {
    console.error("Error uploading PDF:", error)
    throw error
  }
}

export const getNextFlashcard = async (sessionId: string): Promise<NextFlashcardResponse> => {
  try {
    const response = await fetch("http://localhost:8000/next-flashcard", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        session_id: sessionId,
      }),
    })

    if (response.status === 404) {
      console.log("No more flashcards available")
      return { session_id: sessionId, flashcard: null }
    }

    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || "Failed to fetch next flashcard")
    }

    const data: NextFlashcardResponse = await response.json()
    console.log("Next flashcard response:", data) // Log the response for debugging
    return data
  } catch (error) {
    console.error("Error fetching next flashcard:", error)
    throw error
  }
}

export const getAllFlashcards = async (sessionId: string): Promise<Flashcard[]> => {
  const flashcards: Flashcard[] = []
  let hasMore = true
  let retryCount = 0
  const maxRetries = 5
  const retryDelay = 1000 // 1 second

  console.log("Starting to fetch all flashcards for session:", sessionId)

  while (hasMore && retryCount < maxRetries) {
    try {
      const response = await getNextFlashcard(sessionId)
      if (response.flashcard) {
        flashcards.push(response.flashcard)
        console.log(`Fetched flashcard ${flashcards.length}:`, response.flashcard)
        retryCount = 0 // Reset retry count on successful fetch
      } else {
        console.log("No more flashcards available")
        hasMore = false
      }
    } catch (error) {
      console.error(`Error fetching flashcard (attempt ${retryCount + 1}):`, error)
      retryCount++
      if (retryCount < maxRetries) {
        console.log(`Retrying in ${retryDelay}ms...`)
        await new Promise((resolve) => setTimeout(resolve, retryDelay))
      } else {
        console.error("Max retries reached. Stopping flashcard fetch.")
        hasMore = false
      }
    }
  }

  console.log(`Finished fetching flashcards. Total: ${flashcards.length}`)
  return flashcards
}

