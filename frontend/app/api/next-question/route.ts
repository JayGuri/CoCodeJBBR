import { type NextRequest, NextResponse } from "next/server"

// Mock question bank
const questionBank = [
  {
    id: 1,
    question: "What is the capital of France?",
    options: ["London", "Berlin", "Paris", "Madrid"],
    correctAnswer: 2,
  },
  {
    id: 2,
    question: "Which planet is known as the Red Planet?",
    options: ["Venus", "Mars", "Jupiter", "Saturn"],
    correctAnswer: 1,
  },
  {
    id: 3,
    question: "Who painted the Mona Lisa?",
    options: ["Vincent van Gogh", "Pablo Picasso", "Leonardo da Vinci", "Michelangelo"],
    correctAnswer: 2,
  },
  // Add more questions as needed
]

export async function POST(request: NextRequest) {
  const { previousAnswer } = await request.json()

  // In a real application, you would use the previousAnswer to determine the next question
  // For this example, we'll just return a random question from the bank

  const randomIndex = Math.floor(Math.random() * questionBank.length)
  const nextQuestion = questionBank[randomIndex]

  return NextResponse.json(nextQuestion)
}

