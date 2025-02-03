"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { PieChart, Pie, Cell, ResponsiveContainer } from "recharts"

// Sample questions about photosynthesis
const questions = [
  {
    id: 1,
    question: "What is the primary function of photosynthesis?",
    options: [
      "Converting light energy to chemical energy",
      "Breaking down glucose",
      "Cellular respiration",
      "Nitrogen fixation",
    ],
    correct: 0,
  },
  {
    id: 2,
    question: "Which organelle is responsible for photosynthesis in plants?",
    options: ["Mitochondria", "Chloroplast", "Nucleus", "Golgi apparatus"],
    correct: 1,
  },
  // Add more questions here...
]

export default function QuizPage() {
  const [currentQuestion, setCurrentQuestion] = useState(0)
  const [answers, setAnswers] = useState<number[]>([])
  const [showResults, setShowResults] = useState(false)
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null)

  const handleAnswer = (answerIndex: number) => {
    setSelectedAnswer(answerIndex)
  }

  const handleNext = () => {
    if (selectedAnswer !== null) {
      setAnswers([...answers, selectedAnswer])
      setSelectedAnswer(null)
      if (currentQuestion < questions.length - 1) {
        setCurrentQuestion(currentQuestion + 1)
      } else {
        setShowResults(true)
      }
    }
  }

  const handlePrevious = () => {
    if (currentQuestion > 0) {
      setCurrentQuestion(currentQuestion - 1)
      setAnswers(answers.slice(0, -1))
      setSelectedAnswer(null)
    }
  }

  const calculateScore = () => {
    let correct = 0
    answers.forEach((answer, index) => {
      if (answer === questions[index].correct) correct++
    })
    return (correct / questions.length) * 100
  }

  if (showResults) {
    const score = calculateScore()
    const data = [
      { name: "Correct", value: score },
      { name: "Incorrect", value: 100 - score },
    ]
    const COLORS = ["var(--success)", "var(--error)"]

    return (
      <div className="container max-w-4xl py-12">
        <Card>
          <CardHeader>
            <CardTitle>Quiz Results</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={data} cx="50%" cy="50%" labelLine={false} outerRadius={80} fill="#8884d8" dataKey="value">
                    {data.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="space-y-4">
              <h3 className="text-2xl font-bold">Your Score: {score.toFixed(1)}%</h3>
              <div className="space-y-2">
                {questions.map((q, index) => (
                  <div
                    key={q.id}
                    className={`p-4 rounded-lg ${
                      answers[index] === q.correct ? "bg-success/10 border-success" : "bg-error/10 border-error"
                    }`}
                  >
                    <p className="font-medium">{q.question}</p>
                    <p className="text-sm text-muted-foreground">Your answer: {q.options[answers[index]]}</p>
                    <p className="text-sm text-muted-foreground">Correct answer: {q.options[q.correct]}</p>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="container max-w-4xl py-12">
      <Card>
        <CardHeader>
          <CardTitle>Photosynthesis Quiz</CardTitle>
          <Progress value={((currentQuestion + 1) / questions.length) * 100} />
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-4">
            <h2 className="text-xl font-medium">
              Question {currentQuestion + 1} of {questions.length}
            </h2>
            <p>{questions[currentQuestion].question}</p>
            <div className="grid gap-2">
              {questions[currentQuestion].options.map((option, index) => (
                <Button
                  key={index}
                  variant={selectedAnswer === index ? "default" : "outline"}
                  className="justify-start"
                  onClick={() => handleAnswer(index)}
                >
                  {option}
                </Button>
              ))}
            </div>
          </div>
          <div className="flex justify-between">
            <Button variant="outline" onClick={handlePrevious} disabled={currentQuestion === 0}>
              Previous
            </Button>
            <Button onClick={handleNext} disabled={selectedAnswer === null}>
              {currentQuestion === questions.length - 1 ? "Finish" : "Next"}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

