"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts"
import { PerformanceAnalysis } from "../../components/performance-analysis"
import { useTheme } from "next-themes"
import { CheckCircle, XCircle, Clock } from "lucide-react"

// Data Science questions
const questions = [
  {
    id: 1,
    question: "Which of the following is NOT a common Python library used for data analysis?",
    options: ["Pandas", "NumPy", "Matplotlib", "JavaFX"],
    correct: 3,
  },
  {
    id: 2,
    question: "What does the 'fit' method do in scikit-learn?",
    options: [
      "Evaluates model performance",
      "Trains the model on the data",
      "Makes predictions on new data",
      "Optimizes hyperparameters",
    ],
    correct: 1,
  },
  {
    id: 3,
    question: "Which technique is used to prevent overfitting in machine learning models?",
    options: ["Data augmentation", "Regularization", "Feature engineering", "All of the above"],
    correct: 3,
  },
  {
    id: 4,
    question: "What is the purpose of the 'train_test_split' function in scikit-learn?",
    options: [
      "To split data into training and testing sets",
      "To create cross-validation folds",
      "To balance imbalanced datasets",
      "To normalize feature values",
    ],
    correct: 0,
  },
  {
    id: 5,
    question: "Which of the following is a supervised learning algorithm?",
    options: ["K-means clustering", "Principal Component Analysis (PCA)", "Random Forest", "DBSCAN"],
    correct: 2,
  },
  {
    id: 6,
    question: "What does RMSE stand for in the context of model evaluation?",
    options: [
      "Random Mean Squared Error",
      "Root Mean Squared Error",
      "Relative Mean Squared Estimation",
      "Recursive Model State Evaluation",
    ],
    correct: 1,
  },
  {
    id: 7,
    question: "Which visualization would be most appropriate for showing the distribution of a continuous variable?",
    options: ["Pie chart", "Histogram", "Bar chart", "Scatter plot"],
    correct: 1,
  },
  {
    id: 8,
    question: "What is the main purpose of feature scaling in machine learning?",
    options: [
      "To reduce the number of features",
      "To make features have similar ranges",
      "To increase model complexity",
      "To eliminate categorical variables",
    ],
    correct: 1,
  },
  {
    id: 9,
    question: "Which SQL clause is used to filter groups in a GROUP BY query?",
    options: ["WHERE", "HAVING", "FILTER", "GROUP FILTER"],
    correct: 1,
  },
  {
    id: 10,
    question: "What is the purpose of cross-validation in machine learning?",
    options: [
      "To clean the dataset",
      "To evaluate model performance more reliably",
      "To visualize model predictions",
      "To speed up model training",
    ],
    correct: 1,
  },
]

export default function QuizPage() {
  const [currentQuestion, setCurrentQuestion] = useState(0)
  const [answers, setAnswers] = useState<number[]>([])
  const [showResults, setShowResults] = useState(false)
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null)
  const [startTime, setStartTime] = useState<number>(Date.now())
  const [endTime, setEndTime] = useState<number | null>(null)
  const [timePerQuestion, setTimePerQuestion] = useState<number[]>([])
  const { theme } = useTheme()

  useEffect(() => {
    setStartTime(Date.now())
  }, [currentQuestion])

  const handleAnswer = (answerIndex: number) => {
    setSelectedAnswer(answerIndex)
  }

  const handleNext = () => {
    if (selectedAnswer !== null) {
      const currentTime = Date.now()
      const timeSpent = (currentTime - startTime) / 1000
      setTimePerQuestion([...timePerQuestion, timeSpent])

      setAnswers([...answers, selectedAnswer])
      setSelectedAnswer(null)
      if (currentQuestion < questions.length - 1) {
        setCurrentQuestion(currentQuestion + 1)
      } else {
        setEndTime(currentTime)
        setShowResults(true)
      }
    }
  }

  const handlePrevious = () => {
    if (currentQuestion > 0) {
      setCurrentQuestion(currentQuestion - 1)
      setAnswers(answers.slice(0, -1))
      setTimePerQuestion(timePerQuestion.slice(0, -1))
      setSelectedAnswer(null)
    }
  }

  const calculateScore = () => {
    let correct = 0
    answers.forEach((answer, index) => {
      if (answer === questions[index].correct) correct++
    })
    return correct
  }

  if (showResults) {
    const score = calculateScore()
    const correctAnswers = score
    const incorrectAnswers = questions.length - score
    const data = [
      { name: "Correct", value: correctAnswers },
      { name: "Incorrect", value: incorrectAnswers },
    ]
    const COLORS = ["hsl(var(--primary))", theme === "dark" ? "hsl(var(--background))" : "hsl(var(--foreground))"]
    const totalTimeSpent = timePerQuestion.reduce((a, b) => a + b, 0)

    const CustomTooltip = ({ active, payload }: any) => {
      if (active && payload && payload.length) {
        const data = payload[0].payload
        return (
          <div className="bg-background p-2 border rounded shadow">
            <p className="font-semibold">{`${data.name}: ${data.value}`}</p>
            <p>{`${((data.value / questions.length) * 100).toFixed(1)}%`}</p>
          </div>
        )
      }
      return null
    }

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
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} stroke="hsl(var(--border))" />
                    ))}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="space-y-4">
              <h3 className="text-2xl font-bold">Your Score: {((score / questions.length) * 100).toFixed(1)}%</h3>
              <div className="space-y-2">
                {questions.map((q, index) => (
                  <div
                    key={q.id}
                    className={`p-4 rounded-lg ${
                      answers[index] === q.correct ? "bg-primary/10 border-primary" : "bg-muted/50 border-muted"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center">
                        {answers[index] === q.correct ? (
                          <CheckCircle className="mr-2 h-5 w-5 text-green-500 flex-shrink-0" />
                        ) : (
                          <XCircle className="mr-2 h-5 w-5 text-red-500 flex-shrink-0" />
                        )}
                        <p className="font-medium">{q.question}</p>
                      </div>
                      <div className="flex items-center ml-2 flex-shrink-0">
                        <Clock className="mr-1 h-4 w-4 text-blue-500" />
                        <span className="text-sm text-muted-foreground">{timePerQuestion[index].toFixed(1)}s</span>
                      </div>
                    </div>
                    <p className="text-sm text-muted-foreground mt-2">Your answer: {q.options[answers[index]]}</p>
                    <p className="text-sm text-muted-foreground">Correct answer: {q.options[q.correct]}</p>
                  </div>
                ))}
              </div>
            </div>
            <PerformanceAnalysis
              score={score}
              totalQuestions={questions.length}
              timeSpent={totalTimeSpent}
              timePerQuestion={timePerQuestion}
            />
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="container max-w-4xl py-12">
      <Card>
        <CardHeader>
          <CardTitle>Data Science Quiz</CardTitle>
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

