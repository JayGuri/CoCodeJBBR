import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { CheckCircle, Clock, BarChart } from "lucide-react"

interface PerformanceAnalysisProps {
  score: number
  totalQuestions: number
  timeSpent: number
  timePerQuestion: number[]
}

export function PerformanceAnalysis({ score, totalQuestions, timeSpent, timePerQuestion }: PerformanceAnalysisProps) {
  const percentageScore = (score / totalQuestions) * 100
  const averageTimePerQuestion = timeSpent / totalQuestions

  // Find the fastest and slowest questions
  const fastestQuestionIndex = timePerQuestion.indexOf(Math.min(...timePerQuestion))
  const slowestQuestionIndex = timePerQuestion.indexOf(Math.max(...timePerQuestion))

  return (
    <Card className="w-full mt-6">
      <CardHeader>
        <CardTitle className="flex items-center">
          <BarChart className="mr-2 h-5 w-5 text-primary" />
          Performance Analysis
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <h3 className="text-lg font-semibold flex items-center">
            <CheckCircle className="mr-2 h-5 w-5 text-green-500" />
            Score
          </h3>
          <Progress value={percentageScore} className="w-full mt-2" />
          <p className="text-sm text-muted-foreground mt-1">
            You scored {score} out of {totalQuestions} ({percentageScore.toFixed(1)}%)
          </p>
          {percentageScore >= 70 ? (
            <p className="text-sm text-green-500 mt-1">
              Great job! You have a good understanding of data science concepts.
            </p>
          ) : percentageScore >= 50 ? (
            <p className="text-sm text-yellow-500 mt-1">
              Good effort! Review the topics you missed to improve your knowledge.
            </p>
          ) : (
            <p className="text-sm text-red-500 mt-1">
              You might need more practice with data science concepts. Focus on the areas you missed.
            </p>
          )}
        </div>
        <div>
          <h3 className="text-lg font-semibold flex items-center">
            <Clock className="mr-2 h-5 w-5 text-blue-500" />
            Time Analysis
          </h3>
          <p className="text-sm text-muted-foreground">Total time: {timeSpent.toFixed(1)} seconds</p>
          <p className="text-sm text-muted-foreground">
            Average time per question: {averageTimePerQuestion.toFixed(1)} seconds
          </p>

          <div className="mt-3">
            <h4 className="text-md font-medium">Time insights:</h4>
            <ul className="list-disc list-inside mt-1">
              <li className="text-sm text-muted-foreground">
                Fastest question: Question {fastestQuestionIndex + 1} (
                {timePerQuestion[fastestQuestionIndex].toFixed(1)} seconds)
              </li>
              <li className="text-sm text-muted-foreground">
                Slowest question: Question {slowestQuestionIndex + 1} (
                {timePerQuestion[slowestQuestionIndex].toFixed(1)} seconds)
              </li>
            </ul>
          </div>

          <div className="mt-3">
            <h4 className="text-md font-medium">Time per question:</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mt-1">
              {timePerQuestion.map((time, index) => (
                <div key={index} className="flex items-center">
                  <Clock className="mr-1 h-4 w-4 text-blue-500" />
                  <span className="text-sm text-muted-foreground">
                    Q{index + 1}: {time.toFixed(1)}s
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

