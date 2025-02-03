import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { CheckCircle, Clock } from "lucide-react"

interface PerformanceAnalysisProps {
  score: number
  totalQuestions: number
  timeSpent: number
  timePerQuestion: number[]
}

export function PerformanceAnalysis({ score, totalQuestions, timeSpent, timePerQuestion }: PerformanceAnalysisProps) {
  const percentageScore = (score / totalQuestions) * 100
  const averageTimePerQuestion = timeSpent / totalQuestions

  return (
    <Card className="w-full mt-6">
      <CardHeader>
        <CardTitle>Performance Analysis</CardTitle>
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
          <div className="mt-2">
            <h4 className="text-md font-medium">Time per question:</h4>
            <ul className="list-disc list-inside">
              {timePerQuestion.map((time, index) => (
                <li key={index} className="text-sm text-muted-foreground">
                  Question {index + 1}: {time.toFixed(1)} seconds
                </li>
              ))}
            </ul>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

