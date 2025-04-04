import { Loader2 } from "lucide-react"

export function LoadingSpinner() {
  return (
    <div className="fixed inset-0 bg-background/80 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="text-center">
        <Loader2 className="h-10 w-10 animate-spin text-primary mx-auto" />
        <p className="mt-2 text-sm text-muted-foreground">Processing your PDF...</p>
      </div>
    </div>
  )
}