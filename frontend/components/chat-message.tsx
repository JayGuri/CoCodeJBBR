import { cn } from "@/lib/utils"

interface ChatMessageProps {
  role: "user" | "assistant"
  content: string
}

export function ChatMessage({ role, content }: ChatMessageProps) {
  return (
    <div
      className={cn(
        "mb-4 p-3 rounded-lg max-w-[80%] break-words",
        role === "user" ? "bg-primary text-primary-foreground ml-auto" : "bg-secondary text-secondary-foreground",
      )}
    >
      {content}
    </div>
  )
}