"use client"

import { useState, useRef, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "../components/ui/input"
import { Button } from "@/components/ui/button"
import { Send, Loader2 } from "lucide-react"
import { askQuestion } from "../lib/api"

interface ChatMessage {
  role: "user" | "assistant"
  content: string
}

interface ChatbotProps {
  sessionId: string
}

export function Chatbot({ sessionId }: ChatbotProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [previousConversations, setPreviousConversations] = useState<string[]>([])
  const [chatHistory, setChatHistory] = useState<string[]>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Update the handleSendMessage function with better error handling
  const handleSendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: ChatMessage = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      // Update chat history for context
      const updatedChatHistory = [...chatHistory, `User: ${input}`];
      setChatHistory(updatedChatHistory);

      // Call the direct API endpoint for chat
      const response = await askQuestion(input, previousConversations, updatedChatHistory);
      
      const assistantMessage: ChatMessage = {
        role: "assistant",
        content: response.answer,
      };
      setMessages((prev) => [...prev, assistantMessage]);
      
      // Update previous conversations with this Q&A pair
      setPreviousConversations((prev) => [...prev, `Q: ${input}\nA: ${response.answer}`]);
      setChatHistory((prev) => [...prev, `Assistant: ${response.answer}`]);
      
    } catch (error) {
      console.error("Error in chat:", error);
      
      // Check for specific error messages from the API
      let errorMessage = "I'm sorry, but I encountered an error. Please try again later.";
      
      if (error instanceof Error) {
        const errorText = error.message || '';
        
        if (errorText.includes("No documents have been processed")) {
          errorMessage = "No documents have been processed. Please upload a PDF first.";
        } else if (errorText.includes("Failed to connect")) {
          errorMessage = "Could not connect to the AI service. Please check your connection.";
        }
      }
      
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: errorMessage },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card className="mt-6">
      <CardHeader>
        <CardTitle className="text-2xl font-bold">Chat with AI Assistant</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="h-[300px] overflow-y-auto pr-4 space-y-4">
          {messages.length === 0 && (
            <div className="text-center text-muted-foreground">
              Ask questions about the content in your PDF...
            </div>
          )}
          {messages.map((message, index) => (
            <div 
              key={index} 
              className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div 
                className={`max-w-[80%] p-3 rounded-lg ${
                  message.role === "user" 
                    ? "bg-primary text-primary-foreground" 
                    : "bg-muted"
                }`}
              >
                {message.content}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex items-center justify-center">
              <Loader2 className="h-6 w-6 animate-spin text-primary" />
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        <div className="flex items-center space-x-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            onKeyPress={(e) => e.key === "Enter" && handleSendMessage()}
            className="flex-grow"
          />
          <Button onClick={handleSendMessage} disabled={isLoading} size="icon">
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}