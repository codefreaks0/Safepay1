import React, { useState, useEffect, useRef } from 'react';
import { useLocation } from 'wouter';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Loader2, Send, AlertTriangle, User, Bot } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import MainLayout from '@/layouts/main-layout';
import * as GroqModule from 'groq-sdk';
import ReactMarkdown from 'react-markdown';
import EmojiPicker, { Theme as EmojiTheme } from 'emoji-picker-react';
import remarkGfm from 'remark-gfm';

const Groq = GroqModule.default || GroqModule;

console.log('VITE_GROQ_API_KEY:', import.meta.env.VITE_GROQ_API_KEY);
const groq = new Groq({
  apiKey: (import.meta.env.VITE_GROQ_API_KEY || '') as string,
  dangerouslyAllowBrowser: true,
});

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant' | 'system';
  timestamp: Date;
  isLoading?: boolean;
  fileUrl?: string;
  fileType?: string;
  fileName?: string;
}

interface QuickReply {
  id: string;
  text: string;
}

export default function ChatSupport() {
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      content: 'Hello! I\'m your UPI safety assistant. How can I help you today?',
      role: 'assistant',
      timestamp: new Date()
    }
  ]);
  
  const [input, setInput] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  
  const [quickReplies, setQuickReplies] = useState<QuickReply[]>([
    { id: '2', text: 'I think I sent money to a scammer' },
    { id: '1', text: 'How do I verify if a UPI ID is safe?' },
    { id: '3', text: 'How do I report a UPI scam?' },
    { id: '4', text: 'How can I report a scam?' }
  ]);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  
  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  
  const [showEmojiPicker, setShowEmojiPicker] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const handleSubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    
    if (!input.trim()) return;
    
    const userMessageContent = input.trim();
    const userMessageId = Date.now().toString();
    
    setInput('');
    
    setMessages(prev => [
      ...prev,
      {
        id: userMessageId,
        content: userMessageContent,
        role: 'user',
        timestamp: new Date()
      }
    ]);
    
    const loadingMessageId = `loading-${Date.now()}`;
    setMessages(prev => [
      ...prev,
      {
        id: loadingMessageId,
        content: '',
        role: 'assistant',
        timestamp: new Date(),
        isLoading: true
      }
    ]);
    
    setIsSubmitting(true);
    
    try {
      const groqMessages = messages.map(msg => ({
        role: msg.role,
        content: msg.content,
      })).concat({ role: 'user', content: userMessageContent });

      const chatCompletion = await groq.chat.completions.create({
        messages: groqMessages,
        model: "llama3-8b-8192",
        temperature: 0.7,
        max_tokens: 1024,
        top_p: 1,
        stream: false,
        stop: null,
      });

      const assistantResponse = chatCompletion.choices[0]?.message?.content || "I'm sorry, I couldn't generate a response.";
      
      setMessages(prev => 
        prev.filter(msg => msg.id !== loadingMessageId).concat({
          id: Date.now().toString(),
          content: assistantResponse,
          role: 'assistant',
          timestamp: new Date()
        })
      );
      
      setQuickReplies([
        { id: '2', text: 'I think I sent money to a scammer' },
        { id: '1', text: 'How do I verify if a UPI ID is safe?' },
        { id: '3', text: 'How do I report a UPI scam?' },
        { id: '4', text: 'How can I report a scam?' }
      ]);

    } catch (error: any) {
      console.error('Error getting chat response from Groq:', error);
      
      setMessages(prev => prev.filter(msg => msg.id !== loadingMessageId));
      
      setMessages(prev => [
        ...prev,
        {
          id: Date.now().toString(),
          content: "I'm sorry, I'm having trouble connecting right now. Please try again in a moment.",
          role: 'assistant',
          timestamp: new Date()
        }
      ]);
      
      toast({
        title: 'Error',
        description: 'Failed to get response from AI assistant',
        variant: 'destructive'
      });
    } finally {
      setIsSubmitting(false);
    }
  };
  
  const handleQuickReply = (text: string) => {
    setInput(text);
    setTimeout(() => handleSubmit(), 100);
  };

  // Handle emoji selection
  const handleEmojiClick = (emojiData: any) => {
    setInput((prev) => prev + emojiData.emoji);
    setShowEmojiPicker(false);
  };

  // Handle file upload
  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const fileUrl = URL.createObjectURL(file);
    setMessages(prev => [
      ...prev,
      {
        id: Date.now().toString(),
        content: '',
        role: 'user',
        timestamp: new Date(),
        fileUrl,
        fileType: file.type,
        fileName: file.name,
      }
    ]);
    // Optionally, send file to backend here
    // Reset file input
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <MainLayout>
      <div className="flex flex-col flex-1 bg-gray-50 relative min-h-screen"> {/* Main chat container, now relative and min-h-screen for mobile */}
        {/* Header */}
        <div className="bg-white border-b p-4 flex items-center justify-between">
          <h1 className="text-lg font-bold flex items-center">
            <Bot className="h-6 w-6 mr-2 text-primary" />
            AI Safety Assistant
          </h1>
          <Button variant="ghost" onClick={() => setLocation('/home')}>Close</Button>
        </div>

        {/* Chat Messages Container - scrolls independently */}
        <div className="flex-1 p-4 overflow-y-auto space-y-4 pb-[200px]"> {/* Increased padding-bottom for both quick replies and input */}
          {messages.map((message) => (
            <div
              key={message.id}
              className="flex w-full"
            >
              <div className={`flex items-start w-full ${message.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center
                  ${message.role === 'user' ? 'bg-primary text-white ml-2' : 'bg-gray-200 text-gray-700 mr-2'}`}>
                  {message.role === 'user' ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
                </div>
                {/* Improved message bubble for long/overflowing messages, mobile friendly, with file support */}
                <div
                  className={`w-full max-w-[90%] p-3 rounded-lg shadow-sm max-h-[250px] overflow-y-auto break-words whitespace-pre-line
                    ${
                      message.role === 'user'
                        ? 'bg-primary text-white rounded-br-none ml-auto'
                        : message.isLoading
                          ? 'bg-gray-200 text-gray-700 animate-pulse mr-auto'
                          : 'bg-gray-200 text-gray-800 rounded-bl-none mr-auto'
                    }`
                  }
                  style={{
                    wordBreak: 'break-word',
                    lineHeight: 1.6,
                    fontSize: '1rem',
                  }}
                >
                  {message.isLoading ? (
                    <Loader2 className="h-5 w-5 animate-spin" />
                  ) : message.fileUrl ? (
                    message.fileType?.startsWith('image') ? (
                      <img src={message.fileUrl} alt={message.fileName} className="max-w-full max-h-40 rounded mb-2" />
                    ) : (
                      <a href={message.fileUrl} download={message.fileName} className="underline text-blue-600" target="_blank" rel="noopener noreferrer">
                        {message.fileName || 'Download file'}
                      </a>
                    )
                  ) : message.role === 'assistant' ? (
                    <div className="prose prose-sm max-w-none">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
                    </div>
                  ) : (
                    <p className="text-sm whitespace-pre-line">{message.content}</p>
                  )}
                  <p className="text-xs text-right mt-1 opacity-75">
                    {message.timestamp.toLocaleTimeString()}
                  </p>
                </div>
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Quick Replies - fixed just above the input at the bottom */}
        <div className="p-4 border-t bg-white flex flex-wrap gap-2 justify-center fixed bottom-[64px] left-0 w-full z-20">
          {quickReplies.map((reply) => (
            <Button
              key={reply.id}
              variant="outline"
              size="sm"
              onClick={() => handleQuickReply(reply.text)}
              className="rounded-full"
            >
              {reply.text}
            </Button>
          ))}
        </div>

        {/* Message Input - fixed at the very bottom */}
        <form
          onSubmit={handleSubmit}
          className="p-4 border-t bg-white flex flex-col gap-2 fixed bottom-0 left-0 w-full z-30 mb-[22px] shadow-lg"
        >
          <div className="flex items-end gap-2 relative">
            {/* Emoji Picker Button */}
            <Button
              type="button"
              variant="ghost"
              className="rounded-full p-2 h-auto w-auto"
              onClick={() => setShowEmojiPicker((v) => !v)}
              tabIndex={-1}
            >
              <span role="img" aria-label="emoji">😊</span>
            </Button>
            {/* Emoji Picker Popup */}
            {showEmojiPicker && (
              <div className="absolute bottom-12 left-0 z-50">
                <EmojiPicker onEmojiClick={handleEmojiClick} height={350} width={300} theme={EmojiTheme.LIGHT} />
              </div>
            )}
            {/* File Upload Button */}
            <Button
              type="button"
              variant="ghost"
              className="rounded-full p-2 h-auto w-auto"
              onClick={() => fileInputRef.current?.click()}
              tabIndex={-1}
            >
              <span role="img" aria-label="attach file">📎</span>
            </Button>
            <input
              type="file"
              ref={fileInputRef}
              className="hidden"
              onChange={handleFileChange}
              tabIndex={-1}
            />
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder="Type your message..."
              className="flex-1 rounded-xl px-4 py-2 border border-gray-300 focus:ring-primary focus:border-primary resize-none min-h-[44px] max-h-[120px] shadow-sm text-sm"
              rows={1}
              disabled={isSubmitting}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit();
                }
              }}
            />
            <Button type="submit" className="rounded-full p-2 h-auto w-auto" disabled={isSubmitting || !input.trim()}>
              {isSubmitting ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : (
                <Send className="h-5 w-5" />
              )}
            </Button>
          </div>
          <div className="text-xs text-gray-400 text-right pr-2">{input.length}/500</div>
        </form>
      </div>
    </MainLayout>
  );
}