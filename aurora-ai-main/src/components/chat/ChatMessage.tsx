import { cn } from "@/lib/utils";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Bot, User, FileText, Image as ImageIcon, Sparkles, Brain, Activity, Stethoscope, Search, FileSearch, ShieldCheck, ChevronDown, ChevronUp } from "lucide-react";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  agent?: string;
  thinking?: string;
  suggestions?: string[];
  result_image?: string;
  uploaded_image?: string;
  original_image?: string;
  segmentation_image?: string;
  disease_grounding_image?: string;
  all_images?: Array<{ type: string; url: string; label: string }>;
}

interface ChatMessageProps {
  message: Message;
  onSuggestionClick?: (suggestion: string) => void;
}

// Typewriter effect component
const TypewriterText = ({ text, onComplete }: { text: string; onComplete?: () => void }) => {
  const [displayedText, setDisplayedText] = useState("");
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (currentIndex < text.length) {
      const timeout = setTimeout(() => {
        setDisplayedText((prev) => prev + text[currentIndex]);
        setCurrentIndex((prev) => prev + 1);
      }, 10); // Adjust speed here
      return () => clearTimeout(timeout);
    } else {
      onComplete?.();
    }
  }, [currentIndex, text, onComplete]);

  return <span className="whitespace-pre-wrap">{displayedText}</span>;
};

export const ChatMessage = ({ message, onSuggestionClick }: ChatMessageProps) => {
  const isUser = message.role === "user";
  const [isExpanded, setIsExpanded] = useState(true);
  const [activeImageTab, setActiveImageTab] = useState<string>("original");
  const [isThinkingExpanded, setIsThinkingExpanded] = useState(true);

  // Determine agent icon and color
  const getAgentStyle = (agentName?: string) => {
    const name = agentName?.toLowerCase() || "";
    if (name.includes("brain")) return { icon: Brain, color: "text-purple-500", bg: "bg-purple-50", border: "border-purple-200", gradient: "from-purple-500 to-pink-500" };
    if (name.includes("xray") || name.includes("x-ray")) return { icon: FileText, color: "text-blue-500", bg: "bg-blue-50", border: "border-blue-200", gradient: "from-blue-500 to-cyan-500" };
    if (name.includes("skin")) return { icon: Activity, color: "text-rose-500", bg: "bg-rose-50", border: "border-rose-200", gradient: "from-rose-500 to-orange-500" };
    if (name.includes("search")) return { icon: Search, color: "text-amber-500", bg: "bg-amber-50", border: "border-amber-200", gradient: "from-amber-500 to-yellow-500" };
    if (name.includes("rag")) return { icon: FileSearch, color: "text-emerald-500", bg: "bg-emerald-50", border: "border-emerald-200", gradient: "from-emerald-500 to-teal-500" };
    if (name.includes("guard")) return { icon: ShieldCheck, color: "text-slate-500", bg: "bg-slate-50", border: "border-slate-200", gradient: "from-slate-500 to-gray-500" };
    return { icon: Stethoscope, color: "text-indigo-500", bg: "bg-indigo-50", border: "border-indigo-200", gradient: "from-indigo-500 to-violet-500" };
  };

  const agentStyle = getAgentStyle(message.agent);
  const AgentIcon = agentStyle.icon;

  // Handle image tabs
  const images = message.all_images || [];
  const currentImage = images.find(img => img.type === activeImageTab) || images[0];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className={cn(
        "flex gap-4 w-full max-w-4xl mx-auto group",
        isUser ? "flex-row-reverse" : "flex-row"
      )}
    >
      {/* Avatar */}
      <div className="shrink-0 flex flex-col items-center gap-2">
        <Avatar className={cn(
          "h-10 w-10 ring-2 ring-offset-2 transition-all duration-300 shadow-lg",
          isUser ? "ring-blue-500 ring-offset-blue-50" : `ring-${agentStyle.color.split('-')[1]}-500 ring-offset-${agentStyle.color.split('-')[1]}-50`
        )}>
          {isUser ? (
            <AvatarFallback className="bg-gradient-to-br from-blue-500 to-indigo-600 text-white">
              <User className="h-5 w-5" />
            </AvatarFallback>
          ) : (
            <AvatarFallback className={`bg-gradient-to-br ${agentStyle.gradient} text-white`}>
              <AgentIcon className="h-5 w-5 animate-pulse-slow" />
            </AvatarFallback>
          )}
        </Avatar>
      </div>

      {/* Message Content */}
      <div className={cn(
        "flex flex-col gap-2 max-w-[85%] min-w-[200px]",
        isUser ? "items-end" : "items-start"
      )}>
        {/* Agent Badge */}
        {!isUser && message.agent && (
          <motion.div
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium border ${agentStyle.bg} ${agentStyle.color} ${agentStyle.border} shadow-sm backdrop-blur-sm`}
          >
            <AgentIcon className="h-3 w-3" />
            {message.agent.replace(/_/g, " ")}
          </motion.div>
        )}

        {/* Thinking Process */}
        {!isUser && message.thinking && (
          <div className="w-full mb-2">
            <motion.div
              initial={{ height: "auto", opacity: 1 }}
              animate={{ height: isThinkingExpanded ? "auto" : 40 }}
              className="rounded-2xl border border-indigo-100 bg-white/50 backdrop-blur-md overflow-hidden shadow-sm transition-all duration-300"
            >
              <button
                onClick={() => setIsThinkingExpanded(!isThinkingExpanded)}
                className="w-full flex items-center justify-between px-4 py-2.5 bg-indigo-50/50 hover:bg-indigo-50 transition-colors text-xs font-medium text-indigo-600"
              >
                <div className="flex items-center gap-2">
                  <Sparkles className="h-3.5 w-3.5" />
                  <span>Metagentica Reasoning Process</span>
                </div>
                {isThinkingExpanded ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
              </button>

              <AnimatePresence>
                {isThinkingExpanded && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="p-4 text-sm text-slate-600 font-mono bg-white/40"
                  >
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {message.thinking}
                    </ReactMarkdown>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          </div>
        )}

        {/* Message Bubble */}
        <div
          className={cn(
            "relative px-6 py-4 rounded-[2rem] shadow-md backdrop-blur-md transition-all duration-300 hover:shadow-lg",
            isUser
              ? "bg-gradient-to-br from-blue-600 to-indigo-600 text-white rounded-tr-none border border-blue-500/20"
              : "bg-white/80 text-slate-800 rounded-tl-none border border-white/60"
          )}
        >
          {/* Markdown Content */}
          <div className={cn(
            "prose prose-sm max-w-none break-words leading-relaxed",
            isUser ? "prose-invert" : "prose-slate"
          )}>
            {!isUser ? (
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  // Custom styling for markdown elements
                  h1: ({ node, ...props }) => <h1 className="text-xl font-bold mb-3 bg-gradient-to-r from-blue-600 to-violet-600 bg-clip-text text-transparent" {...props} />,
                  h2: ({ node, ...props }) => <h2 className="text-lg font-semibold mb-2 text-slate-800 border-b border-slate-200 pb-1" {...props} />,
                  h3: ({ node, ...props }) => <h3 className="text-md font-medium mb-1 text-slate-700" {...props} />,
                  ul: ({ node, ...props }) => <ul className="list-disc pl-4 space-y-1 my-2" {...props} />,
                  ol: ({ node, ...props }) => <ol className="list-decimal pl-4 space-y-1 my-2" {...props} />,
                  li: ({ node, ...props }) => <li className="pl-1" {...props} />,
                  blockquote: ({ node, ...props }) => <blockquote className="border-l-4 border-blue-200 pl-4 italic my-2 text-slate-500 bg-slate-50/50 py-2 rounded-r-lg" {...props} />,
                  code: ({ node, ...props }) => <code className="bg-slate-100 text-slate-800 px-1.5 py-0.5 rounded text-xs font-mono border border-slate-200" {...props} />,
                  strong: ({ node, ...props }) => <strong className="font-bold text-slate-900" {...props} />,
                  a: ({ node, ...props }) => (
                    <a
                      {...props}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 hover:text-blue-800 underline decoration-blue-300 hover:decoration-blue-800 transition-all duration-200 font-medium relative group/link inline-flex items-center gap-0.5"
                    >
                      {props.children}
                      <span className="inline-block w-3 h-3 opacity-0 group-hover/link:opacity-100 transition-opacity duration-200">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-full h-full"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg>
                      </span>
                    </a>
                  ),
                }}
              >
                {message.content}
              </ReactMarkdown>
            ) : (
              <div className="whitespace-pre-wrap">{message.content}</div>
            )}
          </div>
        </div>

        {/* Image Display Section */}
        {!isUser && images.length > 0 && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="mt-2 w-full max-w-md rounded-2xl overflow-hidden border border-white/60 bg-white/40 backdrop-blur-xl shadow-lg"
          >
            {/* Image Tabs */}
            {images.length > 1 && (
              <div className="flex p-1 gap-1 bg-slate-100/50 backdrop-blur-sm m-2 rounded-xl">
                {images.map((img) => (
                  <button
                    key={img.type}
                    onClick={() => setActiveImageTab(img.type)}
                    className={cn(
                      "flex-1 px-3 py-1.5 text-xs font-medium rounded-lg transition-all duration-200",
                      activeImageTab === img.type
                        ? "bg-white text-blue-600 shadow-sm"
                        : "text-slate-500 hover:text-slate-700 hover:bg-white/50"
                    )}
                  >
                    {img.label}
                  </button>
                ))}
              </div>
            )}

            {/* Active Image */}
            <div className="relative aspect-square md:aspect-video bg-slate-900/5 group overflow-hidden">
              {currentImage && (
                <motion.img
                  key={currentImage.url}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.3 }}
                  src={currentImage.url}
                  alt={currentImage.label}
                  className="w-full h-full object-contain transition-transform duration-500 group-hover:scale-105"
                />
              )}
              <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-end p-4">
                <p className="text-white text-sm font-medium">{currentImage?.label}</p>
              </div>
            </div>
          </motion.div>
        )}

        {/* Suggestions */}
        {!isUser && message.suggestions && message.suggestions.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-2">
            {message.suggestions.map((suggestion, index) => (
              <motion.button
                key={index}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                onClick={() => onSuggestionClick?.(suggestion)}
                className="px-4 py-2 text-xs font-medium bg-white/60 hover:bg-white border border-white/60 hover:border-blue-200 text-slate-600 hover:text-blue-600 rounded-xl shadow-sm hover:shadow-md transition-all duration-300 backdrop-blur-sm"
              >
                {suggestion}
              </motion.button>
            ))}
          </div>
        )}

        {/* Timestamp */}
        <div className="text-[10px] text-slate-400 px-2 font-medium tracking-wide opacity-0 group-hover:opacity-100 transition-opacity duration-300">
          {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </div>
      </div>
    </motion.div>
  );
};
