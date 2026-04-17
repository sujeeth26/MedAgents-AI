import { useEffect, useRef } from "react";
import { Sparkles, Bot, Activity, Wifi, Battery } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { ChatSidebar } from "@/components/chat/ChatSidebar";
import { ChatMessage } from "@/components/chat/ChatMessage";
import { ChatInput } from "@/components/chat/ChatInput";
import { useChat } from "@/hooks/useChat";
import { Particles } from "@/components/ui/Particles";
import { motion } from "framer-motion";

const Chat = () => {
  const {
    conversations,
    selectedChat,
    setSelectedChat,
    messages,
    isLoading,
    uploadedImages,
    imagePreviewUrls,
    createNewChat,
    sendMessage,
    stopGeneration,
    handleImageSelect,
    removeImage,
    deleteChat,
    renameChat,
    userRole,
    setUserRole,
  } = useChat();

  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const currentConversation = conversations.find((c) => c.id === selectedChat);

  return (
    <div className="flex h-screen w-full overflow-hidden relative bg-slate-50 text-slate-900 font-sans selection:bg-blue-100">
      {/* Premium Animated Background - Light Theme */}
      <div className={`absolute inset-0 bg-[radial-gradient(circle_at_top_right,_var(--tw-gradient-stops))] ${userRole === 'patient' ? 'from-blue-100/40 via-slate-50 to-slate-50' : 'from-teal-100/40 via-slate-50 to-slate-50'} pointer-events-none transition-colors duration-1000`} />
      <div className={`absolute inset-0 bg-[radial-gradient(circle_at_bottom_left,_var(--tw-gradient-stops))] ${userRole === 'patient' ? 'from-indigo-100/40 via-slate-50 to-slate-50' : 'from-slate-200/40 via-slate-50 to-slate-50'} pointer-events-none transition-colors duration-1000`} />

      {/* Particle System */}
      <Particles />

      {/* Sidebar */}
      <ChatSidebar
        conversations={conversations}
        selectedChat={selectedChat}
        onSelectChat={setSelectedChat}
        onNewChat={createNewChat}
        onDeleteChat={deleteChat}
        onRenameChat={renameChat}
        userRole={userRole}
        onUserRoleChange={setUserRole}
      />

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col relative z-10 min-w-0">
        {/* Chat Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: "easeOut" }}
          className="h-20 border-b border-white/40 backdrop-blur-xl bg-white/40 px-8 flex items-center justify-between z-20 shadow-sm"
        >
          <div className="flex items-center gap-4">
            <div className="relative group cursor-pointer">
              <div className={`absolute inset-0 bg-gradient-to-r ${userRole === 'patient' ? 'from-blue-400 to-indigo-400' : 'from-teal-400 to-slate-400'} rounded-2xl blur-lg opacity-30 group-hover:opacity-60 transition-opacity duration-500`} />
              <div className="relative h-11 w-11 bg-white/80 rounded-2xl flex items-center justify-center border border-white/60 shadow-sm backdrop-blur-md transition-transform duration-300 group-hover:scale-105">
                <Activity className={`h-6 w-6 ${userRole === 'patient' ? 'text-blue-600' : 'text-teal-600'}`} />
              </div>
            </div>
            <div>
              <motion.h2
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.2 }}
                className="font-bold text-lg text-slate-800 tracking-tight"
              >
                {currentConversation?.title || "Chat"}
              </motion.h2>
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.3 }}
                className="flex items-center gap-2"
              >
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                </span>
                <p className="text-xs text-slate-500 font-medium">System Operational</p>
              </motion.div>
            </div>
          </div>

          <div className="flex items-center gap-6">
            <div className="flex items-center gap-3 text-slate-400">
              <Wifi className="h-4 w-4" />
              <Battery className="h-4 w-4" />
            </div>
            <div className="h-8 w-[1px] bg-slate-200" />
            <div className="flex items-center gap-3">
              <div className="text-right hidden md:block">
                <p className="text-sm font-bold text-slate-800">{userRole === 'patient' ? 'Patient View' : 'Dr. User'}</p>
                <p className={`text-xs ${userRole === 'patient' ? 'text-blue-600' : 'text-teal-600'} font-medium`}>{userRole === 'patient' ? 'Standard Access' : 'Licensed Clinician'}</p>
              </div>
              <Avatar className="h-10 w-10 ring-2 ring-white shadow-lg cursor-pointer hover:scale-105 transition-transform duration-300">
                <AvatarFallback className={`bg-gradient-to-br ${userRole === 'patient' ? 'from-blue-600 to-indigo-600' : 'from-teal-600 to-slate-600'} text-white font-bold`}>
                  {userRole === 'patient' ? 'PT' : 'DR'}
                </AvatarFallback>
              </Avatar>
            </div>
          </div>
        </motion.div>

        {/* Messages */}
        <ScrollArea className="flex-1 px-4 md:px-8">
          <div className="max-w-5xl mx-auto py-8 space-y-8">
            {messages.map((message) => (
              <ChatMessage
                key={message.id}
                message={message}
                onSuggestionClick={sendMessage}
              />
            ))}

            {/* Loading Indicator */}
            {isLoading && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex gap-6 max-w-4xl mx-auto"
              >
                <div className="shrink-0 flex flex-col items-center gap-2">
                  <Avatar className="h-10 w-10 ring-2 ring-white shadow-lg">
                    <AvatarFallback className="bg-gradient-to-br from-blue-500 via-indigo-500 to-violet-500 text-white">
                      <Bot className="h-5 w-5 animate-pulse" />
                    </AvatarFallback>
                  </Avatar>
                </div>
                <div className="flex-1">
                  <div className="inline-block p-6 rounded-[2rem] rounded-tl-none bg-white/80 border border-white/60 backdrop-blur-xl shadow-lg">
                    <div className="flex items-center gap-3">
                      <span className="text-sm text-blue-600 font-bold tracking-wide uppercase text-[10px]">Processing</span>
                      <div className="flex gap-1.5">
                        <motion.div
                          animate={{ scale: [1, 1.5, 1], opacity: [0.5, 1, 0.5], backgroundColor: ["#3b82f6", "#6366f1", "#3b82f6"] }}
                          transition={{ repeat: Infinity, duration: 1, delay: 0 }}
                          className="w-2 h-2 rounded-full"
                        />
                        <motion.div
                          animate={{ scale: [1, 1.5, 1], opacity: [0.5, 1, 0.5], backgroundColor: ["#3b82f6", "#6366f1", "#3b82f6"] }}
                          transition={{ repeat: Infinity, duration: 1, delay: 0.2 }}
                          className="w-2 h-2 rounded-full"
                        />
                        <motion.div
                          animate={{ scale: [1, 1.5, 1], opacity: [0.5, 1, 0.5], backgroundColor: ["#3b82f6", "#6366f1", "#3b82f6"] }}
                          transition={{ repeat: Infinity, duration: 1, delay: 0.4 }}
                          className="w-2 h-2 rounded-full"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
            <div ref={messagesEndRef} className="h-4" />
          </div>
        </ScrollArea>

        {/* Input Area */}
        <ChatInput
          onSend={sendMessage}
          onStop={stopGeneration}
          isLoading={isLoading}
          onImageSelect={handleImageSelect}
          uploadedImages={uploadedImages}
          onRemoveImage={removeImage}
          imagePreviewUrls={imagePreviewUrls}
        />
      </div>
    </div>
  );
};

export default Chat;
