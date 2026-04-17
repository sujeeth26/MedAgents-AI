import { Plus, Search, MoreVertical, ChevronLeft, ChevronRight, Check, X, MessageSquare, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useState } from "react";
import { Message } from "./ChatMessage";
import { motion, AnimatePresence } from "framer-motion";

export interface Conversation {
    id: string;
    title: string;
    preview: string;
    timestamp: Date;
    messages: Message[];
}

interface ChatSidebarProps {
    conversations: Conversation[];
    selectedChat: string;
    onSelectChat: (id: string) => void;
    onNewChat: () => void;
    onDeleteChat: (id: string) => void;
    onRenameChat: (id: string, newTitle: string) => void;
    userRole: "patient" | "clinician";
    onUserRoleChange: (role: "patient" | "clinician") => void;
}

export const ChatSidebar = ({
    conversations,
    selectedChat,
    onSelectChat,
    onNewChat,
    onDeleteChat,
    onRenameChat,
    userRole,
    onUserRoleChange,
}: ChatSidebarProps) => {
    const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
    const [searchQuery, setSearchQuery] = useState("");
    const [editingConversationId, setEditingConversationId] = useState<string | null>(null);
    const [editingTitle, setEditingTitle] = useState<string>("");

    const filteredConversations = conversations.filter((conv) =>
        conv.title.toLowerCase().includes(searchQuery.toLowerCase())
    );

    const handleRenameStart = (conv: Conversation) => {
        setEditingConversationId(conv.id);
        setEditingTitle(conv.title);
    };

    const handleRenameSave = () => {
        if (editingConversationId && editingTitle.trim()) {
            onRenameChat(editingConversationId, editingTitle.trim());
            setEditingConversationId(null);
            setEditingTitle("");
        }
    };

    const handleRenameCancel = () => {
        setEditingConversationId(null);
        setEditingTitle("");
    };

    return (
        <div className="relative h-full z-20 flex">
            <motion.div
                initial={{ width: 320 }}
                animate={{ width: sidebarCollapsed ? 0 : 320 }}
                transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
                className="h-full border-r border-white/40 bg-white/60 backdrop-blur-2xl flex flex-col shadow-2xl overflow-y-auto"
            >
                <div className="flex flex-col h-full min-w-[320px]">
                    {/* Header Section */}
                    <div className="p-6 pb-4 space-y-6">
                        <motion.div
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                        >
                            <Button
                                onClick={onNewChat}
                                className="w-full bg-gradient-to-r from-blue-600 via-indigo-600 to-violet-600 hover:from-blue-700 hover:to-indigo-700 text-white border-0 shadow-lg shadow-blue-500/25 transition-all duration-300 h-12 rounded-2xl font-semibold tracking-wide relative overflow-hidden group"
                            >
                                <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300" />
                                <Plus className="h-5 w-5 mr-2 relative z-10" />
                                <span className="relative z-10">New Chat</span>
                            </Button>
                        </motion.div>

                        <div className="relative group">
                            <div className="absolute inset-0 bg-gradient-to-r from-blue-100 to-indigo-100 rounded-xl blur opacity-20 group-hover:opacity-40 transition-opacity" />
                            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400 group-focus-within:text-blue-500 transition-colors z-10" />
                            <Input
                                placeholder="Search history..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                className="pl-10 bg-white/50 border-white/60 focus:border-blue-500/30 focus:ring-blue-500/20 text-slate-900 placeholder:text-slate-400 h-10 rounded-xl transition-all duration-300 relative z-10 backdrop-blur-sm"
                            />
                        </div>
                    </div>

                    {/* Conversations List */}
                    <ScrollArea className="flex-1 px-4">
                        <div className="space-y-2 py-2">
                            <div className="text-[10px] font-bold text-slate-400 px-4 py-2 uppercase tracking-widest flex items-center gap-2">
                                <Sparkles className="h-3 w-3" />
                                Recent Chats
                            </div>

                            <AnimatePresence mode="popLayout">
                                {filteredConversations.map((conv, index) => (
                                    <motion.div
                                        key={conv.id}
                                        initial={{ opacity: 0, x: -20, y: 10 }}
                                        animate={{ opacity: 1, x: 0, y: 0 }}
                                        exit={{ opacity: 0, x: -20, scale: 0.95 }}
                                        transition={{ duration: 0.3, delay: index * 0.05 }}
                                        className={`group relative rounded-2xl transition-all duration-300 ${selectedChat === conv.id
                                            ? "bg-gradient-to-r from-blue-50/80 to-indigo-50/80 border border-blue-100/50 shadow-sm backdrop-blur-sm"
                                            : "hover:bg-white/40 border border-transparent hover:border-white/40"
                                            }`}
                                    >
                                        {editingConversationId === conv.id ? (
                                            <div className="p-2 flex items-center gap-1">
                                                <Input
                                                    value={editingTitle}
                                                    onChange={(e) => setEditingTitle(e.target.value)}
                                                    onKeyDown={(e) => {
                                                        if (e.key === "Enter") handleRenameSave();
                                                        if (e.key === "Escape") handleRenameCancel();
                                                    }}
                                                    className="h-9 text-sm bg-white/80 border-blue-200 text-slate-900 rounded-lg"
                                                    autoFocus
                                                />
                                                <Button size="icon" variant="ghost" onClick={handleRenameSave} className="h-9 w-9 text-green-600 hover:bg-green-50 rounded-lg">
                                                    <Check className="h-4 w-4" />
                                                </Button>
                                                <Button size="icon" variant="ghost" onClick={handleRenameCancel} className="h-9 w-9 text-red-600 hover:bg-red-50 rounded-lg">
                                                    <X className="h-4 w-4" />
                                                </Button>
                                            </div>
                                        ) : (
                                            <div
                                                onClick={() => onSelectChat(conv.id)}
                                                className="w-full text-left p-3 flex items-center gap-3 relative rounded-2xl cursor-pointer group/item"
                                            >
                                                {selectedChat === conv.id && (
                                                    <motion.div
                                                        layoutId="activeIndicator"
                                                        className="absolute left-0 top-0 bottom-0 w-1 bg-blue-500 rounded-l-2xl"
                                                    />
                                                )}

                                                <div className={`shrink-0 h-10 w-10 rounded-xl flex items-center justify-center transition-all duration-300 ${selectedChat === conv.id
                                                    ? "bg-gradient-to-br from-blue-500 to-indigo-600 text-white shadow-lg shadow-blue-500/30 scale-105"
                                                    : "bg-white/80 text-slate-400 group-hover:bg-white group-hover:text-blue-500 group-hover:shadow-md group-hover:scale-105"
                                                    }`}>
                                                    <MessageSquare className="h-5 w-5" />
                                                </div>

                                                <div className="flex-1 min-w-0">
                                                    <div className={`font-semibold text-sm truncate transition-colors ${selectedChat === conv.id ? "text-slate-900" : "text-slate-600 group-hover:text-slate-900"
                                                        }`}>
                                                        {conv.title}
                                                    </div>
                                                    <div className="text-xs text-slate-400 truncate group-hover:text-slate-500 font-medium">
                                                        {conv.preview}
                                                    </div>
                                                </div>

                                                <div className="shrink-0" onClick={(e) => e.stopPropagation()}>
                                                    <DropdownMenu>
                                                        <DropdownMenuTrigger asChild>
                                                            <Button
                                                                variant="ghost"
                                                                size="icon"
                                                                className={`h-8 w-8 rounded-lg transition-all duration-200 ${selectedChat === conv.id
                                                                    ? "bg-white text-blue-600 shadow-sm hover:bg-blue-50 ring-1 ring-blue-100"
                                                                    : "bg-slate-100 text-slate-500 hover:bg-white hover:text-blue-600 hover:shadow-sm"
                                                                    }`}
                                                            >
                                                                <MoreVertical className="h-4 w-4" />
                                                            </Button>
                                                        </DropdownMenuTrigger>
                                                        <DropdownMenuContent align="end" className="bg-white/90 backdrop-blur-xl border-white/50 text-slate-700 w-48 shadow-xl rounded-xl p-1 z-50">
                                                            <DropdownMenuItem onClick={(e) => { e.stopPropagation(); handleRenameStart(conv); }} className="hover:bg-blue-50 rounded-lg cursor-pointer font-medium">
                                                                Rename
                                                            </DropdownMenuItem>
                                                            <DropdownMenuItem onClick={(e) => { e.stopPropagation(); onDeleteChat(conv.id); }} className="text-red-600 hover:bg-red-50 rounded-lg cursor-pointer font-medium">
                                                                Delete
                                                            </DropdownMenuItem>
                                                        </DropdownMenuContent>
                                                    </DropdownMenu>
                                                </div>
                                            </div>
                                        )}
                                    </motion.div>
                                ))}
                            </AnimatePresence>

                            {filteredConversations.length === 0 && (
                                <div className="text-center text-slate-400 text-sm py-12 italic flex flex-col items-center gap-2">
                                    <div className="h-12 w-12 rounded-full bg-slate-100 flex items-center justify-center">
                                        <Search className="h-6 w-6 text-slate-300" />
                                    </div>
                                    No conversations found
                                </div>
                            )}
                        </div>
                    </ScrollArea>

                    {/* Footer */}
                    <div className="p-4 border-t border-white/40 bg-white/30 backdrop-blur-md">
                        <div className="flex items-center justify-between bg-white/50 rounded-xl p-1 border border-white/60">
                            <button
                                onClick={() => onUserRoleChange("patient")}
                                className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg text-sm font-medium transition-all duration-300 ${userRole === "patient"
                                    ? "bg-white text-blue-600 shadow-sm"
                                    : "text-slate-500 hover:text-slate-700 hover:bg-white/50"
                                    }`}
                            >
                                Patient
                            </button>
                            <button
                                onClick={() => onUserRoleChange("clinician")}
                                className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg text-sm font-medium transition-all duration-300 ${userRole === "clinician"
                                    ? "bg-white text-indigo-600 shadow-sm"
                                    : "text-slate-500 hover:text-slate-700 hover:bg-white/50"
                                    }`}
                            >
                                Clinician
                            </button>
                        </div>
                    </div>
                </div>
            </motion.div>

            {/* Floating Sidebar Toggle Button */}
            <motion.div
                className="absolute -right-5 top-1/2 -translate-y-1/2 z-50"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.5 }}
            >
                <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
                    className="bg-white/90 backdrop-blur-md border border-white/50 hover:bg-white text-slate-600 h-10 w-10 rounded-full shadow-lg hover:shadow-xl hover:scale-110 transition-all duration-300 group"
                >
                    {sidebarCollapsed ? (
                        <ChevronRight className="h-5 w-5 text-blue-500 group-hover:translate-x-0.5 transition-transform" />
                    ) : (
                        <ChevronLeft className="h-5 w-5 text-blue-500 group-hover:-translate-x-0.5 transition-transform" />
                    )}
                </Button>
            </motion.div>
        </div>
    );
};
