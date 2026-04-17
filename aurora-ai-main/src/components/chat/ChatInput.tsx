import { Send, Image, X, Paperclip, Mic, Square, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface ChatInputProps {
    onSend: (message: string) => void;
    onStop?: () => void;
    isLoading: boolean;
    onImageSelect: (files: FileList) => void;
    uploadedImages: File[];
    onRemoveImage: (index: number) => void;
    imagePreviewUrls: string[];
}

export const ChatInput = ({
    onSend,
    onStop,
    isLoading,
    onImageSelect,
    uploadedImages,
    onRemoveImage,
    imagePreviewUrls,
}: ChatInputProps) => {
    const [input, setInput] = useState("");
    const [isFocused, setIsFocused] = useState(false);
    const [isDragOver, setIsDragOver] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const handleSend = () => {
        if (!input.trim() && uploadedImages.length === 0) return;
        onSend(input);
        setInput("");
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            onImageSelect(e.target.files);
        }
    };

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragOver(true);
    };

    const handleDragLeave = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragOver(false);
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragOver(false);
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            onImageSelect(e.dataTransfer.files);
        }
    };

    return (
        <div className="p-6 pb-8 relative z-20">
            <div className="max-w-4xl mx-auto relative">
                {/* Drag & Drop Overlay */}
                <AnimatePresence>
                    {isDragOver && (
                        <motion.div
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.95 }}
                            className="absolute inset-0 -top-20 bg-blue-500/10 backdrop-blur-sm border-2 border-dashed border-blue-500 rounded-[2.5rem] z-50 flex items-center justify-center"
                        >
                            <div className="text-blue-600 font-semibold flex items-center gap-2">
                                <Image className="h-6 w-6" />
                                Drop images here
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Image Previews */}
                <AnimatePresence>
                    {uploadedImages.length > 0 && (
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: 10 }}
                            className="absolute bottom-full left-0 mb-4 flex gap-3 overflow-x-auto pb-2 w-full px-2"
                        >
                            {imagePreviewUrls.map((url, index) => (
                                <motion.div
                                    key={index}
                                    initial={{ scale: 0.8, opacity: 0 }}
                                    animate={{ scale: 1, opacity: 1 }}
                                    exit={{ scale: 0.8, opacity: 0 }}
                                    className="relative group shrink-0"
                                >
                                    <div className="absolute inset-0 bg-gradient-to-tr from-blue-500/20 to-purple-500/20 rounded-xl blur-sm" />
                                    <img
                                        src={url}
                                        alt={`Upload ${index + 1}`}
                                        className="h-24 w-24 object-cover rounded-xl border border-white/50 shadow-lg relative z-10"
                                    />
                                    <button
                                        onClick={() => onRemoveImage(index)}
                                        className="absolute -top-2 -right-2 z-20 bg-red-500 text-white rounded-full p-1.5 shadow-lg opacity-0 group-hover:opacity-100 transition-all hover:bg-red-600 hover:scale-110"
                                    >
                                        <X className="h-3 w-3" />
                                    </button>
                                </motion.div>
                            ))}
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Input Bar */}
                <motion.div
                    animate={{
                        boxShadow: isFocused ? "0 0 40px rgba(59, 130, 246, 0.15)" : "0 8px 32px rgba(0, 0, 0, 0.05)",
                        scale: isFocused ? 1.01 : 1
                    }}
                    className="relative flex items-end gap-2 bg-white/70 backdrop-blur-2xl border border-white/60 rounded-[2.5rem] p-2 transition-all duration-300"
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                >
                    <input
                        type="file"
                        ref={fileInputRef}
                        className="hidden"
                        accept="image/*"
                        multiple
                        onChange={handleFileChange}
                    />

                    <div className="flex gap-1 pb-1 pl-2">
                        <motion.div whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }}>
                            <Button
                                size="icon"
                                variant="ghost"
                                className="h-11 w-11 rounded-full text-slate-400 hover:text-blue-600 hover:bg-blue-50 transition-all duration-300"
                                onClick={() => fileInputRef.current?.click()}
                                disabled={isLoading}
                            >
                                <Paperclip className="h-5 w-5" />
                            </Button>
                        </motion.div>
                        <motion.div whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }}>
                            <Button
                                size="icon"
                                variant="ghost"
                                className="h-11 w-11 rounded-full text-slate-400 hover:text-blue-600 hover:bg-blue-50 transition-all duration-300"
                                disabled={isLoading}
                            >
                                <Mic className="h-5 w-5" />
                            </Button>
                        </motion.div>
                    </div>

                    <Textarea
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        onFocus={() => setIsFocused(true)}
                        onBlur={() => setIsFocused(false)}
                        placeholder="Ask anything about medical analysis..."
                        className="flex-1 min-h-[52px] max-h-32 bg-transparent border-0 focus-visible:ring-0 resize-none py-3.5 px-2 text-slate-800 placeholder:text-slate-400 text-base font-medium"
                        rows={1}
                    />

                    <div className="pb-1 pr-1">
                        <AnimatePresence mode="wait">
                            {isLoading ? (
                                <motion.div
                                    key="stop"
                                    initial={{ scale: 0, rotate: 180 }}
                                    animate={{ scale: 1, rotate: 0 }}
                                    exit={{ scale: 0, rotate: -180 }}
                                >
                                    <Button
                                        size="icon"
                                        className="h-12 w-12 rounded-full bg-red-500 hover:bg-red-600 shadow-lg shadow-red-500/30 hover:shadow-red-500/50"
                                        onClick={onStop}
                                        title="Stop generating"
                                    >
                                        <Square className="h-5 w-5 fill-current" />
                                    </Button>
                                </motion.div>
                            ) : (
                                <motion.div
                                    key="send"
                                    initial={{ scale: 0, rotate: -180 }}
                                    animate={{ scale: 1, rotate: 0 }}
                                    exit={{ scale: 0, rotate: 180 }}
                                >
                                    <Button
                                        size="icon"
                                        className={`h-12 w-12 rounded-full transition-all duration-500 ${input.trim() || uploadedImages.length > 0
                                                ? "bg-gradient-to-tr from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 shadow-lg shadow-blue-500/30 hover:shadow-blue-500/50"
                                                : "bg-slate-100 text-slate-400"
                                            }`}
                                        onClick={handleSend}
                                        disabled={!input.trim() && uploadedImages.length === 0}
                                    >
                                        <Send className="h-5 w-5" />
                                    </Button>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>
                </motion.div>

                <div className="text-center mt-4 flex items-center justify-center gap-2 text-xs text-slate-400 font-medium tracking-wide">
                    <Sparkles className="h-3 w-3 text-blue-400" />
                    MedAgentica AI can make mistakes. Verify important medical information.
                </div>
            </div>
        </div>
    );
};
