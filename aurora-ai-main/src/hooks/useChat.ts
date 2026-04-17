import { useState, useEffect, useRef } from "react";
import { apiService, type ChatResponse } from "@/lib/api";
import { toast } from "sonner";
import { Message } from "@/components/chat/ChatMessage";
import { Conversation } from "@/components/chat/ChatSidebar";

export const useChat = () => {
    // Local Storage Keys
    const STORAGE_KEY = "medagentica-conversations";
    const SELECTED_CHAT_KEY = "medagentica-selected-chat";

    // Helper to parse dates from JSON
    const parseDates = (key: string, value: any) => {
        if (key === "timestamp" && typeof value === "string") {
            return new Date(value);
        }
        return value;
    };

    // Load initial state from local storage
    const loadInitialConversations = (): Conversation[] => {
        // Persistence disabled per user request
        // try {
        //     const saved = localStorage.getItem(STORAGE_KEY);
        //     if (saved) {
        //         return JSON.parse(saved, parseDates);
        //     }
        // } catch (error) {
        //     console.error("Failed to load conversations:", error);
        // }

        // Default welcome conversation
        return [{
            id: "1",
            title: "New Conversation",
            preview: "Start chatting with MedAgentica...",
            timestamp: new Date(),
            messages: [
                {
                    id: "welcome",
                    role: "assistant",
                    content: "👋 Welcome to MedAgentica! I'm your medical assistant powered by multiple specialized agents:\n\n💬 Conversation Agent - General health discussions\n\n📚 RAG Agent - Medical knowledge queries\n\n🌐 Web Search Agent - Latest medical research\n\n🧠 Brain Tumor Agent - MRI analysis\n\n🫁 Chest X-ray Agent - COVID-19 detection\n\n🩺 Skin Lesion Agent - Skin condition analysis\n\nHow can I assist you today?",
                    timestamp: new Date(),
                    agent: "SYSTEM",
                },
            ],
        }];
    };

    const [conversations, setConversations] = useState<Conversation[]>(loadInitialConversations);

    const [selectedChat, setSelectedChat] = useState<string>(() => {
        // return localStorage.getItem(SELECTED_CHAT_KEY) || conversations[0]?.id || "1";
        return "1";
    });

    const [userRole, setUserRole] = useState<"patient" | "clinician">("patient");

    const [uploadedImages, setUploadedImages] = useState<File[]>([]);
    const [imagePreviewUrls, setImagePreviewUrls] = useState<string[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const abortControllerRef = useRef<AbortController | null>(null);

    const currentConversation = conversations.find((c) => c.id === selectedChat);
    const messages = currentConversation?.messages || [];

    // Persist conversations to local storage - DISABLED
    // useEffect(() => {
    //     localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
    // }, [conversations]);

    // Persist selected chat - DISABLED
    // useEffect(() => {
    //     localStorage.setItem(SELECTED_CHAT_KEY, selectedChat);
    // }, [selectedChat]);

    useEffect(() => {
        const handleNewChatEvent = () => createNewChat();
        window.addEventListener('new-chat', handleNewChatEvent);
        return () => window.removeEventListener('new-chat', handleNewChatEvent);
    }, [conversations.length]); // Depend on length to ensure fresh state in closure if needed, though createNewChat uses functional update

    const createNewChat = () => {
        const newId = Date.now().toString();
        const newConv: Conversation = {
            id: newId,
            title: `Chat ${conversations.length + 1}`,
            preview: "New conversation",
            timestamp: new Date(),
            messages: [
                {
                    id: "welcome-" + newId,
                    role: "assistant",
                    content: "👋 Welcome to MedAgentica! I'm your medical assistant powered by multiple specialized agents:\n\n💬 Conversation Agent - General health discussions\n\n📚 RAG Agent - Medical knowledge queries\n\n🌐 Web Search Agent - Latest medical research\n\n🧠 Brain Tumor Agent - MRI analysis\n\n🫁 Chest X-ray Agent - COVID-19 detection\n\n🩺 Skin Lesion Agent - Skin condition analysis\n\nHow can I assist you today?",
                    timestamp: new Date(),
                    agent: "SYSTEM",
                },
            ],
        };

        setConversations((prev) => [newConv, ...prev]);
        setSelectedChat(newId);
    };

    const stopGeneration = () => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
            abortControllerRef.current = null;
            setIsLoading(false);
            toast.info("Generation stopped");
        }
    };

    const sendMessage = async (input: string) => {
        if (!input.trim() && uploadedImages.length === 0) return;

        // Abort previous request if any
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }

        // Create new controller
        const controller = new AbortController();
        abortControllerRef.current = controller;

        setIsLoading(true);

        try {
            if (uploadedImages.length > 0) {
                await handleImageUpload(input);
                return;
            }

            const userMessage: Message = {
                id: Date.now().toString(),
                role: "user",
                content: input,
                timestamp: new Date(),
            };

            setConversations((prev) =>
                prev.map((conv) =>
                    conv.id === selectedChat
                        ? {
                            ...conv,
                            messages: [...conv.messages, userMessage],
                            preview: input.slice(0, 50),
                            timestamp: new Date(),
                        }
                        : conv
                )
            );

            const response: ChatResponse = await apiService.sendMessage(input, [], userRole, selectedChat, controller.signal);

            const assistantMessage: Message = {
                id: (Date.now() + 1).toString(),
                role: "assistant",
                content: response.response,
                timestamp: new Date(),
                agent: response.agent,
                thinking: response.thinking,
                suggestions: response.suggestions,
                result_image: response.result_image,
                all_images: response.all_images,
            };

            setConversations((prev) =>
                prev.map((conv) =>
                    conv.id === selectedChat
                        ? { ...conv, messages: [...conv.messages, assistantMessage] }
                        : conv
                )
            );
        } catch (error: any) {
            if (error.name === 'AbortError') {
                console.log('Request aborted');
                return;
            }
            console.error("Error sending message:", error);
            toast.error("Failed to send message. Please try again.");
        } finally {
            if (abortControllerRef.current === controller) {
                setIsLoading(false);
                abortControllerRef.current = null;
            }
        }
    };

    const handleImageUpload = async (input: string) => {
        // Ensure we have a controller (should be created in sendMessage)
        const controller = abortControllerRef.current || new AbortController();
        abortControllerRef.current = controller;

        try {
            const userMessage: Message = {
                id: Date.now().toString(),
                role: "user",
                content: input || "Please analyze this medical image.",
                timestamp: new Date(),
            };

            setConversations((prev) =>
                prev.map((conv) =>
                    conv.id === selectedChat
                        ? {
                            ...conv,
                            messages: [...conv.messages, userMessage],
                            preview: "Image uploaded",
                            timestamp: new Date(),
                        }
                        : conv
                )
            );

            const response = await apiService.uploadImage(uploadedImages[0], input, userRole, selectedChat, controller.signal);

            const assistantMessage: Message = {
                id: (Date.now() + 1).toString(),
                role: "assistant",
                content: response.response,
                timestamp: new Date(),
                agent: response.agent,
                thinking: response.thinking,
                suggestions: response.suggestions,
                result_image: response.result_image,
                all_images: response.all_images,
            };

            setConversations((prev) =>
                prev.map((conv) =>
                    conv.id === selectedChat
                        ? { ...conv, messages: [...conv.messages, assistantMessage] }
                        : conv
                )
            );

            setUploadedImages([]);
            setImagePreviewUrls([]);
        } catch (error: any) {
            if (error.name === 'AbortError') {
                console.log('Request aborted');
                return;
            }
            console.error("Error uploading image:", error);
            toast.error("Failed to upload image. Please try again.");
        } finally {
            if (abortControllerRef.current === controller) {
                setIsLoading(false);
                abortControllerRef.current = null;
            }
        }
    };

    const handleImageSelect = (files: FileList) => {
        const fileArray = Array.from(files);
        setUploadedImages(fileArray);
        const urls = fileArray.map((file) => URL.createObjectURL(file));
        setImagePreviewUrls(urls);
    };

    const removeImage = (index: number) => {
        setUploadedImages(uploadedImages.filter((_, i) => i !== index));
        URL.revokeObjectURL(imagePreviewUrls[index]);
        setImagePreviewUrls(imagePreviewUrls.filter((_, i) => i !== index));
    };

    const deleteChat = (id: string) => {
        if (window.confirm("Are you sure you want to delete this conversation?")) {
            setConversations((prev) => {
                const filtered = prev.filter((conv) => conv.id !== id);
                if (selectedChat === id) {
                    setSelectedChat(filtered.length > 0 ? filtered[0].id : "");
                }
                return filtered;
            });
            toast.success("Conversation deleted");
        }
    };

    const renameChat = (id: string, newTitle: string) => {
        setConversations((prev) =>
            prev.map((conv) =>
                conv.id === id ? { ...conv, title: newTitle } : conv
            )
        );
    };

    return {
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
    };
};
