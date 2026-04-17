/**
 * API Service for Multi-Agent Medical Assistant
 * Handles all communication with the backend API
 */

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  agent?: string;
  thinking?: string;
  suggestions?: string[];
  result_image?: string;
}

export interface ChatRequest {
  query: string;
  conversation_history?: any[];
}

export interface ChatResponse {
  status: string;
  response: string;
  agent: string;
  thinking?: string;
  confidence?: number;
  suggestions?: string[];
  result_image?: string;
  all_images?: Array<{ type: string; url: string; label: string }>;
}

export interface UploadResponse {
  status: string;
  response: string;
  agent: string;
  thinking?: string;
  suggestions?: string[];
  result_image?: string;
  all_images?: Array<{ type: string; url: string; label: string }>;
}

class ApiService {
  private baseURL = '/api';

  /**
   * Send a chat message to the backend
   */
  async sendMessage(query: string, conversationHistory: any[] = [], userRole: string = "patient", conversationId: string = "1", signal?: AbortSignal): Promise<ChatResponse> {
    try {
      const response = await fetch(`${this.baseURL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include', // Include cookies for session management
        body: JSON.stringify({
          query,
          conversation_history: conversationHistory,
          user_role: userRole,
          conversation_id: conversationId,
        }),
        signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error sending message:', error);
      throw error;
    }
  }

  /**
   * Upload an image with optional text
   */
  async uploadImage(file: File, text: string = '', userRole: string = "patient", conversationId: string = "1", signal?: AbortSignal): Promise<UploadResponse> {
    try {
      const formData = new FormData();
      formData.append('image', file);
      formData.append('text', text);
      formData.append('user_role', userRole);
      formData.append('conversation_id', conversationId);

      const response = await fetch(`${this.baseURL}/upload`, {
        method: 'POST',
        credentials: 'include', // Include cookies for session management
        body: formData,
        signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error uploading image:', error);
      throw error;
    }
  }

  /**
   * Transcribe audio to text
   */
  async transcribeAudio(file: File): Promise<{ text: string }> {
    try {
      const formData = new FormData();
      formData.append('audio_file', file);

      const response = await fetch(`${this.baseURL}/transcribe`, {
        method: 'POST',
        credentials: 'include',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error transcribing audio:', error);
      throw error;
    }
  }

  /**
   * Generate speech from text
   */
  async generateSpeech(text: string, voiceId: string = 'EXAMPLE_VOICE_ID'): Promise<Blob> {
    try {
      const response = await fetch(`${this.baseURL}/generate-speech`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({
          text,
          voice_id: voiceId,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.blob();
    } catch (error) {
      console.error('Error generating speech:', error);
      throw error;
    }
  }

  /**
   * Check backend health
   */
  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    try {
      const response = await fetch(`${this.baseURL}/health`, {
        method: 'GET',
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error checking health:', error);
      throw error;
    }
  }
}

// Export singleton instance
export const apiService = new ApiService();

