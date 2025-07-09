/**
 * API Client Library
 * Provides methods to interact with the backend API endpoints
 */

// Common API request function with error handling
export async function apiRequest(url: string, method: string = 'GET', data?: any) {
  try {
    const options: RequestInit = {
      method,
      headers: {
        'Content-Type': 'application/json'
      }
    };
    
    if (data) {
      options.body = JSON.stringify(data);
    }
    
    const response = await fetch(url, options);
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API request failed: ${response.status} ${errorText}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('API request failed:', error);
    throw error;
  }
}

/**
 * Analyze UPI ID for fraud risk
 */
export async function analyzeUpiId(upiId: string) {
  return apiRequest(`/api/upi-check?upiId=${encodeURIComponent(upiId)}`);
}

/**
 * Process a voice command for fraud detection
 */
export async function processVoiceCommand(command: string) {
  return apiRequest('/api/process-voice', 'POST', { command });
}

/**
 * Analyze transaction risk scoring
 */
export async function analyzeTransaction(transactionData: {
  upiId: string;
  amount: number;
  recipient?: string;
  note?: string;
  location?: { lat: number; lng: number };
  deviceInfo?: any;
}) {
  return apiRequest('/api/analyze-transaction', 'POST', transactionData);
}

/**
 * Analyze WhatsApp message screenshot
 */
export async function analyzeWhatsAppMessage(base64Image: string, description?: string) {
  return apiRequest('/api/analyze-whatsapp', 'POST', { 
    image: base64Image, 
    description 
  });
}

/**
 * Send chat message and get AI response
 */
export async function sendChatMessage(message: string, userId?: number) {
  return apiRequest('/api/chat-message', 'POST', { 
    message, 
    userId
  });
}

/**
 * Get scam news for a location
 */
export async function getScamNews(location?: string) {
  const url = location 
    ? `/api/scam-news?location=${encodeURIComponent(location)}` 
    : '/api/scam-news';
  
  return apiRequest(url);
}

/**
 * Report a scam
 */
export async function reportScam(data: {
  upiId: string;
  scamType: string;
  description: string;
  amountLost?: number;
  userId?: number;
}) {
  return apiRequest('/api/report-scam', 'POST', data);
}

/**
 * Submit police complaint
 */
export async function submitPoliceComplaint(data: {
  upiId: string;
  name: string;
  amount: number;
  policeStation?: string;
  city?: string;
  description: string;
  transactionDate?: string;
  userDetails?: any;
}) {
  return apiRequest('/api/report/police-complaint', 'POST', data);
}