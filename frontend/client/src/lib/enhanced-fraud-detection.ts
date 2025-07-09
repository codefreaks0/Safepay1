import { FraudDetectionResponse } from './fraud-detection';

/**
 * Analyze transaction context with AI for advanced fraud detection
 */
export async function analyzeTransactionContext(description: string, amount: number, recipient: string) {
  try {
    const response = await fetch('/api/ai/analyze-transaction', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ description, amount, recipient })
    });
    
    if (!response.ok) {
      throw new Error('Failed to analyze transaction');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error analyzing transaction context:', error);
    return {
      risk_level: "medium",
      confidence: 0.5,
      suspicious_factors: ["Unable to connect to AI service"],
      recommendation: "Proceed with standard security checks"
    };
  }
}

/**
 * Get personalized security alert for a potentially fraudulent transaction
 */
export async function getSecurityAlert(fraudType: string, riskLevel: number, transaction: any) {
  try {
    const response = await fetch('/api/ai/security-alert', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ fraudType, riskLevel, transaction })
    });
    
    if (!response.ok) {
      throw new Error('Failed to generate security alert');
    }
    
    return await response.text();
  } catch (error) {
    console.error('Error generating security alert:', error);
    return `Security Alert: This transaction to ${transaction.recipient} for ₹${transaction.amount} has unusual characteristics. Please verify before proceeding.`;
  }
}

/**
 * Advanced UPI validation to detect fake or typosquatting UPI IDs
 */
export async function validateUpiId(upiId: string) {
  try {
    const response = await fetch('/api/ai/validate-upi', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ upiId })
    });
    
    if (!response.ok) {
      throw new Error('Failed to validate UPI ID');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error validating UPI ID:', error);
    return {
      is_suspicious: false,
      confidence: 0.5,
      flags: [],
      recommendation: "Unable to analyze with AI, proceed with standard checks"
    };
  }
}

/**
 * Get personalized security tip based on user's transaction history and behavior
 */
export async function getSecurityTip(userActivities: any) {
  try {
    const response = await fetch('/api/ai/security-tip', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ userActivities })
    });
    
    if (!response.ok) {
      throw new Error('Failed to generate security tip');
    }
    
    return await response.text();
  } catch (error) {
    console.error('Error generating security tip:', error);
    return "Always verify the UPI ID before making a payment and never share your UPI PIN with anyone.";
  }
}

/**
 * Advanced QR code analysis to detect potential tampering or manipulation
 */
export async function analyzeQRCode(imageData: string) {
  try {
    const response = await fetch('/api/ai/analyze-qr', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ imageData })
    });
    
    if (!response.ok) {
      throw new Error('Failed to analyze QR code');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error analyzing QR code:', error);
    return {
      is_tampered: false,
      confidence: 0.5,
      observations: ["Unable to analyze image with AI"],
      recommendation: "Proceed with standard security checks"
    };
  }
}

/**
 * Analyze voice transcript for potential scam indicators
 */
export async function analyzeVoice(transcript: string) {
  try {
    const response = await fetch('/api/ai/analyze-voice', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ transcript })
    });
    
    if (!response.ok) {
      throw new Error('Failed to analyze voice transcript');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error analyzing voice transcript:', error);
    return {
      is_scam: false,
      confidence: 0.5,
      scam_indicators: [],
      recommendation: "Unable to analyze with AI, proceed with caution"
    };
  }
}

/**
 * Transcribe audio to text for scam analysis
 */
export async function transcribeAudio(audioFile: File) {
  try {
    const formData = new FormData();
    formData.append('audio', audioFile);
    
    const response = await apiRequest('POST', '/api/ai/transcribe', formData, true);
    
    if (!response.ok) {
      throw new Error('Failed to transcribe audio');
    }
    
    return await response.text();
  } catch (error) {
    console.error('Error transcribing audio:', error);
    return null;
  }
}

/**
 * Enhanced fraud detection with multi-factor analysis
 */
export async function enhancedFraudDetection(
  upiId: string, 
  amount: number, 
  deviceFingerprint: string, 
  transactionDescription?: string
): Promise<FraudDetectionResponse> {
  try {
    // Combine standard fraud detection with AI-enhanced analysis
    const response = await fetch('/api/enhanced-fraud-detection', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        upiId,
        amount,
        deviceFingerprint,
        description: transactionDescription || 'Payment'
      })
    });
    
    if (!response.ok) {
      throw new Error('Failed to perform enhanced fraud detection');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error in enhanced fraud detection:', error);
    
    // Fallback response if the API call fails
    return {
      prediction: false,
      confidence: 0.5,
      features: {
        hourly_reports: 0,
        tx_frequency: 0,
        amount_deviation: 0,
        device_risk: 0,
        platform_reports: 0
      },
      live_data: {
        tx_frequency: 0,
        avg_amount: 0,
        device_mismatches: 0,
        recent_reports: 0
      },
      message: "Unable to connect to fraud detection service",
      meta: { 
        service: "fallback",
        version: "1.0",
        latency_ms: 0
      }
    };
  }
}