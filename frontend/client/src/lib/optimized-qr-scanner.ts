/**
 * Optimized QR Scanner Client Service
 * Uses the backend FastAPI service for QR code analysis
 */

// QR Scan Service Host
// First try the optimized service, fall back to direct implementation
const QR_SCAN_API_URL = '/api/optimized-qr';
const DIRECT_QR_SCAN_API_URL = '/api/direct-qr';

// QR prediction response 
export interface QRPredictionResponse {
  risk_score: number;
  latency_ms: number;
  cached?: boolean;
  features?: {
    length: number;
    has_upi: number;
    num_params: number;
    urgent: number;
    payment: number;
    currency: number;
  };
}

// UPI payment information extracted from QR codes
export interface UPIPaymentInfo {
  pa?: string; // payee address (UPI ID)
  pn?: string; // payee name
  am?: string; // amount
  cu?: string; // currency
  tn?: string; // transaction note
  mc?: string; // merchant code
  mid?: string; // merchant ID
  tr?: string; // transaction reference
  url?: string; // dynamic URL
  mode?: string; // mode of payment
}

// Risk level derived from scores
export type RiskLevel = 'low' | 'medium' | 'high';

/**
 * Extract UPI payment information from QR code text
 * @param qrText - The scanned QR code text
 * @returns UPI payment information object
 */
export function extractUPIPaymentInfo(qrText: string): UPIPaymentInfo | null {
  if (!qrText || !qrText.startsWith('upi://')) {
    return null;
  }

  try {
    // Extract the query parameters
    const paramsText = qrText.split('?')[1];
    if (!paramsText) return null;

    const paramsSegments = paramsText.split('&');
    const params: UPIPaymentInfo = {};

    paramsSegments.forEach(segment => {
      const [key, value] = segment.split('=');
      if (key && value) {
        params[key as keyof UPIPaymentInfo] = decodeURIComponent(value);
      }
    });

    return params;
  } catch (error) {
    console.error('Error parsing UPI QR code:', error);
    return null;
  }
}

/**
 * Classify the QR risk level based on the risk score
 * @param riskScore - The numerical risk score (0-100)
 * @returns Risk level category
 */
export function classifyRiskLevel(riskScore: number): RiskLevel {
  if (riskScore < 30) return 'low';
  if (riskScore < 70) return 'medium';
  return 'high';
}

/**
 * Calculate string entropy as a fallback risk assessment method
 * Used when ML service is unavailable
 * @param str - String to analyze
 * @returns Normalized entropy value (0-1)
 */
function calculateStringEntropy(str: string): number {
  if (!str || str.length === 0) return 0;
  
  // Calculate character frequency
  const charFreq: Record<string, number> = {};
  for (let i = 0; i < str.length; i++) {
    const char = str[i];
    charFreq[char] = (charFreq[char] || 0) + 1;
  }
  
  // Calculate entropy
  let entropy = 0;
  Object.values(charFreq).forEach(freq => {
    const p = freq / str.length;
    entropy -= p * Math.log2(p);
  });

  // Normalize to 0-1 range (typical English text has entropy around 4-5)
  return Math.min(entropy / 6, 1);
}

/**
 * Analyzes QR code text using the ML service
 * @param qrText - The scanned QR code text
 * @returns Promise with risk analysis
 */
export async function analyzeQRCode(qrText: string): Promise<QRPredictionResponse> {
  // First try the optimized ML service
  try {
    // Call the optimized QR scanning service
    const response = await fetch(`${QR_SCAN_API_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ qr_text: qrText }),
    });

    if (!response.ok) {
      throw new Error('QR analysis service unavailable');
    }

    const result = await response.json();
    console.log('Using optimized ML service:', result);
    return result;
  } catch (error) {
    console.warn('Optimized QR service error, trying direct implementation:', error);
    
    // Try the direct TypeScript implementation
    try {
      const directResponse = await fetch(`${DIRECT_QR_SCAN_API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ qr_text: qrText }),
      });
      
      if (directResponse.ok) {
        const directResult = await directResponse.json();
        console.log('Using direct TypeScript implementation:', directResult);
        return directResult;
      }
      
      throw new Error('Direct implementation also failed');
    } catch (directError) {
      console.warn('Direct QR service error, using local fallback:', directError);
      
      // Final fallback to local risk assessment
      const entropy = calculateStringEntropy(qrText);
      
      // UPI-specific risk factors in local fallback
      let baseRisk = 30; // Start with medium risk
      
      // Significantly reduce risk for UPI QR codes
      if (qrText.toLowerCase().startsWith('upi://')) {
        baseRisk -= 25; // Major reduction for UPI QR codes
      }
      
      // Consider entropy for non-standard codes
      if (!qrText.toLowerCase().startsWith('upi://')) {
        baseRisk += entropy * 20; // Only apply entropy factor to non-UPI QRs
      }
      
      // Check for suspicious keywords
      const keywords = ['urgent', 'emergency', 'verify', 'kyc', 'expired', 'blocked'];
      keywords.forEach(keyword => {
        if (qrText.toLowerCase().includes(keyword)) {
          baseRisk += 15; // Increase risk for suspicious keywords
        }
      });
      
      // Final result with minimal features
      return {
        risk_score: Math.max(0, Math.min(100, Math.round(baseRisk))),
        latency_ms: 5,
        features: {
          length: qrText.length,
          has_upi: qrText.toLowerCase().startsWith('upi://') ? 1 : 0,
          num_params: (qrText.match(/&/g) || []).length,
          urgent: keywords.some(k => qrText.toLowerCase().includes(k)) ? 1 : 0,
          payment: qrText.toLowerCase().includes('payment') ? 1 : 0,
          currency: qrText.toLowerCase().includes('inr') ? 1 : 0
        }
      };
    }
  }
}

/**
 * Sends feedback about a scanned QR code
 * @param qrText - The scanned QR code
 * @param isScam - Whether the QR was a scam
 * @returns Promise with feedback status
 */
export async function sendQRFeedback(qrText: string, isScam: boolean): Promise<{ status: string }> {
  try {
    const response = await fetch(`${QR_SCAN_API_URL}/feedback`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        qr_text: qrText, 
        is_scam: isScam 
      }),
    });
    
    if (!response.ok) {
      throw new Error('Failed to send QR feedback');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error sending QR feedback:', error);
    return { status: 'feedback_error' };
  }
}