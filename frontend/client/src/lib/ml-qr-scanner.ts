/**
 * ML-powered QR Scanner Client Library
 * Interfaces with the Python ML microservice via the Node.js backend proxy
 */

// Risk level types
export type RiskLevel = 'Low' | 'Medium' | 'High';
export type RiskRecommendation = 'Allow' | 'Verify' | 'Block';

// QR scan result from ML analysis
export interface QRScanResult {
  risk_score: number;          // 0-100 risk score
  risk_level: RiskLevel;       // Low, Medium, High
  recommendation: RiskRecommendation; // Allow, Verify, Block
  confidence: number;          // 0-1 confidence score
  features: {
    pattern_match: number;     // Score for pattern matching
    domain_check: number;      // Score for domain risk
    syntax_validation: number; // Score for syntax validation
    entropy: number;           // Score for randomness/entropy
    length_score: number;      // Score for length analysis
  };
  scan_time_ms: number;        // Analysis time in milliseconds
  detected_patterns: string[]; // Detected suspicious patterns
}

// UPI payment info from QR code
export interface UPIPaymentInfo {
  valid: boolean;
  upiId: string;
  name?: string;
  amount?: string;
  currency?: string;
}

/**
 * Analyze a QR code text with the ML service
 * @param qrText The raw text from the QR code scan
 * @returns Promise with QR scan risk analysis results
 */
export async function analyzeQRWithML(qrText: string): Promise<QRScanResult> {
  try {
    const response = await fetch('/api/ml/qr-scan', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ qr_text: qrText }),
    });

    if (!response.ok) {
      throw new Error(`ML analysis failed: ${response.statusText}`);
    }

    const result = await response.json();
    
    // Transform API response to match our expected format
    const scanResult: QRScanResult = {
      risk_score: Math.min(100, Math.max(0, Math.round(result.risk_score * 100))),
      risk_level: getRiskLevel(result.risk_score),
      recommendation: getRiskRecommendation(result.risk_score),
      confidence: result.confidence || 0.85,
      features: {
        pattern_match: result.features?.pattern_match || 0,
        domain_check: result.features?.domain_check || 0,
        syntax_validation: result.features?.syntax_validation || 0,
        entropy: result.features?.entropy || 0,
        length_score: result.features?.length_score || 0
      },
      scan_time_ms: result.scan_time_ms || 0,
      detected_patterns: result.detected_patterns || []
    };

    return scanResult;
  } catch (error) {
    console.error('Error analyzing QR code with ML:', error);
    
    // Return a default moderate-risk result in case of API failure
    return {
      risk_score: 35,
      risk_level: 'Medium',
      recommendation: 'Verify',
      confidence: 0.5,
      features: {
        pattern_match: 0.3,
        domain_check: 0.4,
        syntax_validation: 0.6,
        entropy: 0.5,
        length_score: 0.7
      },
      scan_time_ms: 0,
      detected_patterns: ['API_ERROR']
    };
  }
}

/**
 * Extract UPI payment information from QR code text
 * @param qrText Raw text from QR code scan
 * @returns UPI payment info object with parsed information
 */
export function extractUPIPaymentInfo(qrText: string): UPIPaymentInfo {
  // Default return structure
  const result: UPIPaymentInfo = {
    valid: false,
    upiId: ''
  };
  
  // Check for UPI URL format (like upi://pay?pa=abc@bank&pn=Name&am=100)
  if (qrText.startsWith('upi://')) {
    try {
      const url = new URL(qrText);
      const params = new URLSearchParams(url.search);
      
      const upiId = params.get('pa');
      if (upiId) {
        result.valid = true;
        result.upiId = upiId;
        result.name = params.get('pn') || undefined;
        result.amount = params.get('am') || undefined;
        result.currency = params.get('cu') || 'INR';
      }
    } catch (e) {
      // If URL parsing fails, try regex
      const paMatch = qrText.match(/pa=([^&]+)/);
      const pnMatch = qrText.match(/pn=([^&]+)/);
      const amMatch = qrText.match(/am=([^&]+)/);
      const cuMatch = qrText.match(/cu=([^&]+)/);
      
      if (paMatch && paMatch[1]) {
        result.valid = true;
        result.upiId = paMatch[1];
        if (pnMatch && pnMatch[1]) result.name = pnMatch[1];
        if (amMatch && amMatch[1]) result.amount = amMatch[1];
        if (cuMatch && cuMatch[1]) result.currency = cuMatch[1];
      }
    }
  } else if (qrText.includes('@')) {
    // Directly a UPI ID (like abc@bank)
    result.valid = true;
    result.upiId = qrText;
  } else {
    // Try to extract a UPI ID from text
    const match = qrText.match(/([a-zA-Z0-9\.\_\-]+@[a-zA-Z0-9]+)/);
    if (match && match[1]) {
      result.valid = true;
      result.upiId = match[1];
    }
  }
  
  return result;
}

/**
 * Determine risk level based on risk score
 * @param score Risk score (0-1)
 * @returns Risk level (Low, Medium, High)
 */
function getRiskLevel(score: number): RiskLevel {
  if (score < 0.3) return 'Low';
  if (score < 0.7) return 'Medium';
  return 'High';
}

/**
 * Determine recommendation based on risk score
 * @param score Risk score (0-1)
 * @returns Recommendation (Allow, Verify, Block)
 */
function getRiskRecommendation(score: number): RiskRecommendation {
  if (score < 0.25) return 'Allow';
  if (score < 0.65) return 'Verify';
  return 'Block';
}