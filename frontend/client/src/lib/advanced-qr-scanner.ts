/**
 * Advanced ML-Based QR Scanner
 * Integrates with Python backend for enhanced fraud detection
 */

import { apiRequest } from "@/lib/queryClient";
import { QRScanResult } from './ml-qr-scanner';

/**
 * Analyze a QR code using the advanced ML model
 * @param qrText The raw text from the QR code scan
 * @returns Promise with QR scan risk analysis results
 */
export async function analyzeQRWithAdvancedML(qrText: string): Promise<QRScanResult> {
  try {
    console.log('Analyzing QR code with advanced ML service:', qrText);
    
    // Call our advanced ML service
    const response = await apiRequest('POST', '/api/advanced-qr/predict', {
      qr_text: qrText
    });
    
    if (!response.ok) {
      throw new Error(`Advanced QR analysis failed: ${response.statusText}`);
    }
    
    const result = await response.json();
    console.log('Advanced ML analysis result:', result);
    
    // Extract risk score (0-100)
    const riskScore = result.risk_score || 50;
    
    // Determine risk level based on score
    let riskLevel: 'Low' | 'Medium' | 'High';
    if (riskScore < 30) {
      riskLevel = 'Low';
    } else if (riskScore < 70) {
      riskLevel = 'Medium';
    } else {
      riskLevel = 'High';
    }
    
    // Determine recommendation based on risk level
    let recommendation: 'Allow' | 'Verify' | 'Block';
    if (riskLevel === 'Low') {
      recommendation = 'Allow';
    } else if (riskLevel === 'Medium') {
      recommendation = 'Verify';
    } else {
      recommendation = 'Block';
    }
    
    // Extract features or generate defaults
    const features = {
      pattern_match: (result.features?.urgent || 0) * 0.8,
      domain_check: (result.features?.has_upi || 0) * 0.9,
      syntax_validation: 1 - ((riskScore / 100) * 0.3),
      entropy: ((result.features?.length || 0) / 100) * 0.7,
      length_score: Math.min((result.features?.length || 50) / 80, 1)
    };
    
    // Return formatted result compatible with existing UI
    return {
      risk_score: riskScore,
      risk_level: riskLevel,
      recommendation: recommendation,
      confidence: result.fallback ? 0.75 : 0.95, // Lower confidence for fallback results
      features: features,
      scan_time_ms: result.latency_ms || 50,
      detected_patterns: result.labels || []
    };
  } catch (error) {
    console.error('Error analyzing QR code with advanced ML:', error);
    
    // Return a moderate risk assessment as fallback
    return {
      risk_score: 40,
      risk_level: 'Medium',
      recommendation: 'Verify',
      confidence: 0.6,
      features: {
        pattern_match: 0.4,
        domain_check: 0.5,
        syntax_validation: 0.7,
        entropy: 0.5,
        length_score: 0.6
      },
      scan_time_ms: 10,
      detected_patterns: ['ADVANCED_ML_ERROR']
    };
  }
}

/**
 * Submit feedback about a QR code scan
 * @param qrText The scanned QR code text
 * @param isScam Whether the user reported this as a scam
 * @returns Promise with the feedback result
 */
export async function sendAdvancedQRFeedback(
  qrText: string, 
  isScam: boolean
): Promise<{ status: string }> {
  try {
    const response = await apiRequest('POST', '/api/advanced-qr/feedback', {
      qr_text: qrText,
      is_scam: isScam
    });
    
    if (!response.ok) {
      throw new Error('Failed to send QR feedback');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error sending advanced QR feedback:', error);
    return { status: 'feedback_error' };
  }
}