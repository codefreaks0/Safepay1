/**
 * Integration between Enhanced QR Scanner UI and Optimized QR Scanner Service
 * Combines the UI capabilities with the high-performance backend
 */

import { QRScanResult, UPIPaymentInfo, extractUPIPaymentInfo } from './ml-qr-scanner';
import { analyzeQRCode, classifyRiskLevel } from './optimized-qr-scanner';

/**
 * Analyze a QR code with the optimized ML service
 * @param qrText The raw text from the QR code scan
 * @returns Promise with QR scan risk analysis results (compatible with existing UI)
 */
export async function analyzeQRWithOptimizedML(qrText: string): Promise<QRScanResult> {
  try {
    // Use optimized service for analysis
    const optimizedResult = await analyzeQRCode(qrText);
    
    // Get normalized risk score (0-100)
    const riskScore = optimizedResult.risk_score;
    
    // Convert to 0-1 scale for internal functions
    const normalizedScore = riskScore / 100;
    
    // Determine risk level based on classification
    const riskLevel = classifyRiskLevel(riskScore);
    
    // Use existing functions to determine recommendation based on risk level
    let recommendation: 'Allow' | 'Verify' | 'Block';
    if (riskLevel === 'low') {
      recommendation = 'Allow';
    } else if (riskLevel === 'medium') {
      recommendation = 'Verify';
    } else {
      recommendation = 'Block';
    }
    
    // Generate features based on optimized result
    // Use the features from the optimized service if available
    const features = optimizedResult.features 
      ? {
          pattern_match: optimizedResult.features.urgent ? 0.8 : 0.3,
          domain_check: optimizedResult.features.has_upi ? 0.2 : 0.9,
          syntax_validation: 1 - (normalizedScore * 0.3),
          entropy: (optimizedResult.features.length / 100) * 0.9,
          length_score: Math.min(optimizedResult.features.length / 80, 1) // Normalize length
        }
      : {
          // Fallback when features are not available
          pattern_match: normalizedScore * 0.8,
          domain_check: normalizedScore * 0.9,
          syntax_validation: 1 - (normalizedScore * 0.3),
          entropy: normalizedScore * 0.7,
          length_score: 0.7
        };
    
    // Convert risk level to match expected format
    const formattedRiskLevel = riskLevel === 'low' ? 'Low' : 
                              (riskLevel === 'medium' ? 'Medium' : 'High');
    
    // Return formatted result compatible with existing UI
    return {
      risk_score: riskScore,
      risk_level: formattedRiskLevel,
      recommendation: recommendation,
      confidence: optimizedResult.cached ? 0.95 : 0.9, // Higher confidence for cached results
      features: features,
      scan_time_ms: optimizedResult.latency_ms,
      detected_patterns: []
    };
  } catch (error) {
    console.error('Error analyzing QR code with optimized ML:', error);
    
    // Fallback to a moderate risk result
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
      detected_patterns: ['OPTIMIZED_API_ERROR']
    };
  }
}

/**
 * Extract UPI payment information from QR code text
 * This is a wrapper to maintain compatibility with the existing function
 * @param qrText Raw text from QR code scan
 * @returns UPI payment info object with parsed information
 */
export { extractUPIPaymentInfo };

/**
 * Full QR analysis including UPI extraction and optimized ML risk analysis
 * @param qrText Raw text from the QR code
 * @returns Complete analysis with UPI details and risk assessment
 */
export async function analyzeFullQRData(qrText: string): Promise<{
  paymentInfo: UPIPaymentInfo;
  riskAnalysis: QRScanResult;
}> {
  // Extract UPI information
  const paymentInfo = extractUPIPaymentInfo(qrText);
  
  // Get risk analysis from optimized service
  const riskAnalysis = await analyzeQRWithOptimizedML(qrText);
  
  return {
    paymentInfo,
    riskAnalysis
  };
}