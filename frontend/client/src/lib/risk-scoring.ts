/**
 * Real-Time Transaction Risk Scoring Engine
 * 
 * This module provides a client-side implementation of risk scoring for UPI transactions.
 * It simulates the behavior of an ML-based risk scoring system (the actual ML models would run on the server).
 */

// We'll create this file in a moment
import { analyzeTransaction } from './api-client';

// Transaction risk levels
export enum RiskLevel {
  Low = 'low',
  Medium = 'medium',
  High = 'high',
  Critical = 'critical'
}

// Risk score result interface
export interface RiskScoreResult {
  score: number; // 0-1 score where higher means more risky
  level: RiskLevel;
  factors: RiskFactor[];
  recommendation: string;
  needsAdditionalVerification: boolean;
}

// Risk factor that contributed to the score
export interface RiskFactor {
  type: string;
  description: string;
  impact: number; // 0-1 representing how much this contributed to the score
}

/**
 * Calculate risk level based on numeric score
 */
export function getRiskLevelFromScore(score: number): RiskLevel {
  if (score < 0.3) return RiskLevel.Low;
  if (score < 0.6) return RiskLevel.Medium;
  if (score < 0.85) return RiskLevel.High;
  return RiskLevel.Critical;
}

/**
 * Get appropriate color for risk level
 */
export function getRiskColor(level: RiskLevel): string {
  switch (level) {
    case RiskLevel.Low: return 'bg-green-500';
    case RiskLevel.Medium: return 'bg-yellow-500';
    case RiskLevel.High: return 'bg-orange-500';
    case RiskLevel.Critical: return 'bg-red-500';
    default: return 'bg-gray-500';
  }
}

/**
 * Get appropriate text color for risk level
 */
export function getRiskTextColor(level: RiskLevel): string {
  switch (level) {
    case RiskLevel.Low: return 'text-green-700';
    case RiskLevel.Medium: return 'text-yellow-700';
    case RiskLevel.High: return 'text-orange-700';
    case RiskLevel.Critical: return 'text-red-700';
    default: return 'text-gray-700';
  }
}

/**
 * Get appropriate background color for risk level badge
 */
export function getRiskBgColor(level: RiskLevel): string {
  switch (level) {
    case RiskLevel.Low: return 'bg-green-50';
    case RiskLevel.Medium: return 'bg-yellow-50';
    case RiskLevel.High: return 'bg-orange-50';
    case RiskLevel.Critical: return 'bg-red-50';
    default: return 'bg-gray-50';
  }
}

/**
 * Get recommendation based on risk score
 */
export function getRecommendation(score: number): string {
  const level = getRiskLevelFromScore(score);
  
  switch (level) {
    case RiskLevel.Low:
      return "This transaction appears to be safe. You can proceed normally.";
    case RiskLevel.Medium:
      return "Use caution with this transaction. Verify the recipient's details before proceeding.";
    case RiskLevel.High:
      return "This transaction shows significant risk factors. Consider using an alternative payment method or contact the recipient directly to verify.";
    case RiskLevel.Critical:
      return "WARNING: This transaction has critical risk indicators. We strongly recommend cancelling this transaction.";
    default:
      return "Unable to assess risk. Please proceed with caution.";
  }
}

/**
 * Generate risk score for a transaction based on available data
 * In a real implementation, this would call a backend ML service
 */
export async function calculateTransactionRiskScore(
  transactionData: { 
    upiId: string;
    amount: number;
    recipient?: string;
    note?: string;
    location?: { lat: number; lng: number };
    deviceInfo?: any;
  }
): Promise<RiskScoreResult> {
  try {
    // In a real implementation, this would call a server-side ML model
    // Here we'll simulate by checking for known patterns
    const { upiId, amount } = transactionData;
    
    // Try to get analysis from server (which uses OpenAI)
    let serverAnalysis = null;
    try {
      serverAnalysis = await analyzeTransaction(transactionData);
    } catch (error) {
      console.error('Failed to get server analysis, using local fallback', error);
    }
    
    // Initialize risk factors
    const factors: RiskFactor[] = [];
    
    // If we got server analysis, use it
    if (serverAnalysis) {
      return {
        score: serverAnalysis.risk_score,
        level: getRiskLevelFromScore(serverAnalysis.risk_score),
        factors: serverAnalysis.risk_factors.map((f: any) => ({
          type: f.type,
          description: f.description,
          impact: f.impact
        })),
        recommendation: serverAnalysis.recommendation || getRecommendation(serverAnalysis.risk_score),
        needsAdditionalVerification: serverAnalysis.risk_score > 0.6
      };
    }
    
    // Fallback to local basic analysis
    // 1. Check amount (large amounts are riskier)
    let baseScore = 0.1; // Start with baseline risk
    
    if (amount > 50000) {
      baseScore += 0.3;
      factors.push({
        type: 'amount',
        description: 'Large transaction amount',
        impact: 0.3
      });
    } else if (amount > 10000) {
      baseScore += 0.15;
      factors.push({
        type: 'amount',
        description: 'Medium-high transaction amount',
        impact: 0.15
      });
    }
    
    // 2. Check UPI ID for suspicious patterns
    if (!upiId.includes('@')) {
      baseScore += 0.4;
      factors.push({
        type: 'format',
        description: 'Invalid UPI ID format',
        impact: 0.4
      });
    }
    
    // Check for suspicious domains in UPI ID
    const suspiciousDomains = ['wallet', 'secure', 'verify', 'bank', 'payment', 'gov', 'official'];
    const domain = upiId.split('@')[1]?.toLowerCase();
    
    if (domain) {
      for (const suspicious of suspiciousDomains) {
        if (domain.includes(suspicious) && !['oksbi', 'ybl', 'paytm', 'upi'].includes(domain)) {
          baseScore += 0.25;
          factors.push({
            type: 'domain',
            description: 'Suspicious UPI domain',
            impact: 0.25
          });
          break;
        }
      }
    }
    
    // 3. Check note for suspicious keywords if available
    if (transactionData.note) {
      const note = transactionData.note.toLowerCase();
      const suspiciousKeywords = ['urgent', 'verify', 'kyc', 'account will be blocked', 'prize', 'winner', 'gift'];
      
      for (const keyword of suspiciousKeywords) {
        if (note.includes(keyword)) {
          baseScore += 0.35;
          factors.push({
            type: 'note',
            description: 'Suspicious keywords in transaction note',
            impact: 0.35
          });
          break;
        }
      }
    }
    
    // Ensure score is in the range 0-1
    const finalScore = Math.min(1, baseScore);
    
    return {
      score: finalScore,
      level: getRiskLevelFromScore(finalScore),
      factors,
      recommendation: getRecommendation(finalScore),
      needsAdditionalVerification: finalScore > 0.6
    };
  } catch (error) {
    console.error('Error calculating risk score:', error);
    
    // In case of error, return a moderate risk score
    return {
      score: 0.5,
      level: RiskLevel.Medium,
      factors: [{ 
        type: 'error', 
        description: 'Unable to perform complete risk analysis', 
        impact: 0.5 
      }],
      recommendation: 'We could not perform a complete risk analysis. Proceed with caution.',
      needsAdditionalVerification: true
    };
  }
}

/**
 * Simulate device fingerprinting analysis
 */
export function analyzeDeviceFingerprint(): { 
  suspicious: boolean; 
  reasons: string[];
} {
  // In a real implementation, this would collect device information
  // and compare against known patterns
  const userAgent = navigator.userAgent;
  const reasons: string[] = [];
  
  // Check for emulators/simulators
  if (userAgent.includes('Emulator') || userAgent.includes('Simulator')) {
    reasons.push('Emulator detected');
  }
  
  // Check for VPN/Proxy (simplified mock implementation)
  // In real implementation, this would use server-side detection
  if (Math.random() < 0.05) { // 5% chance of simulating VPN detection
    reasons.push('Potential VPN usage detected');
  }
  
  return {
    suspicious: reasons.length > 0,
    reasons
  };
}

/**
 * Analyze behavioral biometrics (user interactions)
 */
export function analyzeBehavioralBiometrics(
  interactions: {
    typingSpeed?: number; // average milliseconds between keystrokes
    touchPressure?: number[]; // normalized pressure values (0-1)
    motionData?: number[][]; // gyroscope/accelerometer readings
  } = {}
): { 
  botProbability: number; 
  indicators: string[] 
} {
  // In a real implementation, this would analyze actual interaction data
  // Here we just simulate the results
  const indicators: string[] = [];
  let botScore = 0;
  
  // 1. Check typing speed if available
  if (interactions.typingSpeed) {
    // Extremely fast typing is suspicious (< 50ms between keystrokes)
    if (interactions.typingSpeed < 50) {
      botScore += 0.6;
      indicators.push('Unusually fast typing speed');
    }
    // Perfectly consistent typing is also suspicious
    else if (interactions.typingSpeed > 0 && interactions.typingSpeed < 60) {
      botScore += 0.3;
      indicators.push('Suspiciously consistent typing rhythm');
    }
  }
  
  // 2. Check touch pressure consistency if available
  if (interactions.touchPressure && interactions.touchPressure.length > 3) {
    // Calculate variance - real humans have variable pressure
    const avg = interactions.touchPressure.reduce((a, b) => a + b, 0) / interactions.touchPressure.length;
    const variance = interactions.touchPressure.reduce((a, b) => a + Math.pow(b - avg, 2), 0) / interactions.touchPressure.length;
    
    if (variance < 0.01) {
      botScore += 0.4;
      indicators.push('Unnaturally consistent touch pressure');
    }
  }
  
  // 3. Check motion data if available (real humans have small hand movements)
  if (interactions.motionData && interactions.motionData.length > 10) {
    // This would involve complex analysis in a real system
    // Simulating here
    const hasMotion = interactions.motionData.some(reading => 
      Math.abs(reading[0]) > 0.01 || Math.abs(reading[1]) > 0.01 || Math.abs(reading[2]) > 0.01
    );
    
    if (!hasMotion) {
      botScore += 0.5;
      indicators.push('No natural device movement detected');
    }
  }
  
  // Ensure score is in the range 0-1
  const finalBotScore = Math.min(1, botScore);
  
  return {
    botProbability: finalBotScore,
    indicators
  };
}