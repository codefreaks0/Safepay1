import { apiRequest } from './queryClient';

export type RiskLevel = 'Low' | 'Medium' | 'High';

export interface UpiRiskAnalysis {
  upiId: string;
  riskPercentage: number;
  riskLevel: RiskLevel;
  reports: number;
  age: string;
  reportedFor: string;
}

export function getRiskLevelFromPercentage(percentage: number): RiskLevel {
  if (percentage >= 75) return 'High';
  if (percentage >= 30) return 'Medium';
  return 'Low';
}

export function getRiskColor(level: RiskLevel): string {
  switch (level) {
    case 'High': return 'red';
    case 'Medium': return 'yellow';
    case 'Low': return 'green';
    default: return 'green';
  }
}

// Quick UPI analysis - fast and minimalist
export async function analyzeUpiRisk(upiId: string): Promise<UpiRiskAnalysis> {
  try {
    console.log(`Analyzing UPI: ${upiId}`);
    const res = await apiRequest('GET', `/api/upi/check/${encodeURIComponent(upiId)}`);
    const data = await res.json();
    console.log('UPI analysis result:', data);
    
    return {
      upiId: data.upiId,
      riskPercentage: data.riskPercentage || 0,
      riskLevel: data.riskLevel === 'high' ? 'High' : 
                data.riskLevel === 'medium' ? 'Medium' : 'Low',
      reports: data.reports || 0,
      age: 'New', // Default value for quick check
      reportedFor: data.reason || 'Unknown'
    };
  } catch (error) {
    console.error('Error analyzing UPI risk:', error);
    // Return a default response to prevent app crash
    return {
      upiId,
      riskPercentage: 50, // Default medium risk
      riskLevel: 'Medium',
      reports: 0,
      age: 'Unknown',
      reportedFor: 'Could not verify'
    };
  }
}

// Mock device fingerprinting function - In production, use device-specific identifiers
export function getDeviceFingerprint(): string {
  // In a real app, this would use device-specific information
  // For example, using a combination of navigator.userAgent, screen resolution, etc.
  return `device_${Date.now()}`;
}

// Function to determine if a transaction should be blocked based on risk level
export function shouldBlockTransaction(riskPercentage: number): boolean {
  return riskPercentage >= 75; // Block if risk is 75% or more
}

// Function to check if a warning should be shown
export function shouldShowWarning(riskPercentage: number): boolean {
  return riskPercentage >= 30 && riskPercentage < 75; // Warning for 30-75% risk
}

export interface FraudDetectionResponse {
  prediction: boolean;
  confidence: number;
  features: {
    hourly_reports: number;
    tx_frequency: number;
    amount_deviation: number;
    device_risk: number;
    platform_reports: number;
  };
  live_data: {
    tx_frequency: number;
    avg_amount: number;
    device_mismatches: number;
    recent_reports: number;
  };
  message: string;
  meta: { 
    service: string;
    version: string;
    latency_ms: number;
  };
}

// Advanced ML-based fraud detection
export async function detectAdvancedFraud(
  upiId: string, 
  amount: number, 
  deviceInfo?: any
): Promise<FraudDetectionResponse> {
  try {
    const res = await apiRequest('POST', '/api/fraud-check', {
      upiId,
      amount,
      deviceInfo: deviceInfo || {
        fingerprint: getDeviceFingerprint(),
        timestamp: Date.now()
      }
    });
    
    return await res.json();
  } catch (error) {
    console.error('Advanced fraud detection error:', error);
    throw error;
  }
}