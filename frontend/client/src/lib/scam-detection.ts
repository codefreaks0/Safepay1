/**
 * Enum for different types of scams
 */
export enum ScamType {
  Banking = "Banking Scam",
  Lottery = "Lottery Scam",
  KYC = "KYC Verification Scam",
  Refund = "Refund Scam",
  Phishing = "Phishing Attempt",
  Reward = "Reward Scam",
  Unknown = "Unknown Scam"
}

/**
 * Voice analysis result interface
 */
export interface VoiceAnalysisResult {
  transcript: string;
  isScam: boolean;
  scamIndicators: string[];
  confidence: number;
  scamType?: string;
}

/**
 * Message analysis result interface
 */
export interface MessageAnalysisResult {
  message: string;
  isScam: boolean;
  scamIndicators: string[];
  confidence: number;
  scamType?: string;
}

/**
 * Interface for voice command processing result
 */
export interface VoiceCommandResult {
  command: string;
  action: string;
  risk_score?: number;
  fraud_type?: string;
  is_scam?: boolean;
}

/**
 * Analyze voice transcript for scam indicators
 * @param transcript The voice transcript to analyze
 * @returns Analysis result with scam indicators
 */
export function analyzeVoiceForScam(transcript: string): VoiceAnalysisResult {
  // Common scam phrases and triggers
  const scamPhrases = [
    { phrase: "verify your account", indicator: "Request for account verification without proper context" },
    { phrase: "suspicious activity", indicator: "Creating false urgency by mentioning suspicious activity" },
    { phrase: "immediate", indicator: "Creating a false sense of urgency" },
    { phrase: "upi pin", indicator: "Requesting sensitive UPI PIN" },
    { phrase: "one time password", indicator: "Requesting OTP" },
    { phrase: "your account will be blocked", indicator: "Threatening account blocking" },
    { phrase: "bank account", indicator: "Reference to bank account in suspicious context" },
    { phrase: "calling from your bank", indicator: "Claims to be calling from your bank" },
    { phrase: "security reasons", indicator: "Vague mention of security reasons" },
    { phrase: "limited time", indicator: "Creating time pressure to act quickly" },
    { phrase: "need your password", indicator: "Direct request for password" },
    { phrase: "verify your identity", indicator: "Identity verification without proper authentication" }
  ];

  // Check for scam indicators
  const indicators = scamPhrases
    .filter(item => transcript.toLowerCase().includes(item.phrase.toLowerCase()))
    .map(item => item.indicator);

  // Determine if it's a scam based on number of indicators
  const isScam = indicators.length >= 2;
  
  // Calculate confidence score based on number of indicators
  const confidence = Math.min(0.5 + (indicators.length * 0.1), 0.99);

  // Determine scam type
  let scamType = undefined;
  if (isScam) {
    if (transcript.toLowerCase().includes("bank") || transcript.toLowerCase().includes("account")) {
      scamType = "Banking Scam";
    } else if (transcript.toLowerCase().includes("reward") || transcript.toLowerCase().includes("prize")) {
      scamType = "Reward Scam";
    } else if (transcript.toLowerCase().includes("refund") || transcript.toLowerCase().includes("return")) {
      scamType = "Refund Scam";
    } else {
      scamType = "Phishing Attempt";
    }
  }

  return {
    transcript,
    isScam,
    scamIndicators: indicators,
    confidence,
    scamType
  };
}

/**
 * Process voice command to extract intent and parameters
 * @param command Voice command text
 * @param userId Optional user ID to associate with the voice command
 * @returns Processed command information with risk assessment
 */
export async function processVoiceCommand(command: string, userId?: number): Promise<VoiceCommandResult> {
  try {
    console.log('Attempting to send voice command to backend:', { command, userId });
    // Make a call to our backend API endpoint
    const response = await fetch('/api/process-voice', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        command,
        userId: userId || undefined,
        language: 'en-US'
      })
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`Server responded with ${response.status}: ${response.statusText}`, errorText);
      throw new Error(`Server responded with ${response.status}: ${response.statusText} - ${errorText}`);
    }

    const result = await response.json();
    console.log('Backend voice analysis successful:', result);
    
    // Transform the server response to match our interface
    return {
      command: result.command,
      action: result.action,
      risk_score: result.risk_score,
      fraud_type: result.scam_type,
      is_scam: result.is_scam
    };
  } catch (error: any) {
    console.error('Error processing voice command:', error);
    // Fallback to local processing in case of API failure
    return {
      command,
      action: inferActionFromCommand(command),
      risk_score: calculateRiskScore(command),
      is_scam: false
    };
  }
}

/**
 * Analyze text message for scam indicators
 * @param message The message text to analyze
 * @returns Analysis result with scam indicators
 */
export function analyzeMessageForScam(message: string): MessageAnalysisResult {
  // Common scam phrases and triggers in messages
  const scamPhrases = [
    { phrase: "verify your account", indicator: "Request for account verification" },
    { phrase: "suspicious activity", indicator: "False urgency about account activity" },
    { phrase: "urgent", indicator: "Creating false sense of urgency" },
    { phrase: "click the link", indicator: "Suspicious link request" },
    { phrase: "login to verify", indicator: "Phishing attempt" },
    { phrase: "your upi pin", indicator: "Request for sensitive UPI PIN" },
    { phrase: "otp", indicator: "Request for OTP" },
    { phrase: "account will be blocked", indicator: "Threatening account blocking" },
    { phrase: "dear customer", indicator: "Generic greeting typical of mass scam attempts" },
    { phrase: "bank account", indicator: "Reference to bank account" },
    { phrase: "kyc", indicator: "KYC verification request" },
    { phrase: "verify details", indicator: "Request to verify personal details" },
    { phrase: "security reasons", indicator: "Vague security references" },
    { phrase: "update information", indicator: "Request to update account information" },
    { phrase: "personal details", indicator: "Request for personal information" },
    { phrase: "click here", indicator: "Suspicious link directive" },
    { phrase: "limited time", indicator: "Time pressure tactics" },
    { phrase: "password", indicator: "Request for password" },
    { phrase: "lottery", indicator: "Lottery or prize scam" },
    { phrase: "won", indicator: "Notification of winning something unexpected" },
    { phrase: "prize", indicator: "Unexpected prize notification" },
    { phrase: "claim your", indicator: "Request to claim something unexpected" }
  ];

  // Check for scam indicators
  const indicators = scamPhrases
    .filter(item => message.toLowerCase().includes(item.phrase.toLowerCase()))
    .map(item => item.indicator);

  // Determine if it's a scam based on number of indicators
  const isScam = indicators.length >= 2;
  
  // Calculate confidence score based on number of indicators
  const confidence = Math.min(0.5 + (indicators.length * 0.1), 0.99);

  // Determine scam type
  let scamType = undefined;
  if (isScam) {
    if (message.toLowerCase().includes("bank") || message.toLowerCase().includes("account")) {
      scamType = "Banking Scam";
    } else if (message.toLowerCase().includes("lottery") || message.toLowerCase().includes("prize") || message.toLowerCase().includes("won")) {
      scamType = "Lottery Scam";
    } else if (message.toLowerCase().includes("kyc") || message.toLowerCase().includes("verify")) {
      scamType = "KYC Verification Scam";
    } else if (message.toLowerCase().includes("refund") || message.toLowerCase().includes("cashback")) {
      scamType = "Refund Scam";
    } else {
      scamType = "Phishing Attempt";
    }
  }

  return {
    message,
    isScam,
    scamIndicators: indicators,
    confidence,
    scamType
  };
}

/**
 * Infer action from voice command
 * @param command Voice command text
 * @returns Inferred action
 */
function inferActionFromCommand(command: string): string {
  const lowerCommand = command.toLowerCase();
  
  if (lowerCommand.includes('send') || lowerCommand.includes('pay') || lowerCommand.includes('transfer')) {
    return 'payment';
  }
  
  if (lowerCommand.includes('check') || lowerCommand.includes('verify')) {
    return 'verification';
  }
  
  if (lowerCommand.includes('history') || lowerCommand.includes('transactions')) {
    return 'history';
  }
  
  if (lowerCommand.includes('help') || lowerCommand.includes('support')) {
    return 'help';
  }
  
  return 'unknown';
}

/**
 * Calculate basic risk score for a command
 * @param command Voice command text
 * @returns Risk score between 0-1
 */
function calculateRiskScore(command: string): number {
  const riskyTerms = ['urgent', 'immediately', 'password', 'verify', 'problem', 'pin', 'otp', 'code', 'block'];
  
  let score = 0;
  const lowerCommand = command.toLowerCase();
  
  riskyTerms.forEach(term => {
    if (lowerCommand.includes(term)) {
      score += 0.1;
    }
  });
  
  return Math.min(score, 1.0);
}