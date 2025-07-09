/**
 * Security education and personalized user tips
 */

import { getSecurityTip } from './enhanced-fraud-detection';

export interface SecurityTip {
  id: string;
  text: string;
  category: 'general' | 'fraud' | 'personal' | 'technical';
  isRead: boolean;
}

export interface UserActivity {
  recentTransactions: {
    amount: number;
    recipient: string;
    timestamp: Date;
  }[];
  loginFrequency: number;
  usesMultiFactorAuth: boolean;
  previousScamReports: number;
  paymentPatterns: {
    frequentRecipients: string[];
    averageAmount: number;
    unusualActivities: string[];
  };
}

/**
 * Generate personalized security tip based on user activity
 */
export async function generatePersonalizedTip(userActivity: UserActivity): Promise<SecurityTip> {
  try {
    const tipText = await getSecurityTip(userActivity);
    
    // Generate a unique ID for the tip
    const id = Date.now().toString(36) + Math.random().toString(36).substring(2);
    
    // Determine the category based on user activity
    let category: 'general' | 'fraud' | 'personal' | 'technical' = 'general';
    
    if (userActivity.previousScamReports > 0) {
      category = 'fraud';
    } else if (userActivity.usesMultiFactorAuth === false) {
      category = 'technical';
    } else if (userActivity.paymentPatterns.unusualActivities.length > 0) {
      category = 'personal';
    }
    
    return {
      id,
      text: tipText,
      category,
      isRead: false
    };
  } catch (error) {
    console.error('Error generating personalized security tip:', error);
    
    // Fallback tip
    return {
      id: Date.now().toString(36),
      text: "Regularly check your transaction history for any unauthorized payments and immediately report suspicious activity.",
      category: 'general',
      isRead: false
    };
  }
}

/**
 * Store tips in local storage
 */
export function saveTip(tip: SecurityTip): void {
  try {
    // Get existing tips
    const tipsJson = localStorage.getItem('security_tips');
    const tips: SecurityTip[] = tipsJson ? JSON.parse(tipsJson) : [];
    
    // Add new tip
    tips.push(tip);
    
    // Save back to local storage
    localStorage.setItem('security_tips', JSON.stringify(tips));
  } catch (error) {
    console.error('Error saving security tip:', error);
  }
}

/**
 * Get all saved tips
 */
export function getSavedTips(): SecurityTip[] {
  try {
    const tipsJson = localStorage.getItem('security_tips');
    return tipsJson ? JSON.parse(tipsJson) : [];
  } catch (error) {
    console.error('Error getting saved security tips:', error);
    return [];
  }
}

/**
 * Mark a tip as read
 */
export function markTipAsRead(tipId: string): void {
  try {
    // Get existing tips
    const tipsJson = localStorage.getItem('security_tips');
    const tips: SecurityTip[] = tipsJson ? JSON.parse(tipsJson) : [];
    
    // Find and update tip
    const updatedTips = tips.map(tip => 
      tip.id === tipId ? { ...tip, isRead: true } : tip
    );
    
    // Save back to local storage
    localStorage.setItem('security_tips', JSON.stringify(updatedTips));
  } catch (error) {
    console.error('Error marking security tip as read:', error);
  }
}

/**
 * Get static security education content for different topics
 */
export function getSecurityEducationContent(topic: string): { title: string, content: string } {
  const educationContent = {
    'upi_basics': {
      title: 'UPI Payment Basics',
      content: `
        Unified Payments Interface (UPI) is a real-time payment system in India that enables instant money transfers between bank accounts.
        
        Key things to remember:
        • UPI IDs are always in the format username@provider (e.g., yourname@okaxis)
        • You never need to share your UPI PIN to receive money
        • Payment requests can be accepted or rejected - never automatically deduct funds
        • Always verify recipient details before confirming a payment
      `
    },
    'common_scams': {
      title: 'Common UPI Scams to Watch For',
      content: `
        Be alert for these common UPI scams:
        
        • QR Code Scams: Fraudsters ask you to scan a QR code to "receive" money, but it's actually a payment request
        • Refund Scams: False promises of refunds that require you to pay a small processing fee first
        • Fake Customer Service: Impersonators claim to be from banks or services requesting sensitive information
        • Remote Access Scams: Asking to install apps that give scammers control of your device
        • UPI PIN Reset: Messages claiming your UPI service needs PIN verification
      `
    },
    'safe_practices': {
      title: 'Safe UPI Practices',
      content: `
        Follow these practices for secure UPI transactions:
        
        • Never share your UPI PIN, OTP, or bank details with anyone
        • Double-check recipient details before confirming payment
        • Be suspicious of unexpected payment requests
        • Use biometric authentication when available
        • Keep your UPI app updated to the latest version
        • Check transaction history regularly for unauthorized payments
        • Register your complaints at cybercrime.gov.in or call 1930 for cyber fraud
      `
    },
    'fake_upi': {
      title: 'How to Spot Fake UPI IDs',
      content: `
        Watch for these signs of fake or fraudulent UPI IDs:
        
        • Spelling variations of known handles (e.g., okicic instead of okicici)
        • Unusual provider names not associated with major banks or payment services
        • Numbers or special characters added to mimic legitimate IDs
        • IDs that don't follow the standard username@provider format
        • Recently created UPI IDs with no transaction history
        • IDs associated with newly registered phone numbers
      `
    }
  };
  
  // Return the requested topic or defaults to UPI basics
  return educationContent[topic as keyof typeof educationContent] || educationContent.upi_basics;
}

/**
 * Generate security quiz questions for user education
 */
export function getSecurityQuizQuestions(): Array<{
  question: string;
  options: string[];
  correctIndex: number;
  explanation: string;
}> {
  return [
    {
      question: "What should you do if someone asks you to scan a QR code to receive money?",
      options: [
        "Scan it immediately to get your money",
        "Ask them to send you a payment request instead",
        "Share your UPI PIN to speed up the process",
        "Give them your phone to let them set it up"
      ],
      correctIndex: 1,
      explanation: "When receiving money via UPI, you should never need to scan a QR code. QR codes are for making payments, not receiving them. Ask the sender to use your UPI ID to send a direct payment instead."
    },
    {
      question: "Which of these is a sign of a potential UPI scam?",
      options: [
        "Receiving a payment notification from your bank",
        "Someone asking for your UPI PIN to send you money",
        "A friend requesting money with a proper description",
        "A payment confirmation SMS from your registered bank number"
      ],
      correctIndex: 1,
      explanation: "You should never share your UPI PIN with anyone, even if they claim it's needed to send you money. Your UPI PIN is only required when you make a payment, not when you receive funds."
    },
    {
      question: "What is the correct format for a UPI ID?",
      options: [
        "username#bankname",
        "bankname@username",
        "username@provider",
        "provider/username"
      ],
      correctIndex: 2,
      explanation: "The correct UPI ID format is username@provider, such as yourname@okaxis or yourname@ybl. The provider part represents your payment service provider (PSP) like banks or payment apps."
    },
    {
      question: "If you realize you've been scammed through UPI, what should you do first?",
      options: [
        "Wait and see if the money returns automatically",
        "Call the cybercrime helpline 1930 immediately",
        "Share the scammer's details on social media",
        "Try to contact the scammer to negotiate"
      ],
      correctIndex: 1,
      explanation: "You should immediately report any UPI fraud by calling the dedicated cybercrime helpline 1930 or filing a complaint on cybercrime.gov.in. Quick reporting increases the chances of recovering your money."
    },
    {
      question: "Which of these is NOT a secure practice when using UPI?",
      options: [
        "Using biometric authentication when available",
        "Saving your UPI PIN in a note-taking app for easy access",
        "Verifying recipient details before payment",
        "Regularly checking your transaction history"
      ],
      correctIndex: 1,
      explanation: "Never save your UPI PIN in digital notes, messaging apps, or anywhere accessible on your device. Your PIN should only be memorized and entered directly in the UPI app when making a transaction."
    }
  ];
}