// Test script for voice scam detection integration
const axios = require('axios');

async function testVoiceAnalysis() {
  console.log('🧪 Testing Voice Scam Detection Integration...\n');
  
  const testCases = [
    {
      name: "Banking Scam",
      transcript: "Your account has been suspended due to suspicious activity. Please verify your identity by providing your account number and password."
    },
    {
      name: "Lottery Scam", 
      transcript: "Congratulations! You've won a lottery prize of 10 lakh rupees. To claim your prize, send us your bank details immediately."
    },
    {
      name: "KYC Scam",
      transcript: "Your KYC verification is pending. Your account will be blocked if not updated within 24 hours. Press 1 to update now."
    },
    {
      name: "Legitimate Call",
      transcript: "Hello, this is regarding your recent transaction. Please confirm if you made this purchase."
    },
    {
      name: "Normal Conversation",
      transcript: "Let's meet for lunch tomorrow at 2 PM. How does that sound?"
    }
  ];
  
  for (const testCase of testCases) {
    try {
      console.log(`📝 Testing: ${testCase.name}`);
      console.log(`   Transcript: "${testCase.transcript}"`);
      
      const response = await axios.post('http://localhost:8082/analyze-voice', {
        transcript: testCase.transcript
      });
      
      const result = response.data;
      console.log(`   Result: ${result.is_scam ? '🚨 SCAM DETECTED' : '✅ LEGITIMATE'}`);
      console.log(`   Confidence: ${(result.confidence * 100).toFixed(1)}%`);
      console.log(`   Risk Score: ${result.risk_score?.toFixed(1) || 'N/A'}%`);
      console.log(`   Scam Type: ${result.scam_type || 'None'}`);
      if (result.scam_indicators && result.scam_indicators.length > 0) {
        console.log(`   Indicators: ${result.scam_indicators.join(', ')}`);
      }
      console.log(`   Method: ${result.analysis_method}\n`);
      
    } catch (error) {
      console.error(`❌ Error testing "${testCase.name}":`, error.message);
      console.log('');
    }
  }
  
  console.log('✅ Voice analysis integration test completed!');
}

// Test the backend endpoint directly
async function testBackendEndpoint() {
  console.log('🔗 Testing Backend Endpoint...\n');
  
  try {
    const response = await axios.post('http://localhost:6900/api/process-audio', {
      transcript: "Your account is suspended, verify now."
    }, {
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    console.log('✅ Backend endpoint working!');
    console.log('Response:', JSON.stringify(response.data, null, 2));
    
  } catch (error) {
    console.error('❌ Backend endpoint error:', error.message);
    if (error.response) {
      console.error('Response data:', error.response.data);
    }
  }
}

// Run tests
async function runTests() {
  console.log('🚀 Starting Voice Scam Detection Tests\n');
  
  // Test AI service directly
  await testVoiceAnalysis();
  
  console.log('\n' + '='.repeat(50) + '\n');
  
  // Test backend endpoint
  await testBackendEndpoint();
}

// Run if this script is executed directly
if (require.main === module) {
  runTests().catch(console.error);
}

module.exports = { testVoiceAnalysis, testBackendEndpoint }; 