/**
 * Voice analysis capabilities for detecting scams in voice messages/calls
 */

import { ScamType } from './scam-detection';
import { analyzeVoice, transcribeAudio } from './enhanced-fraud-detection';

export interface VoiceAnalysisResult {
  isScam: boolean;
  confidence: number;
  transcript: string;
  scamIndicators: string[];
  scamType?: ScamType;
  recommendation: string;
}

/**
 * Process an audio file to detect potential scams
 * 1. Transcribes the audio using OpenAI's Whisper
 * 2. Analyzes the transcript for scam indicators
 */
export async function analyzeAudioForScams(audioFile: File): Promise<VoiceAnalysisResult> {
  try {
    // First transcribe the audio
    const transcript = await transcribeAudio(audioFile);
    
    if (!transcript) {
      throw new Error('Failed to transcribe audio');
    }
    
    // Then analyze the transcript for scam indicators
    return await analyzeTranscriptForScams(transcript);
  } catch (error) {
    console.error('Error analyzing audio for scams:', error);
    return {
      isScam: false,
      confidence: 0,
      transcript: '',
      scamIndicators: ['Error processing audio file'],
      recommendation: 'Unable to analyze audio. Please try again with a clearer recording.'
    };
  }
}

/**
 * Analyze a voice transcript for potential scams
 */
export async function analyzeTranscriptForScams(transcript: string): Promise<VoiceAnalysisResult> {
  try {
    // Use AI-powered analysis of the transcript
    const analysis = await analyzeVoice(transcript);
    
    // Extract the most likely scam type if identified
    let scamType: ScamType | undefined = undefined;
    if (analysis.scam_type) {
      switch(analysis.scam_type.toLowerCase()) {
        case 'fake products':
        case 'product':
        case 'shopping':
          scamType = ScamType.Unknown; // We don't have a specific "Product" scam type
          break;
        case 'phishing':
        case 'impersonation':
        case 'identity theft':
          scamType = ScamType.Phishing;
          break;
        case 'fraud':
        case 'financial':
        case 'investment':
          scamType = ScamType.Banking; // Banking is closest to financial fraud
          break;
        default:
          scamType = ScamType.Unknown;
      }
    }
    
    return {
      isScam: analysis.is_scam,
      confidence: analysis.confidence,
      transcript,
      scamIndicators: analysis.scam_indicators || [],
      scamType,
      recommendation: analysis.recommendation || 'Please review the conversation carefully before proceeding with any payment.'
    };
  } catch (error) {
    console.error('Error analyzing transcript for scams:', error);
    
    // Provide a basic analysis with common red flags for scams
    const commonScamPhrases = [
      'urgent', 'emergency', 'act now', 'limited time', 'secret', 
      'government', 'bank account', 'verify your details', 'suspicious activity',
      'prize', 'lottery', 'won', 'investment opportunity', 'double your money'
    ];
    
    // Simple analysis - check for common scam phrases
    const lowercaseTranscript = transcript.toLowerCase();
    const foundIndicators = commonScamPhrases.filter(phrase => 
      lowercaseTranscript.includes(phrase.toLowerCase())
    );
    
    const isLikelyScam = foundIndicators.length > 2;
    
    return {
      isScam: isLikelyScam,
      confidence: isLikelyScam ? 0.7 : 0.3,
      transcript,
      scamIndicators: foundIndicators,
      recommendation: isLikelyScam 
        ? 'This conversation contains several phrases commonly used in scams. Proceed with extreme caution.'
        : 'Please review the conversation carefully before proceeding with any payment.'
    };
  }
}

/**
 * Record audio for scam analysis
 */
export function startVoiceRecording(): Promise<MediaRecorder> {
  return new Promise((resolve, reject) => {
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(stream => {
        const mediaRecorder = new MediaRecorder(stream);
        resolve(mediaRecorder);
      })
      .catch(err => {
        console.error('Error accessing microphone:', err);
        reject(err);
      });
  });
}

/**
 * Extract the audio data from a MediaRecorder
 */
export function getAudioFromRecorder(mediaRecorder: MediaRecorder): Promise<Blob> {
  return new Promise((resolve) => {
    const audioChunks: BlobPart[] = [];
    
    mediaRecorder.addEventListener('dataavailable', (event) => {
      audioChunks.push(event.data);
    });
    
    mediaRecorder.addEventListener('stop', () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/mp3' });
      resolve(audioBlob);
    });
  });
}

/**
 * Create a File object from a Blob for upload
 */
export function createAudioFile(blob: Blob): File {
  return new File([blob], 'voice-recording.mp3', { type: 'audio/mp3' });
}