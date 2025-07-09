/**
 * QR code analysis for detecting tampering and manipulation
 */

import { analyzeQRCode } from './enhanced-fraud-detection';

export interface QRCodeAnalysisResult {
  isTampered: boolean;
  confidence: number;
  observations: string[];
  recommendation: string;
}

/**
 * Convert an image element to base64 for AI analysis
 */
export function imageToBase64(img: HTMLImageElement): string {
  const canvas = document.createElement('canvas');
  canvas.width = img.width;
  canvas.height = img.height;
  
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Failed to get canvas context');
  }
  
  ctx.drawImage(img, 0, 0);
  
  // Get base64 string without the data URL prefix
  const dataUrl = canvas.toDataURL('image/jpeg');
  return dataUrl.replace(/^data:image\/(png|jpeg);base64,/, '');
}

/**
 * Convert a video frame to base64 for AI analysis
 */
export function videoFrameToBase64(video: HTMLVideoElement): string {
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Failed to get canvas context');
  }
  
  ctx.drawImage(video, 0, 0);
  
  // Get base64 string without the data URL prefix
  const dataUrl = canvas.toDataURL('image/jpeg');
  return dataUrl.replace(/^data:image\/(png|jpeg);base64,/, '');
}

/**
 * Analyze a QR code image for signs of tampering
 * @param imageData Base64 encoded image data
 */
export async function analyzeQRCodeForTampering(imageData: string): Promise<QRCodeAnalysisResult> {
  try {
    const analysis = await analyzeQRCode(imageData);
    
    return {
      isTampered: analysis.is_tampered,
      confidence: analysis.confidence,
      observations: analysis.observations || [],
      recommendation: analysis.recommendation || 'Please verify the QR code source before proceeding with payment.'
    };
  } catch (error) {
    console.error('Error analyzing QR code for tampering:', error);
    return {
      isTampered: false,
      confidence: 0,
      observations: ['Error analyzing QR code'],
      recommendation: 'Unable to analyze QR code. Please verify the source before proceeding.'
    };
  }
}

/**
 * Capture and analyze the current frame from a video element
 * @param videoElement HTML video element to capture from
 */
export async function analyzeVideoFrameQRCode(videoElement: HTMLVideoElement): Promise<QRCodeAnalysisResult> {
  try {
    // Capture the current frame as base64
    const base64Image = videoFrameToBase64(videoElement);
    
    // Analyze the image
    return await analyzeQRCodeForTampering(base64Image);
  } catch (error) {
    console.error('Error capturing and analyzing video frame:', error);
    return {
      isTampered: false,
      confidence: 0,
      observations: ['Error capturing video frame'],
      recommendation: 'Unable to analyze the video frame. Please verify the QR code source before proceeding.'
    };
  }
}

/**
 * Analyze QR code from an uploaded image file
 * @param file Image file containing QR code
 */
export async function analyzeQRCodeFromFile(file: File): Promise<QRCodeAnalysisResult> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    
    reader.onload = async (e) => {
      try {
        const base64Data = (e.target?.result as string)?.split(',')[1];
        if (!base64Data) {
          throw new Error('Failed to read image file');
        }
        
        const result = await analyzeQRCodeForTampering(base64Data);
        resolve(result);
      } catch (error) {
        console.error('Error processing uploaded QR code image:', error);
        reject(error);
      }
    };
    
    reader.onerror = (e) => {
      console.error('Error reading file:', e);
      reject(new Error('Failed to read image file'));
    };
    
    reader.readAsDataURL(file);
  });
}