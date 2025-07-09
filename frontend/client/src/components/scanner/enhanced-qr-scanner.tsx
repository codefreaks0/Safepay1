import React, { useState, useRef, useEffect, useImperativeHandle, forwardRef } from 'react';
import { cn } from '@/lib/utils';
import { ArrowLeft, Camera, Check, Flashlight, X, ShieldAlert, ShieldCheck, Keyboard, Type } from 'lucide-react';
import jsQR from 'jsqr';
import { BrowserMultiFormatReader, DecodeHintType, BarcodeFormat } from '@zxing/library';
import { analyzeQRWithML, extractUPIPaymentInfo, QRScanResult } from '@/lib/ml-qr-scanner';
import { analyzeQRWithOptimizedML } from '@/lib/enhanced-optimized-qr-scanner';
import { analyzeQRWithAdvancedML } from '@/lib/advanced-qr-scanner';

// Define the handle type that will be exposed to parent components
export interface QRScannerHandle {
  stopCamera: () => void;
}

interface QRScannerProps {
  onScan: (data: string) => void;
  onClose: () => void;
  className?: string;
}

export const EnhancedQRScanner = forwardRef<QRScannerHandle, QRScannerProps>(
  function EnhancedQRScanner({ onScan, onClose, className }, ref) {
  const [hasFlash, setHasFlash] = useState(false);
  const [flashOn, setFlashOn] = useState(false);
  const [scanError, setScanError] = useState<string | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [scanProgress, setScanProgress] = useState(0);
  const [scanComplete, setScanComplete] = useState(false);
  const [manualEntry, setManualEntry] = useState(false);
  const [manualUpiId, setManualUpiId] = useState('');
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const codeReaderRef = useRef<BrowserMultiFormatReader | null>(null);
  const animationFrameId = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [showUploadButton, setShowUploadButton] = useState(true);
  const [scanSessionId, setScanSessionId] = useState(0);
  const uploadTimerRef = useRef<NodeJS.Timeout | null>(null);
  const uploadButtonTimerStarted = useRef(false);

  useEffect(() => {
    if (!manualEntry && !scanComplete) {
      setScanSessionId((id) => id + 1);
    }
  }, [manualEntry, scanComplete]);

  useEffect(() => {
    if (!manualEntry && !scanComplete && !uploadButtonTimerStarted.current) {
      setShowUploadButton(true);
      uploadButtonTimerStarted.current = true;
      if (uploadTimerRef.current) clearTimeout(uploadTimerRef.current);
      uploadTimerRef.current = setTimeout(() => {
        setShowUploadButton(false);
      }, 15000);
    }
    if (scanComplete || manualEntry) {
      setShowUploadButton(false);
      if (uploadTimerRef.current) clearTimeout(uploadTimerRef.current);
      uploadButtonTimerStarted.current = false;
    }
    return () => {
      if (uploadTimerRef.current) clearTimeout(uploadTimerRef.current);
    };
  }, [manualEntry, scanComplete]);

  const stopCamera = () => {
    console.log('Stopping camera explicitly');
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => {
        console.log('Stopping track:', track.kind);
        track.stop();
      });
      streamRef.current = null;
    }
    
    if (codeReaderRef.current) {
      codeReaderRef.current.reset();
    }
    
    if (animationFrameId.current) {
      cancelAnimationFrame(animationFrameId.current);
      animationFrameId.current = null;
    }
  };

  useImperativeHandle(ref, () => ({
    stopCamera
  }));

  const scanStartTime = Date.now();

  useEffect(() => {
    const hints = new Map();
    hints.set(DecodeHintType.POSSIBLE_FORMATS, [BarcodeFormat.QR_CODE]);
    
    const codeReader = new BrowserMultiFormatReader(hints);
    codeReaderRef.current = codeReader;

    return () => {
      if (codeReaderRef.current) {
        codeReaderRef.current.reset();
      }
    };
  }, []);
  
  useEffect(() => {
    const setupCamera = async () => {
      try {
        if (!videoRef.current) return;

        const constraints = {
          video: { 
            facingMode: 'environment',
            width: { ideal: 1280 },
            height: { ideal: 720 },
          }
        };

        if (videoRef.current.srcObject) {
          const oldStream = videoRef.current.srcObject as MediaStream;
          oldStream.getTracks().forEach(track => track.stop());
        }

        const newStream = await navigator.mediaDevices.getUserMedia(constraints);
        streamRef.current = newStream;
        
        if (videoRef.current) {
          videoRef.current.srcObject = newStream;
          
          const tracks = newStream.getVideoTracks();
          if (tracks.length > 0) {
            const capabilities = tracks[0].getCapabilities();
            setHasFlash('torch' in capabilities);
          }
        }
      } catch (error) {
        console.error('Error accessing camera:', error);
        setScanError('Camera access error. Please check camera permissions.');
      }
    };
    
    setupCamera();
    
    return () => {
      stopCamera();
    };
  }, []);
  
  const toggleFlash = async () => {
    if (!videoRef.current || !hasFlash) return;
    
    const stream = videoRef.current.srcObject as MediaStream;
    const tracks = stream.getVideoTracks();
    
    if (tracks.length > 0) {
      const track = tracks[0];
      const newFlashState = !flashOn;
      
      try {
        await track.applyConstraints({
          advanced: [{ torch: newFlashState }]
        });
        
        setFlashOn(newFlashState);
      } catch (error) {
        console.error('Error toggling flash:', error);
      }
    }
  };
  
  const [mlScanResult, setMlScanResult] = useState<QRScanResult | null>(null);
  
  const processQrCode = async (qrData: string) => {
    console.log('QR code detected:', qrData);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    setScanProgress(70);
    
    const extractedInfo = extractUPIPaymentInfo(qrData);
    
    let paymentInfo = {
      upi_id: '',
      name: '',
      amount: '',
      currency: 'INR',
      ml_risk_score: 0,
      ml_risk_level: 'Low' as 'Low' | 'Medium' | 'High',
      ml_recommendation: 'Allow' as 'Allow' | 'Verify' | 'Block'
    };
    
    if (extractedInfo && extractedInfo.valid) {
      paymentInfo = {
        ...paymentInfo,
        upi_id: extractedInfo.upiId,
        name: extractedInfo.name || '',
        amount: extractedInfo.amount || '',
        currency: extractedInfo.currency || 'INR'
      };
      
      console.log('Extracted UPI payment info:', paymentInfo);
    } else if (qrData.includes('@')) {
      paymentInfo.upi_id = qrData;
      
      const merchantFromUpi = qrData.split('@')[0];
      if (merchantFromUpi) {
        const formattedName = merchantFromUpi
          .replace(/([A-Z])/g, ' $1')
          .replace(/_/g, ' ')
          .replace(/^\w/, (c) => c.toUpperCase())
          .trim();
          
        paymentInfo.name = formattedName;
      }
      
      console.log('Found direct UPI ID:', qrData, 'with merchant name:', paymentInfo.name);
    } else {
      const match = qrData.match(/([a-zA-Z0-9\.\_\-]+@[a-zA-Z0-9]+)/);
      if (match && match[1]) {
        paymentInfo.upi_id = match[1];
        
        const merchantFromUpi = match[1].split('@')[0];
        if (merchantFromUpi) {
          const formattedName = merchantFromUpi
            .replace(/([A-Z])/g, ' $1')
            .replace(/_/g, ' ')
            .replace(/^\w/, (c) => c.toUpperCase())
            .trim();
            
          paymentInfo.name = formattedName;
        }
        
        console.log('Extracted UPI ID from text:', match[1], 'with merchant name:', paymentInfo.name);
      } else {
        const secondTry = qrData.match(/([^\s\/]+@[^\s\/]+)/);
        if (secondTry && secondTry[1] && secondTry[1].includes('@')) {
          paymentInfo.upi_id = secondTry[1];
          
          const merchantFromUpi = secondTry[1].split('@')[0];
          if (merchantFromUpi) {
            const formattedName = merchantFromUpi
              .replace(/([A-Z])/g, ' $1')
              .replace(/_/g, ' ')
              .replace(/^\w/, (c) => c.toUpperCase())
              .trim();
              
            paymentInfo.name = formattedName;
          }
          
          console.log('Extracted UPI ID with lenient pattern:', secondTry[1], 'with merchant name:', paymentInfo.name);
        } else {
          console.log('No UPI pattern found, using raw data:', qrData);
          
          const cleaned = qrData.trim().replace(/\s+/g, '');
          if (cleaned.length > 0) {
            paymentInfo.upi_id = cleaned;
            if (cleaned.includes('@')) {
              const possibleName = cleaned.split('@')[0];
              if (possibleName) {
                paymentInfo.name = possibleName
                  .replace(/([A-Z])/g, ' $1')
                  .replace(/_/g, ' ')
                  .replace(/^\w/, (c) => c.toUpperCase())
                  .trim();
              }
            }
          } else {
            paymentInfo.upi_id = 'unknown';
            
            setScanError('Could not detect a valid UPI ID. Try manual entry.');
          }
        }
      }
    }
    
    try {
      setScanProgress(85);
      console.log('Analyzing QR code with ML service...');
      
      let mlResult;
      
      try {
        mlResult = await analyzeQRWithAdvancedML(qrData);
        console.log('Advanced ML analysis result:', mlResult);
        console.log('Using advanced ML analysis result');
      } catch (advancedError) {
        console.warn('Advanced ML analysis failed, falling back to optimized scanner:', advancedError);
        
        mlResult = await analyzeQRWithOptimizedML(qrData);
        console.log('Fallback ML analysis result:', mlResult);
      }
      
      setMlScanResult(mlResult);
      
      paymentInfo.ml_risk_score = mlResult.risk_score;
      paymentInfo.ml_risk_level = mlResult.risk_level;
      paymentInfo.ml_recommendation = mlResult.recommendation;
      
      setScanComplete(true);
      setScanProgress(100);
      
      setTimeout(() => {
        const audio = new Audio('/sounds/qr-success.mp3');
        audio.play().catch(err => console.log('Audio play error', err));
        
        onScan(JSON.stringify(paymentInfo));
      }, 800);
      
    } catch (error) {
      console.error('Error analyzing QR code with ML:', error);
      
      setScanComplete(true);
      setScanProgress(100);
      
      setTimeout(() => {
        const audio = new Audio('/sounds/qr-success.mp3');
        audio.play().catch(err => console.log('Audio play error', err));
        
        onScan(JSON.stringify(paymentInfo));
      }, 800);
    }
  };

  const startZxingDetection = () => {
    if (!codeReaderRef.current || !videoRef.current || scanComplete) return;
    
    const videoElement = videoRef.current;
    
    try {
      setIsScanning(true);
      
      codeReaderRef.current.decodeFromVideoDevice(
        null,
        videoElement,
        (result, error) => {
          if (result) {
            processQrCode(result.getText());
            
            codeReaderRef.current?.reset();
          }
          
          if (error && !(error instanceof TypeError)) {
            console.error('ZXing error:', error);
          }

          if (!scanComplete) {
            setScanProgress(prev => {
              const fluctuation = Math.sin(Date.now() / 300) * 5;
              return Math.min(Math.max(50 + fluctuation, 40), 60);
            });
          }
        }
      );
    } catch (err) {
      console.error('Error starting ZXing detection:', err);
      
      startJsQrDetection();
    }
  };
  
  const startJsQrDetection = () => {
    if (scanComplete || !videoRef.current || !canvasRef.current) return;
    
    const detectQRCode = () => {
      if (scanComplete) {
        if (animationFrameId.current) {
          cancelAnimationFrame(animationFrameId.current);
        }
        return;
      }
      
      const video = videoRef.current;
      const canvas = canvasRef.current;
      
      if (!video || !canvas) return;
      
      const width = video.videoWidth;
      const height = video.videoHeight;
      
      if (width === 0 || height === 0) {
        animationFrameId.current = requestAnimationFrame(detectQRCode);
        return;
      }
      
      canvas.width = width;
      canvas.height = height;
      
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        console.error('Could not get canvas context');
        return;
      }
      
      ctx.drawImage(video, 0, 0, width, height);
      
      try {
        const imageData = ctx.getImageData(0, 0, width, height);
        
        const code = jsQR(imageData.data, imageData.width, imageData.height, {
          inversionAttempts: "dontInvert",
        });
        
        if (code) {
          processQrCode(code.data);
          return;
        }
        
        setScanProgress(prev => {
          const fluctuation = Math.sin(Date.now() / 300) * 5;
          return Math.min(Math.max(50 + fluctuation, 40), 60);
        });
        
        if (Date.now() - scanStartTime > 10000 && !scanComplete) {
          setScanError('QR code not detected. Please try again or use manual entry.');
          setIsScanning(false);
          return;
        }
        
        animationFrameId.current = requestAnimationFrame(detectQRCode);
      } catch (err) {
        console.error('jsQR scanning error:', err);
        
        setScanError('Error analyzing camera feed');
        
        setTimeout(() => {
          animationFrameId.current = requestAnimationFrame(detectQRCode);
        }, 500);
      }
    };
    
    setIsScanning(true);
    detectQRCode();
  };
  
  const handleVideoPlay = () => {
    if (!isScanning) {
      setIsScanning(true);
      
      startZxingDetection();
    }
  };
  
  const handleManualEntry = async () => {
    const upiPattern = /^[\w.-]+@[\w]+$/;
    
    if (!manualUpiId.trim()) {
      setScanError('Please enter a UPI ID');
      return;
    }
    
    let processedUpiId = manualUpiId;
    if (!processedUpiId.includes('@')) {
      processedUpiId += '@okaxis';
    }
    
    setScanProgress(75);
    
    let paymentInfo = {
      upi_id: processedUpiId,
      name: 'Demo Merchant',
      amount: '100',
      currency: 'INR',
      ml_risk_score: 0,
      ml_risk_level: 'Low' as 'Low' | 'Medium' | 'High',
      ml_recommendation: 'Allow' as 'Allow' | 'Verify' | 'Block'
    };
    
    console.log('Processing manual UPI entry:', processedUpiId);
    
    const upiUrl = `upi://pay?pa=${processedUpiId}&pn=Demo%20Merchant&am=100&cu=INR&tn=Payment`;
    
    try {
      setScanProgress(85);
      console.log('Analyzing manual UPI entry with ML service...');
      
      const mlResult = await analyzeQRWithOptimizedML(upiUrl);
      console.log('ML analysis result for manual entry:', mlResult);
      
      setMlScanResult(mlResult);
      
      paymentInfo.ml_risk_score = mlResult.risk_score;
      paymentInfo.ml_risk_level = mlResult.risk_level;
      paymentInfo.ml_recommendation = mlResult.recommendation;
    } catch (error) {
      console.error('Error analyzing manual UPI entry with ML:', error);
    }
    
    setScanComplete(true);
    setScanProgress(100);
    
    setTimeout(() => {
      onScan(JSON.stringify(paymentInfo));
    }, 500);
  };
  
  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = async (event) => {
      const img = new window.Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(img, 0, 0, img.width, img.height);
          const imageData = ctx.getImageData(0, 0, img.width, img.height);
          const qrCode = jsQR(imageData.data, imageData.width, imageData.height);
          if (qrCode && qrCode.data) {
            processQrCode(qrCode.data);
            setScanError(null);
            return;
          }
          try {
            const codeReader = new BrowserMultiFormatReader();
            codeReader.decodeFromImage(undefined, img.src).then(result => {
              if (result && result.getText()) {
                processQrCode(result.getText());
                setScanError(null);
              } else {
                setScanError('No QR code found in the uploaded image.');
              }
            }).catch(() => {
              setScanError('No QR code found in the uploaded image.');
            });
          } catch (err) {
            setScanError('No QR code found in the uploaded image.');
          }
        }
      };
      if (event.target?.result) {
        img.src = event.target.result as string;
      }
    };
    reader.readAsDataURL(file);
  };

  useEffect(() => {
    return () => {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
        clearTimeout(animationFrameId.current);
      }
      
      if (codeReaderRef.current) {
        codeReaderRef.current.reset();
      }
    };
  }, []);

  return (
    <div className={cn("relative flex flex-col h-screen bg-black", className)}>
      <div className="w-full flex justify-between items-center p-6">
        <button 
          onClick={onClose}
          className="w-10 h-10 bg-black/50 rounded-full flex items-center justify-center"
        >
          <X className="w-6 h-6 text-white" />
        </button>
        <p className="text-white font-medium">
          {manualEntry ? 'Enter UPI ID' : 'Scan QR Code'}
        </p>
        <div className="w-10"></div>
      </div>
      
      {!manualEntry ? (
        <>
          <div className="flex-1 flex items-center justify-center">
            <video 
              ref={videoRef}
              autoPlay 
              playsInline 
              muted 
              className="absolute inset-0 w-full h-full object-cover"
              onPlay={handleVideoPlay}
            />
            
            <div className="border-2 border-white rounded-3xl w-[250px] h-[250px] relative z-10 overflow-hidden flex items-center justify-center">
              <canvas ref={canvasRef} className="hidden" />
              
              {isScanning && !scanComplete && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-black">
                  <div className="w-16 h-16 mb-4">
                    <svg className="w-full h-full animate-pulse" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M3 9V5.25C3 4.00736 4.00736 3 5.25 3H9M9 21H5.25C4.00736 21 3 19.9926 3 18.75V15M21 15V18.75C21 19.9926 19.9926 21 18.75 21H15M15 3H18.75C19.9926 3 21 4.00736 21 5.25V9" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </div>
                  <div className="w-3/4 bg-gray-700 rounded-full h-1.5 mb-2">
                    <div 
                      className="bg-white h-1.5 rounded-full transition-all duration-100 ease-in-out" 
                      style={{ width: `${scanProgress}%` }}
                    />
                  </div>
                  <p className="text-white text-sm">Processing QR code...</p>
                </div>
              )}
              
              {scanComplete && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/30">
                  <div className="w-16 h-16 text-green-500 animate-pulse mb-2">
                    <Check className="w-full h-full" />
                  </div>
                  <p className="text-white font-medium text-center">QR Code Detected!</p>
                  <p className="text-white text-sm mt-1">Starting security verification...</p>
                </div>
              )}
            </div>
          </div>
          
          <div className="w-full p-6 flex flex-col items-center">
            <div className="flex justify-center mb-4">
              {hasFlash && (
                <button 
                  onClick={toggleFlash}
                  className={cn(
                    "w-12 h-12 rounded-full flex items-center justify-center mr-4",
                    flashOn ? "bg-yellow-500" : "bg-white/20"
                  )}
                >
                  <Flashlight className="w-6 h-6 text-white" />
                </button>
              )}
              
              <button 
                onClick={() => setManualEntry(true)}
                className="w-12 h-12 bg-white rounded-full flex items-center justify-center"
              >
                <Keyboard className="w-6 h-6 text-black" />
              </button>
            </div>
            
            <div className="flex flex-col gap-3 w-full">
              {!isScanning && !scanComplete && (
                <button
                  onClick={() => {
                    setScanProgress(0);
                    setScanComplete(false);
                    setScanError(null);
                    
                    setIsScanning(true);
                    
                    startZxingDetection();
                  }}
                  className="mt-4 bg-primary text-white px-6 py-3 rounded-lg text-lg font-medium border-2 border-white shadow-lg animate-pulse"
                >
                  👉 Tap to Scan QR Code
                </button>
              )}
              
              {scanError && (
                <div className="bg-red-500/70 text-white px-4 py-2 rounded-lg text-center mt-2">
                  {scanError} 
                  <button 
                    className="underline ml-2"
                    onClick={() => {
                      setScanError(null);
                      setIsScanning(false);
                    }}
                  >
                    Retry
                  </button>
                </div>
              )}
              
              <button
                onClick={() => {
                  stopCamera();
                  setManualEntry(true);
                }}
                className="bg-white text-primary hover:bg-primary hover:text-white transition-colors flex items-center justify-center gap-2 mt-3 py-3 rounded-lg font-medium w-full border-2 border-white shadow-md"
              >
                <Keyboard className="w-5 h-5" /> Enter UPI ID manually
              </button>
            </div>
          </div>

          {showUploadButton && !scanComplete && !manualEntry && (
            <div className="w-full flex justify-center mt-2 mb-4">
              <input
                type="file"
                accept="image/*"
                ref={fileInputRef}
                style={{ display: 'none' }}
                onChange={handleImageUpload}
              />
              <button
                type="button"
                className="px-4 py-2 bg-blue-600 text-white rounded shadow hover:bg-blue-700 transition"
                onClick={() => fileInputRef.current?.click()}
              >
                Upload QR Image
              </button>
            </div>
          )}
        </>
      ) : (
        <div className="flex-1 flex flex-col p-6">
          <div className="flex-1 flex flex-col items-center justify-center">
            <div className="w-full max-w-md">
              <div className="mb-6">
                <label className="block text-white text-sm font-medium mb-2">Enter UPI ID</label>
                <div className="relative">
                  <input
                    type="text"
                    value={manualUpiId}
                    onChange={(e) => setManualUpiId(e.target.value)}
                    placeholder="username@bank"
                    className="bg-white/10 border border-white/40 text-white w-full p-4 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
                  />
                  {manualUpiId && (
                    <button
                      onClick={() => setManualUpiId('')}
                      className="absolute right-4 top-1/2 transform -translate-y-1/2 text-white/60"
                    >
                      <X size={16} />
                    </button>
                  )}
                </div>
                <p className="text-white/60 text-xs mt-2">
                  Example: johndoe@okicici or 9876543210@paytm
                </p>
                
                {scanError && (
                  <div className="bg-red-500/70 text-white px-4 py-2 rounded-lg text-center mt-2">
                    {scanError}
                  </div>
                )}
              </div>
              
              <button
                onClick={handleManualEntry}
                className="w-full bg-primary text-white px-6 py-4 rounded-lg text-lg font-medium"
              >
                Verify UPI
              </button>
              
              <button
                onClick={() => {
                  setManualEntry(false);
                  setScanError(null);
                }}
                className="text-white flex items-center justify-center gap-2 mt-4"
              >
                <ArrowLeft size={16} />
                Back to QR Scanner
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
});

interface EnhancedQRScannerProps {
  onScan: (qrData: string) => void;
}

export const EnhancedQRScannerMock: React.FC<EnhancedQRScannerProps> = ({ onScan }) => {
  const handleMockScan = () => {
    const mockQRData = "upi://pay?pa=merchant@upi&pn=Merchant&am=100&cu=INR";
    onScan(mockQRData);
  };

  return (
    <div className="scanner-container">
      <button onClick={handleMockScan} className="mock-scan-button">
        Simulate QR Scan
      </button>
    </div>
  );
};