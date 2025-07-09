import React, { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useToast } from "@/hooks/use-toast";
import { Loader2, Smartphone } from "lucide-react";
import { 
  RadioGroup, 
  RadioGroupItem 
} from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { SafetyVerification } from "./safety-verification";
import { TransactionConfirmation } from "./transaction-confirmation";

// Import Instascan when it's available in the window object
declare global {
  interface Window {
    Instascan: any;
  }
}

interface QRPaymentFlowProps {
  onProcessQR: (qrData: any) => void;
  onCancel: () => void;
}

export function QRPaymentFlow({ onProcessQR, onCancel }: QRPaymentFlowProps) {
  const [scannerActive, setScannerActive] = useState(false);
  const [scanner, setScanner] = useState<any>(null);
  const [paymentData, setPaymentData] = useState<any>(null);
  const [amount, setAmount] = useState("");
  const [selectedApp, setSelectedApp] = useState("gpay");
  const [showAppSelection, setShowAppSelection] = useState(false);
  const [showSafetyVerification, setShowSafetyVerification] = useState(false);
  const [showTransactionConfirmation, setShowTransactionConfirmation] = useState(false);
  const [merchantRiskScore, setMerchantRiskScore] = useState(85); // Default to medium-high safety
  const [isRegisteredBusiness, setIsRegisteredBusiness] = useState(true);
  const { toast } = useToast();

  const initScanner = () => {
    setScannerActive(true);
    
    // Debug log to check what scanner libraries are available
    // Declare global libraries for TypeScript
    const w = window as any;
    
    console.log("Scanner libraries:", {
      Instascan: !!w.Instascan,
      ZXing: typeof w.ZXing !== 'undefined' ? 'Available' : 'Not available',
      jsQR: typeof w.jsQR !== 'undefined' ? 'Available' : 'Not available',
    });
    
    if (!w.Instascan) {
      toast({
        title: "Scanner Not Available",
        description: "QR scanner libraries not loaded. Using demo mode.",
        variant: "destructive",
      });
      
      // Auto switch to mock data if scanner not available
      setTimeout(() => {
        useMockQRData();
      }, 1500);
      
      setScannerActive(false);
      return;
    }
    
    let newScanner = new w.Instascan.Scanner({ 
      video: document.getElementById('scanner'),
      mirror: false
    });

    setScanner(newScanner);

    w.Instascan.Camera.getCameras()
      .then((cameras: any[]) => {
        if (cameras.length > 0) {
          newScanner.start(cameras[0]);
        } else {
          toast({
            title: "Camera Error",
            description: "No cameras found on your device",
            variant: "destructive",
          });
          resetScanner();
        }
      })
      .catch((err: any) => {
        console.error('Camera error:', err);
        toast({
          title: "Camera Error",
          description: "Unable to access camera. Please check permissions.",
          variant: "destructive",
        });
        resetScanner();
      });

    newScanner.addListener('scan', (content: string) => {
      if (validateUPIQr(content)) {
        processValidQR(content);
        newScanner.stop();
      } else {
        toast({
          title: "Invalid QR Code",
          description: "Please scan a valid UPI payment QR code",
          variant: "destructive",
        });
      }
    });
  };

  const validateUPIQr = (content: string): boolean => {
    // Basic UPI QR validation
    return content.startsWith('upi://pay?') || 
           content.startsWith('https://upi://pay?');
  };

  const processValidQR = (content: string) => {
    const params = new URLSearchParams(content.split('?')[1]);
    
    // Store payment details
    const qrData = {
      upiId: params.get('pa') || 'unknown@upi',
      name: params.get('pn') || 'Merchant',
      amount: params.get('am') || ''
    };
    
    setPaymentData(qrData);
    setScannerActive(false);
    
    // Get UPI risk data for this merchant
    fetchUpiRiskData(qrData.upiId);
  };
  
  // Fetch UPI risk data from API
  const fetchUpiRiskData = async (upiId: string) => {
    try {
      const encodedUpiId = encodeURIComponent(upiId);
      const response = await fetch(`/api/upi/check/${encodedUpiId}`);
      
      if (response.ok) {
        const data = await response.json();
        // Update risk score from API
        setMerchantRiskScore(data.riskPercentage || 85);
        // Check if this is a registered business
        setIsRegisteredBusiness(data.status === 'SAFE' || data.status === 'VERIFIED');
      } else {
        // Use default values if API fails
        console.error('Failed to fetch UPI risk data');
      }
    } catch (error) {
      console.error('Error fetching UPI risk data:', error);
    }
  };

  const resetScanner = () => {
    if (scanner) {
      scanner.stop();
    }
    setScannerActive(false);
    setPaymentData(null);
  };

  const handleAmountChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    // Only allow numbers and a single decimal point
    const value = e.target.value.replace(/[^0-9.]/g, '');
    const parts = value.split('.');
    if (parts.length > 2) {
      // Don't allow multiple decimal points
      return;
    }
    setAmount(value);
  };

  // Redirects to UPI payment app with deep link
  const redirectToPaymentApp = () => {
    if (!paymentData || !paymentData.upiId || !amount) {
      toast({
        title: "Invalid Payment",
        description: "Missing UPI ID or amount",
        variant: "destructive",
      });
      return;
    }
    
    // Create UPI deep link
    const upiId = paymentData.upiId;
    const payeeName = paymentData.name || "Merchant";
    const amountValue = amount;
    const transactionNote = "Payment via SafePay";
    
    // Format the UPI link
    const upiLink = `upi://pay?pa=${upiId}&pn=${encodeURIComponent(payeeName)}&am=${amountValue}&tn=${encodeURIComponent(transactionNote)}`;
    
    // Log the link (for debugging)
    console.log("UPI Link:", upiLink);
    
    // Redirect to UPI app
    window.location.href = upiLink;
    
    // Fallback if UPI app not installed
    setTimeout(() => {
      if (!document.hidden) {
        toast({
          title: "No UPI App Found",
          description: `Install ${selectedApp === 'gpay' ? 'Google Pay' : selectedApp === 'phonepe' ? 'PhonePe' : 'Paytm'} to complete this payment`,
          variant: "destructive",
        });
      }
    }, 2000);
  };

  const handleProceed = () => {
    if (!amount || isNaN(parseFloat(amount)) || parseFloat(amount) <= 0) {
      toast({
        title: "Invalid Amount",
        description: "Please enter a valid payment amount",
        variant: "destructive",
      });
      return;
    }

    // Show safety verification first
    setShowSafetyVerification(true);
  };
  
  const handleContinueToTransaction = () => {
    // Show transaction confirmation screen
    setShowSafetyVerification(false);
    setShowTransactionConfirmation(true);
  };
  
  const handleBackToSafety = () => {
    // Go back to safety verification
    setShowTransactionConfirmation(false);
    setShowSafetyVerification(true);
  };
  
  const handleConfirmTransaction = () => {
    // Move to payment app selection
    setShowTransactionConfirmation(false);
    setShowAppSelection(true);
  };
  
  const handlePaymentAppSelected = () => {
    // Redirect to payment app
    redirectToPaymentApp();
    
    // Also notify parent component
    onProcessQR({
      ...paymentData,
      amount,
      selectedApp,
      merchantRiskScore,
      isRegisteredBusiness
    });
  };

  // Mock QR data for testing
  const useMockQRData = () => {
    const mockData = {
      upiId: 'merchant@upi',
      name: 'Test Merchant',
      amount: '100'
    };
    setPaymentData(mockData);
    fetchUpiRiskData(mockData.upiId);
  };

  return (
    <div className="space-y-6">
      {!scannerActive && !paymentData && (
        <div className="text-center space-y-4">
          <Button onClick={initScanner} className="w-full">
            Scan QR Code
          </Button>
          <Button variant="outline" onClick={useMockQRData} className="w-full">
            Use Demo QR
          </Button>
        </div>
      )}

      {scannerActive && (
        <div className="relative aspect-square w-full max-w-md mx-auto">
          <video id="scanner" className="w-full h-full object-cover rounded-lg"></video>
          <div className="absolute inset-0 border-2 border-primary rounded-lg pointer-events-none"></div>
        </div>
      )}

      {paymentData && !showSafetyVerification && !showTransactionConfirmation && !showAppSelection && (
        <Card>
          <CardContent className="pt-6">
            <div className="space-y-4">
              <div>
                <Label>UPI ID</Label>
                <div className="font-medium">{paymentData.upiId}</div>
              </div>
              <div>
                <Label>Merchant Name</Label>
                <div className="font-medium">{paymentData.name}</div>
              </div>
              <div>
                <Label>Amount</Label>
                <Input
                  type="text"
                  value={amount}
                  onChange={handleAmountChange}
                  placeholder="Enter amount"
                />
              </div>
              <Button onClick={handleProceed} className="w-full">
                Continue
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {showSafetyVerification && (
        <SafetyVerification
          merchantRiskScore={merchantRiskScore}
          isRegisteredBusiness={isRegisteredBusiness}
          onContinue={handleContinueToTransaction}
          onBack={resetScanner}
        />
      )}

      {showTransactionConfirmation && (
        <TransactionConfirmation
          paymentData={paymentData}
          amount={amount}
          onConfirm={handleConfirmTransaction}
          onBack={handleBackToSafety}
        />
      )}

      {showAppSelection && (
        <Card>
          <CardContent className="pt-6">
            <div className="space-y-4">
              <Label>Select Payment App</Label>
              <RadioGroup value={selectedApp} onValueChange={setSelectedApp}>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="gpay" id="gpay" />
                  <Label htmlFor="gpay">Google Pay</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="phonepe" id="phonepe" />
                  <Label htmlFor="phonepe">PhonePe</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="paytm" id="paytm" />
                  <Label htmlFor="paytm">Paytm</Label>
                </div>
              </RadioGroup>
              <Button onClick={handlePaymentAppSelected} className="w-full">
                Pay Now
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}