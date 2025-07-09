import React from 'react';
import { EnhancedQRScanner } from '@/components/scanner/enhanced-qr-scanner';
import { useLocation } from 'wouter';

export default function QRScan() {
  const [, setLocation] = useLocation();

  const handleScan = (qrData: string) => {
    try {
      console.log('QR code scanned:', qrData);
      sessionStorage.setItem('lastScannedQR', qrData);
      setLocation('/scan?qrData=' + encodeURIComponent(qrData));
    } catch (error) {
      console.error('Error handling QR scan:', error);
    }
  };

  const handleClose = () => {
    setLocation('/'); // Navigate to home or any other appropriate route
  };

  return (
    <div className="h-screen w-full flex flex-col">
      <EnhancedQRScanner onScan={handleScan} onClose={handleClose} />
    </div>
  );
}