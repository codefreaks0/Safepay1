import React, { useState, useEffect } from 'react';
import { useLocation } from 'wouter';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useToast } from '@/hooks/use-toast';
import { ArrowLeft, CheckCircle } from 'lucide-react';

export default function Checkout() {
  const [location, setLocation] = useLocation();
  const [paymentDetails, setPaymentDetails] = useState<{
    upiId: string;
    amount: string;
    note: string;
    app: string;
    name: string;
  }>({
    upiId: '',
    amount: '',
    note: '',
    app: '',
    name: ''
  });
  const { toast } = useToast();

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    setPaymentDetails({
      upiId: params.get('upiId') || '',
      amount: params.get('amount') || '',
      note: params.get('note') || '',
      app: params.get('app') || '',
      name: params.get('name') || ''
    });
  }, []);

  const handleConfirm = () => {
    // Generate a unique transaction ID
    const transactionId = `TXN-${Math.random().toString(36).substring(2, 10).toUpperCase()}`;
    
    // Format current date
    const formattedDate = new Date().toLocaleString('en-US', {
      day: 'numeric',
      month: 'short',
      year: 'numeric'
    });
    
    // Create complete payment details
    const completePaymentDetails = {
      ...paymentDetails,
      timestamp: new Date().toISOString(),
      transactionId,
      date: formattedDate,
      fromUpiId: 'guest@upi'
    };
    
    // Store complete payment details in session storage for the success screen
    sessionStorage.setItem('lastPayment', JSON.stringify(completePaymentDetails));
    
    // Navigate to success screen
    setLocation('/success');
  };

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-md mx-auto">
        <button
          onClick={() => setLocation('/payment')}
          className="flex items-center text-gray-600 mb-6"
        >
          <ArrowLeft className="h-5 w-5 mr-2" />
          Back
        </button>

        <Card>
          <CardHeader>
            <CardTitle>Confirm Payment</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Amount</span>
                  <span className="font-medium">₹{paymentDetails.amount}</span>
                </div>
                
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">To</span>
                  <span className="font-medium">{paymentDetails.name || paymentDetails.upiId}</span>
                </div>
                
                {paymentDetails.note && (
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Note</span>
                    <span className="font-medium">{paymentDetails.note}</span>
                  </div>
                )}
                
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Payment App</span>
                  <span className="font-medium capitalize">{paymentDetails.app}</span>
                </div>
              </div>

              <div className="pt-4">
                <Button
                  onClick={handleConfirm}
                  className="w-full"
                >
                  Confirm Payment
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}