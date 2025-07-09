import React, { useState, useEffect } from 'react';
import { useLocation } from 'wouter';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useToast } from '@/hooks/use-toast';
import { ArrowLeft, CreditCard, Wallet, Smartphone } from 'lucide-react';

export default function Payment() {
  const [location, setLocation] = useLocation();
  const [amount, setAmount] = useState('');
  const [note, setNote] = useState('');
  const [selectedApp, setSelectedApp] = useState<string | null>(null);
  const { toast } = useToast();

  // Extract payment details from URL parameters
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const upiId = params.get('upiId');
    const name = params.get('name');
    const amount = params.get('amount');
    
    if (amount) {
      setAmount(amount);
    }
  }, []);

  const handleAmountChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    // Only allow numbers and decimal point
    if (/^\d*\.?\d*$/.test(value)) {
      setAmount(value);
    }
  };

  const handleNoteChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setNote(e.target.value);
  };

  const handleAppSelect = (app: string) => {
    setSelectedApp(app);
  };

  const handleProceed = () => {
    if (!amount || parseFloat(amount) <= 0) {
      toast({
        title: "Invalid Amount",
        description: "Please enter a valid amount greater than 0",
        variant: "destructive",
      });
      return;
    }

    if (!selectedApp) {
      toast({
        title: "No App Selected",
        description: "Please select a payment app to continue",
        variant: "destructive",
      });
      return;
    }

    // Construct payment URL with all necessary parameters
    const params = new URLSearchParams(window.location.search);
    params.append('amount', amount);
    if (note) params.append('note', note);
    params.append('app', selectedApp);
    params.append('fromUpiId', 'guest@upi'); // Use a fixed guest UPI ID

    // Navigate to checkout page
    setLocation(`/checkout?${params.toString()}`);
  };

  const paymentApps = [
    { id: 'gpay', name: 'Google Pay', icon: <Smartphone className="h-6 w-6" /> },
    { id: 'phonepe', name: 'PhonePe', icon: <Wallet className="h-6 w-6" /> },
    { id: 'paytm', name: 'Paytm', icon: <CreditCard className="h-6 w-6" /> },
  ];

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-md mx-auto">
        <button
          onClick={() => setLocation('/home')}
          className="flex items-center text-gray-600 mb-6"
        >
          <ArrowLeft className="h-5 w-5 mr-2" />
          Back
        </button>

        <Card>
          <CardHeader>
            <CardTitle>Enter Payment Details</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Amount (₹)
                </label>
                <Input
                  type="text"
                  value={amount}
                  onChange={handleAmountChange}
                  placeholder="0.00"
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Note (Optional)
                </label>
                <Input
                  type="text"
                  value={note}
                  onChange={handleNoteChange}
                  placeholder="Add a note"
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Select Payment App
                </label>
                <div className="grid grid-cols-3 gap-3">
                  {paymentApps.map((app) => (
                    <button
                      key={app.id}
                      onClick={() => handleAppSelect(app.id)}
                      className={`p-3 rounded-lg border-2 flex flex-col items-center justify-center ${
                        selectedApp === app.id
                          ? 'border-primary bg-primary/5'
                          : 'border-gray-200 hover:border-primary/50'
                      }`}
                    >
                      {app.icon}
                      <span className="text-sm mt-1">{app.name}</span>
                    </button>
                  ))}
                </div>
              </div>

              <Button
                onClick={handleProceed}
                className="w-full"
              >
                Proceed to Payment
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}