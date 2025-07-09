import React, { useState } from 'react';
import { useLocation } from 'wouter';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { isValidPhoneNumber } from '@/lib/upi';
import { useToast } from '@/hooks/use-toast';
import { useAuthState } from '@/hooks/use-auth-state';

export default function Login() {
  const [phone, setPhone] = useState('');
  const [location, setLocation] = useLocation();
  const [error, setError] = useState('');
  const { login } = useAuthState();
  const { toast } = useToast();

  const handlePhoneChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value.replace(/\D/g, '');
    setPhone(value);
    
    if (value && !isValidPhoneNumber(value) && value.length === 10) {
      setError('Please enter a valid 10-digit phone number');
    } else {
      setError('');
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!phone) {
      setError('Phone number is required');
      return;
    }
    
    if (!isValidPhoneNumber(phone)) {
      setError('Please enter a valid 10-digit phone number');
      return;
    }
    
    try {
      const response = await fetch('http://localhost:6900/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ phone })
      });

      if (!response.ok) {
        const errorData = await response.json();
        console.error('Backend error response (non-2xx status):', response.status, errorData);
        setError(errorData.error || `Login failed: Server responded with status ${response.status}`);
        return;
      }

      const data = await response.json();
      console.log('Backend login response data:', data);

      if (data.success) {
        // User exists and login successful, proceed with OTP redirect
        login(data.user._id, data.user.phone);
        toast({
          title: "Login Successful",
          description: "Redirecting to OTP verification.",
        });
        setLocation("/otp");
      } else if (data.redirect === 'signup') {
        // User does not exist, redirect to signup page
        toast({
          title: "User Not Found",
          description: "Please sign up for a new account.",
          variant: "destructive"
        });
        setLocation(`/signup?phone=${phone}`);
      } else {
        // Fallback for unexpected data structure
        setError('Unexpected response from server.');
        console.error('Unexpected backend response:', data);
      }
      
    } catch (err) {
      console.error('Network or parsing error during login:', err);
      setError('Failed to connect to the server or an unexpected error occurred. Please check your backend.');
    }
  };

  return (
    <div className="flex flex-col px-6 py-12 min-h-screen">
      <div className="mt-16 mb-12">
        <h1 className="text-3xl font-bold mb-2">Sign Up</h1>
        <p className="text-gray-500">Enter your phone number to continue</p>
      </div>
      
      <form onSubmit={handleSubmit}>
        <div className="mb-8">
          <Label htmlFor="phone" className="block text-sm font-medium text-gray-500 mb-2">
            Phone Number
          </Label>
          <div className="flex items-center bg-[#F5F6FA] rounded-xl px-4 py-3 shadow-sm">
            <span className="text-gray-900 font-medium mr-2">+91</span>
            <Input
              type="tel"
              id="phone"
              maxLength={10}
              value={phone}
              onChange={handlePhoneChange}
              className="flex-1 bg-transparent border-none focus:ring-0 text-lg"
              placeholder="Enter phone number"
            />
          </div>
          {error && <p className="text-error text-sm mt-1">{error}</p>}
        </div>
        
        <Button 
          type="submit" 
          className="w-full bg-primary text-white font-semibold py-4 px-6 rounded-xl shadow-md"
        >
          Sign Up
        </Button>
      </form>
      
      <p className="text-center text-sm text-gray-500 mt-8">
        By continuing, you agree to our{' '}
        <a href="#" className="text-primary">Terms of Service</a> and{' '}
        <a href="#" className="text-primary">Privacy Policy</a>
      </p>
    </div>
  );
}
