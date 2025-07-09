import React, { useState, useEffect } from 'react';
import { useLocation } from 'wouter';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Link } from 'wouter';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { useAuthState } from '@/hooks/use-auth-state';

export default function OTP() {
  console.log("OTP Component Mounted!"); // Debug log to see if component renders
  const [otp, setOtp] = useState<string>("");
  const [phoneNumber, setPhoneNumber] = useState<string>("");
  const [displayOtp, setDisplayOtp] = useState<string>("");
  const [timer, setTimer] = useState<number>(60);
  const [resendDisabled, setResendDisabled] = useState<boolean>(true);
  const [, setLocation] = useLocation();
  const { login } = useAuthState();

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const phone = params.get("phone");
    if (phone) {
      setPhoneNumber(phone);
    }

    const generated = Math.floor(100000 + Math.random() * 900000).toString();
    setDisplayOtp(generated);

    const interval = setInterval(() => {
      setTimer((prev) => {
        if (prev === 1) {
          clearInterval(interval);
          setResendDisabled(false);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const handleOtpComplete = async () => {
    if (otp === displayOtp) {
      console.log("OTP Verified. Proceeding to home page.");
      // Always redirect to home after OTP verification for this debugging phase.
      // The useAuthState hook now forces isLoggedIn: true.
      setLocation('/home');
    } else {
      alert("Incorrect OTP. Please try again.");
    }
  };

  const handleResend = () => {
    console.log("Resending OTP...");
    setTimer(60);
    setResendDisabled(true);
    const generated = Math.floor(100000 + Math.random() * 900000).toString();
    setDisplayOtp(generated);

    const interval = setInterval(() => {
      setTimer((prev) => {
        if (prev === 1) {
          clearInterval(interval);
          setResendDisabled(false);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen dark:bg-gray-900 bg-gray-100 p-4">
      <Card className="w-full max-w-md dark:bg-gray-800 dark:text-white">
        <CardHeader className="text-center">
          <CardTitle className="text-2xl font-bold">Verify OTP</CardTitle>
          <CardDescription className="text-gray-600 dark:text-gray-400">
            A 6-digit OTP has been sent to +91 {phoneNumber}
          </CardDescription>
          <p className="mt-4 text-lg font-semibold dark:text-white">Your OTP is: <span className="text-blue-500 dark:text-blue-400">{displayOtp}</span></p>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="otp">Enter OTP</Label>
            <Input
              id="otp"
              type="number"
              placeholder="Enter 6-digit OTP"
              value={otp}
              onChange={(e) => setOtp(e.target.value)}
              className="dark:bg-gray-700 dark:text-white dark:border-gray-600"
            />
          </div>
          <Button onClick={handleOtpComplete} className="w-full">
            Verify OTP
          </Button>
          <div className="text-center text-sm text-gray-500 dark:text-gray-400">
            Didn't receive OTP?{" "}
            <Button
              variant="link"
              onClick={handleResend}
              disabled={resendDisabled}
              className="p-0 h-auto dark:text-blue-400 dark:hover:text-blue-300"
            >
              {resendDisabled ? `Resend in ${timer}s` : "Resend OTP"}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
