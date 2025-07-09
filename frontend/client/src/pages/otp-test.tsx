import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";

export default function OtpTest() {
  const [phoneNumber, setPhoneNumber] = useState("");
  const [otp, setOtp] = useState("");
  const [generatedOtp, setGeneratedOtp] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();

  const handleRequestOtp = () => {
    if (!phoneNumber) {
      toast({
        title: "Error",
        description: "Please enter a phone number",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);

    // Generate a random 6-digit OTP
    const newOtp = Math.floor(100000 + Math.random() * 900000).toString();
    setGeneratedOtp(newOtp);

    // Save the OTP to local storage
    localStorage.setItem("otp", newOtp);

    toast({
      title: "OTP Generated",
      description: `Your OTP is: ${newOtp}`,
    });

    setIsLoading(false);
  };

  const handleVerifyOtp = () => {
    const storedOtp = localStorage.getItem("otp");

    if (otp === storedOtp) {
      toast({
        title: "Success",
        description: "OTP verified successfully. You are logged in!",
      });
    } else {
      toast({
        title: "Error",
        description: "Invalid OTP. Please try again.",
        variant: "destructive",
      });
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-50 p-4">
      <Card className="w-full max-w-md mx-auto">
        <CardHeader className="space-y-1">
          <CardTitle className="text-2xl font-bold text-center">
            OTP Test Page
          </CardTitle>
          <CardDescription className="text-center">
            Test the OTP generation and verification functionality
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-4">
          <div className="flex rounded-md border border-input overflow-hidden">
            <div className="bg-muted px-3 py-2 flex items-center text-sm">
              +91
            </div>
            <Input
              type="tel"
              placeholder="10 digit mobile number"
              value={phoneNumber}
              onChange={(e) => {
                const value = e.target.value.replace(/\D/g, "");
                setPhoneNumber(value);
              }}
              className="flex-1 border-0 focus-visible:ring-0 focus-visible:ring-offset-0"
              maxLength={10}
            />
          </div>

          <Button
            onClick={handleRequestOtp}
            className="w-full"
            disabled={isLoading}
          >
            {isLoading ? "Requesting..." : "Request OTP"}
          </Button>

          <Input
            type="text"
            placeholder="Enter OTP"
            value={otp}
            onChange={(e) => setOtp(e.target.value)}
            className="w-full mt-4"
          />

          <Button
            onClick={handleVerifyOtp}
            className="w-full mt-2"
          >
            Verify OTP
          </Button>
        </CardContent>

        <CardFooter className="flex justify-center">
          <p className="text-xs text-muted-foreground text-center">
            This is a test page to debug OTP functionality
          </p>
        </CardFooter>
      </Card>
    </div>
  );
}