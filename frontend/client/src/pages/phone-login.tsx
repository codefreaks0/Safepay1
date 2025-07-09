import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { OtpInput } from "@/components/ui/otp-input";
import { useLocation } from "wouter";
import { Phone, LockKeyhole } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

export default function PhoneLogin() {
  const [step, setStep] = useState<"phone" | "otp">("phone");
  const [phoneNumber, setPhoneNumber] = useState("");
  const [otp, setOtp] = useState("");
  const [generatedOtp, setGeneratedOtp] = useState("");
  const { toast } = useToast();
  const [, setLocation] = useLocation();

  const handleRequestOtp = () => {
    console.log("handleRequestOtp called, phone number:", phoneNumber);
    if (!phoneNumber) {
      toast({
        title: "Error",
        description: "Please enter a phone number",
        variant: "destructive",
      });
      return;
    }

    // Generate a random 6-digit OTP
    const newOtp = Math.floor(100000 + Math.random() * 900000).toString();
    setGeneratedOtp(newOtp);

    // Save the OTP to local storage
    localStorage.setItem("otp", newOtp);

    toast({
      title: "OTP Generated",
      description: `Your OTP is: ${newOtp}`,
    });

    setStep("otp");
    console.log("Step set to OTP");
  };

  const handleVerifyOtp = () => {
    const storedOtp = localStorage.getItem("otp");

    if (otp === storedOtp) {
      toast({
        title: "Success",
        description: "OTP verified successfully. You are logged in!",
      });
      setLocation("/home");
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
            {step === "phone" ? "Login with Phone" : "Verify OTP"}
          </CardTitle>
          <CardDescription className="text-center">
            {step === "phone"
              ? "Enter your phone number to receive a one-time password"
              : `We've sent a 6-digit code to ${phoneNumber}`}
          </CardDescription>
        </CardHeader>

        <CardContent>
          {step === "phone" ? (
            <div className="space-y-4">
              <div className="flex rounded-md border border-input overflow-hidden">
                <div className="bg-muted px-3 py-2 flex items-center text-sm">
                  <Phone className="h-4 w-4 mr-2" />
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
                disabled={phoneNumber.length !== 10}
              >
                Get OTP
              </Button>
            </div>
          ) : (
            <div className="space-y-4">
              <OtpInput
                length={6}
                onComplete={(otp) => setOtp(otp)}
                className="flex justify-center gap-2"
              />

              <Button
                onClick={handleVerifyOtp}
                className="w-full"
                disabled={otp.length !== 6}
              >
                Verify OTP
              </Button>
            </div>
          )}
        </CardContent>

        <CardFooter className="flex justify-center">
          <p className="text-xs text-muted-foreground text-center">
            By continuing, you agree to our Terms of Service and Privacy Policy.
          </p>
        </CardFooter>
      </Card>
    </div>
  );
}