import React, { useState, useEffect } from 'react';
import { useLocation } from 'wouter';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useToast } from '@/hooks/use-toast';
import { apiRequest } from '@/lib/queryClient';
import { useAuthState } from '@/hooks/use-auth-state';

// Define the report types as per backend enum
enum BackendReportType {
  Voice = 'voice',
  Message = 'message',
  Whatsapp = 'whatsapp',
  Other = 'other',
}

export default function ReportScam() {
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  const { authState } = useAuthState();
  
  const [reportType, setReportType] = useState<BackendReportType>(BackendReportType.Other); // New state for backend reportType
  const [scamContact, setScamContact] = useState(''); // Renamed from upiId
  const [scamPlatform, setScamPlatform] = useState(''); // New state for scamPlatform
  const [scamDetails, setScamDetails] = useState(''); // Renamed from description
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [screenshot, setScreenshot] = useState<File | null>(null); // New state for screenshot

  const handleSubmit = async () => {
    // Basic validation
    if (!reportType || !scamDetails) {
      toast({
        title: "Missing Information",
        description: "Please select a report type and provide scam details.",
        variant: "destructive",
        duration: 3000,
      });
      return;
    }

    if (reportType !== BackendReportType.Other && !scamContact) {
      toast({
        title: "Contact Required",
        description: "Scam contact is required for this report type.",
        variant: "destructive",
        duration: 3000,
      });
      return;
    }

    if (!authState.isLoggedIn || !authState.userId) {
      toast({
        title: "Login Required",
        description: "Please login to submit a scam report",
        variant: "destructive",
        duration: 3000,
      });
      setLocation('/login');
      return;
    }
    
    setIsSubmitting(true);
    
    try {
      // NOTE: userId is not sent in body; backend links report to session userId for security
      const response = await fetch('http://localhost:6900/api/scam-reports', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include', // Always include credentials for session
        body: JSON.stringify({
          reportType,
          scamContact: scamContact || null,
          scamPlatform: scamPlatform || null,
          scamDetails,
          // screenshot: not sent yet, backend not implemented
        })
      });

      if (response.ok) {
        toast({
          title: "Report Submitted",
          description: "Thank you for helping make UPI safer for everyone.",
          variant: "default",
          duration: 3000,
        });
        setLocation('/my-reports'); // Redirect to my reports page
      } else {
        const errorData = await response.json();
        throw new Error(errorData.error || errorData.message || 'Failed to submit report');
      }
    } catch (err) {
      const error = err as Error;
      console.error("Error submitting scam report:", err); // Log the error
      toast({
        title: "Submission Failed",
        description: error.message || "An error occurred. Please try again.",
        variant: "destructive",
        duration: 3000,
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="flex flex-col px-6 py-8">
      <div className="flex justify-between items-center mb-6">
        <button
          onClick={() => setLocation('/home')}
          className="w-10 h-10 bg-[#F5F6FA] rounded-full flex items-center justify-center"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.5}
            stroke="currentColor"
            className="w-6 h-6"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M15.75 19.5L8.25 12l7.5-7.5"
            />
          </svg>
        </button>
        <h1 className="text-xl font-bold">Report Scam</h1>
        <div className="w-10"></div>
      </div>
      
      <Card className="bg-white rounded-2xl p-6 shadow-md border border-gray-100 mb-6">
        <Label htmlFor="report-type" className="block text-sm text-gray-500 mb-2">Report Type</Label>
        <Select value={reportType} onValueChange={(value) => setReportType(value as BackendReportType)}>
          <SelectTrigger className="w-full bg-[#F5F6FA] rounded-xl px-4 py-3 border-none focus:ring-0">
            <SelectValue placeholder="Select how the scam occurred" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={BackendReportType.Voice}>Voice Call</SelectItem>
            <SelectItem value={BackendReportType.Message}>SMS/Text Message</SelectItem>
            <SelectItem value={BackendReportType.Whatsapp}>WhatsApp</SelectItem>
            <SelectItem value={BackendReportType.Other}>Other</SelectItem>
          </SelectContent>
        </Select>

        {reportType !== BackendReportType.Other && (
          <div className="mt-4">
            <Label htmlFor="scam-contact" className="block text-sm text-gray-500 mb-2">Scam Contact (Phone, Email, UPI ID, etc.)</Label>
            <Input
              id="scam-contact"
              type="text"
              value={scamContact}
              onChange={(e) => setScamContact(e.target.value)}
              className="w-full bg-[#F5F6FA] rounded-xl px-4 py-3 border-none focus:ring-0"
              placeholder="e.g. +919876543210, scammer@email.com, or upi@id"
            />
          </div>
        )}

        <div className="mt-4">
          <Label htmlFor="scam-platform" className="block text-sm text-gray-500 mb-2">Platform (e.g. WhatsApp, Phone Call)</Label>
          <Input
            id="scam-platform"
            type="text"
            value={scamPlatform}
            onChange={(e) => setScamPlatform(e.target.value)}
            className="w-full bg-[#F5F6FA] rounded-xl px-4 py-3 border-none focus:ring-0"
            placeholder="e.g. WhatsApp, Phone Call, SMS, Email"
          />
        </div>

        <Label htmlFor="scam-details" className="block text-sm text-gray-500 mt-4 mb-2">Scam Details</Label>
        <Textarea
          id="scam-details"
          value={scamDetails}
          onChange={(e) => setScamDetails(e.target.value)}
          className="w-full bg-[#F5F6FA] rounded-xl px-4 py-3 h-24 border-none focus:ring-0"
          placeholder="Describe what happened..."
        />
      </Card>

      <Card className="bg-white rounded-2xl p-6 shadow-md border border-gray-100 mb-6">
        <Label className="block text-sm text-gray-500 mb-4">Add Screenshots (optional)</Label>
        <div className="border-2 border-dashed border-gray-200 rounded-xl p-6 flex flex-col items-center justify-center">
          <input
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            id="screenshot-upload"
            onChange={e => setScreenshot(e.target.files?.[0] || null)}
          />
          <label htmlFor="screenshot-upload" className="cursor-pointer flex flex-col items-center">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={1.5}
              stroke="currentColor"
              className="w-10 h-10 text-gray-500 mb-3"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
              />
            </svg>
            <p className="text-gray-500 text-sm text-center">Tap to upload screenshots</p>
            {screenshot && <span className="text-xs mt-2">{screenshot.name}</span>}
          </label>
        </div>
      </Card>
      
      <Button
        onClick={handleSubmit}
        disabled={isSubmitting}
        className="bg-primary text-white font-semibold py-4 px-6 rounded-xl shadow-md text-center"
      >
        {isSubmitting ? "Submitting..." : "Submit Report"}
      </Button>
    </div>
  );
}
