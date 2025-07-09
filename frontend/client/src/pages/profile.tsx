import React, { useState, useEffect } from 'react';
import { useLocation } from 'wouter';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useToast } from '@/hooks/use-toast';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Plus, Trash2, CheckCircle2, Circle, ArrowLeft, Calendar, Shield, Phone, Globe, Wallet, CreditCard, Building2 } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { format } from 'date-fns';
import { useAuthState } from '@/hooks/use-auth-state';

// Define interfaces for populated data (should match backend models)
interface PaymentMethod {
  _id: string;
  type: 'upi' | 'card' | 'bank_account';
  name: string;
  upiId?: string;
  cardNumber?: string;
  expiryDate?: string;
  accountNumber?: string;
  ifscCode?: string;
  isDefault: boolean;
}

interface ScamReport {
  _id: string;
  reportType: 'voice' | 'message' | 'whatsapp' | 'other';
  scamContact?: string;
  scamPlatform?: string;
  scamDetails: string;
  status: 'pending' | 'reviewed' | 'resolved' | 'rejected';
  createdAt: string;
}

export default function Profile() {
  console.log("Profile component rendered.");
  const { authState } = useAuthState();
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [address, setAddress] = useState('');
  const [dob, setDob] = useState('');
  const [paymentMethods, setPaymentMethods] = useState<PaymentMethod[]>([]);
  const [scamReports, setScamReports] = useState<ScamReport[]>([]);
  const [, setLocation] = useLocation();
  const { toast } = useToast();

  // Function to get appropriate icon for payment method type
  function getPaymentMethodIcon(type: string) {
    switch (type) {
      case 'upi':
        return <Wallet className="h-5 w-5" />;
      case 'card':
        return <CreditCard className="h-5 w-5" />;
      case 'bank_account':
        return <Building2 className="h-5 w-5" />;
      default:
        return <CreditCard className="h-5 w-5" />;
    }
  }

  // Function to get human-readable type name for payment methods
  function getPaymentTypeName(type: string) {
    switch (type) {
      case 'upi':
        return 'UPI';
      case 'card':
        return 'Card';
      case 'bank_account':
        return 'Bank Account';
      default:
        return type;
    }
  }

  // Helper to get badge color based on reportType (copied from my-reports.tsx)
  const getReportTypeColor = (reportType: ScamReport['reportType']) => {
    switch(reportType) {
      case 'voice':
        return 'bg-red-100 text-red-800 border-red-200 dark:bg-red-900/30 dark:text-red-300';
      case 'message':
        return 'bg-blue-100 text-blue-800 border-blue-200 dark:bg-blue-900/30 dark:text-blue-300';
      case 'whatsapp':
        return 'bg-green-100 text-green-800 border-green-200 dark:bg-green-900/30 dark:text-green-300';
      case 'other':
        return 'bg-gray-100 text-gray-800 border-gray-200 dark:bg-gray-800 dark:text-gray-300';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200 dark:bg-gray-800 dark:text-gray-300';
    }
  };
  
  // Helper to get status badge color (copied from my-reports.tsx)
  const getStatusColor = (status: ScamReport['status']) => {
    switch(status) {
      case 'pending':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200 dark:bg-yellow-900/30 dark:text-yellow-300';
      case 'reviewed':
        return 'bg-blue-100 text-blue-800 border-blue-200 dark:bg-blue-900/30 dark:text-blue-300';
      case 'resolved':
        return 'bg-green-100 text-green-800 border-green-200 dark:bg-green-900/30 dark:text-green-300';
      case 'rejected':
        return 'bg-red-100 text-red-800 border-red-200 dark:bg-red-900/30 dark:text-red-300';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200 dark:bg-gray-800 dark:text-gray-300';
    }
  };

  // Fetch user data on component mount
  useEffect(() => {
    console.log("Profile useEffect triggered.");
    console.log("Current authState for profile fetch:", authState);

    const fetchUserProfile = async () => {
      if (!authState.isLoggedIn || !authState.userId) {
        console.log("User not logged in or userId not available, skipping profile fetch and redirecting.");
        setLocation('/login'); // Redirect if not truly logged in
        toast({ title: 'Error', description: 'Please log in to view your profile.', variant: 'destructive', duration: 3000 });
        return;
      }

      try {
        console.log(`Attempting to fetch profile for userId: ${authState.userId}`);
        const response = await fetch(`http://localhost:6900/profile/${authState.userId}`, {
          method: 'GET',
          credentials: 'include',
        });

        console.log("Profile fetch response status:", response.status);
        if (!response.ok) {
          const errorData = await response.json();
          console.error('Backend error during profile fetch:', response.status, errorData);
          if (response.status === 401) {
            setLocation('/login');
            toast({ title: 'Error', description: 'Your session has expired. Please log in again.', variant: 'destructive', duration: 3000 });
          } else {
            toast({ title: 'Error', description: errorData.error || 'Failed to fetch profile.', variant: 'destructive', duration: 3000 });
          }
          return;
        }

        const data = await response.json();
        console.log("Data received from profile API:", data);

        if (data.user) {
          setFullName(data.user.name || '');
          setEmail(data.user.email || '');
          setAddress(data.user.address || '');
          setDob(data.user.dob ? format(new Date(data.user.dob), 'yyyy-MM-dd') : '');
          setPaymentMethods(data.user.paymentMethods || []);
          setScamReports(data.user.scamReports || []);
          console.log("Profile states updated with data:", { name: data.user.name, email: data.user.email, dob: data.user.dob });
        } else {
          console.log("Data received but no user object found.", data);
        }
      } catch (err) {
        console.error('Error fetching profile (network/parsing):', err);
        toast({ title: 'Error', description: 'Network error or server unreachable.', variant: 'destructive', duration: 3000 });
      }
    };

    fetchUserProfile();
  }, [authState.isLoggedIn, authState.userId, setLocation, toast]);

  const handleSaveProfile = async (e: React.FormEvent) => {
    e.preventDefault();
    // Re-enabling actual save functionality
    if (!authState.isLoggedIn || !authState.userId) {
      toast({ title: 'Error', description: 'Not logged in to save profile.', variant: 'destructive', duration: 3000 });
      setLocation('/login');
      return;
    }

    try {
      const response = await fetch(`http://localhost:6900/profile/${authState.userId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          name: fullName,
          email: email,
          address: address,
          dob: dob,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        toast({ title: 'Error', description: errorData.error || 'Failed to save profile.', variant: 'destructive', duration: 3000 });
        return;
      }

      const data = await response.json();
      if (data.success) {
        toast({ title: 'Success', description: 'Profile updated successfully!', duration: 3000 });
      } else {
        toast({ title: 'Error', description: data.error || 'Failed to save profile.', variant: 'destructive', duration: 3000 });
      }
    } catch (err) {
      console.error('Error saving profile:', err);
      toast({ title: 'Error', description: 'Network error or server unreachable.', variant: 'destructive', duration: 3000 });
    }
  };

  return (
    <div className="flex flex-col px-6 py-12 min-h-screen">
      <div className="mt-16 mb-12">
        <h1 className="text-3xl font-bold mb-2">My Profile</h1>
        <p className="text-gray-500">Manage your personal information</p>
      </div>

      <form onSubmit={handleSaveProfile}>
        <div className="mb-8">
          <Label htmlFor="fullName" className="block text-sm font-medium text-gray-500 mb-2">
            Full Name
          </Label>
          <Input
            type="text"
            id="fullName"
            value={fullName}
            onChange={(e) => setFullName(e.target.value)}
            className="bg-[#F5F6FA] rounded-xl px-4 py-3 shadow-sm text-lg"
            placeholder="Enter your full name"
          />
        </div>

        <div className="mb-8">
          <Label htmlFor="email" className="block text-sm font-medium text-gray-500 mb-2">
            Email Address
          </Label>
          <Input
            type="email"
            id="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="bg-[#F5F6FA] rounded-xl px-4 py-3 shadow-sm text-lg"
            placeholder="Enter your email address"
          />
        </div>

        <div className="mb-8">
          <Label htmlFor="address" className="block text-sm font-medium text-gray-500 mb-2">
            Address
          </Label>
          <Input
            type="text"
            id="address"
            value={address}
            onChange={(e) => setAddress(e.target.value)}
            className="bg-[#F5F6FA] rounded-xl px-4 py-3 shadow-sm text-lg"
            placeholder="Enter your address"
          />
        </div>

        <div className="mb-8">
          <Label htmlFor="dob" className="block text-sm font-medium text-gray-500 mb-2">
            Date of Birth
          </Label>
          <Input
            type="date"
            id="dob"
            value={dob}
            onChange={(e) => setDob(e.target.value)}
            className="bg-[#F5F6FA] rounded-xl px-4 py-3 shadow-sm text-lg"
            placeholder="YYYY-MM-DD"
          />
        </div>

        <Button 
          type="submit" 
          className="w-full bg-primary text-white font-semibold py-4 px-6 rounded-xl shadow-md"
        >
          Save Profile
        </Button>
      </form>

      {/* Payment Methods Section */}
      <div className="mt-12 mb-8">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold">Payment Methods</h2>
          <Button variant="outline" size="sm" onClick={() => setLocation('/payment-methods')}>
            <Plus className="h-4 w-4 mr-2" /> Add New
          </Button>
        </div>
        {paymentMethods.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {paymentMethods.map((method) => (
              <Card key={method._id} className="p-4 flex items-center justify-between">
                <div className="flex items-center">
                  <div className="bg-gray-100 p-2 rounded-full mr-3">
                    {getPaymentMethodIcon(method.type)}
                  </div>
                  <div>
                    <h3 className="font-medium">{method.name}</h3>
                    <p className="text-sm text-gray-500">{getPaymentTypeName(method.type)}</p>
                  </div>
                </div>
                <Button variant="ghost" size="icon" className="text-gray-400 hover:text-red-500">
                  <Trash2 className="h-5 w-5" />
                </Button>
              </Card>
            ))}
          </div>
        ) : (
          <p className="text-gray-500">No payment methods added yet.</p>
        )}
      </div>

      {/* Scam Reports Section */}
      <div className="mt-12 mb-8">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold">Scam Reports</h2>
          <Button variant="outline" size="sm" onClick={() => setLocation('/report-scam')}>
            <Plus className="h-4 w-4 mr-2" /> Report New
          </Button>
        </div>
        {scamReports.length > 0 ? (
          <div className="space-y-4">
            {scamReports.map((report) => (
              <Card key={report._id} className="p-4">
                <CardHeader className="pb-2">
                  <div className="flex justify-between items-start">
                    <CardTitle className="text-base">Report Type: {report.reportType}</CardTitle>
                    <Badge className={`${getReportTypeColor(report.reportType)} border`}>
                      {report.reportType.charAt(0).toUpperCase() + report.reportType.slice(1)}
                    </Badge>
                  </div>
                  <CardDescription className="flex items-center gap-1 text-xs">
                    <Calendar className="h-3 w-3" />
                    Reported on {format(new Date(report.createdAt), 'MMM d, yyyy')}
                  </CardDescription>
                </CardHeader>
                <CardContent className="pb-4">
                  <div className="flex items-center gap-1 text-gray-600 text-sm mb-1">
                    <Shield className="h-4 w-4" />
                    <span>Status: </span>
                    <Badge className={`${getStatusColor(report.status)}`}>
                      {report.status.charAt(0).toUpperCase() + report.status.slice(1)}
                    </Badge>
                  </div>
                  <p className="text-sm text-gray-700 line-clamp-2">
                    {report.scamDetails}
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>
        ) : (
          <p className="text-gray-500">No scam reports submitted yet.</p>
        )}
      </div>
    </div>
  );
}