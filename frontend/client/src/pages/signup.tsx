import React, { useState, useEffect } from 'react';
import { useLocation } from 'wouter';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useToast } from '@/hooks/use-toast';
import { useAuthState } from '@/hooks/use-auth-state';

export default function Signup() {
  const [name, setName] = useState('');
  const [phone, setPhone] = useState('');
  const [error, setError] = useState('');
  const [location, setLocation] = useLocation();
  const { toast } = useToast();
  const { login } = useAuthState();

  useEffect(() => {
    console.log('Signup Page: window.location.search', window.location.search);
    const queryParams = new URLSearchParams(window.location.search);
    const phoneFromQuery = queryParams.get('phone');
    console.log('Signup Page: phoneFromQuery', phoneFromQuery);
    if (phoneFromQuery) {
      setPhone(phoneFromQuery);
      console.log('Signup Page: Phone state set to', phoneFromQuery);
    }
  }, []);

  const handleNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setName(e.target.value);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!name) {
      setError('Name is required');
      return;
    }
    if (!phone) {
      setError('Phone number is missing. Please go back to login.');
      return;
    }

    try {
      const response = await fetch('http://localhost:6900/signup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ phone, name }),
      });
      const data = await response.json();

      if (data.success) {
        login(data.user._id, data.user.phone);
        toast({ title: 'Success', description: 'Account created successfully!' });
        setLocation('/profile');
      } else {
        setError(data.error || 'Signup failed. Please try again.');
        toast({ title: 'Error', description: data.error || 'Signup failed.', variant: 'destructive' });
      }
    } catch (err) {
      console.error('Signup error:', err);
      setError('Failed to connect to the server or an error occurred.');
      toast({ title: 'Error', description: 'Network error or server unreachable.', variant: 'destructive' });
    }
  };

  return (
    <div className="flex flex-col px-6 py-12 min-h-screen">
      <div className="mt-16 mb-12">
        <h1 className="text-3xl font-bold mb-2">Complete Sign Up</h1>
        <p className="text-gray-500">Enter your name to create your account</p>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="mb-8">
          <Label htmlFor="phone" className="block text-sm font-medium text-gray-500 mb-2">
            Phone Number (Read-only)
          </Label>
          <div className="flex items-center bg-[#F5F6FA] rounded-xl px-4 py-3 shadow-sm">
            <span className="text-gray-900 font-medium mr-2">+91</span>
            <Input
              type="tel"
              id="phone"
              value={phone}
              readOnly
              className="flex-1 bg-transparent border-none focus:ring-0 text-lg text-gray-900"
            />
          </div>
        </div>

        <div className="mb-8">
          <Label htmlFor="name" className="block text-sm font-medium text-gray-500 mb-2">
            Your Name
          </Label>
          <Input
            type="text"
            id="name"
            value={name}
            onChange={handleNameChange}
            className="flex-1 bg-transparent border-none focus:ring-0 text-lg"
            placeholder="Enter your name"
          />
        </div>
        {error && <p className="text-error text-sm mt-1">{error}</p>}

        <Button
          type="submit"
          className="w-full bg-primary text-white font-semibold py-4 px-6 rounded-xl shadow-md"
        >
          Create Account
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