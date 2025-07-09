import { loadStripe } from '@stripe/stripe-js';

// Load Stripe outside of component render to avoid recreating Stripe object on each render
let stripePromiseInstance: Promise<any> | null = null;

/**
 * Get a Stripe instance initialized with the public key
 * @returns Promise that resolves to a Stripe instance
 */
export const getStripePromise = () => {
  if (!stripePromiseInstance && import.meta.env.VITE_STRIPE_PUBLIC_KEY) {
    stripePromiseInstance = loadStripe(import.meta.env.VITE_STRIPE_PUBLIC_KEY);
  }
  return stripePromiseInstance;
};

/**
 * Create a payment intent with the provided details
 * @param amount Payment amount (in Indian Rupees)
 * @param upiId UPI ID of the recipient
 * @param description Payment description
 * @returns Promise that resolves to response with clientSecret and paymentIntentId
 */
export const createPaymentIntent = async (
  amount: number,
  upiId: string,
  description: string
) => {
  try {
    const response = await fetch('/api/create-payment-intent', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        amount,
        upiId,
        description,
        currency: 'inr',
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to create payment intent');
    }

    return await response.json();
  } catch (error: any) {
    console.error('Error creating payment intent:', error);
    throw error;
  }
};

/**
 * Check the status of a payment
 * @param paymentIntentId ID of the payment intent to check
 * @param userId Optional user ID
 * @returns Promise that resolves to payment status
 */
export const checkPaymentStatus = async (
  paymentIntentId: string,
  userId?: number
) => {
  try {
    const url = new URL(`/api/payment/${paymentIntentId}`, window.location.origin);
    if (userId) {
      url.searchParams.append('userId', userId.toString());
    }

    const response = await fetch(url.toString(), {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to check payment status');
    }

    return await response.json();
  } catch (error: any) {
    console.error('Error checking payment status:', error);
    throw error;
  }
};