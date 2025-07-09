import React, { useState, useEffect } from 'react';
import { useLocation } from 'wouter';
import { Input } from '@/components/ui/input';
import { Card, CardContent } from '@/components/ui/card';
import { format, formatDistanceToNow, isToday, isYesterday } from 'date-fns';
import { 
  Search, 
  ArrowLeft, 
  ReceiptText, 
  ShoppingBag, 
  Calendar, 
  Package, 
  User, 
  ArrowUpRight, 
  ChevronRight, 
  CreditCard, 
  Smartphone, 
  Clock, 
  Building, 
  IndianRupee,
  FilterX,
  Loader2,
  AlertCircle,
  ChevronLeft,
  Info,
  Filter,
  ArrowDownLeft
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { useAuthState } from '@/hooks/use-auth-state';
import { apiRequest } from '@/lib/queryClient';
import { Badge } from '@/components/ui/badge';
import { 
  Pagination, 
  PaginationContent, 
  PaginationItem, 
  PaginationLink, 
  PaginationNext, 
  PaginationPrevious 
} from '@/components/ui/pagination';
import { Skeleton } from '@/components/ui/skeleton';
import { 
  Dialog, 
  DialogContent, 
  DialogDescription, 
  DialogHeader, 
  DialogTitle, 
  DialogFooter 
} from '@/components/ui/dialog';
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from '@/components/ui/select';

// Transaction interface matching server schema
interface Transaction {
  _id: string;
  userId: string;
  fromUpiId: string;
  toUpiId: string;
  amount: number;
  status: 'success' | 'failed' | 'pending';
  transactionDate: string;
  transactionId: string;
}

// Payment method icons
const PaymentMethodIcon = ({ method }: { method: string }) => {
  switch(method.toLowerCase()) {
    case 'upi':
      return <Smartphone className="h-4 w-4" />;
    case 'card':
      return <CreditCard className="h-4 w-4" />;
    case 'bank':
      return <Building className="h-4 w-4" />;
    default:
      return <IndianRupee className="h-4 w-4" />;
  }
};

// Format an amount in paise to rupees with ₹ symbol
const formatAmount = (amount: number, currency = "inr") => {
  const value = Math.abs(amount) / 100; // Convert paise to rupees
  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: currency.toUpperCase(),
    minimumFractionDigits: 2
  }).format(value);
};

export default function History() {
  const [, setLocation] = useLocation();
  const [searchQuery, setSearchQuery] = useState('');
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [filteredTransactions, setFilteredTransactions] = useState<Transaction[]>([]);
  const [selectedTransaction, setSelectedTransaction] = useState<Transaction | null>(null);
  const [isDetailOpen, setIsDetailOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [filter, setFilter] = useState("all");
  const itemsPerPage = 10;
  
  // Get userId from auth state
  const { authState } = useAuthState();
  
  // Fetch user transactions when component mounts
  useEffect(() => {
    const fetchTransactions = async () => {
      if (!authState.isLoggedIn || !authState.userId) {
        setError("You need to be logged in to view transaction history");
        setIsLoading(false);
        return;
      }
      
      try {
        setIsLoading(true);
        const response = await apiRequest('GET', `/api/transactions/${authState.userId}`);
        
        if (!response.ok) {
          throw new Error('Failed to fetch transaction history');
        }
        
        const data: Transaction[] = await response.json();
        setTransactions(data);
        setIsLoading(false);
      } catch (err) {
        console.error('Error fetching transactions:', err);
        setError('Failed to load transaction history');
        setIsLoading(false);
      }
    };
    
    fetchTransactions();
  }, [authState.isLoggedIn, authState.userId]);
  
  // Apply filters and search
  useEffect(() => {
    let result = [...transactions];
    
    // Apply search filter
    if (searchQuery) {
      result = result.filter(tx => 
        tx.toUpiId.toLowerCase().includes(searchQuery.toLowerCase()) ||
        tx.fromUpiId.toLowerCase().includes(searchQuery.toLowerCase()) ||
        tx.amount.toString().includes(searchQuery)
      );
    }
    
    // Apply type filter
    if (filter !== "all") {
      result = result.filter(tx => tx.status === filter);
    }
    
    setFilteredTransactions(result);
  }, [transactions, searchQuery, filter]);

  const handleTransactionClick = (transaction: Transaction) => {
    setSelectedTransaction(transaction);
  };

  const handleBackToList = () => {
    setSelectedTransaction(null);
  };

  // Transaction detail view
  if (selectedTransaction) {
    return (
      <div className="flex flex-col px-6 py-8">
        <div className="flex items-center mb-6">
          <Button 
            variant="ghost" 
            size="icon" 
            onClick={handleBackToList}
            className="p-2 mr-2"
          >
            <ArrowLeft className="w-5 h-5" />
          </Button>
          <h1 className="text-xl font-bold">Transaction Details</h1>
        </div>
        
        <Card className="bg-white rounded-2xl p-6 shadow-md border border-gray-100 w-full mb-8">
          <div className="flex justify-center mb-6">
            <div className="w-16 h-16 rounded-full bg-[#F5F6FA] flex items-center justify-center">
              <ReceiptText className="w-8 h-8 text-primary" />
            </div>
          </div>
          
          <div className="text-center mb-6">
            <p className={`text-2xl font-bold ${selectedTransaction.amount > 0 ? 'text-green-600' : 'text-red-600'}`}>
              {selectedTransaction.amount > 0 ? '+ ' : '- '}
              ₹{Math.abs(selectedTransaction.amount).toFixed(2)}
            </p>
            <p className="text-sm text-gray-500 mt-1">
              {selectedTransaction.status === 'success' ? 'Received from' : 'Paid to'} {selectedTransaction.toUpiId}
            </p>
            <p className="text-xs text-gray-400 mt-1">
              {new Date(selectedTransaction.transactionDate).toLocaleString('en-US', {
                day: 'numeric',
                month: 'short',
                year: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
                hour12: true
              })}
            </p>
          </div>
          
          <div className="border-t border-gray-100 pt-4">
            <div className="flex justify-between py-2">
              <p className="text-gray-500">Status</p>
              <p className="font-medium text-green-600">{selectedTransaction.status === 'success' ? 'Completed' : selectedTransaction.status === 'failed' ? 'Failed' : 'Pending'}</p>
            </div>
            <div className="flex justify-between py-2">
              <p className="text-gray-500">Transaction ID</p>
              <p className="font-medium text-sm">{selectedTransaction.transactionId}</p>
            </div>
            <div className="flex justify-between py-2">
              <p className="text-gray-500">UPI ID</p>
              <p className="font-medium text-sm">{selectedTransaction.fromUpiId}</p>
            </div>
            {selectedTransaction.toUpiId && (
              <div className="flex justify-between py-2">
                <p className="text-gray-500">Payment App</p>
                <p className="font-medium">{selectedTransaction.toUpiId}</p>
              </div>
            )}
          </div>
        </Card>
        
        <Button
          variant="outline"
          className="w-full mb-4"
          onClick={() => {
            if (navigator.share) {
              navigator.share({
                title: 'Transaction Receipt',
                text: `${selectedTransaction.status === 'success' ? 'Received' : 'Paid'} ₹${Math.abs(selectedTransaction.amount).toFixed(2)} ${selectedTransaction.status === 'success' ? 'from' : 'to'} ${selectedTransaction.toUpiId} (${selectedTransaction.fromUpiId}). Transaction ID: ${selectedTransaction.transactionId}`
              });
            } else {
              alert('Share functionality not supported on this browser');
            }
          }}
        >
          Share Receipt
        </Button>
        
        <Button
          variant="default"
          className="w-full bg-primary"
          onClick={() => setLocation('/home')}
        >
          Done
        </Button>
      </div>
    );
  }

  // Group transactions by date
  const groupedTransactions = () => {
    // Get today and yesterday dates for comparison
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    yesterday.setHours(0, 0, 0, 0);
    
    // Create groups
    const groups: Record<string, Transaction[]> = {
      today: [],
      yesterday: [],
      earlier: []
    };
    
    // Group transactions
    filteredTransactions.forEach(transaction => {
      const txDate = new Date(transaction.transactionDate);
      txDate.setHours(0, 0, 0, 0);
      
      if (txDate.getTime() === today.getTime()) {
        groups.today.push(transaction);
      } else if (txDate.getTime() === yesterday.getTime()) {
        groups.yesterday.push(transaction);
      } else {
        groups.earlier.push(transaction);
      }
    });
    
    return groups;
  };
  
  const transactionGroups = groupedTransactions();
  
  // Helper function to get transaction icon
  const getTransactionIcon = (title: string) => {
    const titleLower = title.toLowerCase();
    
    if (titleLower.includes('shop') || titleLower.includes('market') || titleLower.includes('store')) {
      return <ShoppingBag className="h-5 w-5 text-white" />;
    } else if (titleLower.includes('food') || titleLower.includes('restaurant') || titleLower.includes('delivery')) {
      return <Package className="h-5 w-5 text-white" />;
    } else if (titleLower.includes('transfer') || titleLower.includes('send') || titleLower.includes('receive')) {
      return <ArrowUpRight className="h-5 w-5 text-white" />;
    } else {
      return <User className="h-5 w-5 text-white" />;
    }
  };
  
  // Helper function to get icon background color
  const getIconBgColor = (title: string) => {
    const titleLower = title.toLowerCase();
    
    if (titleLower.includes('shop') || titleLower.includes('market') || titleLower.includes('store')) {
      return 'bg-cyan-400';
    } else if (titleLower.includes('food') || titleLower.includes('restaurant') || titleLower.includes('delivery')) {
      return 'bg-indigo-500';
    } else if (titleLower.includes('transfer') || titleLower.includes('send') || titleLower.includes('receive')) {
      return 'bg-green-500';
    } else {
      return 'bg-gray-400';
    }
  };
  
  // Transaction list view with new design based on the UI screenshot
  return (
    <div className="flex flex-col min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 p-4 shadow-sm flex items-center">
        <Button 
          variant="ghost" 
          size="icon" 
          onClick={() => window.history.back()}
          className="mr-2"
        >
          <ChevronLeft className="h-5 w-5 text-gray-700 dark:text-gray-300" />
        </Button>
        <h1 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Transaction History</h1>
      </div>
      
      {/* Search and Filter */}
      <div className="p-4 bg-white dark:bg-gray-800 shadow-sm mb-4">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-500" />
          <Input
            type="text"
            placeholder="Search by UPI ID or amount..."
            className="pl-9 pr-4 py-2 rounded-lg bg-gray-100 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 focus:ring-blue-500 focus:border-blue-500 w-full text-gray-900 dark:text-gray-100"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
                </div>
              </div>
              
      {/* Transaction List */}
      <div className="flex-1 overflow-y-auto p-4">
        {isLoading ? (
          <p className="text-center text-gray-600 dark:text-gray-400">Loading transactions...</p>
        ) : error ? (
          <p className="text-center text-red-600 dark:text-red-400">Error: {error}</p>
        ) : filteredTransactions.length === 0 ? (
          <p className="text-center text-gray-600 dark:text-gray-400">No transactions found.</p>
        ) : (
          <div className="space-y-4">
            {filteredTransactions.map((transaction) => (
              <Card key={transaction._id} className="rounded-lg shadow-sm dark:bg-gray-800 border-none">
                <CardContent className="p-4 flex items-center space-x-4">
                  <div className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center
                    ${transaction.status === 'success' ? 'bg-green-100 dark:bg-green-900/30' : 'bg-red-100 dark:bg-red-900/30'}`}>
                    {transaction.status === 'success' ? (
                      <ArrowUpRight className="h-5 w-5 text-green-600 dark:text-green-400" />
                    ) : (
                      <ArrowDownLeft className="h-5 w-5 text-red-600 dark:text-red-400" />
                    )}
                  </div>
                  <div className="flex-1">
                    <div className="flex justify-between items-center">
                      <p className="text-sm font-medium text-gray-900 dark:text-gray-100">{transaction.toUpiId}</p>
                      <p className={`text-base font-semibold ${transaction.status === 'success' ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                        ₹{transaction.amount.toFixed(2)}
                      </p>
                  </div>
                    <p className="text-xs text-gray-500 dark:text-gray-400">From: {transaction.fromUpiId}</p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      {format(new Date(transaction.transactionDate), 'dd MMM yyyy, hh:mm a')}
                    </p>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>

      {/* Bottom Navigation (if applicable) */}
      {/* <BottomNav /> */}
    </div>
  );
}
