import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Shield, AlertTriangle, CheckCircle } from 'lucide-react';
import { apiRequest } from "@/lib/queryClient";
import { 
  Dialog, 
  DialogContent, 
  DialogDescription, 
  DialogHeader, 
  DialogTitle,
  DialogTrigger
} from "@/components/ui/dialog";
import { useToast } from "@/hooks/use-toast";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";

// Types for UPI check response
interface DomainAnalysis {
  domain: string;
  count: number;
}

interface UpiCheckResponse {
  upiId: string;
  status: 'SAFE' | 'SUSPICIOUS' | 'SCAM';
  riskPercentage: number;
  riskLevel: 'Low' | 'Medium' | 'High';
  reports: number;
  reason: string;
  confidence_score: number;
  risk_factors?: string[];
  recommendations?: string[];
  age?: string;
  reportedFor?: string;
  safety_score?: number; // AI-generated safety score
  ai_analysis?: string; // AI-generated explanation
}

export default function UpiCheckButton() {
  const [upiId, setUpiId] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<UpiCheckResponse | null>(null);
  const [isOpen, setIsOpen] = useState(false);
  const { toast } = useToast();

  const handleUpiIdChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setUpiId(e.target.value);
  };

  const checkUpi = async () => {
    if (!upiId) {
      toast({
        title: "UPI ID Required",
        description: "Please enter a UPI ID to check",
        variant: "destructive"
      });
      return;
    }
    setIsLoading(true);
    try {
      // Call Flask backend directly
      const response = await apiRequest('POST', 'http://localhost:5005/predict-upi-fraud', { upi_id: upiId });
      const data = await response.json();
      if (response.ok) {
        setResults(data);
        toast({
          title: data.safety_status === 'Safe' ? 'Safe UPI ID' : 'Risky UPI ID',
          description: `Fraud Probability: ${data.fraud_probability}% | Risk: ${data.risk_level}`,
          variant: data.safety_status === 'Safe' ? 'default' : 'destructive'
        });
      } else {
        toast({
          title: "Check Failed",
          description: data.error || "Failed to analyze UPI ID",
          variant: "destructive"
        });
      }
    } catch (error) {
      console.error('UPI check error:', error);
      toast({
        title: "Check Failed",
        description: "An error occurred while checking the UPI ID",
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  };

  const getRiskColor = (level: string) => {
    switch (level.toLowerCase()) {
      case 'high':
        return 'bg-destructive text-destructive-foreground';
      case 'medium':
        return 'bg-amber-500 text-white';
      case 'low':
        return 'bg-green-500 text-white';
      default:
        return 'bg-muted text-muted-foreground';
    }
  };

  const getProgressColor = (score: number) => {
    if (score >= 70) return 'bg-destructive';
    if (score >= 40) return 'bg-amber-500';
    return 'bg-green-500';
  };

  const getRiskIcon = (level: string) => {
    switch (level.toLowerCase()) {
      case 'high':
        return <AlertTriangle className="h-6 w-6 text-destructive" />;
      case 'medium':
        return <AlertTriangle className="h-6 w-6 text-amber-500" />;
      case 'low':
        return <CheckCircle className="h-6 w-6 text-green-500" />;
      default:
        return <Shield className="h-6 w-6 text-muted-foreground" />;
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button 
          className="w-full flex items-center gap-2 bg-primary text-white" 
          size="lg"
        >
          <Shield className="h-5 w-5" />
          UPI Scam Check
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Check UPI ID For Scams</DialogTitle>
          <DialogDescription>
            Enter a UPI ID to check its safety and fraud risk
          </DialogDescription>
        </DialogHeader>

        <div className="flex items-center space-x-2">
          <Input
            placeholder="Enter UPI ID (e.g., name@upi)"
            value={upiId}
            onChange={handleUpiIdChange}
            className="flex-1"
          />
          <Button 
            onClick={checkUpi} 
            disabled={isLoading}
          >
            {isLoading ? "Checking..." : "Check"}
          </Button>
        </div>

        {results && (
          <div className="mt-4 space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center justify-between">
                  <span>Risk Assessment</span>
                  <span className={`px-2 py-1 text-xs rounded-full ${getRiskColor(results.risk_level)}`}>
                    {results.safety_status}
                  </span>
                </CardTitle>
                <CardDescription>
                  {results.upi_id}
                </CardDescription>
              </CardHeader>
              <CardContent className="pb-2">
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Fraud Probability</span>
                    <span className="font-medium">{results.fraud_probability}%</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Risk Level</span>
                    <span className="font-medium">{results.risk_level}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Safe Percentage</span>
                    <span className="font-medium">{results.safe_percentage}%</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Total Transactions</span>
                    <span className="font-medium">{results.total_transactions}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Failed Transactions</span>
                    <span className="font-medium">{results.failed_transactions}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Average Amount</span>
                    <span className="font-medium">₹{results.average_amount}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Payer Failed Ratio</span>
                    <span className="font-medium">{results.payer_failed_ratio}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Payer Unique Beneficiaries</span>
                    <span className="font-medium">{results.payer_unique_beneficiaries}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Beneficiary Unique Payers</span>
                    <span className="font-medium">{results.beneficiary_unique_payers}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Payer Recent Frauds</span>
                    <span className="font-medium">{results.payer_recent_frauds}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Beneficiary Recent Frauds</span>
                    <span className="font-medium">{results.beneficiary_recent_frauds}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Payer Fraud Ratio</span>
                    <span className="font-medium">{results.payer_fraud_ratio}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Beneficiary Fraud Ratio</span>
                    <span className="font-medium">{results.beneficiary_fraud_ratio}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Total Fraud Ratio</span>
                    <span className="font-medium">{results.total_fraud_ratio}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Has Recent Fraud</span>
                    <span className="font-medium">{results.has_recent_fraud ? 'Yes' : 'No'}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>High Fraud Payer</span>
                    <span className="font-medium">{results.is_high_fraud_payer ? 'Yes' : 'No'}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>High Fraud Beneficiary</span>
                    <span className="font-medium">{results.is_high_fraud_beneficiary ? 'Yes' : 'No'}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Amount x Failed</span>
                    <span className="font-medium">{results.amt_x_failed}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Freq x Failed</span>
                    <span className="font-medium">{results.freq_x_failed}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Amount x Suspicious</span>
                    <span className="font-medium">{results.amt_x_suspicious}</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}