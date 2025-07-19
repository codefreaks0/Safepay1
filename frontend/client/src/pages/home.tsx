import React, { useState } from 'react';
import { useLocation } from 'wouter';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { BottomNav } from '@/components/navigation/bottom-nav';
import { 
  Bell, ShieldAlert, Search, ArrowRight, MoonIcon, SunIcon, Video, 
  Zap, Calculator, CreditCard, FileText, Gauge, Settings, ChevronRight,
  Gift, HelpCircle, Lock, MessageSquare, Phone, Users, AlertTriangle, X
} from 'lucide-react';
import { NotificationBar } from '@/components/ui/notification-bar';
import { useToast } from '@/hooks/use-toast';
import { useTheme } from '@/hooks/useTheme';
import { useAuthState } from '@/hooks/use-auth-state';
import { analyzeUpiRisk, UpiRiskAnalysis } from '@/lib/fraud-detection';
// import { VideoDetectionHomeButton } from '@/components/ui/video-detection-home-button';

export default function Home() {
  const [, setLocation] = useLocation();
  const [showNotification, setShowNotification] = useState(false);
  const [upiInput, setUpiInput] = useState('');
  const { toast } = useToast();
  const { isDark, setTheme } = useTheme();
  const { authState } = useAuthState();
  const [riskResult, setRiskResult] = useState<UpiRiskAnalysis | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showRiskModal, setShowRiskModal] = useState(false);

  const handleAlertClick = () => {
    setShowNotification(true);
  };
  
  const handleUpiSearch = async () => {
    if (!upiInput.trim()) {
      toast({
        title: "Empty Input",
        description: "Please enter a UPI ID to search",
        variant: "destructive",
      });
      setRiskResult(null);
      setShowRiskModal(false);
      return;
    }
    setIsLoading(true);
    setRiskResult(null);
    setShowRiskModal(false);
    let upiId = upiInput;
    if (!upiId.includes('@')) {
      upiId = upiId + '@okaxis';
      toast({
        title: "Processing",
        description: `Using demo format: ${upiId}`,
      });
    }
    try {
      const result = await analyzeUpiRisk(upiId);
      setRiskResult(result);
      setShowRiskModal(true);
    } catch (e) {
      toast({
        title: "Error",
        description: "Failed to fetch UPI risk data",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Close modal on outside click
  const handleBackdropClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) {
      setShowRiskModal(false);
    }
  };

  return (
    <div className="dark-bg-secondary h-screen overflow-hidden fixed inset-0 flex flex-col">
      {/* UPI Risk Modal at top, with backdrop */}
      {riskResult && showRiskModal && (
        <div
          className="fixed inset-0 z-50 flex items-start justify-center bg-black/30 backdrop-blur-sm"
          onClick={handleBackdropClick}
        >
          <div className="relative mt-8 flex flex-col items-center justify-center p-8 bg-white rounded-2xl shadow-2xl w-full max-w-md mx-auto animate-fade-in border border-gray-100">
            <button
              className="absolute top-3 right-3 text-gray-400 hover:text-gray-700"
              onClick={() => setShowRiskModal(false)}
              aria-label="Close"
            >
              <X className="w-6 h-6" />
            </button>
            {riskResult.riskLevel === 'High' && (
              <div className="mb-2">
                <AlertTriangle className="w-10 h-10 text-[#dc2626] drop-shadow-lg animate-bounce" />
              </div>
            )}
            <div
              className="text-2xl font-extrabold mb-2 tracking-wide"
              style={{ color: riskResult.riskLevel === 'High' ? '#dc2626' : riskResult.riskLevel === 'Medium' ? '#fbbf24' : '#22c55e' }}
            >
              {riskResult.riskLevel === 'High' ? 'Risky!' : riskResult.riskLevel === 'Medium' ? 'Caution!' : 'Safe!'}
            </div>
            <div className="relative flex items-center justify-center mb-2">
              <svg width="130" height="130">
                <defs>
                  <linearGradient id="red-glow" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0%" stopColor="#ff4d4f" />
                    <stop offset="100%" stopColor="#dc2626" />
                  </linearGradient>
                </defs>
                <circle cx="65" cy="65" r="58" stroke="#f3f4f6" strokeWidth="12" fill="none" />
                <circle
                  cx="65"
                  cy="65"
                  r="58"
                  stroke={riskResult.riskLevel === 'High' ? 'url(#red-glow)' : riskResult.riskLevel === 'Medium' ? '#fbbf24' : '#22c55e'}
                  strokeWidth="12"
                  fill="none"
                  strokeDasharray={2 * Math.PI * 58}
                  strokeDashoffset={2 * Math.PI * 58 * (1 - riskResult.riskPercentage / 100)}
                  strokeLinecap="round"
                  style={riskResult.riskLevel === 'High' ? { filter: 'drop-shadow(0 0 8px #dc2626aa)' } : {}}
                />
              </svg>
              <span
                className="absolute text-4xl font-extrabold"
                style={{
                  color: riskResult.riskLevel === 'High' ? '#dc2626' : riskResult.riskLevel === 'Medium' ? '#fbbf24' : '#22c55e',
                  textShadow: riskResult.riskLevel === 'High' ? '0 2px 8px #dc2626aa' : undefined,
                }}
              >
                {riskResult.riskPercentage}%
              </span>
            </div>
            <div className="text-base font-semibold mt-4 mb-1 text-gray-800">
              Reported {riskResult.reports} users for fraud
            </div>
            <div className="text-xs text-gray-500 mb-2">in last 6 months</div>
            <div className="text-xs text-gray-600 flex items-center gap-1">
              UPI ID:
              <span className="font-mono bg-gray-100 px-2 py-0.5 rounded cursor-pointer select-all" title="Copy UPI ID">
                {riskResult.upiId}
              </span>
            </div>
          </div>
        </div>
      )}
      {/* Top bar with search */}
      <div className="p-4 dark-bg-primary z-10 shadow-sm">
        <div className="flex items-center gap-2">
          <div className="flex-1 bg-slate-100 dark:bg-gray-700 rounded-full px-3 py-1.5 flex items-center transition-colors duration-300">
            <Search className="w-4 h-4 dark-text-tertiary mr-2 flex-shrink-0" />
            <Input 
              className="border-0 bg-transparent focus-visible:ring-0 focus-visible:ring-offset-0 placeholder:text-slate-500 dark:placeholder:text-gray-400 text-sm w-full h-8 dark-text-primary"
              placeholder="Enter UPI ID to check..."
              value={upiInput}
              onChange={(e) => {
                setUpiInput(e.target.value);
                if (!e.target.value) setRiskResult(null);
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  handleUpiSearch();
                }
              }}
            />
            {upiInput && (
              <Button 
                size="sm" 
                className="rounded-full h-7 w-7 p-0 flex items-center justify-center bg-blue-600 hover:bg-blue-700 dark:bg-blue-700 dark:hover:bg-blue-800"
                onClick={handleUpiSearch}
                disabled={isLoading}
              >
                {isLoading ? (
                  <span className="w-4 h-4 animate-spin border-2 border-white border-t-transparent rounded-full"></span>
                ) : (
                  <ArrowRight className="w-4 h-4" />
                )}
              </Button>
            )}
          </div>
          <button 
            onClick={() => setTheme(isDark ? 'light' : 'dark')}
            className="w-8 h-8 rounded-full flex items-center justify-center bg-slate-100 dark:bg-gray-700 text-slate-500 dark:text-gray-300 hover:bg-slate-200 dark:hover:bg-gray-600 transition-colors duration-300"
          >
            {isDark ? <SunIcon size={16} /> : <MoonIcon size={16} />}
          </button>
        </div>
      </div>
      
      {/* Show UPI Risk Result below search bar if available */}
      {riskResult && (
        <></>
      )}
      
      {/* Main content area - fixed height and scrollable if needed */}
      <div className="flex-1 overflow-y-auto pb-16">
        {/* Alert button */}
        <div className="px-4 py-6 flex justify-center">
          <Button 
            className="w-16 h-16 rounded-full bg-red-600 hover:bg-red-700 dark:bg-red-700 dark:hover:bg-red-800 text-white transition-colors duration-300"
            onClick={handleAlertClick}
          >
            <ShieldAlert className="w-6 h-6" />
          </Button>
        </div>
        
        {/* Menu items - symmetrically arranged grid */}
        <div className="px-4">

          
          {/* Grid layout with 4 columns and 2 rows (8 items total) */}
          <div className="grid grid-cols-4 gap-3 mb-6">
            {/* Row 1 - 4 items */}
            {/* Item 1: Scan and Pay */}
            <button 
              onClick={() => setLocation('/qr-scan')}
              className="flex flex-col items-center"
            >
              <div className="w-14 h-14 bg-blue-50 dark:bg-blue-900/30 rounded-xl flex items-center justify-center mb-1 shadow-sm">
                <svg 
                  xmlns="http://www.w3.org/2000/svg" 
                  viewBox="0 0 24 24" 
                  fill="none" 
                  stroke="currentColor" 
                  strokeWidth="2" 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  className="w-6 h-6 text-blue-500"
                >
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                  <rect x="7" y="7" width="3" height="3"></rect>
                  <rect x="14" y="7" width="3" height="3"></rect>
                  <rect x="7" y="14" width="3" height="3"></rect>
                  <rect x="14" y="14" width="3" height="3"></rect>
                </svg>
              </div>
              <span className="text-[10px] text-center font-medium dark-text-secondary">Scan & Pay</span>
            </button>
            
            {/* Item 2: Scam News */}
            <button 
              onClick={() => setLocation('/scam-news')}
              className="flex flex-col items-center"
            >
              <div className="w-14 h-14 bg-blue-50 dark:bg-blue-900/30 rounded-xl flex items-center justify-center mb-1 shadow-sm">
                <svg 
                  xmlns="http://www.w3.org/2000/svg" 
                  fill="none" 
                  viewBox="0 0 24 24" 
                  strokeWidth={1.5} 
                  stroke="currentColor" 
                  className="w-6 h-6 text-blue-500"
                >
                  <path 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                    d="M12 7.5h1.5m-1.5 3h1.5m-7.5 3h7.5m-7.5 3h7.5m3-9h3.375c.621 0 1.125.504 1.125 1.125V18a2.25 2.25 0 01-2.25 2.25M16.5 7.5V18a2.25 2.25 0 002.25 2.25M16.5 7.5V4.875c0-.621-.504-1.125-1.125-1.125H4.125C3.504 3.75 3 4.254 3 4.875V18a2.25 2.25 0 002.25 2.25h13.5M6 7.5h3v3H6v-3z"
                  />
                </svg>
              </div>
              <span className="text-[10px] text-center font-medium dark-text-secondary">Scam News</span>
            </button>
            
            {/* Item 3: Voice Check */}
            <button 
              onClick={() => setLocation('/voice-check')}
              className="flex flex-col items-center"
            >
              <div className="w-14 h-14 bg-blue-50 dark:bg-blue-900/30 rounded-xl flex items-center justify-center mb-1 shadow-sm">
                <Phone className="w-6 h-6 text-blue-500" />
              </div>
              <span className="text-[10px] text-center font-medium dark-text-secondary">Voice Check</span>
            </button>
            
            {/* Item 4: Report Scam */}
            <button 
              onClick={() => setLocation('/report-scam')}
              className="flex flex-col items-center"
            >
              <div className="w-14 h-14 bg-blue-50 dark:bg-blue-900/30 rounded-xl flex items-center justify-center mb-1 shadow-sm">
                <AlertTriangle className="w-6 h-6 text-blue-500" />
              </div>
              <span className="text-[10px] text-center font-medium dark-text-secondary">Report Scam</span>
            </button>
            
            {/* Row 2 - 4 items */}
            {/* Item 5: Security */}
            <button 
              onClick={() => setLocation('/security-settings')}
              className="flex flex-col items-center"
            >
              <div className="w-14 h-14 bg-blue-50 dark:bg-blue-900/30 rounded-xl flex items-center justify-center mb-1 shadow-sm">
                <Lock className="w-6 h-6 text-blue-500" />
              </div>
              <span className="text-[10px] text-center font-medium dark-text-secondary">Security</span>
            </button>
            
            {/* Item 6: Legal Support */}
            <button 
              onClick={() => setLocation('/legal-help')}
              className="flex flex-col items-center"
            >
              <div className="w-14 h-14 bg-blue-50 dark:bg-blue-900/30 rounded-xl flex items-center justify-center mb-1 shadow-sm">
                <FileText className="w-6 h-6 text-blue-500" />
              </div>
              <span className="text-[10px] text-center font-medium dark-text-secondary">Legal Support</span>
            </button>
            
            {/* Item 7: WhatsApp Check */}
            <button 
              onClick={() => setLocation('/whatsapp-check')}
              className="flex flex-col items-center"
            >
              <div className="w-14 h-14 bg-blue-50 dark:bg-blue-900/30 rounded-xl flex items-center justify-center mb-1 shadow-sm">
                <MessageSquare className="w-6 h-6 text-blue-500" />
              </div>
              <span className="text-[10px] text-center font-medium dark-text-secondary">WhatsApp</span>
            </button>
            
            {/* Item 8: View All Services */}
            <button 
              onClick={() => setLocation('/all-services')}
              className="flex flex-col items-center"
            >
              <div className="w-14 h-14 bg-blue-50 dark:bg-blue-900/30 rounded-xl flex items-center justify-center mb-1 shadow-sm">
                <ChevronRight className="w-6 h-6 text-blue-500" />
              </div>
              <span className="text-[10px] text-center font-medium dark-text-secondary">More Services</span>
            </button>
          </div>
        </div>
      </div>
      
      {/* Notification Bar */}
      {showNotification && (
        <NotificationBar
          message="Recent suspicious activity has been detected in your area. Please be vigilant with unknown UPI requests."
          onClose={() => setShowNotification(false)}
        />
      )}
      
      {/* Bottom Navigation */}
      <BottomNav />
    </div>
  );
}
