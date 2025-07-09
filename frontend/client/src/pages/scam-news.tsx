import React, { useState, useEffect } from 'react';
import { useLocation } from 'wouter';
import { 
  Card, 
  CardContent, 
  CardFooter, 
  CardHeader, 
  CardTitle 
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { 
  AlertOctagon, 
  AlertTriangle, 
  Shield, 
  Activity, 
  MapPin, 
  ChevronRight, 
  ArrowUpRight, 
  Search, 
  Loader2, 
  CheckCircle,
  Info,
  ArrowLeft
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { NewsDetail } from '@/components/scam-news/news-detail';
import { Dialog, DialogContent } from '@/components/ui/dialog';
import * as GroqModule from 'groq-sdk';

const Groq = GroqModule.default || GroqModule;

// Initialize Groq client
const groq = new Groq({
  apiKey: (import.meta.env.VITE_GROQ_API_KEY || '') as string,
  dangerouslyAllowBrowser: true, // Required for browser environments
});

interface ScamAlert {
  title: string;
  type: string;
  description: string;
  affected_areas: string[];
  risk_level: 'High' | 'Medium' | 'Low';
  date_reported: string;
  verification_status: 'Verified' | 'Investigating' | 'Unverified';
}

interface HeatmapPoint {
  city: string;
  coordinates: [number, number];
  intensity: number;
  scam_type: string;
  trend: 'Increasing' | 'Stable' | 'Decreasing';
}

interface PreventionTip {
  tip: string;
  category: string;
}

interface ReportsSummary {
  total_reports: number;
  most_reported: string[];
  financial_loss: string;
  emerging_patterns: string[];
  hotspot_areas: string[];
}

interface UpiAnalysis {
  risk_level: string;
  confidence?: number;
  analysis: string;
  flags?: string[];
  recommendations?: string[];
}

interface ScamNewsData {
  alerts: ScamAlert[];
  geo_spread: HeatmapPoint[];
  prevention_tips: PreventionTip[];
  reports_summary: ReportsSummary;
  upi_analysis: UpiAnalysis | null;
  trust_score: number;
  last_updated: string;
}

export default function ScamNews() {
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  
  const [data, setData] = useState<ScamNewsData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('alerts');
  const [upiSearch, setUpiSearch] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [selectedAlert, setSelectedAlert] = useState<ScamAlert | null>(null);
  const [showDetailDialog, setShowDetailDialog] = useState(false);
  
  // Check if we have a saved alert from the home screen
  useEffect(() => {
    const savedAlert = localStorage.getItem('selectedScamAlert');
    if (savedAlert) {
      try {
        const alertData = JSON.parse(savedAlert);
        setSelectedAlert(alertData);
        setShowDetailDialog(true);
        // Clear the stored data after using it
        localStorage.removeItem('selectedScamAlert');
      } catch (error) {
        console.error('Error parsing saved alert:', error);
      }
    }
  }, []);
  
  // Fetch scam news data from Groq
  const fetchScamNews = async () => {
    if (isLoading) return;
    
    setIsLoading(true);
    
    try {
      const randomSeed = Math.random().toString(36).substring(2, 10); // 8-char random string
      const now = new Date().toISOString();
      const prompt = `Give me top 5 fraud alerts nearby Bhopal, India as of ${now}. For each alert, include:
      - title (concise summary)
      - type (e.g., UPI, ATM, Phishing, Loan, Job, Lottery)
      - description (1-2 sentences detailed description)
      - affected_areas (array of strings, including 'Bhopal')
      - risk_level ('High', 'Medium', or 'Low')
      - date_reported (recent date, e.g., '2023-10-27')
      - verification_status ('Verified', 'Investigating', or 'Unverified')

      Format the response as a JSON array of objects, strictly following the provided structure. Ensure variety and realistic details for each alert. DO NOT include any text outside the JSON.

      Randomization seed: ${randomSeed}
      `;

      const chatCompletion = await groq.chat.completions.create({
        messages: [
          { role: "system", content: "You are a helpful assistant that generates realistic scam alerts." },
          { role: "user", content: prompt },
        ],
        model: "llama3-8b-8192", 
        temperature: 0.8, // Slightly higher temperature for more varied responses
        max_tokens: 1500,
        response_format: { type: "json_object" }, // Request JSON object
      });

      const rawResponse = chatCompletion.choices[0]?.message?.content;
      
      if (!rawResponse) {
        throw new Error("Groq did not return any content.");
      }

      // Groq might return JSON wrapped in markdown or extra text. Try to extract just the JSON.
      let jsonString = rawResponse;
      const jsonMatch = rawResponse.match(/```json\n([\s\S]*?)\n```/);
      if (jsonMatch && jsonMatch[1]) {
        jsonString = jsonMatch[1];
      } else if (rawResponse.startsWith('{') && rawResponse.endsWith('}')) {
        // If it's a direct JSON object string without markdown
        // do nothing, jsonString is already rawResponse
      } else if (rawResponse.startsWith('[') && rawResponse.endsWith(']')) {
        // If it's a direct JSON array string without markdown
        // do nothing, jsonString is already rawResponse
      } else {
        // Fallback: assume the entire response is the JSON content if no markdown block found
        console.warn("Groq response did not contain a JSON markdown block. Attempting to parse raw response.", rawResponse);
      }

      const parsed = JSON.parse(jsonString);
      const generatedAlerts: ScamAlert[] = Array.isArray(parsed.alerts)
        ? parsed.alerts
        : Array.isArray(parsed)
          ? parsed
          : parsed.alerts
            ? [parsed.alerts]
            : [];
      
      // Mock other data fields as Groq is only providing alerts
      setData(prevData => ({
        alerts: [...(prevData?.alerts || []), ...generatedAlerts],
        geo_spread: prevData?.geo_spread || [
          { city: 'Bhopal', coordinates: [23.2599, 77.4126], intensity: 0.8, scam_type: 'Phishing', trend: 'Increasing' },
          { city: 'Indore', coordinates: [22.7196, 75.8577], intensity: 0.6, scam_type: 'UPI', trend: 'Stable' },
        ],
        prevention_tips: prevData?.prevention_tips || [
          { tip: 'Always verify the identity of the caller/sender.', category: 'General' },
          { tip: 'Never share your UPI PIN or OTP with anyone.', category: 'UPI' },
          { tip: 'Be wary of unsolicited messages offering lottery wins.', category: 'Lottery' },
        ],
        reports_summary: prevData?.reports_summary || {
          total_reports: Math.floor(Math.random() * 100) + 50,
          most_reported: ['UPI fraud', 'OTP scam'],
          financial_loss: `₹${(Math.random() * 500000).toFixed(2)}`,
          emerging_patterns: ['Fake job offers', 'Online shopping scams'],
          hotspot_areas: ['Bhopal', 'Indore'],
        },
        upi_analysis: prevData?.upi_analysis || null,
        trust_score: Math.floor(Math.random() * 20) + 80,
        last_updated: new Date().toLocaleString(),
      }));
      
      toast({
        title: 'Scam alerts updated',
        description: `Found ${generatedAlerts.length} new alerts for Bhopal.`,
      });

    } catch (error: any) {
      console.error('Error fetching scam alerts from Groq:', error);
      toast({
        title: 'Error',
        description: 'Failed to fetch scam alerts. Please try again.',
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };
  
  // Search UPI ID for potential scams
  const searchUpiId = async () => {
    if (!upiSearch.trim() || isSearching) return;
    
    setIsSearching(true);
    
    try {
      // Use Groq to analyze UPI ID. This will require a new prompt similar to fetchScamNews
      // For now, I'll mock this part, as the primary request was for scam alerts.
      const mockUpiAnalysis: UpiAnalysis = {
        risk_level: Math.random() > 0.7 ? 'High' : Math.random() > 0.3 ? 'Medium' : 'Low',
        confidence: parseFloat((Math.random()).toFixed(2)),
        analysis: "This UPI ID has some suspicious patterns based on our simulated analysis.",
        flags: ['Frequent sender', 'Unverified merchant'],
        recommendations: ['Verify recipient details', 'Proceed with caution'],
      };

      setData(prevData => prevData ? {
        ...prevData,
        upi_analysis: mockUpiAnalysis
      } : null);
      
      // Switch to UPI tab to show results
      setActiveTab('upi');
    } catch (error) {
      console.error('Error analyzing UPI ID:', error);
      toast({
        title: 'Analysis Error',
        description: 'Failed to analyze UPI ID. Please try again.',
        variant: 'destructive',
      });
    } finally {
      setIsSearching(false);
    }
  };
  
  // Fetch data on component mount
  useEffect(() => {
    fetchScamNews();
    
    // Set up auto refresh every 24 hours (for alerts only, not UPI search)
    const refreshInterval = setInterval(fetchScamNews, 24 * 60 * 60 * 1000); // 24 hours
    
    // Cleanup on unmount
    return () => clearInterval(refreshInterval);
  }, []); // Empty dependency array means this runs once on mount

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    searchUpiId();
  };

  const handleAlertSelect = (alert: ScamAlert) => {
    setSelectedAlert(alert);
    setShowDetailDialog(true);
  };

  const handleReportSimilar = () => {
    // Implement logic to report similar scam
    toast({
      title: 'Reported',
      description: 'Similar scam reported. Thank you for your contribution!',
    });
  };

  const renderRiskBadge = (level: string) => {
    let color = '';
    switch (level) {
      case 'High':
        color = 'bg-red-500 hover:bg-red-600';
        break;
      case 'Medium':
        color = 'bg-orange-500 hover:bg-orange-600';
        break;
      case 'Low':
        color = 'bg-green-500 hover:bg-green-600';
        break;
      default:
        color = 'bg-gray-500 hover:bg-gray-600';
    }
    return <Badge className={`text-white ${color}`}>{level}</Badge>;
  };

  const renderVerificationBadge = (status: string) => {
    let color = '';
    switch (status) {
      case 'Verified':
        color = 'bg-green-100 text-green-700';
        break;
      case 'Investigating':
        color = 'bg-orange-100 text-orange-700';
        break;
      case 'Unverified':
        color = 'bg-gray-100 text-gray-700';
        break;
      default:
        color = 'bg-gray-100 text-gray-700';
    }
    return <Badge variant="secondary" className={`${color}`}>{status}</Badge>;
  };

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-4xl mx-auto">
        <button
          onClick={() => setLocation('/home')}
          className="flex items-center text-gray-600 mb-6"
        >
          <ArrowLeft className="h-5 w-5 mr-2" />
          Back
        </button>

        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="text-2xl font-bold flex items-center">
              <AlertOctagon className="mr-2 text-primary" /> Scam News & Alerts
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-gray-700 mb-4">
              Stay informed about the latest UPI fraud attempts and scams.
              Our AI assistant provides up-to-date alerts and analysis.
            </p>
            <div className="flex space-x-2">
              <Input
                placeholder="Enter UPI ID to analyze..."
                className="flex-1"
                value={upiSearch}
                onChange={(e) => setUpiSearch(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    searchUpiId();
                  }
                }}
              />
              <Button onClick={searchUpiId} disabled={isSearching}>
                {isSearching ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Search className="h-4 w-4 mr-2" />}
                Analyze
              </Button>
            </div>
          </CardContent>
        </Card>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="mb-6">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="alerts">Alerts</TabsTrigger>
            <TabsTrigger value="reports">Reports</TabsTrigger>
            <TabsTrigger value="upi">UPI Analysis</TabsTrigger>
          </TabsList>

          <TabsContent value="alerts" className="mt-4">
            {isLoading ? (
              <Card className="flex items-center justify-center p-8">
                <Loader2 className="h-8 w-8 animate-spin text-primary mr-3" />
                <span className="text-lg text-gray-600">Fetching latest scam alerts...</span>
              </Card>
            ) : data && data.alerts.length > 0 ? (
              <div className="space-y-4">
                {data.alerts.map((alert, index) => (
                  <Card key={index} className="cursor-pointer hover:shadow-lg transition-shadow"
                        onClick={() => handleAlertSelect(alert)}>
                    <CardContent className="p-4">
                      <div className="flex justify-between items-start mb-2">
                        <h3 className="font-semibold text-lg flex items-center">
                          {alert.title}
                          {renderRiskBadge(alert.risk_level)}
                        </h3>
                        <span className="text-sm text-gray-500">{alert.date_reported}</span>
                      </div>
                      <p className="text-gray-700 text-sm mb-2">{alert.description}</p>
                      <div className="flex flex-wrap gap-2 text-xs">
                        {alert.affected_areas.map((area, areaIndex) => (
                          <Badge key={areaIndex} variant="secondary" className="flex items-center">
                            <MapPin className="h-3 w-3 mr-1" /> {area}
                          </Badge>
                        ))}
                        <Badge variant="secondary">Type: {alert.type}</Badge>
                        {renderVerificationBadge(alert.verification_status)}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : (
              <Card className="text-center p-8">
                <Info className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <p className="text-lg text-gray-600">No scam alerts found for your area</p>
                <p className="text-sm text-gray-500 mt-2">Try refreshing the page or check back later for updates.</p>
                <Button onClick={fetchScamNews} className="mt-4">
                  <Loader2 className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
                  Refresh Alerts
                </Button>
              </Card>
            )}
            <div className="mt-6 text-center">
              <Button onClick={fetchScamNews} disabled={isLoading}>
                {isLoading ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Activity className="h-4 w-4 mr-2" />}
                Load More Alerts
              </Button>
            </div>
          </TabsContent>

          <TabsContent value="reports" className="mt-4">
            {/* Reports Summary Content */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Activity className="mr-2" /> Reports Summary
                </CardTitle>
              </CardHeader>
              <CardContent>
                {data ? (
                  <div className="space-y-4 text-gray-700">
                    <p>Total reports received: <span className="font-semibold">{data.reports_summary.total_reports}</span></p>
                    <p>Most reported scams: <span className="font-semibold">{data.reports_summary.most_reported.join(', ')}</span></p>
                    <p>Estimated financial loss: <span className="font-semibold text-red-600">{data.reports_summary.financial_loss}</span></p>
                    <p>Emerging patterns: <span className="font-semibold">{data.reports_summary.emerging_patterns.join(', ')}</span></p>
                    <p>Hotspot areas: <span className="font-semibold">{data.reports_summary.hotspot_areas.join(', ')}</span></p>
                  </div>
                ) : (
                  <p className="text-gray-600">No reports data available.</p>
                )}
              </CardContent>
              <CardFooter>
                <Button onClick={fetchScamNews} disabled={isLoading}>
                  <Loader2 className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
                  Refresh Reports
                </Button>
              </CardFooter>
            </Card>
          </TabsContent>

          <TabsContent value="upi" className="mt-4">
            {/* UPI Analysis Content */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Shield className="mr-2" /> UPI ID Analysis
                </CardTitle>
              </CardHeader>
              <CardContent>
                {data && data.upi_analysis ? (
                  <div className="space-y-4 text-gray-700">
                    <p>Risk Level: {renderRiskBadge(data.upi_analysis.risk_level)}</p>
                    {data.upi_analysis.confidence && <p>Confidence: <span className="font-semibold">{(data.upi_analysis.confidence * 100).toFixed(2)}%</span></p>}
                    <p>Analysis: <span className="font-semibold">{data.upi_analysis.analysis}</span></p>
                    {data.upi_analysis.flags && data.upi_analysis.flags.length > 0 && (
                      <p>Flags: <span className="font-semibold">{data.upi_analysis.flags.join(', ')}</span></p>
                    )}
                    {data.upi_analysis.recommendations && data.upi_analysis.recommendations.length > 0 && (
                      <p>Recommendations: <span className="font-semibold">{data.upi_analysis.recommendations.join(', ')}</span></p>
                    )}
                  </div>
                ) : ( 
                  <p className="text-gray-600">Enter a UPI ID above to get an analysis.</p>
                )}
              </CardContent>
              <CardFooter>
                <Button onClick={() => setUpiSearch('')} disabled={isSearching}>
                  Clear Search
                </Button>
              </CardFooter>
            </Card>
          </TabsContent>
        </Tabs>

        <Dialog open={showDetailDialog} onOpenChange={setShowDetailDialog}>
          <DialogContent className="sm:max-w-[800px]">
            {selectedAlert && (
              <NewsDetail 
                alert={selectedAlert} 
                onReportSimilar={handleReportSimilar} 
                onClose={() => setShowDetailDialog(false)}
              />
            )}
          </DialogContent>
        </Dialog>
      </div>
    </div>
  );
}