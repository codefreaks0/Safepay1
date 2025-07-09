import React from 'react';
import { useLocation } from 'wouter';
import { useQuery } from '@tanstack/react-query';
import { format } from 'date-fns';
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardHeader, 
  CardTitle 
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  ArrowLeft, 
  AlertTriangle, 
  Calendar, 
  Shield,
  Phone,
  MessageSquare,
  Globe
} from 'lucide-react';
import { apiRequest } from '@/lib/queryClient';
import { useAuthState } from '@/hooks/use-auth-state';
import { 
  Dialog, 
  DialogContent, 
  DialogDescription,
  DialogHeader, 
  DialogTitle
} from '@/components/ui/dialog';

// Define the interface to match backend ScamReport model
interface BackendScamReport {
  _id: string;
  userId: string;
  reportType: 'voice' | 'message' | 'whatsapp' | 'other';
  scamContact?: string; // Optional
  scamPlatform?: string; // Optional
  scamDetails: string;
  status: 'pending' | 'reviewed' | 'resolved' | 'rejected';
  createdAt: string; // ISO date string
  updatedAt: string; // ISO date string
}

// Formatted report for display, extending backend model
interface FormattedReport extends BackendScamReport {
  date: string;
  time: string;
}

export default function MyReports() {
  const [, setLocation] = useLocation();
  const [selectedReport, setSelectedReport] = React.useState<FormattedReport | null>(null);
  const [dialogOpen, setDialogOpen] = React.useState(false);
  const { authState } = useAuthState();
  
  // Removed useEffect for redirect if not logged in (for debugging purposes)
  // React.useEffect(() => {
  //   if (!authState.isLoggedIn) {
  //     setLocation('/login?returnUrl=/my-reports');
  //   }
  // }, [authState.isLoggedIn, setLocation]);
  
  // Get user ID from auth state (it's already a string)
  const userId = authState.userId;
  
  // Fetch user scam reports
  const { data: reports, isLoading, error } = useQuery<FormattedReport[], Error>({
    queryKey: ['myScamReports', userId], // Updated query key for clarity
    queryFn: async () => {
      if (!userId) {
        // If userId is null, it means not logged in, error will be handled by useEffect redirect
        throw new Error('User ID not available for fetching reports.');
      }
      
      // Corrected API endpoint URL with full path
      const res = await apiRequest('GET', `http://localhost:6900/api/scam-reports/${userId}`);
      
      // No need to await res.json() here if apiRequest already handles it, 
      // but let's assume apiRequest returns the raw Response object as before. 
      // If apiRequest resolves with already parsed JSON, remove this line.
      const data: BackendScamReport[] = await res.json();
      
      // Format dates for display
      return data.map((report: BackendScamReport) => {
        const timestamp = new Date(report.createdAt); // Use createdAt for timestamp
        return {
          ...report,
          date: format(timestamp, 'MMM d, yyyy'),
          time: format(timestamp, 'h:mm a'),
        };
      });
    },
    // Only enable query if userId is present
    enabled: !!userId,
    // Retry once if there's an error (e.g., transient network issue)
    retry: 1,
    // Keep old data while refetching in the background
    placeholderData: (previousData) => previousData,
  });
  
  const handleReportClick = (report: FormattedReport) => {
    setSelectedReport(report);
    setDialogOpen(true);
  };
  
  // Helper to get badge color based on reportType (simplified from original scamType)
  const getReportTypeColor = (reportType: BackendScamReport['reportType']) => {
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
  
  // Helper to get status badge color
  const getStatusColor = (status: BackendScamReport['status']) => {
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

  return (
    <div className="flex flex-col px-6 py-8 min-h-screen overflow-y-auto">
      <div className="flex items-center mb-6">
        <Button 
          variant="ghost" 
          size="icon" 
          className="mr-2" 
          onClick={() => setLocation('/account')}
        >
          <ArrowLeft className="h-5 w-5" />
        </Button>
        <h1 className="text-2xl font-bold">My Scam Reports</h1>
      </div>
      
      {isLoading ? (
        <div className="flex justify-center items-center h-60">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
        </div>
      ) : error ? (
        <Card className="bg-red-50 dark:bg-red-900/20 border-red-100 dark:border-red-800">
          <CardContent className="py-6">
            <div className="flex items-center justify-center gap-2 text-red-700 dark:text-red-400">
              <AlertTriangle />
              <p>Error loading your reports. Please try again later.</p>
            </div>
          </CardContent>
        </Card>
      ) : reports && reports.length > 0 ? (
        <div className="space-y-4 pb-6">
          {reports.map((report: FormattedReport) => (
            <Card 
              key={report._id} // Use _id from MongoDB
              className="overflow-hidden transition-all duration-200 hover:shadow-md cursor-pointer"
              onClick={() => handleReportClick(report)}
            >
              <CardHeader className="pb-2">
                <div className="flex justify-between items-start">
                  <CardTitle className="text-base">Report Type: {report.reportType}</CardTitle>
                  <Badge className={`${getReportTypeColor(report.reportType)} border`}>
                    {report.reportType.charAt(0).toUpperCase() + report.reportType.slice(1)}
                  </Badge>
                </div>
                <CardDescription className="flex items-center gap-1 text-xs">
                  <Calendar className="h-3 w-3" />
                  Reported on {report.date} at {report.time}
                </CardDescription>
              </CardHeader>
              <CardContent className="pb-4">
                <div className="flex justify-between items-center text-sm">
                  <div className="flex items-center gap-1 text-gray-600 dark:text-gray-400">
                    <Shield className="h-4 w-4" />
                    <span>Status: </span>
                    <Badge className={`${getStatusColor(report.status)}`}>
                      {report.status.charAt(0).toUpperCase() + report.status.slice(1)}
                    </Badge>
                  </div>
                  {report.scamContact && (
                    <div className="flex items-center gap-1 font-medium">
                      <Phone className="h-4 w-4 text-gray-500" />
                      <span>{report.scamContact}</span>
                    </div>
                  )}
                </div>
                {report.scamPlatform && (
                  <div className="flex items-center gap-1 text-sm mt-2 text-gray-600 dark:text-gray-400">
                    <Globe className="h-4 w-4" />
                    <span>Platform: {report.scamPlatform}</span>
                  </div>
                )}
                <p className="text-sm text-gray-700 dark:text-gray-300 mt-2 line-clamp-2">
                  {report.scamDetails}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center h-60 text-center">
          <Shield className="h-16 w-16 text-gray-300 dark:text-gray-600 mb-4" />
          <h3 className="text-xl font-medium mb-2">No Reports Yet</h3>
          <p className="text-gray-500 dark:text-gray-400 mb-6 max-w-md">
            You haven't submitted any scam reports yet. If you encounter any suspicious activities, report them to help the community.
          </p>
          <Button onClick={() => setLocation('/report-scam')}>
            Report a Scam
          </Button>
        </div>
      )}
      
      {/* Report Detail Dialog */}
      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent className="max-h-[90vh] overflow-y-auto">
          {selectedReport && (
            <>
              <DialogHeader>
                <DialogTitle className="text-xl mb-1">Report Details</DialogTitle>
                <DialogDescription>
                  Reported on {selectedReport.date} at {selectedReport.time}
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4 mt-4">
                <div>
                  <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">Report Type</h3>
                  <p className="text-lg font-semibold">{selectedReport.reportType.charAt(0).toUpperCase() + selectedReport.reportType.slice(1)}</p>
                </div>
                
                {selectedReport.scamContact && (
                  <div>
                    <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">Scam Contact</h3>
                    <p className="text-lg font-semibold">{selectedReport.scamContact}</p>
                  </div>
                )}

                {selectedReport.scamPlatform && (
                  <div>
                    <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">Scam Platform</h3>
                    <p className="text-lg font-semibold">{selectedReport.scamPlatform}</p>
                  </div>
                )}
                
                <div>
                  <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">Scam Details</h3>
                  <p className="text-lg">{selectedReport.scamDetails}</p>
                </div>
                
                <div>
                  <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">Status</h3>
                  <Badge className={`${getStatusColor(selectedReport.status)}`}>
                    {selectedReport.status.charAt(0).toUpperCase() + selectedReport.status.slice(1)}
                  </Badge>
                </div>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}