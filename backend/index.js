const express = require('express');
const session = require('express-session');
const mongoose = require('mongoose');
const cors = require('cors');
const fileUpload = require('express-fileupload');
const User = require('./models/User');
const PaymentMethod = require('./models/PaymentMethod');
const ScamReport = require('./models/ScamReport');
const Transaction = require('./models/Transaction');
const MongoStore = require('connect-mongo');
const path = require('path');
const axios = require('axios');
const fs = require('fs');
const multer = require('multer');
const FormData = require('form-data');

const app = express();

// --- BEGIN CORE MIDDLEWARE CONFIGURATION ---

// 1. CORS Middleware (MUST come before session middleware when using credentials)
app.use(cors({
  origin: 'http://localhost:5173', // Explicitly allow your frontend origin
  credentials: true // Crucial: Allows cookies to be sent with cross-origin requests
}));

// 2. JSON Body Parser
app.use(express.json());

// 3. File Upload Middleware
// app.use(fileUpload({
//   limits: { fileSize: 5 * 1024 * 1024 }, // 5MB max file size
//   abortOnLimit: true,
//   responseOnLimit: 'File size limit has been reached',
//   useTempFiles: true,
//   tempFileDir: '/tmp/'
// }));

// MongoDB connection (moved up to be available for session store)
mongoose.connect('mongodb+srv://yashsajwan2004:1NV2Y7QwL6YU8nEF@cluster0.ycb55bx.mongodb.net/', {
})
.then(() => console.log('MongoDB connected successfully.'))
.catch(err => console.error('MongoDB connection error:', err));

// 3. Express Session Middleware
app.use(session({
  name: 'safepay.sid', // Explicitly set a custom session cookie name
  secret: 'd01c0b3a3c9b7e7a7f4a2b9d8e6c5a4f3b2c1d0e9a8f7b6a5c4d3e2f1a0b9c8d7e6f5a4b3c2d1e0f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3',
  resave: false,
  saveUninitialized: false,
  proxy: true,
  store: MongoStore.create({
    mongoUrl: 'mongodb+srv://yashsajwan2004:1NV2Y7QwL6YU8nEF@cluster0.ycb55bx.mongodb.net/',
    collectionName: 'sessions',
    ttl: 1000 * 60 * 60 * 24 // 24 hours (matches cookie maxAge)
  }),
  cookie: {
    maxAge: 1000 * 60 * 60 * 24,
    secure: false,
    httpOnly: true,
    sameSite: 'lax',
    path: '/'
  }
}));

// Add a test middleware to confirm session is initialized
app.use((req, res, next) => {
  console.log('Middleware: Session ID on incoming request:', req.sessionID);
  console.log('Middleware: req.session initialized:', !!req.session);
  if (req.session && req.session.userId) {
    console.log('Middleware: req.session.userId:', req.session.userId);
  }
  next();
});

// --- END CORE MIDDLEWARE CONFIGURATION ---

// Login route
app.post('/login', async (req, res) => {
  const { phone } = req.body;
  console.log('Login attempt for phone:', phone);
  if (!phone) return res.status(400).json({ error: 'Phone required' });
  try {
    const user = await User.findOne({ phone });
    console.log('User found (null if not found): ', user);
    if (user) {
      req.session.userId = user._id; // Set user ID in session
      console.log('Login Success: Session userId set to: ', req.session.userId);
      res.json({ success: true, user });

    } else {
      // User not found, instruct frontend to redirect to signup
      console.log('User not found, redirecting to signup...');
      res.json({ exists: false, redirect: 'signup' });
    }
  } catch (err) {
    console.error('DB error during login:', err);
    res.status(500).json({ error: 'DB error', details: err.message });
  }
});

// Signup route
app.post('/signup', async (req, res) => {
  const { phone, name } = req.body;
  if (!phone || !name) return res.status(400).json({ error: 'Phone and name required' });
  try {
    const newUser = await User.create({ phone, name });
    req.session.userId = newUser._id; // Set user ID in session
    console.log('Signup Success: Session userId set to: ', req.session.userId);
    res.json({ success: true, user: newUser });
  } catch (err) {
    console.error('Error during signup:', err); // Log the full error for debugging
    if (err.code === 11000) {
      // Duplicate key error (e.g., phone number already exists)
      res.status(400).json({ error: 'User with this phone number already exists.', details: err.message });
    } else {
      // Other database errors
      res.status(500).json({ error: 'Failed to create user due to a database error.', details: err.message });
    }
  }
});

// Profile route (protected)
app.get('/profile/:userId', async (req, res) => {
  console.log('Profile request received. Session userId: ', req.session.userId, 'Requested userId:', req.params.userId);
  
  if (!req.session.userId || req.session.userId.toString() !== req.params.userId) {
    console.log('Profile: Session userId mismatch or missing.');
    return res.status(401).json({ error: 'Unauthorized', message: 'You are not authorized to view this profile or your session has expired.' });
  }

  try {
    const user = await User.findById(req.params.userId)
      .populate('paymentMethods') // Populate payment methods
      .populate('scamReports'); // Populate scam reports

    if (!user) {
      console.log('Profile: User not found for userId:', req.params.userId);
      return res.status(404).json({ error: 'User not found' });
    }
    console.log('Profile: User data fetched successfully.');
    res.json({ user });
  } catch (err) {
    console.error('DB error during profile fetch:', err);
    res.status(500).json({ error: 'DB error', details: err.message });
  }
});

// Logout route
app.post('/logout', (req, res) => {
  req.session.destroy(err => {
    if (err) {
      console.error('Error destroying session:', err);
      return res.status(500).json({ error: 'Logout failed' });
    }
    res.clearCookie('safepay.sid'); // Clear the specific session cookie name
    res.json({ success: true });
  });
});

// Update Profile route
app.put('/profile/:userId', async (req, res) => {
  if (!req.session.userId || req.session.userId.toString() !== req.params.userId) {
    return res.status(401).json({ error: 'Unauthorized' });
  }
  const { name, email, address, dob } = req.body;

  if (!name && !email && !address && !dob) {
    return res.status(400).json({ error: 'No fields to update provided' });
  }

  try {
    const updatedUser = await User.findByIdAndUpdate(
      req.params.userId,
      { $set: { name, email, address, dob } }, // Use $set to update specific fields
      { new: true, runValidators: true } // Return the updated document and run schema validators
    );

    if (!updatedUser) {
      return res.status(404).json({ error: 'User not found' });
    }
    res.json({ success: true, user: updatedUser });
  } catch (err) {
    console.error('DB error during profile update:', err);
    res.status(500).json({ error: 'DB error', details: err.message });
  }
});

// Add Payment Method route
app.post('/api/payment-methods', async (req, res) => {
  if (!req.session.userId) {
    return res.status(401).json({ error: 'Not logged in' });
  }

  const { type, name, upiId, cardNumber, expiryDate, accountNumber, ifscCode } = req.body;

  // --- DEBUG LOGS START ---
  console.log('--- Receiving Payment Method POST Request ---');
  console.log('Session User ID:', req.session.userId);
  console.log('Received Body:', req.body);
  // --- DEBUG LOGS END ---

  try {
    const newPaymentMethod = new PaymentMethod({
      userId: req.session.userId,
      type,
      name,
      upiId,
      cardNumber,
      expiryDate,
      accountNumber,
      ifscCode,
    });

    // --- DEBUG LOGS START ---
    console.log('PaymentMethod object before save:', newPaymentMethod);
    // --- DEBUG LOGS END ---

    await newPaymentMethod.save();

    // Update the User document with the new payment method reference
    await User.findByIdAndUpdate(
      req.session.userId,
      { $push: { paymentMethods: newPaymentMethod._id } },
      { new: true } // Return the updated user document (optional, but good practice)
    );

    // --- DEBUG LOGS START ---
    console.log('PaymentMethod saved successfully:', newPaymentMethod);
    // --- DEBUG LOGS END ---

    res.status(201).json({ success: true, method: newPaymentMethod });
  } catch (err) {
    console.error('Error adding payment method:', err);
    res.status(500).json({ error: 'Failed to add payment method', details: err.message });
  }
});

// Get Payment Methods for a user
app.get('/api/payment-methods/:userId', async (req, res) => {
  // Convert userId to string for proper comparison and security
  if (!req.session.userId || req.session.userId.toString() !== req.params.userId) {
    return res.status(401).json({ error: 'Unauthorized' });
  }
  try {
    const paymentMethods = await PaymentMethod.find({ userId: req.params.userId });
    console.log(`Fetched ${paymentMethods.length} payment methods for user ${req.params.userId}`);
    res.json(paymentMethods);
  } catch (err) {
    console.error('Error fetching payment methods:', err);
    res.status(500).json({ error: 'Failed to fetch payment methods', details: err.message });
  }
});

// Delete Payment Method
app.delete('/api/payment-methods/:methodId', async (req, res) => {
  try {
    const method = await PaymentMethod.findById(req.params.methodId);
    if (!method) {
      return res.status(404).json({ error: 'Payment method not found' });
    }
    // Ensure the method belongs to the logged-in user
    if (!req.session.userId || method.userId.toString() !== req.session.userId.toString()) {
      return res.status(401).json({ error: 'Unauthorized to delete this method' });
    }
    await PaymentMethod.findByIdAndDelete(req.params.methodId);
    res.json({ success: true, message: 'Payment method deleted successfully' });
  } catch (err) {
    console.error('Error deleting payment method:', err);
    res.status(500).json({ error: 'Failed to delete payment method', details: err.message });
  }
});

// Set Default Payment Method
app.post('/api/payment-methods/:userId/set-default/:methodId', async (req, res) => {
  // Convert userId to string for proper comparison
  if (!req.session.userId || req.session.userId.toString() !== req.params.userId) {
    return res.status(401).json({ error: 'Unauthorized' });
  }
  try {
    // Unset current default for this user
    await PaymentMethod.updateMany({ userId: req.params.userId, isDefault: true }, { isDefault: false });

    // Set new default
    const updatedMethod = await PaymentMethod.findByIdAndUpdate(
      req.params.methodId,
      { isDefault: true },
      { new: true }
    );

    if (!updatedMethod) {
      return res.status(404).json({ error: 'Payment method not found' });
    }

    res.json({ success: true, method: updatedMethod });
  } catch (err) {
    console.error('Error setting default payment method:', err);
    res.status(500).json({ error: 'Failed to set default payment method', details: err.message });
  }
});

// Add Scam Report route
app.post('/api/scam-reports', async (req, res) => {
  // Ensure user is logged in
  if (!req.session.userId) {
    return res.status(401).json({ error: 'Not logged in' });
  }

  // Validate required fields
  const { reportType, scamContact, scamPlatform, scamDetails } = req.body;
  if (!reportType || !scamDetails) {
    return res.status(400).json({ error: 'Missing required fields: reportType and scamDetails are required.' });
  }
  if (reportType !== 'other' && !scamContact) {
    return res.status(400).json({ error: 'Scam contact is required for this report type.' });
  }

  try {
    // Prepare new scam report
    const newScamReport = new ScamReport({
      userId: req.session.userId,
      reportType,
      scamContact: scamContact || null,
      scamPlatform: scamPlatform || null,
      scamDetails,
      // screenshot: not implemented yet
    });

    await newScamReport.save();

    // Link report to user
    await User.findByIdAndUpdate(
      req.session.userId,
      { $push: { scamReports: newScamReport._id } },
      { new: true }
    );

    res.status(201).json({ success: true, report: newScamReport });
  } catch (err) {
    console.error('Error adding scam report:', err);
    res.status(500).json({ error: 'Failed to add scam report', details: err.message });
  }
});

// Get Scam Reports for a user
app.get('/api/scam-reports/:userId', async (req, res) => {
  if (!req.session.userId || req.session.userId.toString() !== req.params.userId) {
    return res.status(401).json({ error: 'Unauthorized' });
  }
  try {
    const scamReports = await ScamReport.find({ userId: req.params.userId }).sort({ createdAt: -1 }); // Sort by newest first
    res.json(scamReports);
  } catch (err) {
    console.error('Error fetching scam reports:', err);
    res.status(500).json({ error: 'Failed to fetch scam reports', details: err.message });
  }
});

// Process audio for voice analysis
const ASSEMBLY_API_KEY = 'd328086b73264cd39534ba4e82a1046f';
app.post('/api/process-audio', async (req, res) => {
  if (!req.session.userId) {
    return res.status(401).json({ error: 'Not logged in' });
  }

  // If transcript is sent directly (JSON)
  if (req.body && req.body.transcript) {
    try {
      // Call your AI service with the transcript
      const response = await axios.post('https://safepay1-3sms.onrender.com/analyze-voice', {
        transcript: req.body.transcript
      });
      
      // Return both transcript and analysis results
      return res.json({
        transcript: req.body.transcript,
        analysis: response.data
      });
    } catch (error) {
      console.error('Error analyzing transcript:', error);
      return res.status(500).json({ error: 'Failed to analyze transcript', details: error.message });
    }
  }

  // AssemblyAI audio transcription
  const audioFile = req.files?.audio;
  if (!audioFile) {
    return res.status(400).json({ error: 'No audio file provided' });
  }

  // Ensure uploads directory exists
  const uploadsDir = path.join(__dirname, 'uploads');
  if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir);
  }

  // Save the audio file temporarily
  const uploadPath = path.join(uploadsDir, `${Date.now()}-${audioFile.name}`);
  await audioFile.mv(uploadPath);

  try {
    // 1. Upload audio to AssemblyAI
    const audioData = fs.readFileSync(uploadPath);
    const uploadRes = await axios.post(
      'https://api.assemblyai.com/v2/upload',
      audioData,
      {
        headers: {
          'authorization': ASSEMBLY_API_KEY,
          'transfer-encoding': 'chunked'
        }
      }
    );
    const audioUrl = uploadRes.data.upload_url;

    // 2. Request transcription
    const transcriptRes = await axios.post(
      'https://api.assemblyai.com/v2/transcript',
      { audio_url: audioUrl },
      { headers: { 'authorization': ASSEMBLY_API_KEY } }
    );
    const transcriptId = transcriptRes.data.id;

    // 3. Poll for completion
    let transcript;
    for (let i = 0; i < 30; i++) {
      await new Promise(r => setTimeout(r, 2000));
      const pollingRes = await axios.get(
        `https://api.assemblyai.com/v2/transcript/${transcriptId}`,
        { headers: { 'authorization': ASSEMBLY_API_KEY } }
      );
      if (pollingRes.data.status === 'completed') {
        transcript = pollingRes.data.text;
        break;
      } else if (pollingRes.data.status === 'failed') {
        throw new Error('Transcription failed');
      }
    }

    // Clean up temp file
    fs.unlinkSync(uploadPath);

    if (!transcript) {
      return res.status(500).json({ error: 'Transcription timed out' });
    }

    // 4. Run scam analysis on the transcribed text
    try {
      const analysisResponse = await axios.post('https://safepay1-3sms.onrender.com/analyze-voice', {
        transcript: transcript
      });
      
      // Return both transcript and analysis results
      return res.json({
        transcript: transcript,
        analysis: analysisResponse.data
      });
    } catch (analysisError) {
      console.error('Error analyzing transcribed text:', analysisError);
      // Return transcript even if analysis fails
      return res.json({
        transcript: transcript,
        analysis: {
          is_scam: false,
          confidence: 0,
          risk_score: 0,
          scam_type: null,
          scam_indicators: [],
          analysis_method: 'transcription_only'
        }
      });
    }

  } catch (error) {
    // Clean up temp file if it exists
    if (fs.existsSync(uploadPath)) {
      fs.unlinkSync(uploadPath);
    }
    console.error('AssemblyAI error:', error);
    return res.status(500).json({ error: 'Failed to transcribe audio', details: error.message });
  }
});

// Process a new payment transaction
app.post('/api/transactions/process', async (req, res) => {
  if (!req.session.userId) {
    return res.status(401).json({ error: 'Not logged in' });
  }

  const { fromUpiId, toUpiId, amount, status = 'success' } = req.body; // Default status to 'success' for now

  if (!fromUpiId || !toUpiId || !amount) {
    return res.status(400).json({ error: 'From UPI ID, To UPI ID, and Amount are required' });
  }

  try {
    // Generate a 12-digit random alphanumeric transaction ID
    const randomPart = (Math.random().toString(36).substring(2) + Math.random().toString(36).substring(2) + Math.random().toString(36).substring(2)).substring(0, 12).toUpperCase();
    const transactionId = `TXN-${randomPart}`;

    const newTransaction = new Transaction({
      userId: req.session.userId,
      fromUpiId,
      toUpiId,
      amount,
      status,
      transactionId,
    });

    await newTransaction.save();
    console.log('Transaction saved successfully:', newTransaction);

    res.status(201).json({ success: true, transaction: newTransaction });
  } catch (err) {
    console.error('Error processing transaction:', err);
    res.status(500).json({ error: 'Failed to process transaction', details: err.message });
  }
});

// Get transaction history for a user
app.get('/api/transactions/:userId', async (req, res) => {
  if (!req.session.userId || req.session.userId.toString() !== req.params.userId) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  try {
    const transactions = await Transaction.find({ userId: req.params.userId }).sort({ transactionDate: -1 });
    console.log(`Fetched ${transactions.length} transactions for user ${req.params.userId}`);
    res.json(transactions);
  } catch (err) {
    console.error('Error fetching transactions:', err);
    res.status(500).json({ error: 'Failed to fetch transactions', details: err.message });
  }
});

// Test session routes (temporary for debugging)
app.get('/set-session', (req, res) => {
  req.session.test = 'Session is working!';
  res.json({ message: 'Session variable set' });
});

app.get('/get-session', (req, res) => {
  if (req.session.test) {
    res.json({ message: `Session variable: ${req.session.test}` });
  } else {
    res.status(404).json({ message: 'Session variable not found or session expired' });
  }
});

// Get MFA/SIM Swap settings for a user
app.get('/api/security/mfa-settings/:userId', async (req, res) => {
  if (!req.session.userId || req.session.userId.toString() !== req.params.userId) {
    return res.status(401).json({ error: 'Unauthorized' });
  }
  // Dummy response for now (customize as needed)
  res.json({
    simSwapEnabled: false,
    lastChecked: null,
    mfaEnabled: false,
    mfaMethods: []
  });
});

// Proxy OCR Extract to Flask
app.post('/api/ocr-extract', multer().single('image'), async (req, res) => {
  try {
    const form = new FormData();
    form.append('image', req.file.buffer, req.file.originalname);
    const flaskRes = await fetch('http://localhost:8090/ocr-extract', {
      method: 'POST',
      body: form,
      headers: form.getHeaders(),
    });
    const data = await flaskRes.json();
    res.status(flaskRes.status).json(data);
  } catch (err) {
    res.status(500).json({ error: 'Proxy OCR failed', details: err.message });
  }
});

// Proxy Analyze Text to Flask
app.post('/api/analyze-text', express.json(), async (req, res) => {
  try {
    const flaskRes = await fetch('http://localhost:8090/predict-text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body),
    });
    const data = await flaskRes.json();
    res.status(flaskRes.status).json(data);
  } catch (err) {
    res.status(500).json({ error: 'Proxy analyze failed', details: err.message });
  }
});

// Proxy Analyze Video to FastAPI video detection service
app.post('/api/analyze-video', multer().single('video'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No video file uploaded' });
    }
    const form = new FormData();
    // Send as 'video_file' to FastAPI
    form.append('video_file', req.file.buffer, req.file.originalname);
    const fastapiRes = await fetch('http://localhost:8083/analyze-video', {
      method: 'POST',
      body: form,
      headers: form.getHeaders(),
    });
    const data = await fastapiRes.json();
    res.status(fastapiRes.status).json(data);
  } catch (err) {
    res.status(500).json({ error: 'Proxy video analysis failed', details: err.message });
  }
});

const upload = multer();

app.post('/api/analyze-whatsapp', upload.single('screenshot'), async (req, res) => {
  try {
    const formData = new FormData();
    formData.append('screenshot', req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype,
    });
    const response = await axios.post('http://localhost:8090/analyze-whatsapp', formData, {
      headers: formData.getHeaders(),
    });
    res.json(response.data);
  } catch (error) {
    res.status(500).json({ error: 'Failed to analyze WhatsApp screenshot.' });
  }
});

// --- BEGIN UPI RISK CHECK ENDPOINT ---
/**
 * GET /api/upi/check/:upiId
 * Proxies to Flask AI service for real UPI risk analysis.
 */
app.get('/api/upi/check/:upiId', async (req, res) => {
  console.log('UPI param:', req.params.upiId);
  const { upiId } = req.params; // Express automatically decodes %40 to @
  try {
    const flaskRes = await fetch('http://localhost:5005/predict-upi-fraud', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ upi_id: upiId })
    });
    const data = await flaskRes.json();
    if (!flaskRes.ok || data.error) {
      return res.status(400).json({ error: data.error || 'Failed to analyze UPI ID' });
    }
    // Only send required fields to frontend
    res.json({
      upiId: data.upi_id,
      fraudProbability: data.fraud_probability,
      riskLevel: data.risk_level,
      safetyStatus: data.safety_status,
      beneficiaryRecentFrauds: data.beneficiary_recent_frauds,
      payerRecentFrauds: data.payer_recent_frauds
    });
  } catch (err) {
    res.status(500).json({ error: 'Proxy UPI risk check failed', details: err.message });
  }
});
// --- END UPI RISK CHECK ENDPOINT ---

// --- BEGIN DUMMY AI VALIDATE UPI ENDPOINT ---
app.post('/api/ai/validate-upi', async (req, res) => {
  // Always return safe for now
  res.json({
    is_suspicious: false,
    confidence: 0.1,
    flags: [],
    recommendation: 'Safe UPI ID'
  });
});
// --- END DUMMY AI VALIDATE UPI ENDPOINT ---

// --- BEGIN DUMMY ENHANCED FRAUD DETECTION ENDPOINT ---
app.post('/api/enhanced-fraud-detection', async (req, res) => {
  // Always return safe for now
  res.json({
    prediction: false,
    confidence: 0.1,
    message: 'No fraud detected',
    features: {},
    live_data: {},
    meta: { service: 'dummy-enhanced-fraud', latency_ms: 10 }
  });
});
// --- END DUMMY ENHANCED FRAUD DETECTION ENDPOINT ---

const PORT = 6900;
app.listen(PORT, () => {
  console.log(`Node.js backend running on port ${PORT}`);
});

app.get('/test-flask', async (req, res) => {
  try {
    const flaskRes = await fetch('http://localhost:5005/predict-upi-fraud', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ upi_id: 'test@upi' })
    });
    const data = await flaskRes.json();
    res.json(data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});
