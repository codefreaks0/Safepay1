import React, { useState } from 'react';
import { Users, MessageCircle, PlusCircle, ShieldCheck, X } from 'lucide-react';
import { Button } from '@/components/ui/button';

export default function Community() {
  const [showModal, setShowModal] = useState(false);
  const [title, setTitle] = useState('');
  const [message, setMessage] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    alert('Discussion posted! (Demo only)\nTitle: ' + title + '\nMessage: ' + message);
    setShowModal(false);
    setTitle('');
    setMessage('');
  };

  return (
    <div className="min-h-screen bg-gray-50 p-4 flex flex-col items-center">
      <div className="w-full max-w-xl">
        <div className="flex items-center mb-6">
          <Users className="text-blue-500 mr-2" size={28} />
          <h1 className="text-2xl font-bold">Community</h1>
        </div>
        <div className="bg-white rounded-lg shadow-sm p-4 mb-6">
          <h2 className="text-lg font-semibold mb-2 flex items-center">
            <ShieldCheck className="text-green-500 mr-2" size={18} />
            Welcome to the SafePay Community!
          </h2>
          <p className="text-gray-700 mb-2">
            Connect with other users, share your experiences, and help each other stay safe from scams and frauds. Please follow our community guidelines to keep this space helpful and respectful.
          </p>
          <ul className="list-disc pl-6 text-sm text-gray-600 mb-2">
            <li>Be respectful and supportive to all members.</li>
            <li>Do not share personal or sensitive information.</li>
            <li>Report suspicious activity to moderators.</li>
            <li>Stay on topic: fintech safety, scam awareness, and digital security.</li>
          </ul>
        </div>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-md font-semibold flex items-center">
            <MessageCircle className="mr-2 text-blue-400" size={18} />
            Recent Discussions
          </h3>
          <Button className="flex items-center gap-1 text-sm px-3 py-1.5" onClick={() => setShowModal(true)}>
            <PlusCircle size={16} /> Start New
          </Button>
        </div>
        <div className="space-y-3">
          <div className="bg-blue-50 rounded-lg p-3 flex flex-col">
            <span className="font-medium text-blue-800">How to spot a UPI scam?</span>
            <span className="text-xs text-gray-600 mt-1">12 replies • 2 hours ago</span>
          </div>
          <div className="bg-green-50 rounded-lg p-3 flex flex-col">
            <span className="font-medium text-green-800">Share your experience: QR code fraud</span>
            <span className="text-xs text-gray-600 mt-1">8 replies • 1 day ago</span>
          </div>
          <div className="bg-yellow-50 rounded-lg p-3 flex flex-col">
            <span className="font-medium text-yellow-800">Tips for safe online payments</span>
            <span className="text-xs text-gray-600 mt-1">5 replies • 3 days ago</span>
          </div>
        </div>
        <div className="mt-8 text-center text-xs text-gray-400">
          &copy; {new Date().getFullYear()} SafePay Community
        </div>
      </div>
      {/* Modal for new discussion */}
      {showModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-30">
          <div className="bg-white rounded-lg shadow-lg p-6 w-full max-w-md relative">
            <button className="absolute top-2 right-2 text-gray-400 hover:text-gray-600" onClick={() => setShowModal(false)}>
              <X size={20} />
            </button>
            <h2 className="text-lg font-semibold mb-4">Start a New Discussion</h2>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">Title</label>
                <input
                  type="text"
                  className="w-full border rounded px-3 py-2 focus:outline-none focus:ring focus:border-blue-300"
                  value={title}
                  onChange={e => setTitle(e.target.value)}
                  required
                  maxLength={80}
                  placeholder="Enter a short, clear title"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Message</label>
                <textarea
                  className="w-full border rounded px-3 py-2 focus:outline-none focus:ring focus:border-blue-300"
                  value={message}
                  onChange={e => setMessage(e.target.value)}
                  required
                  rows={4}
                  maxLength={500}
                  placeholder="Describe your question, tip, or experience..."
                />
              </div>
              <Button type="submit" className="w-full">Post Discussion</Button>
            </form>
          </div>
        </div>
      )}
    </div>
  );
} 