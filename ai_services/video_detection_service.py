from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from video_detection import ScamVideoDetector # Assuming video_detection.py is in the same directory
import os
import shutil
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Video Scam Detection API",
    description="API for analyzing videos for scam indicators using multi-modal AI."
)

# Initialize the scam video detector globally
# This will load models once when the service starts
try:
    video_detector = ScamVideoDetector()
    logger.info("ScamVideoDetector initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize ScamVideoDetector: {e}")
    video_detector = None # Ensure it's None if initialization fails

@app.get("/")
async def read_root():
    return {"message": "Video Scam Detection Service is running!"}

@app.post("/analyze-video")
async def analyze_video_endpoint(video_file: UploadFile = File(...)):
    if video_detector is None:
        raise HTTPException(status_code=500, detail="Video detector not initialized.")
    
    # Create a temporary file to save the uploaded video
    temp_file_path = f"temp_uploaded_video_{video_file.filename}"
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
        
        logger.info(f"Received video file: {video_file.filename}. Saved to {temp_file_path}")
        
        # Call the video analysis function
        # For simplicity, we are not passing audio_text here for now.
        # If needed, frontend can send it as another form field.
        analysis_results = video_detector.analyze_video(temp_file_path)
        
        logger.info(f"Video analysis completed for {video_file.filename}")
        return JSONResponse(content=analysis_results)
    except Exception as e:
        logger.error(f"Error processing video file {video_file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process video: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Cleaned up temporary file: {temp_file_path}")

if __name__ == "__main__":
    # You can specify the port here, e.g., 8083 for video service
    # Make sure this port is not conflicting with 8081 or 8082
    uvicorn.run(app, host="0.0.0.0", port=8083) 